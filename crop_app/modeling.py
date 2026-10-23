from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

FEATURES = ["N", "P", "K", "ph"]
TARGET = "label"
DATASET_URL = "https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset"
DATASET_METADATA_URL = (
    "https://www.kaggle.com/api/v1/datasets/view/atharvaingle/"
    "crop-recommendation-dataset"
)
DATASET_LICENSE = "Apache 2.0 (as declared on Kaggle; upstream provenance is limited)"
LIMITATIONS = [
    "The dataset publisher says it was assembled from Indian rainfall, climate, and "
    "fertilizer datasets, but does not identify a complete source-level lineage.",
    "The dataset does not document the time scale for rainfall, temperature, or humidity. "
    "These weather columns are excluded from this soil-only model.",
    "The N/P/K fields are described as ratios without units; model inputs must use the "
    "dataset's numeric scale and training ranges.",
    "Dataset labels and test performance do not establish real-world agronomic reliability "
    "or suitability outside the dataset's coverage.",
    "Weather displayed by the app is contextual only and is not used to rank crops.",
]


def load_dataset(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {*FEATURES, TARGET}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    data = data.copy()
    for feature in FEATURES:
        converted = pd.to_numeric(data[feature], errors="coerce")
        invalid_values = data[feature].notna() & converted.isna()
        if invalid_values.any():
            raise ValueError(
                f"Feature {feature} contains non-numeric values; inspect the source CSV."
            )
        data[feature] = converted
    data[TARGET] = data[TARGET].astype("string").str.strip().replace("", pd.NA)
    data = data.dropna(subset=[TARGET])
    if data.empty:
        raise ValueError("No labeled rows remain in the dataset.")

    feature_values = data[FEATURES].to_numpy(dtype=float)
    if np.isinf(feature_values).any():
        raise ValueError("Dataset contains infinite soil feature values.")
    invalid_ph = data["ph"].notna() & ~data["ph"].between(0, 14)
    negative_soil = data[["N", "P", "K"]].lt(0).any(axis=1)
    if invalid_ph.any() or negative_soil.any():
        raise ValueError(
            "Dataset contains invalid soil values (pH outside 0–14 or negative N/P/K). "
            "Inspect and resolve these rows before training."
        )
    empty_features = data[FEATURES].columns[data[FEATURES].isna().all()].tolist()
    if empty_features:
        raise ValueError(
            "No usable values were found for soil features: "
            + ", ".join(empty_features)
        )
    return data.drop_duplicates(subset=[*FEATURES, TARGET]).reset_index(drop=True)


def split_dataset(
    data: pd.DataFrame, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    x = data[FEATURES]
    y = data[TARGET].astype(str)
    groups = x.groupby(FEATURES, dropna=False, sort=False).ngroup()
    outer_split = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=random_state
    )
    development_indices, test_indices = next(outer_split.split(x, y, groups))
    x_development = x.iloc[development_indices]
    y_development = y.iloc[development_indices]
    development_groups = groups.iloc[development_indices]

    validation_split = StratifiedGroupKFold(
        n_splits=4, shuffle=True, random_state=random_state + 1
    )
    train_indices, validation_indices = next(
        validation_split.split(x_development, y_development, development_groups)
    )
    x_train = x_development.iloc[train_indices]
    x_validation = x_development.iloc[validation_indices]
    x_test = x.iloc[test_indices]
    y_train = y_development.iloc[train_indices]
    y_validation = y_development.iloc[validation_indices]
    y_test = y.iloc[test_indices]
    return x_train, x_validation, x_test, y_train, y_validation, y_test


def build_candidates(random_state: int = 42) -> dict[str, Pipeline]:
    def pipeline(classifier: Any, scale: bool = False) -> Pipeline:
        steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
        if scale:
            steps.append(("scaler", StandardScaler()))
        steps.append(("classifier", classifier))
        return Pipeline(steps)

    return {
        "Logistic Regression": pipeline(
            LogisticRegression(max_iter=2000, random_state=random_state), scale=True
        ),
        "Decision Tree": pipeline(
            DecisionTreeClassifier(random_state=random_state)
        ),
        "Random Forest": pipeline(
            RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=1)
        ),
    }


def compare_candidates(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    candidates = build_candidates(random_state)
    rows: list[dict[str, float | str]] = []
    for name, estimator in candidates.items():
        estimator.fit(x_train, y_train)
        prediction = estimator.predict(x_validation)
        rows.append(
            {
                "model": name,
                "validation_accuracy": accuracy_score(y_validation, prediction),
                "validation_macro_f1": f1_score(
                    y_validation, prediction, average="macro", zero_division=0
                ),
            }
        )
    results = pd.DataFrame(rows).sort_values(
        ["validation_macro_f1", "validation_accuracy", "model"],
        ascending=[False, False, True],
    )
    return results.reset_index(drop=True), candidates


def calibrate_selected_model(
    estimator: Pipeline, x_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42
) -> CalibratedClassifierCV:
    groups = x_train.groupby(FEATURES, dropna=False, sort=False).ngroup()
    group_counts = pd.DataFrame(
        {"label": y_train.to_numpy(), "group": groups.to_numpy()}
    ).drop_duplicates().groupby("label")["group"].nunique()
    if group_counts.empty or int(group_counts.min()) < 5:
        raise ValueError(
            "Sigmoid calibration requires at least five distinct soil-feature groups "
            "per crop label in the training partition."
        )
    calibration_folds = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=random_state
    )
    calibration_splits = list(
        calibration_folds.split(x_train, y_train, groups)
    )
    calibrated = CalibratedClassifierCV(
        estimator=estimator,
        method="sigmoid",
        cv=calibration_splits,
    )
    return calibrated.fit(x_train, y_train)


def calibration_diagnostics(
    model: Any,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    bins: int = 10,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    probabilities = model.predict_proba(x_test)
    classes = list(model.classes_)
    label_codes = pd.Categorical(y_test, categories=classes).codes
    if (label_codes < 0).any():
        raise ValueError("Test labels contain classes unknown to the calibrated model.")

    one_hot = np.eye(len(classes))[label_codes]
    confidences = probabilities.max(axis=1)
    predictions = np.asarray(classes)[probabilities.argmax(axis=1)]
    correct = (predictions == np.asarray(y_test)).astype(float)
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    expected_calibration_error = 0.0
    rows: list[dict[str, float | int]] = []
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (confidences >= lower) & (
            (confidences < upper) if upper < 1 else (confidences <= upper)
        )
        if in_bin.any():
            mean_confidence = float(confidences[in_bin].mean())
            observed_accuracy = float(correct[in_bin].mean())
            expected_calibration_error += float(in_bin.mean()) * abs(
                mean_confidence - observed_accuracy
            )
            rows.append(
                {
                    "mean confidence": mean_confidence,
                    "observed accuracy": observed_accuracy,
                    "count": int(in_bin.sum()),
                }
            )

    diagnostics: dict[str, float | int] = {
        "log_loss": float(log_loss(y_test, probabilities, labels=classes)),
        "multiclass_brier": float(
            np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))
        ),
        "top_label_ece": float(expected_calibration_error),
        "ece_bins": bins,
    }
    return diagnostics, pd.DataFrame(
        rows, columns=["mean confidence", "observed accuracy", "count"]
    )


def make_metadata(
    selected_model: str,
    x_train: pd.DataFrame,
    calibration_metrics: dict[str, float | int],
    source_data: pd.DataFrame | None = None,
) -> dict[str, Any]:
    ranges = {
        feature: {
            "min": float(x_train[feature].min()),
            "max": float(x_train[feature].max()),
        }
        for feature in FEATURES
    }
    medians = {
        feature: float(x_train[feature].median()) for feature in FEATURES
    }
    crop_profiles: dict[str, dict[str, float | int]] = {}
    if source_data is not None and TARGET in source_data.columns:
        profile_features = [
            feature
            for feature in (*FEATURES, "temperature", "rainfall")
            if feature in source_data.columns
        ]
        for crop, crop_rows in source_data.groupby(TARGET, observed=True):
            profile: dict[str, float | int] = {
                "sample_count": int(len(crop_rows))
            }
            for feature in profile_features:
                profile[feature] = float(crop_rows[feature].median())
            crop_profiles[str(crop)] = profile
    return {
        "feature_order": FEATURES,
        "training_ranges": ranges,
        "training_medians": medians,
        "crop_profiles": crop_profiles,
        "selected_model": selected_model,
        "dataset_url": DATASET_URL,
        "dataset_metadata_url": DATASET_METADATA_URL,
        "dataset_version": 1,
        "dataset_license": DATASET_LICENSE,
        "random_state": 42,
        "environment": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "weather_is_model_input": False,
        "limitations": LIMITATIONS,
        "calibration_diagnostics": calibration_metrics,
        "probability_note": (
            "The score is the model's sigmoid-calibrated class probability among "
            "the training dataset's crop labels. It is not a probability of crop "
            "success or a measure of agronomic suitability. On the held-out test "
            "split, top-label expected calibration error was "
            f"{calibration_metrics['top_label_ece']:.3f} using "
            f"{calibration_metrics['ece_bins']} fixed-width bins, so interpret "
            "percentages cautiously."
        ),
        "explanation_note": (
            "Local factors compare the selected crop's predicted probability at the "
            "submitted values with the probability after replacing one feature at a "
            "time by its training median. This is an association/sensitivity aid, "
            "not a causal explanation."
        ),
    }


def save_artifacts(
    model: Any, metadata: dict[str, Any], output_dir: str | Path
) -> tuple[Path, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    model_path = destination / "crop_model.joblib"
    metadata_path = destination / "model_metadata.json"
    joblib.dump(model, model_path)
    import json

    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return model_path, metadata_path


def evaluate_model(
    model: Any, x_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, Any]:
    predictions = model.predict(x_test)
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(y_test, predictions, average="macro", zero_division=0),
        "classification_report": classification_report(
            y_test, predictions, zero_division=0
        ),
        "predictions": predictions,
    }

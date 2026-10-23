from __future__ import annotations

import argparse
from pathlib import Path
import sys

from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from crop_app.modeling import (
    calibrate_selected_model,
    calibration_diagnostics,
    compare_candidates,
    evaluate_model,
    load_dataset,
    make_metadata,
    save_artifacts,
    split_dataset,
)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the soil-only crop classification pipeline."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "Crop_recommendation.csv",
        help="Path to the downloaded Kaggle CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts",
        help="Directory for the model and metadata.",
    )
    args = parser.parse_args()

    data = load_dataset(args.data)
    x_train, x_validation, x_test, y_train, y_validation, y_test = split_dataset(data)
    validation_results, candidates = compare_candidates(
        x_train, y_train, x_validation, y_validation
    )
    print("Validation results (model selection uses macro-F1):")
    print(validation_results.to_string(index=False))
    selected_name = str(validation_results.iloc[0]["model"])

    calibrated = calibrate_selected_model(
        candidates[selected_name], x_train, y_train
    )
    evaluation = evaluate_model(calibrated, x_test, y_test)
    calibration_metrics, _ = calibration_diagnostics(calibrated, x_test, y_test)
    print(f"\nSelected model: {selected_name}")
    print(f"Held-out test accuracy: {evaluation['accuracy']:.4f}")
    print(f"Held-out test macro-F1: {evaluation['macro_f1']:.4f}")
    print(f"Held-out test log loss: {calibration_metrics['log_loss']:.4f}")
    print(
        "Held-out test multiclass Brier score: "
        f"{calibration_metrics['multiclass_brier']:.4f}"
    )
    print(
        "Held-out test top-label ECE "
        f"({calibration_metrics['ece_bins']} bins): "
        f"{calibration_metrics['top_label_ece']:.4f}"
    )
    print("\nHeld-out test classification report:")
    print(evaluation["classification_report"])
    labels = list(calibrated.classes_)
    print("\nHeld-out test confusion matrix (label order below):")
    print(labels)
    print(confusion_matrix(y_test, evaluation["predictions"], labels=labels))

    metadata = make_metadata(
        selected_name, x_train, calibration_metrics, source_data=data
    )
    model_path, metadata_path = save_artifacts(calibrated, metadata, args.output)
    print(f"\nSaved model: {model_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()

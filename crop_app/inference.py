from __future__ import annotations

from typing import Any

import pandas as pd

from crop_app.validation import FEATURES


def rank_crops(
    model: Any, values: dict[str, float], top_k: int = 5
) -> list[dict[str, float | str]]:
    row = pd.DataFrame([[values[name] for name in FEATURES]], columns=FEATURES)
    probabilities = model.predict_proba(row)[0]
    ranked_indices = probabilities.argsort()[::-1][:top_k]
    return [
        {
            "crop": str(model.classes_[index]),
            "probability": float(probabilities[index]),
        }
        for index in ranked_indices
    ]


def explain_top_crop(
    model: Any, values: dict[str, float], training_medians: dict[str, float]
) -> list[dict[str, float | str]]:
    top_crop = rank_crops(model, values, top_k=1)[0]["crop"]
    class_index = list(model.classes_).index(top_crop)
    original = pd.DataFrame([[values[name] for name in FEATURES]], columns=FEATURES)
    original_probability = float(model.predict_proba(original)[0][class_index])

    factors: list[dict[str, float | str]] = []
    for feature in FEATURES:
        ablated = original.copy()
        ablated.loc[0, feature] = float(training_medians[feature])
        baseline_probability = float(model.predict_proba(ablated)[0][class_index])
        factors.append(
            {
                "feature": feature,
                "change": original_probability - baseline_probability,
                "submitted_value": values[feature],
                "training_median": float(training_medians[feature]),
            }
        )
    return sorted(factors, key=lambda factor: abs(float(factor["change"])), reverse=True)

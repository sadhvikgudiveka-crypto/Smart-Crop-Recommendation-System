from __future__ import annotations

import math
from collections.abc import Mapping

FEATURES = ("N", "P", "K", "ph")


class InputValidationError(ValueError):
    """Raised when submitted soil values cannot be used by the model."""


def validate_soil_inputs(
    values: Mapping[str, object], metadata: Mapping[str, object]
) -> dict[str, float]:
    ranges = metadata.get("training_ranges")
    if not isinstance(ranges, Mapping):
        raise ValueError("Model metadata is missing training feature ranges.")

    validated: dict[str, float] = {}
    issues: list[str] = []
    for feature in FEATURES:
        raw_value = values.get(feature)
        try:
            number = float(raw_value)
        except (TypeError, ValueError):
            issues.append(f"{feature} is required and must be a number.")
            continue

        if not math.isfinite(number):
            issues.append(f"{feature} must be a finite number.")
            continue
        if feature == "ph" and not 0 <= number <= 14:
            issues.append("pH must be between 0 and 14.")
            continue
        if feature in ("N", "P", "K") and number < 0:
            issues.append(f"{feature} cannot be negative.")
            continue

        feature_range = ranges.get(feature)
        if not isinstance(feature_range, Mapping):
            raise ValueError(f"Model metadata has no training range for {feature}.")
        minimum = float(feature_range["min"])
        maximum = float(feature_range["max"])
        if number < minimum or number > maximum:
            issues.append(
                f"{feature}={number:g} is outside the training data range "
                f"({minimum:g}–{maximum:g}); the model cannot reliably extrapolate."
            )
            continue
        validated[feature] = number

    if issues:
        raise InputValidationError(" ".join(issues))
    return validated

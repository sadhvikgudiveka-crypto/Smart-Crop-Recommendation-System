import unittest

from crop_app.validation import InputValidationError, validate_soil_inputs


METADATA = {
    "training_ranges": {
        "N": {"min": 0, "max": 140},
        "P": {"min": 5, "max": 145},
        "K": {"min": 5, "max": 205},
        "ph": {"min": 3.5, "max": 10},
    }
}


class SoilValidationTests(unittest.TestCase):
    def test_accepts_complete_values_inside_training_ranges(self) -> None:
        values = {"N": 40, "P": 50, "K": 60, "ph": 6.5}
        self.assertEqual(validate_soil_inputs(values, METADATA), values)

    def test_rejects_missing_and_non_finite_values(self) -> None:
        with self.assertRaises(InputValidationError):
            validate_soil_inputs(
                {"N": None, "P": 50, "K": float("inf"), "ph": 6.5}, METADATA
            )

    def test_rejects_p_h_outside_physical_domain(self) -> None:
        with self.assertRaisesRegex(InputValidationError, "pH must be between 0 and 14"):
            validate_soil_inputs(
                {"N": 40, "P": 50, "K": 60, "ph": 14.1}, METADATA
            )

    def test_rejects_values_outside_training_range(self) -> None:
        with self.assertRaisesRegex(InputValidationError, "outside the training data range"):
            validate_soil_inputs(
                {"N": 141, "P": 50, "K": 60, "ph": 6.5}, METADATA
            )


if __name__ == "__main__":
    unittest.main()

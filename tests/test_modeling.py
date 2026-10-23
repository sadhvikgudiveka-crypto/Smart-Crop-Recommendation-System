import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from crop_app.modeling import (
    FEATURES,
    calibration_diagnostics,
    load_dataset,
    make_metadata,
    split_dataset,
)


class ModelingDataTests(unittest.TestCase):
    def test_metadata_contains_descriptive_per_crop_medians_from_source_data(self) -> None:
        source_data = pd.DataFrame(
            [
                {"N": 10, "P": 20, "K": 30, "ph": 6.0, "temperature": 22.0, "rainfall": 80.0, "label": "rice"},
                {"N": 30, "P": 40, "K": 50, "ph": 8.0, "temperature": 26.0, "rainfall": 120.0, "label": "rice"},
            ]
        )
        metadata = make_metadata(
            "test",
            source_data[["N", "P", "K", "ph"]],
            {"top_label_ece": 0.1, "ece_bins": 10},
            source_data=source_data,
        )
        self.assertEqual(
            metadata["crop_profiles"]["rice"],
            {
                "sample_count": 2,
                "N": 20.0,
                "P": 30.0,
                "K": 40.0,
                "ph": 7.0,
                "temperature": 24.0,
                "rainfall": 100.0,
            },
        )
        self.assertIn("not a probability of crop success", metadata["probability_note"])

    def test_loader_removes_repeated_model_rows_and_rejects_invalid_features(self) -> None:
        rows = [
            {"N": 10, "P": 20, "K": 30, "ph": 6, "label": " crop_a ", "rainfall": 3},
            {"N": 10, "P": 20, "K": 30, "ph": 6, "label": "crop_a", "rainfall": 4},
            {"N": 10, "P": 20, "K": 30, "ph": 6, "label": "crop_b", "rainfall": 5},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            loaded = load_dataset(path)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(set(loaded["label"]), {"crop_a", "crop_b"})

            pd.DataFrame([{**rows[0], "N": "not numeric"}]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "non-numeric values"):
                load_dataset(path)

    def test_grouped_split_does_not_split_identical_soil_vectors(self) -> None:
        rows = []
        for class_index, label in enumerate(("crop_a", "crop_b", "crop_c")):
            for value in range(25):
                row = {
                    "N": class_index * 100 + value,
                    "P": 10 + value,
                    "K": 20 + value,
                    "ph": 4 + value / 10,
                    "label": label,
                }
                rows.extend([row, row.copy()])

        data = pd.DataFrame(rows)
        x_train, x_validation, x_test, _, _, _ = split_dataset(data)
        groups = [
            set(map(tuple, partition[FEATURES].to_numpy()))
            for partition in (x_train, x_validation, x_test)
        ]
        self.assertFalse(groups[0] & groups[1])
        self.assertFalse(groups[0] & groups[2])
        self.assertFalse(groups[1] & groups[2])

    def test_calibration_diagnostics_computes_top_label_error(self) -> None:
        class ProbabilityModel:
            classes_ = np.array(["crop_a", "crop_b"])

            def predict_proba(self, _rows):
                return np.array([[0.8, 0.2], [0.3, 0.7]])

        features = pd.DataFrame({"N": [1, 2], "P": [1, 2], "K": [1, 2], "ph": [6, 7]})
        labels = pd.Series(["crop_a", "crop_a"])
        metrics, table = calibration_diagnostics(ProbabilityModel(), features, labels)

        self.assertAlmostEqual(metrics["top_label_ece"], 0.45)
        self.assertEqual(int(table["count"].sum()), 2)


if __name__ == "__main__":
    unittest.main()

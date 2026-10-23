import unittest

import numpy as np

from crop_app.inference import explain_top_crop, rank_crops


class FakeModel:
    classes_ = np.array(["maize", "rice"])

    def predict_proba(self, rows):
        first_value = rows.iloc[0]["N"]
        return np.array([[0.8, 0.2]]) if first_value > 20 else np.array([[0.3, 0.7]])


class InferenceTests(unittest.TestCase):
    def test_rankings_are_sorted_by_probability(self) -> None:
        result = rank_crops(FakeModel(), {"N": 30, "P": 1, "K": 1, "ph": 6})
        self.assertEqual([item["crop"] for item in result[:2]], ["maize", "rice"])

    def test_top_k_never_returns_more_available_classes(self) -> None:
        result = rank_crops(
            FakeModel(), {"N": 30, "P": 1, "K": 1, "ph": 6}, top_k=3
        )
        self.assertEqual(len(result), 2)

    def test_local_explanation_reports_ablation_change(self) -> None:
        result = explain_top_crop(
            FakeModel(),
            {"N": 30, "P": 1, "K": 1, "ph": 6},
            {"N": 10, "P": 1, "K": 1, "ph": 6},
        )
        nitrogen = next(item for item in result if item["feature"] == "N")
        self.assertAlmostEqual(nitrogen["change"], 0.5)


if __name__ == "__main__":
    unittest.main()

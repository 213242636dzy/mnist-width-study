import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ResultConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = pd.read_csv(PROJECT_ROOT / "results" / "raw_results.csv")
        cls.summary = pd.read_csv(
            PROJECT_ROOT / "results" / "summary_results.csv"
        )
        cls.combined = pd.read_csv(
            PROJECT_ROOT / "results" / "combined_results.csv"
        )

    def test_widths_and_seed_counts(self):
        self.assertEqual(
            sorted(self.raw["hidden_size"].unique().tolist()),
            [4, 8, 16, 32, 64, 128, 256],
        )
        seed_counts = self.raw.groupby("hidden_size")["seed"].nunique()
        self.assertTrue((seed_counts == 3).all())

    def test_resource_formulas(self):
        expected_parameters = 795 * self.raw["hidden_size"] + 10
        expected_macs = 794 * self.raw["hidden_size"]
        self.assertTrue(
            (self.raw["parameter_count"] == expected_parameters).all()
        )
        self.assertTrue((self.raw["macs"] == expected_macs).all())

    def test_reported_working_points(self):
        h32 = self.combined.loc[
            self.combined["hidden_size"] == 32
        ].iloc[0]
        h64 = self.combined.loc[
            self.combined["hidden_size"] == 64
        ].iloc[0]

        self.assertAlmostEqual(h32["accuracy_drop_pp"], 0.933333, places=5)
        self.assertAlmostEqual(
            h32["macs_reduction_percent"], 87.5, places=6
        )
        self.assertAlmostEqual(h64["accuracy_drop_pp"], 0.5, places=6)
        self.assertAlmostEqual(
            h64["macs_reduction_percent"], 75.0, places=6
        )


if __name__ == "__main__":
    unittest.main()

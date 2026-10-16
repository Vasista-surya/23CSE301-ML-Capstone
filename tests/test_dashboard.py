"""Streamlit-facing regression checks for the dashboard shell."""

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

import streamlit_app as app


PROJECT_DIR = Path(__file__).resolve().parents[1]


class DashboardSmokeTests(unittest.TestCase):
    def test_regression_normalizer_keeps_cv_metrics_distinct(self) -> None:
        frame = pd.DataFrame(
            {
                "Model": ["Gradient Boosting Regressor"],
                "R2": [0.9],
                "RMSE": [1.0],
                "MAE": [0.8],
                "CV_R2_Mean": [0.88],
                "CV_R2_Std": [0.02],
            }
        )
        normalized = app._normalise_metric_columns(frame)
        self.assertEqual(
            list(normalized.columns),
            ["Algorithm", "R2", "RMSE", "MAE", "Mean CV R2", "Std CV R2"],
        )
        self.assertEqual(normalized.columns.duplicated().sum(), 0)

    def test_all_dashboard_pages_render_without_streamlit_exceptions(self) -> None:
        pages = [
            "Project Overview",
            "Regression Lab",
            "Classification Lab",
            "Clustering Lab",
            "Model Comparison",
            "Dataset Explorer",
        ]
        test_app = AppTest.from_file(str(PROJECT_DIR / "streamlit_app.py"))
        test_app.run(timeout=120)
        for page in pages:
            test_app.radio[0].set_value(page)
            test_app.run(timeout=120)
            self.assertFalse(
                test_app.exception,
                f"{page} raised: {[str(error.value) for error in test_app.exception]}",
            )


if __name__ == "__main__":
    unittest.main()

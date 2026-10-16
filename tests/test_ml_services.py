"""Backend contract and smoke tests.

These tests deliberately exercise the public service layer with the real CSV
files.  The first result request may generate local joblib artifacts under
``models/``; subsequent requests use the compatible cached artifacts.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import ml_services as services


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "datasets"


class BackendContractTests(unittest.TestCase):
    def test_import_and_dataset_discovery(self) -> None:
        summary = services.get_project_summary()
        self.assertEqual(summary["status"], "ok")
        self.assertTrue(summary["datasets"]["fifa.csv"]["discoverable"])
        self.assertTrue(summary["datasets"]["nutrition.csv"]["discoverable"])
        self.assertTrue(summary["datasets"]["mall_customers.csv"]["discoverable"])

    def test_preview_contract(self) -> None:
        result = services.get_dataset_preview("nutrition", n_rows=3)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["data"]), 3)
        self.assertIn("category", result["columns"])
        self.assertEqual(services.get_dataset_preview("does-not-exist")["status"], "error")

    def test_all_result_tables_have_stable_contracts(self) -> None:
        regression = services.get_regression_results()
        classification = services.get_classification_results()
        clustering = services.get_clustering_results()
        self.assertEqual(regression["status"], "ok", regression.get("error"))
        self.assertEqual(classification["status"], "ok", classification.get("error"))
        self.assertEqual(clustering["status"], "ok", clustering.get("error"))
        self.assertTrue(
            {"Model", "R2", "RMSE", "MAE"}.issubset(regression["comparison"].columns)
        )
        self.assertTrue(
            {
                "Model",
                "Accuracy",
                "Precision_weighted",
                "Recall_weighted",
                "F1_weighted",
                "ROC_AUC_OvR",
            }.issubset(classification["comparison"].columns)
        )
        self.assertTrue(
            {
                "Algorithm",
                "Silhouette_Score",
                "Davies_Bouldin_Index",
                "Calinski_Harabasz_Index",
                "n_clusters",
            }.issubset(clustering["comparison"].columns)
        )

    def test_all_ten_regressors_have_numeric_results(self) -> None:
        result = services.get_regression_results()
        self.assertEqual(result["status"], "ok", result.get("error"))
        self.assertEqual(len(result["comparison"]), 10)
        self.assertTrue(np.isfinite(result["comparison"][["R2", "RMSE", "MAE"]].to_numpy()).all())

    def test_all_ten_classifiers_have_metrics_and_confusion_matrices(self) -> None:
        result = services.get_classification_results()
        self.assertEqual(result["status"], "ok", result.get("error"))
        self.assertEqual(len(result["comparison"]), 10)
        self.assertTrue(
            np.isfinite(
                result["comparison"]
                [["Accuracy", "Precision_weighted", "Recall_weighted", "F1_weighted", "ROC_AUC_OvR"]]
                .to_numpy()
            ).all()
        )
        self.assertEqual(len(result["confusion_matrices"]), 11)
        for matrix in result["confusion_matrices"].values():
            self.assertEqual(matrix.shape, (len(result["class_labels"]), len(result["class_labels"])))

    def test_clustering_outputs_and_internal_metrics(self) -> None:
        result = services.get_clustering_results()
        self.assertEqual(result["status"], "ok", result.get("error"))
        self.assertEqual(len(result["comparison"]), 2)
        self.assertEqual(set(result["comparison"]["n_clusters"]), {6})
        self.assertEqual(len(result["elbow"]), 9)
        self.assertEqual(result["pca_coordinates"].shape[1], 4)
        self.assertEqual(result["tsne_coordinates"].shape[1], 3)
        self.assertTrue(np.isfinite(result["comparison"].select_dtypes("number").to_numpy()).all())

    def test_inference_reuses_feature_engineering_and_preprocessing(self) -> None:
        fifa_row = pd.read_csv(DATA_DIR / "fifa.csv").iloc[0].to_dict()
        nutrition_row = pd.read_csv(DATA_DIR / "nutrition.csv").iloc[0].to_dict()
        customer_row = pd.read_csv(DATA_DIR / "mall_customers.csv").iloc[0].to_dict()

        regression = services.predict_regression(fifa_row)
        self.assertEqual(regression["status"], "ok", regression.get("error"))
        self.assertTrue(np.isfinite(regression["prediction"]))
        stringified_fifa = {key: str(value) for key, value in fifa_row.items()}
        stringified_regression = services.predict_regression(stringified_fifa, "Polynomial Regression")
        self.assertEqual(stringified_regression["status"], "ok", stringified_regression.get("error"))

        classification = services.predict_classification(nutrition_row, "Bagging")
        self.assertEqual(classification["status"], "ok", classification.get("error"))
        self.assertIn(classification["predicted_class"], classification["class_order"])
        self.assertTrue(classification["probability_supported"])
        self.assertAlmostEqual(sum(classification["probabilities"].values()), 1.0, places=6)
        stringified_nutrition = {key: str(value) for key, value in nutrition_row.items()}
        stringified_classification = services.predict_classification(stringified_nutrition, "Random Forest")
        self.assertEqual(stringified_classification["status"], "ok", stringified_classification.get("error"))

        kmeans = services.assign_cluster(customer_row, "KMeans")
        agglomerative = services.assign_cluster(customer_row, "Agglomerative")
        self.assertEqual(kmeans["status"], "ok", kmeans.get("error"))
        self.assertEqual(agglomerative["status"], "ok", agglomerative.get("error"))
        self.assertIn(kmeans["cluster"], range(6))
        self.assertIn(agglomerative["cluster"], range(6))
        self.assertIn("nearest centroid", agglomerative["assignment_strategy"])

    def test_invalid_inputs_fail_gracefully(self) -> None:
        self.assertEqual(services.predict_regression({})["status"], "error")
        self.assertEqual(services.predict_classification({}, "not-a-model")["status"], "error")
        self.assertEqual(
            services.assign_cluster({"Age": 30}, "not-an-algorithm")["status"], "error"
        )


if __name__ == "__main__":
    unittest.main()

"""Stable backend services for the 23CSE301 capstone dashboard.

The UI should call only the functions in this module.  Result services return
small dictionaries containing pandas objects and metadata; prediction
services return dictionaries that are safe to render directly in Streamlit.
Model training is lazy and cached by :mod:`model_utils`, so a Streamlit rerun
does not refit models when the generated artifact is still compatible with the
installed scikit-learn version.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from model_utils import (
    CLASSIFICATION_MODEL_NAMES,
    CLASSIFICATION_TUNED_NAME,
    CLUSTER_ALGORITHMS,
    CLUSTER_FEATURES,
    DATA_DIR,
    MODEL_DIR,
    NUTRITION_FEATURES,
    REGRESSION_FEATURES,
    REGRESSION_TARGET,
    ModelInputError,
    get_classification_bundle,
    get_clustering_bundle,
    get_regression_bundle,
    prepare_classification_input,
    prepare_cluster_input,
    prepare_regression_input,
)


def _ok(**values: Any) -> dict[str, Any]:
    return {"status": "ok", **values}


def _error(message: str, **values: Any) -> dict[str, Any]:
    return {"status": "error", "error": message, **values}


def _csv_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def _canonical_regression_name(model_name: Optional[str], selected: str) -> str:
    if not model_name:
        return selected
    aliases = {
        "Linear Regression": "Linear Regression",
        "Ridge Regression": "Ridge Regression",
        "Lasso Regression": "Lasso Regression",
        "ElasticNet Regression": "ElasticNet Regression",
        "Polynomial Regression": "Polynomial Regression",
        "Decision Tree Regressor": "Decision Tree Regressor",
        "Random Forest Regressor": "Random Forest Regressor",
        "Gradient Boosting Regressor": "Gradient Boosting Regressor",
        "Support Vector Regressor": "Support Vector Regressor",
        "K-Nearest Neighbors Regressor": "K-Nearest Neighbors Regressor",
        "Tuned Random Forest Regressor": "Tuned Random Forest Regressor",
        "Tuned Gradient Boosting Regressor": "Tuned Gradient Boosting Regressor",
        # Friendly notebook labels are accepted without making them the
        # canonical names returned by the comparison table.
        "1. Linear Regression": "Linear Regression",
        "2. Ridge Regression": "Ridge Regression",
        "3. Lasso Regression": "Lasso Regression",
        "4. ElasticNet Regression": "ElasticNet Regression",
        "5. Polynomial Regression": "Polynomial Regression",
        "6. Decision Tree Regressor": "Decision Tree Regressor",
        "7. Random Forest Regressor": "Random Forest Regressor",
        "8. Gradient Boosting Regressor": "Gradient Boosting Regressor",
        "9. Support Vector Regressor": "Support Vector Regressor",
        "10. KNN Regressor": "K-Nearest Neighbors Regressor",
        "KNN Regressor": "K-Nearest Neighbors Regressor",
    }
    if model_name not in aliases:
        raise ModelInputError(f"Unknown regression model: {model_name}")
    return aliases[model_name]


def _canonical_classification_name(model_name: Optional[str], selected: str) -> str:
    if not model_name:
        return selected
    aliases = {
        "Logistic Regression": "Logistic Regression",
        "K-Nearest Neighbors": "K-Nearest Neighbors",
        "KNN": "K-Nearest Neighbors",
        "Gaussian Naive Bayes": "Gaussian Naive Bayes",
        "Decision Tree Classifier": "Decision Tree Classifier",
        "Decision Tree": "Decision Tree Classifier",
        "Support Vector Classifier": "Support Vector Classifier",
        "SVC": "Support Vector Classifier",
        "Random Forest Classifier": "Random Forest Classifier",
        "Random Forest": "Random Forest Classifier",
        "AdaBoost Classifier": "AdaBoost Classifier",
        "AdaBoost": "AdaBoost Classifier",
        "Gradient Boosting Classifier": "Gradient Boosting Classifier",
        "Gradient Boosting": "Gradient Boosting Classifier",
        "Bagging Classifier": "Bagging Classifier",
        "Bagging": "Bagging Classifier",
        "MLP Classifier": "MLP Classifier",
        "Tuned Bagging Classifier": "Tuned Bagging Classifier",
        "Tuned Bagging": "Tuned Bagging Classifier",
    }
    if model_name in aliases:
        return aliases[model_name]
    # Accept the exact numbered labels from the notebook as a convenience.
    numbered = {
        "1. Logistic Regression": "Logistic Regression",
        "2. K-Nearest Neighbors": "K-Nearest Neighbors",
        "3. Gaussian Naive Bayes": "Gaussian Naive Bayes",
        "4. Decision Tree": "Decision Tree Classifier",
        "5. Support Vector Classifier": "Support Vector Classifier",
        "6. Random Forest": "Random Forest Classifier",
        "7. AdaBoost": "AdaBoost Classifier",
        "8. Gradient Boosting": "Gradient Boosting Classifier",
        "9. Bagging": "Bagging Classifier",
        "10. MLP Classifier": "MLP Classifier",
    }
    if model_name not in numbered:
        raise ModelInputError(f"Unknown classification model: {model_name}")
    return numbered[model_name]


def _canonical_cluster_algorithm(algorithm_name: str) -> str:
    aliases = {
        "K-Means": "K-Means",
        "KMeans": "K-Means",
        "K-means": "K-Means",
        "Agglomerative Hierarchical (Ward)": "Agglomerative Hierarchical (Ward)",
        "Agglomerative Hierarchical": "Agglomerative Hierarchical (Ward)",
        "Agglomerative Hierarchical Clustering": "Agglomerative Hierarchical (Ward)",
        "Agglomerative Clustering (Ward)": "Agglomerative Hierarchical (Ward)",
        "Agglomerative": "Agglomerative Hierarchical (Ward)",
        "Agglomerative Clustering": "Agglomerative Hierarchical (Ward)",
    }
    if algorithm_name not in aliases:
        raise ModelInputError(f"Unknown clustering algorithm: {algorithm_name}")
    return aliases[algorithm_name]


def get_project_summary() -> dict[str, Any]:
    """Return source datasets, exact task definitions, and artifact status."""

    datasets = {}
    for name in ["fifa.csv", "nutrition.csv", "mall_customers.csv"]:
        path = DATA_DIR / name
        if path.exists():
            try:
                frame = pd.read_csv(path, nrows=0)
                rows = _csv_row_count(path)
                datasets[name] = {
                    "path": str(path),
                    "rows": max(rows, 0),
                    "columns": list(frame.columns),
                    "discoverable": True,
                }
            except Exception as exc:
                datasets[name] = {"path": str(path), "discoverable": False, "error": str(exc)}
        else:
            datasets[name] = {"path": str(path), "discoverable": False}
    return _ok(
        project="23CSE301 Machine Learning Capstone Dashboard",
        source_of_truth={
            "regression": "notebooks/fifa-performance-regression.ipynb",
            "classification": "notebooks/nutrition-classification.ipynb",
            "clustering": "notebooks/clustering.ipynb",
            "original_datasets_unchanged": True,
        },
        datasets=datasets,
        artifacts={
            "directory": str(MODEL_DIR),
            "regression_generated": (MODEL_DIR / "regression_bundle.joblib").exists(),
            "classification_generated": (MODEL_DIR / "classification_bundle.joblib").exists(),
            "clustering_generated": (MODEL_DIR / "clustering_bundle.joblib").exists(),
            "generation": "lazy, reproducible training from the source CSV files",
        },
        regression={
            "target": REGRESSION_TARGET,
            "n_features": len(REGRESSION_FEATURES),
            "features": list(REGRESSION_FEATURES),
            "train_test_split": {"test_size": 0.20, "random_state": 42, "stratified_by": "10 target quantiles"},
            "models": [
                "Linear Regression",
                "Ridge Regression",
                "Lasso Regression",
                "ElasticNet Regression",
                "Polynomial Regression",
                "Decision Tree Regressor",
                "Random Forest Regressor",
                "Gradient Boosting Regressor",
                "Support Vector Regressor",
                "K-Nearest Neighbors Regressor",
            ],
        },
        classification={
            "target": "category_group",
            "n_features": len(NUTRITION_FEATURES),
            "features": list(NUTRITION_FEATURES),
            "models": list(CLASSIFICATION_MODEL_NAMES),
            "selected_tuned_model": CLASSIFICATION_TUNED_NAME,
            "train_test_split": {"test_size": 0.20, "random_state": 42, "stratified": True},
        },
        clustering={
            "features": list(CLUSTER_FEATURES),
            "customer_id_used_as_feature": False,
            "gender_used_as_feature": False,
            "algorithms": list(CLUSTER_ALGORITHMS),
            "selected_n_clusters": 6,
        },
    )


def get_regression_results() -> dict[str, Any]:
    """Return the computed ten-model regression comparison and CV/tuning tables."""

    try:
        bundle = get_regression_bundle()
        comparison = bundle["comparison"].copy()
        cv_results = bundle["cv_results"].copy()
        # Keep explicit canonical names and the notebook/UI-compatible aliases.
        cv_results["Mean CV R2"] = cv_results["CV_R2_Mean"]
        cv_results["Std CV R2"] = cv_results["CV_R2_Std"]
        ordered = ["Model", "R2", "RMSE", "MAE", "CV_R2_Mean", "CV_R2_Std"]
        comparison = comparison.loc[:, [column for column in ordered if column in comparison.columns]]
        return _ok(
            comparison=comparison,
            cv_results=cv_results,
            tuning_results=bundle["tuning_results"].copy(),
            feature_names=list(bundle["feature_names"]),
            target_name=bundle["target_name"],
            selected_model=bundle["selected_model"],
            split=bundle["split"],
            n_rows=bundle["n_rows"],
            message="Metrics were computed from the notebook workflow on the held-out test split.",
        )
    except Exception as exc:
        return _error(f"Unable to load or train regression models: {exc}")


def get_classification_results() -> dict[str, Any]:
    """Return all ten classifier metrics, confusion matrices, CV, and tuning."""

    try:
        bundle = get_classification_bundle()
        comparison = bundle["comparison"].copy()
        columns = [
            "Model",
            "Accuracy",
            "Precision_weighted",
            "Recall_weighted",
            "F1_weighted",
            "ROC_AUC_OvR",
        ]
        comparison = comparison.loc[:, [column for column in columns if column in comparison.columns]]
        return _ok(
            comparison=comparison,
            confusion_matrices={
                name: matrix.copy() for name, matrix in bundle["confusion_matrices"].items()
            },
            class_labels=list(bundle["classes"]),
            cv_results=bundle["cv_results"].copy(),
            baseline_vs_tuned=bundle["baseline_vs_tuned"].copy(),
            tuning=dict(bundle["tuning"]),
            feature_names=list(bundle["feature_names"]),
            target_name=bundle["target_name"],
            selected_model=bundle["selected_model"],
            split=bundle["split"],
            n_rows=bundle["n_rows"],
            target_counts=dict(bundle["target_counts"]),
            message=(
                "Metrics use the notebook's 11-class category_group target; "
                "ROC-AUC is weighted one-vs-rest."
            ),
        )
    except Exception as exc:
        return _error(f"Unable to load or train classification models: {exc}")


def get_clustering_results() -> dict[str, Any]:
    """Return final clustering metrics, curves, assignments, and projections."""

    try:
        bundle = get_clustering_bundle()
        return _ok(
            comparison=bundle["comparison"].copy(),
            elbow=bundle["elbow"].copy(),
            linkage_comparison=bundle["linkage_comparison"].copy(),
            assignments=bundle["assignments"].copy(),
            profiles={name: profile.copy() for name, profile in bundle["profiles"].items()},
            pca_coordinates=bundle["pca_coordinates"].copy(),
            pca_explained_variance_ratio=list(bundle["pca_explained_variance_ratio"]),
            tsne_coordinates=bundle["tsne_coordinates"].copy(),
            feature_names=list(bundle["feature_names"]),
            selected_n_clusters=bundle["selected_n_clusters"],
            n_rows=bundle["n_rows"],
            message=(
                "Agglomerative assignment for a new customer uses the nearest centroid "
                "of its fitted cluster representation because sklearn has no predict() method."
            ),
        )
    except Exception as exc:
        return _error(f"Unable to load or train clustering models: {exc}")


def predict_regression(features: Any, model_name: Optional[str] = None) -> dict[str, Any]:
    """Predict ``performance_score`` from the exact 52 model features.

    ``features`` may be a mapping or one-row DataFrame.  It may also contain
    the raw notebook source fields ``match_date`` and the three engineered
    source columns; missing engineered fields are derived before inference.
    """

    try:
        bundle = get_regression_bundle()
        name = _canonical_regression_name(model_name, bundle["selected_model"])
        if name not in bundle["models"]:
            raise ModelInputError(f"Regression model is unavailable: {name}")
        clean, scaled = prepare_regression_input(features, bundle["preprocessing"])
        model_info = bundle["models"][name]
        if model_info["input_space"] == "polynomial":
            model_input = model_info["poly"].transform(scaled[:, model_info["poly_indices"]])
        else:
            model_input = scaled
        prediction = float(model_info["estimator"].predict(model_input)[0])
        return _ok(
            prediction=prediction,
            predicted_value=prediction,
            model=name,
            target_name=bundle["target_name"],
            feature_order=list(bundle["feature_names"]),
            preprocessing="training medians -> training IQR caps -> training StandardScaler",
        )
    except (ModelInputError, ValueError, KeyError) as exc:
        return _error(str(exc))
    except Exception as exc:
        return _error(f"Regression prediction failed: {exc}")


def predict_classification(features: Any, model_name: Optional[str] = None) -> dict[str, Any]:
    """Predict ``category_group`` and return probabilities in class order."""

    try:
        bundle = get_classification_bundle()
        name = _canonical_classification_name(model_name, bundle["selected_model"])
        if name not in bundle["models"]:
            raise ModelInputError(f"Classification model is unavailable: {name}")
        clean, scaled = prepare_classification_input(features, bundle["preprocessing"])
        model_info = bundle["models"][name]
        model_input = scaled if model_info["input_space"] == "scaled" else clean
        estimator = model_info["estimator"]
        predicted = estimator.predict(model_input)[0]
        result: dict[str, Any] = {
            "status": "ok",
            "predicted_class": str(predicted),
            "model": name,
            "class_order": list(bundle["classes"]),
            "feature_order": list(bundle["feature_names"]),
            "preprocessing": "training medians -> training IQR caps -> model-specific scaling",
        }
        if hasattr(estimator, "predict_proba"):
            probabilities = np.asarray(estimator.predict_proba(model_input), dtype=float)[0]
            model_classes = list(getattr(estimator, "classes_", bundle["classes"]))
            aligned = {label: 0.0 for label in bundle["classes"]}
            for label, probability in zip(model_classes, probabilities):
                if label in aligned:
                    aligned[label] = float(probability)
            result["probabilities"] = aligned
            result["probability_supported"] = True
            result["probability_message"] = "Class probabilities are ordered by class_order."
        else:
            result["probabilities"] = None
            result["probability_supported"] = False
            result["probability_message"] = (
                f"{name} does not expose predict_proba; a probability estimate is unavailable."
            )
        return result
    except (ModelInputError, ValueError, KeyError) as exc:
        return _error(str(exc))
    except Exception as exc:
        return _error(f"Classification prediction failed: {exc}")


def assign_cluster(features: Any, algorithm_name: str = "K-Means") -> dict[str, Any]:
    """Assign one customer to a fitted cluster without using CustomerID/Gender."""

    try:
        bundle = get_clustering_bundle()
        algorithm = _canonical_cluster_algorithm(algorithm_name)
        scaled = prepare_cluster_input(features, bundle["scaler"])
        if algorithm == "K-Means":
            cluster = int(bundle["models"][algorithm].predict(scaled)[0])
            strategy = "K-Means fitted centroid prediction"
        else:
            centroids = np.asarray(bundle["centroids"][algorithm])
            distances = np.linalg.norm(centroids - scaled[0], axis=1)
            cluster = int(np.argmin(distances))
            strategy = (
                "nearest centroid in the fitted Agglomerative Ward cluster representation; "
                "AgglomerativeClustering.predict is not available"
            )
        return _ok(
            algorithm=algorithm,
            cluster=cluster,
            cluster_label=cluster,
            n_clusters=bundle["selected_n_clusters"],
            feature_order=list(bundle["feature_names"]),
            assignment_strategy=strategy,
        )
    except (ModelInputError, ValueError, KeyError) as exc:
        return _error(str(exc))
    except Exception as exc:
        return _error(f"Cluster assignment failed: {exc}")


def get_dataset_preview(dataset_name: str, n_rows: int = 10) -> dict[str, Any]:
    """Return a bounded preview of one original or supplied preprocessed CSV."""

    allowed = {
        "fifa": "fifa.csv",
        "fifa.csv": "fifa.csv",
        "nutrition": "nutrition.csv",
        "nutrition.csv": "nutrition.csv",
        "mall_customers": "mall_customers.csv",
        "mall_customers.csv": "mall_customers.csv",
        "fifa_preprocessed": "fifa_preprocessed.csv",
        "nutrition_preprocessed": "nutrition_preprocessed.csv",
        "mall_customers_preprocessed": "mall_customers_preprocessed.csv",
    }
    try:
        if dataset_name not in allowed:
            raise ModelInputError(
                f"Unknown dataset '{dataset_name}'. Choose one of: {', '.join(sorted(allowed))}"
            )
        if not isinstance(n_rows, int) or n_rows < 1 or n_rows > 100:
            raise ModelInputError("n_rows must be an integer between 1 and 100")
        path = DATA_DIR / allowed[dataset_name]
        if not path.exists():
            raise FileNotFoundError(f"Dataset was not found: {path}")
        preview = pd.read_csv(path, nrows=n_rows)
        return _ok(
            dataset=dataset_name,
            path=str(path),
            data=preview,
            columns=list(preview.columns),
            shape=[_csv_row_count(path), int(len(preview.columns))],
        )
    except (ModelInputError, ValueError, FileNotFoundError) as exc:
        return _error(str(exc))
    except Exception as exc:
        return _error(f"Dataset preview failed: {exc}")


__all__ = [
    "get_project_summary",
    "get_regression_results",
    "get_classification_results",
    "get_clustering_results",
    "predict_regression",
    "predict_classification",
    "assign_cluster",
    "get_dataset_preview",
]

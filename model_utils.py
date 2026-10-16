"""Reusable data preparation and model-training utilities for the capstone.

The notebooks are the source of truth for this module.  The code below keeps
their feature definitions, split seeds, preprocessing order, and estimator
hyperparameters while making the workflows safe to call from a dashboard.

Models are generated lazily and persisted below ``models/``.  No model file
was present in the supplied project, so an artifact is only described as
generated after this module has trained it from the source CSV files.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import tempfile
import threading
from typing import Any, Mapping, Optional

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    calinski_harabasz_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from scipy.cluster.hierarchy import linkage
from sklearn.manifold import TSNE


SEED = 42
ARTIFACT_SCHEMA_VERSION = 2
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "datasets"
MODEL_DIR = PROJECT_DIR / "models"

REGRESSION_TARGET = "performance_score"
REGRESSION_FEATURES = [
    "age",
    "jersey_number",
    "height_cm",
    "weight_kg",
    "market_value_eur",
    "goals_team",
    "goals_opponent",
    "minutes_played",
    "goals",
    "assists",
    "shots",
    "shots_on_target",
    "expected_goals_xg",
    "expected_assists_xa",
    "key_passes",
    "successful_passes",
    "total_passes",
    "pass_accuracy",
    "dribbles_attempted",
    "successful_dribbles",
    "crosses",
    "successful_crosses",
    "tackles",
    "interceptions",
    "clearances",
    "blocks",
    "aerial_duels_won",
    "aerial_duels_lost",
    "recoveries",
    "defensive_actions",
    "fouls_committed",
    "fouls_suffered",
    "yellow_cards",
    "red_cards",
    "offsides",
    "saves",
    "save_percentage",
    "punches",
    "clean_sheet",
    "goals_conceded",
    "penalty_saves",
    "distance_covered_km",
    "sprint_distance_km",
    "top_speed_kmh",
    "accelerations",
    "decelerations",
    "stamina_score",
    "match_year",
    "match_month",
    "goal_contribution_rate",
    "work_rate_intensity",
    "passing_efficiency",
]
REGRESSION_POLYNOMIAL_FEATURES = [
    "minutes_played",
    "goals",
    "assists",
    "distance_covered_km",
    "expected_goals_xg",
]
REGRESSION_DROPPED_COLUMNS = [
    "player_id",
    "player_name",
    "match_id",
    "player_rating",
    "tournament_rating",
    "offensive_contribution",
    "defensive_contribution",
    "possession_impact",
    "pressure_resistance",
    "creativity_score",
    "consistency_score",
    "clutch_performance_score",
    "total_goals_tournament",
    "total_assists_tournament",
    "total_minutes_tournament",
    "player_of_match_awards",
]
REGRESSION_MODEL_NAMES = [
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
]

NUTRITION_RAW_FEATURES = [
    "calories",
    "protein_g",
    "carbohydrates_g",
    "fat_g",
    "fiber_g",
    "sugars_g",
    "calcium_mg",
    "iron_mg",
    "sodium_mg",
    "potassium_mg",
    "magnesium_mg",
    "phosphorus_mg",
    "zinc_mg",
    "vitamin_a_iu",
    "vitamin_c_mg",
    "vitamin_b1_mg",
    "vitamin_b2_mg",
    "vitamin_b3_mg",
    "vitamin_b5_mg",
    "vitamin_b6_mg",
    "vitamin_e_mg",
    "folate_mcg",
]
NUTRITION_ENGINEERED_FEATURES = [
    "total_macros_g",
    "fat_carb_ratio",
    "protein_carb_ratio",
]
NUTRITION_FEATURES = NUTRITION_RAW_FEATURES + NUTRITION_ENGINEERED_FEATURES
CLASSIFICATION_MODEL_NAMES = [
    "Logistic Regression",
    "K-Nearest Neighbors",
    "Gaussian Naive Bayes",
    "Decision Tree Classifier",
    "Support Vector Classifier",
    "Random Forest Classifier",
    "AdaBoost Classifier",
    "Gradient Boosting Classifier",
    "Bagging Classifier",
    "MLP Classifier",
]
CLASSIFICATION_TUNED_NAME = "Tuned Bagging Classifier"

CLUSTER_FEATURES = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
CLUSTER_ALGORITHMS = ["K-Means", "Agglomerative Hierarchical (Ward)"]
CLUSTER_COUNT = 6

_BUNDLE_LOCKS = {
    "regression": threading.Lock(),
    "classification": threading.Lock(),
    "clustering": threading.Lock(),
}


class ModelInputError(ValueError):
    """Raised internally when a service input cannot be validated."""


@dataclass
class PreprocessingState:
    """Train-only preprocessing statistics persisted with each artifact."""

    feature_names: list[str]
    medians: pd.Series
    lower: pd.Series
    upper: pd.Series
    scaler: StandardScaler


def _data_path(filename: str) -> Path:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Required dataset was not found: {path}")
    return path


def _ensure_model_dir() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _artifact_path(name: str) -> Path:
    _ensure_model_dir()
    return MODEL_DIR / name


def _source_fingerprint(kind: str) -> str:
    source_files = {
        "regression": ["fifa.csv"],
        "classification": ["nutrition.csv"],
        "clustering": ["mall_customers.csv"],
    }[kind]
    parts = []
    for filename in source_files:
        path = _data_path(filename)
        stat = path.stat()
        parts.append(f"{filename}:{stat.st_size}:{stat.st_mtime_ns}")
    return "|".join(parts)


def _save_artifact(path: Path, artifact: dict[str, Any]) -> None:
    """Write a complete artifact in one operation so partial files are avoided."""

    # A unique temporary path prevents two Streamlit processes from sharing a
    # fixed ``.tmp`` filename on Windows.  os.replace is atomic on one volume.
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=f"{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    ) as handle:
        temporary_name = handle.name
    temporary = Path(temporary_name)
    try:
        joblib.dump(artifact, temporary, compress=3)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_artifact(path: Path, kind: str) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    try:
        artifact = joblib.load(path)
        if artifact.get("kind") != kind:
            return None
        if artifact.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
            return None
        if artifact.get("sklearn_version") != sklearn.__version__:
            # Refit instead of hiding a potentially incompatible estimator.
            return None
        if artifact.get("source_fingerprint") != _source_fingerprint(kind):
            return None
        return artifact
    except Exception:
        # A corrupt or incompatible local cache is recoverable by retraining.
        return None


def _input_frame(features: Any) -> pd.DataFrame:
    if isinstance(features, pd.DataFrame):
        frame = features.copy()
    elif isinstance(features, Mapping):
        frame = pd.DataFrame([dict(features)])
    else:
        raise ModelInputError("features must be a mapping or a one-row pandas DataFrame")
    if frame.empty:
        raise ModelInputError("features must contain at least one row")
    if len(frame) != 1:
        raise ModelInputError("prediction services accept exactly one feature row")
    return frame


def _numeric_frame(frame: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    missing = [name for name in feature_names if name not in frame.columns]
    if missing:
        raise ModelInputError(f"Missing required feature(s): {', '.join(missing)}")
    result = frame.loc[:, feature_names].copy()
    for name in feature_names:
        result[name] = pd.to_numeric(result[name], errors="coerce")
    # NaNs are intentionally retained here: fitted training medians handle
    # missing numeric values exactly as they did in the notebooks.
    non_numeric = [name for name in feature_names if result[name].isna().all() and frame[name].notna().any()]
    if non_numeric:
        raise ModelInputError(f"Feature value(s) must be numeric: {', '.join(non_numeric)}")
    return result


def _fit_train_preprocessing(X_train: pd.DataFrame) -> tuple[pd.DataFrame, PreprocessingState]:
    medians = X_train.median(numeric_only=True)
    filled = X_train.fillna(medians)
    q1 = filled.quantile(0.25)
    q3 = filled.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    clipped = filled.clip(lower=lower, upper=upper, axis=1)
    scaler = StandardScaler()
    scaler.fit(clipped)
    return clipped, PreprocessingState(list(X_train.columns), medians, lower, upper, scaler)


def _transform_with_state(X: pd.DataFrame, state: PreprocessingState) -> tuple[pd.DataFrame, np.ndarray]:
    ordered = X.loc[:, state.feature_names].copy()
    ordered = ordered.fillna(state.medians)
    ordered = ordered.clip(lower=state.lower, upper=state.upper, axis=1)
    return ordered, state.scaler.transform(ordered)


class _NotebookPreprocessor(BaseEstimator, TransformerMixin):
    """Fold-local equivalent of the notebook median/IQR/scaler sequence."""

    def __init__(self, scale: bool = True):
        self.scale = scale

    def fit(self, X: Any, y: Any = None) -> "_NotebookPreprocessor":
        frame = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.feature_names_in_ = list(frame.columns)
        frame = frame.apply(pd.to_numeric, errors="coerce")
        self.medians_ = frame.median(numeric_only=True)
        filled = frame.fillna(self.medians_)
        q1 = filled.quantile(0.25)
        q3 = filled.quantile(0.75)
        iqr = q3 - q1
        self.lower_ = q1 - 1.5 * iqr
        self.upper_ = q3 + 1.5 * iqr
        clipped = filled.clip(lower=self.lower_, upper=self.upper_, axis=1)
        self.scaler_ = StandardScaler().fit(clipped) if self.scale else None
        return self

    def transform(self, X: Any) -> np.ndarray:
        frame = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names_in_)
        frame = frame.loc[:, self.feature_names_in_].apply(pd.to_numeric, errors="coerce")
        frame = frame.fillna(self.medians_)
        frame = frame.clip(lower=self.lower_, upper=self.upper_, axis=1)
        if self.scaler_ is not None:
            return self.scaler_.transform(frame)
        return frame.to_numpy(dtype=float)


class _PolynomialSubsetTransformer(BaseEstimator, TransformerMixin):
    """Apply the notebook polynomial transform to the five selected columns."""

    def __init__(self, indices: list[int]):
        self.indices = indices

    def fit(self, X: Any, y: Any = None) -> "_PolynomialSubsetTransformer":
        self.poly_ = PolynomialFeatures(degree=2, include_bias=False)
        self.poly_.fit(np.asarray(X)[:, self.indices])
        return self

    def transform(self, X: Any) -> np.ndarray:
        return self.poly_.transform(np.asarray(X)[:, self.indices])


def _cv_regression_pipeline(name: str, estimator: Any) -> Pipeline:
    steps: list[tuple[str, Any]] = [("preprocess", _NotebookPreprocessor(scale=True))]
    if name == "Polynomial Regression":
        indices = [REGRESSION_FEATURES.index(value) for value in REGRESSION_POLYNOMIAL_FEATURES]
        steps.append(("polynomial", _PolynomialSubsetTransformer(indices)))
    steps.append(("model", clone(estimator)))
    return Pipeline(steps)


def _coerce_numeric_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    invalid: list[str] = []
    for name in columns:
        if name not in result.columns:
            continue
        original = result[name]
        converted = pd.to_numeric(original, errors="coerce")
        if (converted.isna() & original.notna()).any():
            invalid.append(name)
        result[name] = converted
    if invalid:
        raise ModelInputError(f"Feature value(s) must be numeric: {', '.join(invalid)}")
    return result


def _prepare_regression_dataset() -> dict[str, Any]:
    df = pd.read_csv(_data_path("fifa.csv")).drop_duplicates().copy()
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df["match_year"] = df["match_date"].dt.year
    df["match_month"] = df["match_date"].dt.month
    df = df.drop(columns=["match_date"])
    df["goal_contribution_rate"] = (df["goals"] + df["assists"]) / (
        df["minutes_played"].replace(0, np.nan) / 90
    )
    df["work_rate_intensity"] = df["sprint_distance_km"] / (df["distance_covered_km"] + 0.001)
    df["passing_efficiency"] = df["successful_passes"] / (df["total_passes"] + 1)
    df = df.replace([np.inf, -np.inf], np.nan)

    model_df = df.drop(columns=[c for c in REGRESSION_DROPPED_COLUMNS if c in df.columns])
    model_df = model_df.select_dtypes(include=np.number)
    X = model_df.drop(columns=[REGRESSION_TARGET])
    # Reindex makes a drift in a source CSV an explicit, actionable error.
    missing = [name for name in REGRESSION_FEATURES if name not in X.columns]
    if missing:
        raise ValueError(f"Regression source is missing expected feature(s): {missing}")
    X = X.loc[:, REGRESSION_FEATURES]
    y = model_df[REGRESSION_TARGET]
    y_bins = pd.qcut(y, q=10, labels=False, duplicates="drop")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y_bins
    )
    X_train_clean, state = _fit_train_preprocessing(X_train)
    X_test_clean, _ = _transform_with_state(X_test, state)
    X_train_scaled = state.scaler.transform(X_train_clean)
    X_test_scaled = state.scaler.transform(X_test_clean)
    return {
        "X_train_raw": X_train,
        "X_test_raw": X_test,
        "X_train": X_train_clean,
        "X_test": X_test_clean,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "state": state,
        "n_rows": len(df),
    }


def prepare_regression_input(features: Any, state: PreprocessingState) -> tuple[pd.DataFrame, np.ndarray]:
    """Validate model-ready FIFA features, deriving notebook fields when possible."""

    frame = _input_frame(features)
    frame = _coerce_numeric_columns(
        frame,
        [
            name
            for name in REGRESSION_FEATURES
            if name not in {"goal_contribution_rate", "work_rate_intensity", "passing_efficiency"}
        ]
        + ["goals", "assists", "minutes_played", "sprint_distance_km", "distance_covered_km", "successful_passes", "total_passes"],
    )
    if "match_year" not in frame.columns or "match_month" not in frame.columns:
        if "match_date" not in frame.columns:
            missing = [name for name in ("match_year", "match_month") if name not in frame.columns]
            raise ModelInputError(
                "Provide match_year and match_month, or provide match_date so they can be derived"
            )
        date = pd.to_datetime(frame["match_date"], errors="coerce")
        if date.isna().any():
            raise ModelInputError("match_date must be a valid date")
        frame["match_year"] = date.dt.year
        frame["match_month"] = date.dt.month
    derived = {
        "goal_contribution_rate": lambda f: (f["goals"] + f["assists"])
        / (f["minutes_played"].replace(0, np.nan) / 90),
        "work_rate_intensity": lambda f: f["sprint_distance_km"] / (f["distance_covered_km"] + 0.001),
        "passing_efficiency": lambda f: f["successful_passes"] / (f["total_passes"] + 1),
    }
    for name, function in derived.items():
        if name not in frame.columns:
            try:
                frame[name] = function(frame)
            except KeyError as exc:
                raise ModelInputError(
                    f"Missing source feature(s) needed to derive {name}: {exc.args[0]}"
                ) from exc
    frame = frame.replace([np.inf, -np.inf], np.nan)
    numeric = _numeric_frame(frame, state.feature_names)
    return _transform_with_state(numeric, state)


def _regression_estimators() -> dict[str, Any]:
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.05, max_iter=10000),
        "ElasticNet Regression": ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=10000),
        "Decision Tree Regressor": DecisionTreeRegressor(max_depth=6, random_state=SEED),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=50, max_depth=12, random_state=SEED, n_jobs=1
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=4, random_state=SEED
        ),
        "Support Vector Regressor": SVR(C=10, kernel="rbf", epsilon=0.1),
        "K-Nearest Neighbors Regressor": KNeighborsRegressor(n_neighbors=9, weights="distance"),
    }


def _regression_metrics(y_true: pd.Series, prediction: np.ndarray, name: str) -> dict[str, Any]:
    return {
        "Model": name,
        "R2": float(r2_score(y_true, prediction)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, prediction))),
        "MAE": float(mean_absolute_error(y_true, prediction)),
    }


def train_regression_bundle(save: bool = True) -> dict[str, Any]:
    """Train every notebook regression estimator and the documented tuned models."""

    data = _prepare_regression_dataset()
    models: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    estimators = _regression_estimators()
    for name, estimator in estimators.items():
        if name == "Decision Tree Regressor" or name in {
            "Random Forest Regressor",
            "Gradient Boosting Regressor",
        }:
            # The notebook scales every baseline estimator, including trees.
            estimator.fit(data["X_train_scaled"], data["y_train"])
            prediction = estimator.predict(data["X_test_scaled"])
            models[name] = {"estimator": estimator, "input_space": "scaled"}
        else:
            estimator.fit(data["X_train_scaled"], data["y_train"])
            prediction = estimator.predict(data["X_test_scaled"])
            models[name] = {"estimator": estimator, "input_space": "scaled"}
        rows.append(_regression_metrics(data["y_test"], prediction, name))

    poly_indices = [REGRESSION_FEATURES.index(name) for name in REGRESSION_POLYNOMIAL_FEATURES]
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(data["X_train_scaled"][:, poly_indices])
    X_test_poly = poly.transform(data["X_test_scaled"][:, poly_indices])
    poly_model = LinearRegression().fit(X_train_poly, data["y_train"])
    poly_prediction = poly_model.predict(X_test_poly)
    models["Polynomial Regression"] = {
        "estimator": poly_model,
        "input_space": "polynomial",
        "poly": poly,
        "poly_indices": poly_indices,
    }
    rows.append(_regression_metrics(data["y_test"], poly_prediction, "Polynomial Regression"))

    comparison = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
    top_two = comparison.head(2)["Model"].tolist()
    cv_rows = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for name in top_two:
        # Unlike the notebook's executed CV cell, this pipeline refits the
        # median, IQR caps, and scaler inside every fold to avoid leakage.
        estimator = _cv_regression_pipeline(name, models[name]["estimator"])
        scores = cross_val_score(
            estimator,
            data["X_train_raw"],
            data["y_train"],
            cv=kfold,
            scoring="r2",
            n_jobs=1,
        )
        cv_rows.append(
            {
                "Model": name,
                "CV_R2_Mean": float(scores.mean()),
                "CV_R2_Std": float(scores.std()),
                "CV_R2_Scores": [float(value) for value in scores],
            }
        )
    cv_results = pd.DataFrame(cv_rows)
    comparison = comparison.merge(cv_results[["Model", "CV_R2_Mean", "CV_R2_Std"]], on="Model", how="left")

    tuning_rows = []
    # The executed notebook already reports the best parameters from its
    # bounded GridSearchCV.  Refitting those documented parameters here avoids
    # rerunning an expensive search on every cold Streamlit start while keeping
    # the selected model reproducible.  The source table remains the evidence
    # for the search result; this backend does not claim to rerun that search.
    gb_params = {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 3}
    gb_tuned_estimator = GradientBoostingRegressor(random_state=SEED, **gb_params)
    gb_tuned_estimator.fit(data["X_train_scaled"], data["y_train"])
    gb_tuned_pred = gb_tuned_estimator.predict(data["X_test_scaled"])
    models["Tuned Gradient Boosting Regressor"] = {
        "estimator": gb_tuned_estimator,
        "input_space": "scaled",
    }
    gb_tuned = _regression_metrics(
        data["y_test"], gb_tuned_pred, "Tuned Gradient Boosting Regressor"
    )
    gb_tuned["Best_Params"] = gb_params
    tuning_rows.append(gb_tuned)

    rf_params = {"n_estimators": 50, "max_depth": 10, "min_samples_split": 2}
    rf_tuned_estimator = RandomForestRegressor(random_state=SEED, n_jobs=1, **rf_params)
    rf_tuned_estimator.fit(data["X_train_scaled"], data["y_train"])
    rf_tuned_pred = rf_tuned_estimator.predict(data["X_test_scaled"])
    models["Tuned Random Forest Regressor"] = {
        "estimator": rf_tuned_estimator,
        "input_space": "scaled",
    }
    rf_tuned = _regression_metrics(data["y_test"], rf_tuned_pred, "Tuned Random Forest Regressor")
    rf_tuned["Best_Params"] = rf_params
    tuning_rows.append(rf_tuned)
    tuning_results = pd.DataFrame(tuning_rows)
    selected_model = str(tuning_results.sort_values("R2", ascending=False).iloc[0]["Model"])

    artifact = {
        "kind": "regression",
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "sklearn_version": sklearn.__version__,
        "source_fingerprint": _source_fingerprint("regression"),
        "feature_names": REGRESSION_FEATURES,
        "target_name": REGRESSION_TARGET,
        "preprocessing": data["state"],
        "models": models,
        "comparison": comparison,
        "cv_results": cv_results,
        "tuning_results": tuning_results,
        "tuning_procedure": {
            "method": "notebook-reported bounded GridSearchCV parameters, reproducible refit",
            "scoring": "r2",
            "cv": 3,
            "search_rerun": False,
        },
        "selected_model": selected_model,
        "split": {"test_size": 0.20, "random_state": SEED, "stratified_by": "10 target quantiles"},
        "n_rows": data["n_rows"],
    }
    if save:
        _save_artifact(_artifact_path("regression_bundle.joblib"), artifact)
    return artifact


@lru_cache(maxsize=1)
def get_regression_bundle() -> dict[str, Any]:
    with _BUNDLE_LOCKS["regression"]:
        artifact = _load_artifact(_artifact_path("regression_bundle.joblib"), "regression")
        if artifact is None:
            artifact = train_regression_bundle(save=True)
        return artifact


def _prepare_classification_dataset() -> dict[str, Any]:
    df = pd.read_csv(_data_path("nutrition.csv")).drop_duplicates().copy()
    top_categories = df["category"].value_counts().head(10).index
    df["category_group"] = df["category"].where(df["category"].isin(top_categories), "Other")
    df["total_macros_g"] = df["protein_g"] + df["carbohydrates_g"] + df["fat_g"]
    df["fat_carb_ratio"] = df["fat_g"] / (df["carbohydrates_g"] + 1)
    df["protein_carb_ratio"] = df["protein_g"] / (df["carbohydrates_g"] + 1)
    X = df[NUTRITION_FEATURES].copy()
    y = df["category_group"].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    X_train_clean, state = _fit_train_preprocessing(X_train)
    X_test_clean, _ = _transform_with_state(X_test, state)
    X_train_scaled = state.scaler.transform(X_train_clean)
    X_test_scaled = state.scaler.transform(X_test_clean)
    classes = sorted(y.unique().tolist())
    return {
        "X_train_raw": X_train,
        "X_test_raw": X_test,
        "X_train": X_train_clean,
        "X_test": X_test_clean,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "state": state,
        "classes": classes,
        "n_rows": len(df),
        "target_counts": y.value_counts().to_dict(),
    }


def prepare_classification_input(
    features: Any, state: PreprocessingState
) -> tuple[pd.DataFrame, np.ndarray]:
    """Validate nutrition features and reproduce the notebook feature engineering."""

    frame = _input_frame(features)
    frame = _coerce_numeric_columns(frame, NUTRITION_RAW_FEATURES + NUTRITION_ENGINEERED_FEATURES)
    engineered = {
        "total_macros_g": lambda f: f["protein_g"] + f["carbohydrates_g"] + f["fat_g"],
        "fat_carb_ratio": lambda f: f["fat_g"] / (f["carbohydrates_g"] + 1),
        "protein_carb_ratio": lambda f: f["protein_g"] / (f["carbohydrates_g"] + 1),
    }
    for name, function in engineered.items():
        if name not in frame.columns:
            try:
                frame[name] = function(frame)
            except KeyError as exc:
                raise ModelInputError(
                    f"Missing source feature(s) needed to derive {name}: {exc.args[0]}"
                ) from exc
    frame = frame.replace([np.inf, -np.inf], np.nan)
    numeric = _numeric_frame(frame, state.feature_names)
    return _transform_with_state(numeric, state)


def _make_bagging_classifier(**kwargs: Any) -> BaggingClassifier:
    """Use the current estimator keyword and remain readable on older sklearn."""

    base = DecisionTreeClassifier(random_state=SEED)
    try:
        return BaggingClassifier(estimator=base, **kwargs)
    except TypeError:  # pragma: no cover - exercised only by older sklearn
        return BaggingClassifier(base_estimator=base, **kwargs)


def _classification_estimators() -> dict[str, tuple[Any, str]]:
    return {
        "Logistic Regression": (LogisticRegression(max_iter=2000, random_state=SEED), "scaled"),
        "K-Nearest Neighbors": (KNeighborsClassifier(n_neighbors=5), "scaled"),
        "Gaussian Naive Bayes": (GaussianNB(), "scaled"),
        "Decision Tree Classifier": (DecisionTreeClassifier(max_depth=6, random_state=SEED), "unscaled"),
        "Support Vector Classifier": (
            SVC(C=1.0, kernel="rbf", probability=True, random_state=SEED),
            "scaled",
        ),
        "Random Forest Classifier": (
            RandomForestClassifier(n_estimators=50, max_depth=None, random_state=SEED, n_jobs=1),
            "unscaled",
        ),
        "AdaBoost Classifier": (
            AdaBoostClassifier(n_estimators=50, learning_rate=0.5, random_state=SEED),
            "unscaled",
        ),
        "Gradient Boosting Classifier": (
            GradientBoostingClassifier(
                n_estimators=50, learning_rate=0.1, max_depth=3, random_state=SEED
            ),
            "unscaled",
        ),
        "Bagging Classifier": (_make_bagging_classifier(n_estimators=50, random_state=SEED, n_jobs=1), "unscaled"),
        "MLP Classifier": (
            MLPClassifier(hidden_layer_sizes=(32,), activation="relu", max_iter=200, random_state=SEED),
            "scaled",
        ),
    }


def _cv_classification_pipeline(name: str, estimator: Any) -> Pipeline:
    input_space = _classification_estimators()[name][1]
    return Pipeline(
        [
            ("preprocess", _NotebookPreprocessor(scale=input_space == "scaled")),
            ("model", clone(estimator)),
        ]
    )


def _classification_probabilities(model: Any, X: Any, classes: list[str]) -> Optional[np.ndarray]:
    if not hasattr(model, "predict_proba"):
        return None
    probabilities = np.asarray(model.predict_proba(X), dtype=float)
    model_classes = list(getattr(model, "classes_", classes))
    if model_classes != classes:
        aligned = np.zeros((len(probabilities), len(classes)), dtype=float)
        for index, label in enumerate(model_classes):
            if label in classes:
                aligned[:, classes.index(label)] = probabilities[:, index]
        probabilities = aligned
    return probabilities


def _classification_metrics(
    model: Any,
    X: Any,
    y_true: pd.Series,
    name: str,
    classes: list[str],
) -> tuple[dict[str, Any], np.ndarray, Optional[np.ndarray]]:
    prediction = model.predict(X)
    probabilities = _classification_probabilities(model, X, classes)
    if probabilities is None:
        roc_auc = float("nan")
    else:
        y_binary = label_binarize(y_true, classes=classes)
        roc_auc = float(
            roc_auc_score(y_binary, probabilities, multi_class="ovr", average="weighted")
        )
    metrics = {
        "Model": name,
        "Accuracy": float(accuracy_score(y_true, prediction)),
        "Precision_weighted": float(
            precision_score(y_true, prediction, average="weighted", zero_division=0)
        ),
        "Recall_weighted": float(
            recall_score(y_true, prediction, average="weighted", zero_division=0)
        ),
        "F1_weighted": float(f1_score(y_true, prediction, average="weighted", zero_division=0)),
        "ROC_AUC_OvR": roc_auc,
    }
    matrix = confusion_matrix(y_true, prediction, labels=classes)
    return metrics, matrix, probabilities


def train_classification_bundle(save: bool = True) -> dict[str, Any]:
    """Train all ten required classifiers plus the notebook's tuned Bagging model."""

    data = _prepare_classification_dataset()
    models: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    confusion_matrices: dict[str, pd.DataFrame] = {}
    estimators = _classification_estimators()
    for name, (estimator, input_space) in estimators.items():
        train_X = data["X_train_scaled"] if input_space == "scaled" else data["X_train"]
        test_X = data["X_test_scaled"] if input_space == "scaled" else data["X_test"]
        estimator.fit(train_X, data["y_train"])
        metrics, matrix, _ = _classification_metrics(
            estimator, test_X, data["y_test"], name, data["classes"]
        )
        models[name] = {"estimator": estimator, "input_space": input_space}
        rows.append(metrics)
        confusion_matrices[name] = pd.DataFrame(
            matrix, index=data["classes"], columns=data["classes"]
        )

    comparison = pd.DataFrame(rows).sort_values("F1_weighted", ascending=False).reset_index(drop=True)
    top_two = comparison.head(2)["Model"].tolist()
    cv_rows = []
    stratified = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    for name in top_two:
        # Refit imputation, IQR caps, and (where applicable) scaling per fold.
        train_X = data["X_train_raw"]
        scores = cross_val_score(
            _cv_classification_pipeline(name, models[name]["estimator"]),
            train_X,
            data["y_train"],
            cv=stratified,
            scoring="f1_weighted",
            n_jobs=1,
        )
        cv_rows.append(
            {
                "Model": name,
                "CV_F1_weighted_Mean": float(scores.mean()),
                "CV_F1_weighted_Std": float(scores.std()),
                "CV_F1_weighted_Scores": [float(value) for value in scores],
                "CV_n_splits": 3,
            }
        )
    cv_results = pd.DataFrame(cv_rows)

    bagging_grid = {
        "n_estimators": [50, 100, 150],
        "max_samples": [0.7, 1.0],
        "max_features": [0.7, 1.0],
    }
    bagging_pipeline = Pipeline(
        [
            ("preprocess", _NotebookPreprocessor(scale=False)),
            ("model", _make_bagging_classifier(n_estimators=50, random_state=SEED, n_jobs=1)),
        ]
    )
    bagging_search = GridSearchCV(
        bagging_pipeline,
        {f"model__{key}": value for key, value in bagging_grid.items()},
        cv=3,
        scoring="f1_weighted",
        n_jobs=1,
    )
    bagging_search.fit(data["X_train_raw"], data["y_train"])
    bag_params = {
        key.removeprefix("model__"): value for key, value in bagging_search.best_params_.items()
    }
    tuned = _make_bagging_classifier(**bag_params)
    tuned.fit(data["X_train"], data["y_train"])
    tuned_metrics, tuned_matrix, _ = _classification_metrics(
        tuned,
        data["X_test"],
        data["y_test"],
        CLASSIFICATION_TUNED_NAME,
        data["classes"],
    )
    models[CLASSIFICATION_TUNED_NAME] = {"estimator": tuned, "input_space": "unscaled"}
    confusion_matrices[CLASSIFICATION_TUNED_NAME] = pd.DataFrame(
        tuned_matrix, index=data["classes"], columns=data["classes"]
    )
    baseline = comparison.loc[comparison["Model"] == "Bagging Classifier"].iloc[0].to_dict()
    baseline_tuned_rows = [
        {"Version": "Baseline Bagging", **{key: baseline[key] for key in rows[0] if key != "Model"}},
        {"Version": "Tuned Bagging", **{key: tuned_metrics[key] for key in tuned_metrics if key != "Model"}},
    ]
    baseline_vs_tuned = pd.DataFrame(baseline_tuned_rows)

    artifact = {
        "kind": "classification",
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "sklearn_version": sklearn.__version__,
        "source_fingerprint": _source_fingerprint("classification"),
        "feature_names": NUTRITION_FEATURES,
        "target_name": "category_group",
        "classes": data["classes"],
        "preprocessing": data["state"],
        "models": models,
        "comparison": comparison,
        "confusion_matrices": confusion_matrices,
        "cv_results": cv_results,
        "baseline_vs_tuned": baseline_vs_tuned,
        "selected_model": CLASSIFICATION_TUNED_NAME,
        "split": {"test_size": 0.20, "random_state": SEED, "stratified": True},
        "n_rows": data["n_rows"],
        "target_counts": data["target_counts"],
        "tuning": {"best_params": bag_params, "cv": 3, "scoring": "f1_weighted"},
    }
    if save:
        _save_artifact(_artifact_path("classification_bundle.joblib"), artifact)
    return artifact


@lru_cache(maxsize=1)
def get_classification_bundle() -> dict[str, Any]:
    with _BUNDLE_LOCKS["classification"]:
        artifact = _load_artifact(_artifact_path("classification_bundle.joblib"), "classification")
        if artifact is None:
            artifact = train_classification_bundle(save=True)
        return artifact


def _prepare_clustering_dataset() -> dict[str, Any]:
    df = pd.read_csv(_data_path("mall_customers.csv")).copy()
    missing = [name for name in CLUSTER_FEATURES if name not in df.columns]
    if missing:
        raise ValueError(f"Clustering source is missing expected feature(s): {missing}")
    X = df.loc[:, CLUSTER_FEATURES].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        raise ValueError("Clustering source contains missing or non-numeric feature values")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    elbow_rows = []
    for n_clusters in range(2, 11):
        model = KMeans(n_clusters=n_clusters, init="k-means++", random_state=SEED, n_init=10)
        labels = model.fit_predict(X_scaled)
        elbow_rows.append(
            {
                "k": n_clusters,
                "Inertia": float(model.inertia_),
                "Silhouette": float(silhouette_score(X_scaled, labels)),
            }
        )
    elbow = pd.DataFrame(elbow_rows)

    kmeans = KMeans(n_clusters=CLUSTER_COUNT, init="k-means++", random_state=SEED, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    linkage_rows = []
    for method in ["ward", "complete", "average"]:
        candidate = AgglomerativeClustering(n_clusters=CLUSTER_COUNT, linkage=method)
        labels = candidate.fit_predict(X_scaled)
        linkage_rows.append(
            {
                "Linkage": method,
                "Silhouette": float(silhouette_score(X_scaled, labels)),
                "Davies_Bouldin": float(davies_bouldin_score(X_scaled, labels)),
                "Calinski_Harabasz": float(calinski_harabasz_score(X_scaled, labels)),
            }
        )
    linkage_comparison = pd.DataFrame(linkage_rows)
    agglomerative = AgglomerativeClustering(n_clusters=CLUSTER_COUNT, linkage="ward")
    agglomerative_labels = agglomerative.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=SEED)
    pca_coordinates = pca.fit_transform(X_scaled)
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=SEED,
        init="pca",
        learning_rate="auto",
    )
    tsne_coordinates = tsne.fit_transform(X_scaled)

    assignments = df.copy()
    assignments["KMeans_Cluster"] = kmeans_labels.astype(int)
    assignments["Agglomerative_Cluster"] = agglomerative_labels.astype(int)
    assignments["PCA_1"] = pca_coordinates[:, 0]
    assignments["PCA_2"] = pca_coordinates[:, 1]
    assignments["TSNE_1"] = tsne_coordinates[:, 0]
    assignments["TSNE_2"] = tsne_coordinates[:, 1]

    profiles: dict[str, pd.DataFrame] = {}
    for name, labels in [
        ("K-Means", kmeans_labels),
        ("Agglomerative Hierarchical (Ward)", agglomerative_labels),
    ]:
        profile = X.copy()
        profile["Cluster"] = labels.astype(int)
        profiles[name] = profile.groupby("Cluster", as_index=False)[CLUSTER_FEATURES].mean()

    def metrics(name: str, labels: np.ndarray) -> dict[str, Any]:
        return {
            "Algorithm": name,
            "Silhouette_Score": float(silhouette_score(X_scaled, labels)),
            "Davies_Bouldin_Index": float(davies_bouldin_score(X_scaled, labels)),
            "Calinski_Harabasz_Index": float(calinski_harabasz_score(X_scaled, labels)),
            "n_clusters": CLUSTER_COUNT,
        }

    comparison = pd.DataFrame(
        [
            metrics("K-Means", kmeans_labels),
            metrics("Agglomerative Hierarchical (Ward)", agglomerative_labels),
        ]
    )
    # Nearest-centroid assignment is the documented fallback for the fitted
    # agglomerative representation; sklearn does not expose predict().
    centroids = {
        "K-Means": np.asarray(kmeans.cluster_centers_),
        "Agglomerative Hierarchical (Ward)": np.vstack(
            [X_scaled[agglomerative_labels == label].mean(axis=0) for label in range(CLUSTER_COUNT)]
        ),
    }
    artifact = {
        "kind": "clustering",
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "sklearn_version": sklearn.__version__,
        "source_fingerprint": _source_fingerprint("clustering"),
        "feature_names": CLUSTER_FEATURES,
        "scaler": scaler,
        "models": {"K-Means": kmeans, "Agglomerative Hierarchical (Ward)": agglomerative},
        "centroids": centroids,
        "comparison": comparison,
        "elbow": elbow,
        "linkage_comparison": linkage_comparison,
        "assignments": assignments,
        "profiles": profiles,
        "pca_coordinates": pd.DataFrame(
            {
                "PCA_1": pca_coordinates[:, 0],
                "PCA_2": pca_coordinates[:, 1],
                "KMeans_Cluster": kmeans_labels.astype(int),
                "Agglomerative_Cluster": agglomerative_labels.astype(int),
            }
        ),
        "pca_explained_variance_ratio": [float(value) for value in pca.explained_variance_ratio_],
        "tsne_coordinates": pd.DataFrame(
            {
                "TSNE_1": tsne_coordinates[:, 0],
                "TSNE_2": tsne_coordinates[:, 1],
                "KMeans_Cluster": kmeans_labels.astype(int),
            }
        ),
        "n_rows": len(df),
        "selected_n_clusters": CLUSTER_COUNT,
    }
    return artifact


def prepare_cluster_input(features: Any, scaler: StandardScaler) -> np.ndarray:
    frame = _input_frame(features)
    numeric = _numeric_frame(frame, CLUSTER_FEATURES)
    if numeric.isna().any().any():
        raise ModelInputError("All clustering feature values are required and must be numeric")
    return scaler.transform(numeric)


def train_clustering_bundle(save: bool = True) -> dict[str, Any]:
    artifact = _prepare_clustering_dataset()
    if save:
        _save_artifact(_artifact_path("clustering_bundle.joblib"), artifact)
    return artifact


@lru_cache(maxsize=1)
def get_clustering_bundle() -> dict[str, Any]:
    with _BUNDLE_LOCKS["clustering"]:
        artifact = _load_artifact(_artifact_path("clustering_bundle.joblib"), "clustering")
        if artifact is None:
            artifact = train_clustering_bundle(save=True)
        return artifact


def clear_cached_bundles() -> None:
    """Clear in-process caches; useful after intentionally regenerating artifacts."""

    get_regression_bundle.cache_clear()
    get_classification_bundle.cache_clear()
    get_clustering_bundle.cache_clear()

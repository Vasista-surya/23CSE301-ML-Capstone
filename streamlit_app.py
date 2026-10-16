"""Machine Learning Capstone dashboard.

This entry point owns navigation, local read-only dataset/result access, and
the presentation of the ML service contract.  It never fabricates predictions:
live prediction controls call ``ml_services.py`` when that module is present
and show an explicit setup state when it is not.
"""

from __future__ import annotations

import base64
import importlib
import io
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    import streamlit as st
except ImportError:  # pragma: no cover - enables syntax checks before installing UI deps
    st = None

try:
    import plotly.express as px
except ImportError:  # pragma: no cover - app_components shows the user the dependency state
    px = None

from app_components import (
    AMBER,
    BLUE,
    INK,
    PALETTE,
    RED,
    TEAL,
    VIOLET,
    callout,
    comparison_bar,
    empty_state,
    heatmap_chart,
    histogram_chart,
    inject_styles,
    metric_grid,
    metric_line,
    render_brand,
    render_figure,
    render_notebook_image,
    render_page_header,
    render_service_status,
    render_workflow,
    scatter_chart,
    section_header,
    show_table,
    source_note,
)


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "datasets"
NOTEBOOK_DIR = APP_DIR / "notebooks"

DATASET_FILES: dict[str, str] = {
    "FIFA performance · raw": "fifa.csv",
    "FIFA performance · preprocessed": "fifa_preprocessed.csv",
    "Nutrition · raw": "nutrition.csv",
    "Nutrition · preprocessed": "nutrition_preprocessed.csv",
    "Mall customers · raw": "mall_customers.csv",
    "Mall customers · preprocessed": "mall_customers_preprocessed.csv",
}

REGRESSION_FEATURES = [
    "age", "jersey_number", "height_cm", "weight_kg", "market_value_eur",
    "goals_team", "goals_opponent", "minutes_played", "goals", "assists", "shots",
    "shots_on_target", "expected_goals_xg", "expected_assists_xa", "key_passes",
    "successful_passes", "total_passes", "pass_accuracy", "dribbles_attempted",
    "successful_dribbles", "crosses", "successful_crosses", "tackles", "interceptions",
    "clearances", "blocks", "aerial_duels_won", "aerial_duels_lost", "recoveries",
    "defensive_actions", "fouls_committed", "fouls_suffered", "yellow_cards", "red_cards",
    "offsides", "saves", "save_percentage", "punches", "clean_sheet", "goals_conceded",
    "penalty_saves", "distance_covered_km", "sprint_distance_km", "top_speed_kmh",
    "accelerations", "decelerations", "stamina_score", "match_year", "match_month",
    "goal_contribution_rate", "work_rate_intensity", "passing_efficiency",
]

CLASSIFICATION_FEATURES = [
    "calories", "protein_g", "carbohydrates_g", "fat_g", "fiber_g", "sugars_g",
    "calcium_mg", "iron_mg", "sodium_mg", "potassium_mg", "magnesium_mg",
    "phosphorus_mg", "zinc_mg", "vitamin_a_iu", "vitamin_c_mg", "vitamin_b1_mg",
    "vitamin_b2_mg", "vitamin_b3_mg", "vitamin_b5_mg", "vitamin_b6_mg", "vitamin_e_mg",
    "folate_mcg", "total_macros_g", "fat_carb_ratio", "protein_carb_ratio",
]

CLUSTER_FEATURES = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]

CLASSIFICATION_ALGORITHMS = [
    "Logistic Regression", "K-Nearest Neighbors", "Gaussian Naive Bayes", "Decision Tree",
    "Support Vector Classifier", "Random Forest", "AdaBoost", "Gradient Boosting", "Bagging",
    "MLP Classifier",
]

REGRESSION_ALGORITHMS = [
    "Linear Regression", "Ridge Regression", "Lasso Regression", "ElasticNet Regression",
    "Polynomial Regression", "Decision Tree Regressor", "Random Forest Regressor",
    "Gradient Boosting Regressor", "Support Vector Regressor", "KNN Regressor",
]

PLOT_CELL_MAP = {
    "regression_predicted_actual": ("fifa-performance-regression.ipynb", 37, 0),
    "regression_residuals": ("fifa-performance-regression.ipynb", 35, 0),
    "regression_importance": ("fifa-performance-regression.ipynb", 39, 0),
    "classification_tuned_cm": ("nutrition-classification.ipynb", 34, 0),
    "clustering_elbow": ("clustering.ipynb", 9, 0),
    "clustering_dendrogram": ("clustering.ipynb", 13, 0),
    "clustering_pca": ("clustering.ipynb", 17, 0),
    "clustering_tsne": ("clustering.ipynb", 20, 0),
}


class ServiceUnavailable(RuntimeError):
    """Raised only for a missing or incomplete optional service layer."""


def _require_streamlit() -> None:
    if st is None:
        raise RuntimeError(
            "Streamlit is not installed. Run `python -m pip install -r requirements.txt` first."
        )


@st.cache_data(show_spinner=False) if st is not None else (lambda fn: fn)
def load_dataset(dataset_name: str) -> pd.DataFrame:
    """Load a supplied CSV without ever modifying it."""

    filename = DATASET_FILES.get(dataset_name)
    if not filename:
        raise FileNotFoundError(f"Unknown dataset: {dataset_name}")
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


@st.cache_data(show_spinner=False) if st is not None else (lambda fn: fn)
def load_notebook(notebook_name: str) -> dict[str, Any]:
    path = NOTEBOOK_DIR / notebook_name
    if not path.exists():
        raise FileNotFoundError(f"Notebook not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _service_module() -> Any | None:
    """Import Agent 2's service layer only when it has been supplied."""

    try:
        return importlib.import_module("ml_services")
    except (ImportError, ModuleNotFoundError):
        return None


def service_available() -> bool:
    return _service_module() is not None


def _service_function(function_name: str) -> Any | None:
    module = _service_module()
    return getattr(module, function_name, None) if module is not None else None


def call_service(function_name: str, *args: Any, **kwargs: Any) -> Any:
    """Call a real service function, keeping ordinary UI errors user-friendly."""

    function = _service_function(function_name)
    if function is None:
        raise ServiceUnavailable(
            f"{function_name} is unavailable until ml_services.py exposes the agreed contract."
        )
    try:
        return function(*args, **kwargs)
    except TypeError:
        # Agent 2's recommended clustering signature uses ``algorithm`` while
        # the project brief names the positional argument ``algorithm_name``.
        if function_name == "assign_cluster" and "algorithm_name" in kwargs:
            retry_kwargs = dict(kwargs)
            retry_kwargs["algorithm"] = retry_kwargs.pop("algorithm_name")
            return function(*args, **retry_kwargs)
        raise


def _friendly_service_error(error: Exception) -> str:
    if isinstance(error, ServiceUnavailable):
        return "The prediction service is not installed in this workspace yet."
    return f"The prediction service returned an error ({type(error).__name__}). Check ml_services.py and try again."


def _as_frame(payload: Any) -> pd.DataFrame | None:
    """Normalize common service response containers without inventing values."""

    if payload is None:
        return None
    if isinstance(payload, pd.DataFrame):
        return payload.copy()
    if isinstance(payload, Mapping):
        if str(payload.get("status", "ok")).lower() == "error":
            return None
        for key in ("results", "comparison", "data", "records", "rows"):
            if key in payload:
                nested = _as_frame(payload[key])
                if nested is not None:
                    return nested
        if payload and all(not isinstance(value, (list, tuple, dict, pd.DataFrame)) for value in payload.values()):
            return pd.DataFrame([dict(payload)])
    if isinstance(payload, (list, tuple)):
        try:
            return pd.DataFrame(payload)
        except (TypeError, ValueError):
            return None
    return None


def _clean_model_name(value: Any) -> str:
    return re.sub(r"^\d+\.\s*", "", str(value)).strip()


def _normalise_metric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    replacements: dict[str, str] = {}
    for column in frame.columns:
        text = str(column)
        lowered = text.lower().replace("²", "2").strip()
        compact = re.sub(r"[\s\-]+", "_", lowered)
        if text in {"Model", "model"}:
            replacements[column] = "Algorithm"
        elif compact in {"cv_r2_mean", "mean_cv_r2", "mean_cv_r2_score"}:
            replacements[column] = "Mean CV R2"
        elif compact in {"cv_r2_std", "std_cv_r2", "std_cv_r2_score"}:
            replacements[column] = "Std CV R2"
        elif compact in {"r2", "r_score", "r2_score", "r�"}:
            replacements[column] = "R2"
        elif (
            "f1" in lowered
            and "weighted" in lowered
            and not any(token in lowered for token in ("mean cv", "std cv", "cv_", "cv "))
        ):
            replacements[column] = "F1-score (weighted)"
    frame = frame.rename(columns=replacements)
    if "Algorithm" in frame.columns:
        frame["Algorithm"] = frame["Algorithm"].map(_clean_model_name)
    if frame.columns.duplicated().any():
        # Keep the first occurrence only. Explicit aliases above should prevent
        # this for supported service payloads; this guard protects old notebook
        # exports from taking down the entire dashboard.
        frame = frame.loc[:, ~frame.columns.duplicated()].copy()
    return frame


@st.cache_data(show_spinner=False) if st is not None else (lambda fn: fn)
def load_regression_notebook_tables() -> dict[str, pd.DataFrame]:
    """Read the actual regression tables embedded in the supplied notebook."""

    notebook = load_notebook("fifa-performance-regression.ipynb")
    candidates: list[pd.DataFrame] = []
    for cell in notebook.get("cells", []):
        for output in cell.get("outputs", []):
            html = output.get("data", {}).get("text/html")
            if not html:
                continue
            try:
                tables = pd.read_html(io.StringIO("".join(html)))
            except (ImportError, ValueError):
                continue
            for table in tables:
                normalised = _normalise_metric_columns(table)
                if "Algorithm" in normalised.columns:
                    candidates.append(normalised)

    baseline = pd.DataFrame()
    cv = pd.DataFrame()
    tuned = pd.DataFrame()
    for table in candidates:
        columns = {str(column) for column in table.columns}
        if {"Algorithm", "Mean CV R2", "Std CV R2"}.issubset(columns):
            cv = table[["Algorithm", "Mean CV R2", "Std CV R2"]].copy()
        elif {"Algorithm", "R2", "RMSE", "MAE"}.issubset(columns):
            compact = table[["Algorithm", "R2", "RMSE", "MAE"]].copy()
            compact["R2"] = pd.to_numeric(compact["R2"], errors="coerce")
            # The tuned table is the only two-row table in the notebook.
            if len(compact) <= 3 and {"Random Forest Regressor", "Gradient Boosting Regressor"}.issubset(set(compact["Algorithm"])):
                tuned = compact
            elif baseline.empty or len(compact) > len(baseline):
                baseline = compact

    if not baseline.empty:
        baseline = baseline.sort_values("R2", ascending=False).reset_index(drop=True)
    return {"baseline": baseline, "cv": cv, "tuned": tuned}


def local_results(task: str) -> dict[str, pd.DataFrame]:
    if task == "regression":
        return load_regression_notebook_tables()
    if task == "classification":
        return {
            "baseline": _safe_dataset("classification_review2_comparison.csv"),
            "cv": _safe_dataset("classification_review2_cv_top2.csv"),
            "tuned": _safe_dataset("bagging_review2_tuning.csv"),
        }
    if task == "clustering":
        return {
            "baseline": _safe_dataset("clustering_review2_comparison.csv"),
            "profiles": _safe_dataset("clustering_cluster_profiles.csv"),
        }
    return {}


@st.cache_data(show_spinner=False) if st is not None else (lambda fn: fn)
def _safe_dataset(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def results_for(task: str) -> dict[str, pd.DataFrame]:
    """Prefer Agent 2's service results, then use verified persisted outputs."""

    fallback = local_results(task)
    function_name = {
        "regression": "get_regression_results",
        "classification": "get_classification_results",
        "clustering": "get_clustering_results",
    }.get(task)
    if not function_name or _service_function(function_name) is None:
        return fallback
    # Result pages should stay fast on a clean checkout.  Agent 2's result
    # services intentionally train lazily; use the already verified notebook
    # outputs until that service has a persisted bundle.  Prediction forms can
    # still request a lazy bundle explicitly on submit.
    summary = service_project_summary()
    artifact_key = {
        "regression": "regression_generated",
        "classification": "classification_generated",
        "clustering": "clustering_generated",
    }.get(task)
    artifacts = summary.get("artifacts", {}) if isinstance(summary, Mapping) else {}
    if fallback.get("baseline") is not None and not artifacts.get(artifact_key, False):
        return fallback
    try:
        payload = call_service(function_name)
    except Exception:
        return fallback
    if isinstance(payload, Mapping):
        normalised: dict[str, pd.DataFrame] = {}
        key_aliases = {
            "comparison": "baseline",
            "cv_results": "cv",
            "tuning_results": "tuned",
            "baseline_vs_tuned": "tuned",
        }
        for source_key, target_key in key_aliases.items():
            frame = _as_frame(payload.get(source_key))
            if frame is None:
                continue
            frame = _normalise_metric_columns(frame)
            if task == "classification":
                frame = frame.rename(
                    columns={
                        "Model": "Algorithm",
                        "Precision_weighted": "Precision",
                        "Recall_weighted": "Recall",
                        "F1_weighted": "F1-score (weighted)",
                        "ROC_AUC_OvR": "ROC-AUC (OvR)",
                        "CV_F1_weighted_Mean": "Mean 3-Fold Weighted F1",
                        "CV_F1_weighted_Std": "Std",
                    }
                )
                if target_key == "cv":
                    frame = frame[[column for column in ["Algorithm", "Mean 3-Fold Weighted F1", "Std"] if column in frame.columns]]
            elif task == "clustering":
                frame = frame.rename(
                    columns={
                        "Silhouette_Score": "Silhouette Score",
                        "Davies_Bouldin_Index": "Davies-Bouldin Index",
                        "Calinski_Harabasz_Index": "Calinski-Harabasz Index",
                    }
                )
            elif task == "regression":
                if target_key == "baseline":
                    frame = frame[[column for column in ["Algorithm", "R2", "RMSE", "MAE"] if column in frame.columns]]
                elif target_key == "cv":
                    frame = frame[[column for column in ["Algorithm", "Mean CV R2", "Std CV R2"] if column in frame.columns]]
            if "Algorithm" in frame.columns:
                frame["Algorithm"] = frame["Algorithm"].map(_clean_model_name)
            normalised[target_key] = frame
        if normalised:
            return {**fallback, **normalised}
    frame = _as_frame(payload)
    if frame is not None:
        return {**fallback, "baseline": frame}
    return fallback


def service_project_summary() -> Any | None:
    """Read the optional summary payload when Agent 2 exposes it."""

    if _service_function("get_project_summary") is None:
        return None
    try:
        return call_service("get_project_summary")
    except Exception:
        return None


def service_confusion_matrix(model_label: str) -> pd.DataFrame | None:
    """Return a numeric confusion matrix when the fitted service exposes one."""

    if _service_function("get_classification_results") is None:
        return None
    aliases = {
        "Decision Tree": "Decision Tree Classifier",
        "Random Forest": "Random Forest Classifier",
        "AdaBoost": "AdaBoost Classifier",
        "Gradient Boosting": "Gradient Boosting Classifier",
        "Bagging": "Bagging Classifier",
    }
    try:
        payload = call_service("get_classification_results")
    except Exception:
        return None
    if not isinstance(payload, Mapping):
        return None
    matrices = payload.get("confusion_matrices")
    if not isinstance(matrices, Mapping):
        return None
    matrix = matrices.get(aliases.get(model_label, model_label))
    if isinstance(matrix, pd.DataFrame):
        return matrix.copy()
    if isinstance(matrix, Mapping):
        try:
            return pd.DataFrame(matrix)
        except (TypeError, ValueError):
            return None
    return None


def service_dataset_preview(dataset_name: str) -> pd.DataFrame | None:
    """Read an optional service preview without making the explorer depend on it."""

    if _service_function("get_dataset_preview") is None:
        return None
    candidates = [dataset_name, DATASET_FILES.get(dataset_name, "").replace(".csv", "")]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            frame = _as_frame(call_service("get_dataset_preview", candidate))
        except Exception:
            continue
        if frame is not None and not frame.empty:
            return frame
    return None


def _safe_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.apply(pd.to_numeric, errors="coerce") if not frame.empty else frame


def _classification_frame() -> pd.DataFrame:
    frame = results_for("classification").get("baseline", pd.DataFrame()).copy()
    if frame.empty:
        return frame
    frame = _normalise_metric_columns(frame)
    if "Algorithm" in frame:
        frame["Algorithm"] = frame["Algorithm"].map(_clean_model_name)
    return frame


def _clustering_frame() -> pd.DataFrame:
    frame = results_for("clustering").get("baseline", pd.DataFrame()).copy()
    if frame.empty:
        return frame
    frame = _normalise_metric_columns(frame)
    return frame


def _dataset_or_empty(name: str) -> pd.DataFrame:
    try:
        return load_dataset(name)
    except (FileNotFoundError, pd.errors.ParserError, UnicodeDecodeError, OSError):
        return pd.DataFrame()


def _engineer_nutrition(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if frame.empty:
        return frame
    required = ["protein_g", "carbohydrates_g", "fat_g"]
    if not set(required).issubset(frame.columns):
        return pd.DataFrame()
    frame["total_macros_g"] = frame[required].sum(axis=1)
    frame["fat_carb_ratio"] = frame["fat_g"] / (frame["carbohydrates_g"] + 1)
    frame["protein_carb_ratio"] = frame["protein_g"] / (frame["carbohydrates_g"] + 1)
    return frame


def _engineer_fifa(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if frame.empty:
        return frame
    required = {"match_date", "minutes_played", "goals", "assists", "sprint_distance_km", "distance_covered_km", "successful_passes", "total_passes"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    dates = pd.to_datetime(frame.get("match_date"), errors="coerce")
    frame["match_year"] = dates.dt.year
    frame["match_month"] = dates.dt.month
    minutes = pd.to_numeric(frame["minutes_played"], errors="coerce")
    frame["goal_contribution_rate"] = (frame["goals"] + frame["assists"]) / (minutes / 90)
    frame["work_rate_intensity"] = frame["sprint_distance_km"] / (frame["distance_covered_km"] + 0.001)
    frame["passing_efficiency"] = frame["successful_passes"] / (frame["total_passes"] + 1)
    return frame


def numeric_defaults(dataset_name: str, features: Sequence[str], engineer: str | None = None) -> dict[str, float]:
    frame = _dataset_or_empty(dataset_name)
    if engineer == "nutrition":
        frame = _engineer_nutrition(frame)
    elif engineer == "fifa":
        frame = _engineer_fifa(frame)
    defaults: dict[str, float] = {}
    for feature in features:
        if feature in frame.columns:
            value = pd.to_numeric(frame[feature], errors="coerce").median()
        else:
            value = 0.0
        defaults[feature] = float(value) if pd.notna(value) and math.isfinite(float(value)) else 0.0
    return defaults


def _read_notebook_image(notebook_name: str, cell_index: int, occurrence: int = 0) -> bytes | None:
    try:
        notebook = load_notebook(notebook_name)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    cell = notebook.get("cells", [])[cell_index] if cell_index < len(notebook.get("cells", [])) else {}
    found = 0
    for output in cell.get("outputs", []):
        raw = output.get("data", {}).get("image/png")
        if raw is None:
            continue
        if found == occurrence:
            try:
                return base64.b64decode("".join(raw) if isinstance(raw, list) else raw)
            except (ValueError, TypeError):
                return None
        found += 1
    return None


def _read_notebook_images(notebook_name: str, cell_index: int) -> list[bytes]:
    """Return all PNG outputs from a notebook cell in their authored order."""

    try:
        notebook = load_notebook(notebook_name)
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    if cell_index >= len(notebook.get("cells", [])):
        return []
    images: list[bytes] = []
    for output in notebook["cells"][cell_index].get("outputs", []):
        raw = output.get("data", {}).get("image/png")
        if raw is None:
            continue
        try:
            images.append(base64.b64decode("".join(raw) if isinstance(raw, list) else raw))
        except (ValueError, TypeError):
            continue
    return images


def notebook_plot(key: str) -> bytes | None:
    spec = PLOT_CELL_MAP.get(key)
    if not spec:
        return None
    notebook_name, cell_index, occurrence = spec
    return _read_notebook_image(notebook_name, cell_index, occurrence)


def _format_pct(value: Any, digits: int = 1) -> str:
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def _format_num(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _metric_value(frame: pd.DataFrame, column: str, mode: str = "max") -> Any:
    if frame.empty or column not in frame.columns:
        return None
    series = pd.to_numeric(frame[column], errors="coerce").dropna()
    if series.empty:
        return None
    return series.max() if mode == "max" else series.min()


def render_sidebar() -> str:
    _require_streamlit()
    with st.sidebar:
        render_brand()
        st.markdown('<div class="sidebar-meta">A faculty-ready view of the capstone experiments, evaluation outputs, and deployment boundary.</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-rule"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-meta" style="letter-spacing:.1em;text-transform:uppercase;font-weight:700;margin-bottom:.55rem;">Workspace</div>', unsafe_allow_html=True)
        page = st.radio(
            "Dashboard section",
            [
                "Project Overview",
                "Regression Lab",
                "Classification Lab",
                "Clustering Lab",
                "Model Comparison",
                "Dataset Explorer",
            ],
            label_visibility="collapsed",
        )
        st.markdown('<div class="sidebar-rule"></div>', unsafe_allow_html=True)
        render_service_status(
            service_available(),
            "Predictions use Agent 2's service functions when available." if service_available() else "CSV results and notebook outputs are available; live prediction endpoints are still pending.",
        )
        st.markdown('<div class="sidebar-rule"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-meta"><strong>Source boundary</strong><br>Read-only access to supplied datasets, comparison CSVs, and notebook outputs. No source asset is overwritten by this app.</div>', unsafe_allow_html=True)
        st.markdown('<div class="footer-note" style="text-align:left;padding-top:2rem;">Machine Learning Capstone<br>23CSE301 · Academic year 2026–27</div>', unsafe_allow_html=True)
    return page


def _safe_count(dataset_name: str) -> int | None:
    frame = _dataset_or_empty(dataset_name)
    return int(len(frame)) if not frame.empty else None


def _best_regression() -> tuple[str, Any] | None:
    frame = results_for("regression").get("baseline", pd.DataFrame())
    if frame.empty or "R2" not in frame.columns or "Algorithm" not in frame.columns:
        return None
    row = frame.sort_values("R2", ascending=False).iloc[0]
    return str(row["Algorithm"]), row["R2"]


def _best_classification() -> tuple[str, Any] | None:
    frame = _classification_frame()
    metric = "F1-score (weighted)"
    if frame.empty or metric not in frame.columns or "Algorithm" not in frame.columns:
        return None
    row = frame.assign(_metric=pd.to_numeric(frame[metric], errors="coerce")).sort_values("_metric", ascending=False).iloc[0]
    return str(row["Algorithm"]), row[metric]


def _best_clustering() -> tuple[str, Any] | None:
    frame = _clustering_frame()
    metric = "Silhouette Score"
    if frame.empty or metric not in frame.columns or "Algorithm" not in frame.columns:
        return None
    row = frame.assign(_metric=pd.to_numeric(frame[metric], errors="coerce")).sort_values("_metric", ascending=False).iloc[0]
    return str(row["Algorithm"]), row[metric]


def render_overview() -> None:
    render_page_header(
        "Executive view / 01",
        "Machine Learning Capstone",
        "A single, evidence-led workspace for the FIFA performance regression, nutrition classification, and mall customer segmentation tracks.",
        [("23CSE301 Machine Learning", "blue"), ("Academic year 2026–27", "violet"), ("Read-only analysis layer", "teal")],
    )

    fifa_rows = _safe_count("FIFA performance · raw")
    nutrition_rows = _safe_count("Nutrition · raw")
    mall_rows = _safe_count("Mall customers · raw")
    classification_best = _best_classification()
    regression_best = _best_regression()
    clustering_best = _best_clustering()
    metric_grid(
        [
            {"label": "Datasets in scope", "value": "3", "caption": "FIFA · Nutrition · Mall Customers", "accent": BLUE},
            {"label": "Records analysed", "value": f"{(fifa_rows or 0) + (nutrition_rows or 0) + (mall_rows or 0):,}", "caption": "Across the three supplied raw files", "accent": TEAL},
            {"label": "Algorithms compared", "value": "22", "caption": "10 regression · 10 classification · 2 clustering", "accent": VIOLET},
            {"label": "Live service", "value": "Ready" if service_available() else "Pending", "caption": "Agent 2 prediction layer", "accent": TEAL if service_available() else AMBER},
        ],
        columns=4,
    )
    if service_project_summary() is not None:
        source_note("Agent 2 project summary is connected. The cards above still use local CSV/notebook evidence so the dashboard remains useful when the service is offline.")

    section_header("Three tracks, one evaluation story", "The interface follows the project structure rather than forcing incompatible metrics into one score.")
    track_cols = st.columns(3)
    track_content = [
        ("01", "Regression", "FIFA player match records", "Continuous performance_score prediction with R², RMSE, MAE, cross-validation, and residual diagnostics.", "10 algorithms", BLUE),
        ("02", "Classification", "Nutrition composition records", "Eleven grouped food classes evaluated with weighted metrics and one-vs-rest ROC-AUC.", "10 algorithms", TEAL),
        ("03", "Clustering", "Mall customer profiles", "Unsupervised segmentation using three scaled behavioural features and internal validation.", "2 algorithms", VIOLET),
    ]
    for col, (index, title, dataset, description, count, accent) in zip(track_cols, track_content):
        with col:
            st.markdown(
                f'<div class="track-card" style="--accent:{accent}"><div class="track-index">{index}</div><h3>{title}</h3>'
                f'<div class="pill" style="display:inline-flex">{dataset}</div><p>{description}</p>'
                f'<div class="track-stat"><span>Implemented</span><strong>{count}</strong></div></div>',
                unsafe_allow_html=True,
            )

    section_header("Project workflow", "Every page makes the boundary between persisted evidence and deployable service behaviour visible.")
    render_workflow(["Data", "Preprocessing", "Training", "Evaluation", "Insights"])

    section_header("Dataset footprint", "Counts and feature dimensions are calculated from the supplied files, not hard-coded claims.")
    dataset_rows = pd.DataFrame(
        [
            {"Track": "Regression", "Dataset": "FIFA performance", "Records": fifa_rows, "Raw columns": 75, "Model features": 52, "Target / structure": "performance_score"},
            {"Track": "Classification", "Dataset": "Nutrition", "Records": nutrition_rows, "Raw columns": 24, "Model features": 25, "Target / structure": "category_group · 11 classes"},
            {"Track": "Clustering", "Dataset": "Mall customers", "Records": mall_rows, "Raw columns": 5, "Model features": 3, "Target / structure": "No target · k = 6"},
        ]
    )
    show_table(dataset_rows, height=188)

    section_header("Verified result snapshot", "The snapshot below uses the notebook-embedded regression table and persisted Review 2 comparison CSVs.")
    result_cols = st.columns(3)
    snapshots = [
        (result_cols[0], "Regression leader", regression_best, "R²", BLUE, "Higher is better"),
        (result_cols[1], "Classification leader", classification_best, "Weighted F1", TEAL, "Baseline comparison"),
        (result_cols[2], "Clustering leader", clustering_best, "Silhouette", VIOLET, "Higher is better"),
    ]
    for col, label, result, metric, accent, note in snapshots:
        with col:
            if result:
                name, value = result
                value_text = _format_pct(value, 2) if metric in {"R²", "Weighted F1"} else _format_num(value, 4)
                metric_grid([{"label": label, "value": value_text, "caption": f"{name} · {note}", "accent": accent}], columns=1)
            else:
                empty_state(label, "No persisted comparison table is available.")

    if not service_available():
        callout(
            "The evidence layer is ready, but live prediction endpoints are not yet present in this workspace. The prediction forms remain connected to the agreed Agent 2 contract and will activate automatically when ml_services.py is added.",
            "warn",
        )


def _pretty_feature_name(feature: str) -> str:
    return feature.replace("_", " ").replace("(k$)", "(k$)").strip().title()


def _input_step(value: float) -> float:
    if abs(value) >= 1000:
        return 100.0
    if abs(value) >= 10 and float(value).is_integer():
        return 1.0
    return 0.01


def render_feature_form(
    *,
    form_key: str,
    features: Sequence[str],
    defaults: Mapping[str, float],
    title: str,
    help_text: str,
    columns: int = 3,
) -> tuple[bool, dict[str, float]]:
    """Render a compact, explicit-submit feature vector form."""

    values: dict[str, float] = {}
    with st.form(form_key, clear_on_submit=False):
        st.markdown(f"**{title}**")
        st.caption(help_text)
        form_cols = st.columns(columns)
        for index, feature in enumerate(features):
            default = float(defaults.get(feature, 0.0))
            with form_cols[index % columns]:
                values[feature] = st.number_input(
                    _pretty_feature_name(feature),
                    value=default,
                    step=_input_step(default),
                    format="%.4f",
                    key=f"{form_key}_{feature}",
                    help=f"Service feature key: {feature}",
                )
        submitted = st.form_submit_button("Run prediction", type="primary", width="stretch")
    return submitted, values


def render_prediction_payload(payload: Any, *, title: str = "Prediction returned") -> None:
    """Display a service response without assuming one serialization shape."""

    if payload is None:
        empty_state("No prediction returned", "The service returned an empty response.")
        return
    if isinstance(payload, Mapping):
        if str(payload.get("status", "ok")).lower() == "error":
            callout(str(payload.get("error", "The service returned an error.")), "warn")
            return
        prediction = payload.get(
            "prediction",
            payload.get(
                "predicted_value",
                payload.get("predicted_class", payload.get("cluster", payload.get("cluster_label"))),
            ),
        )
        model = payload.get("model", payload.get("algorithm", "Configured service"))
        if prediction is not None:
            metric_grid(
                [{"label": title, "value": str(prediction), "caption": str(model), "accent": TEAL}],
                columns=1,
            )
        probabilities = payload.get("probabilities", payload.get("class_probabilities"))
        if probabilities is not None:
            if isinstance(probabilities, Mapping):
                probability_frame = pd.DataFrame(
                    {"Class": list(probabilities.keys()), "Probability": list(probabilities.values())}
                )
            else:
                probability_frame = pd.DataFrame(probabilities)
            if not probability_frame.empty:
                if "Probability" in probability_frame.columns:
                    probability_frame["Probability"] = pd.to_numeric(probability_frame["Probability"], errors="coerce")
                    probability_frame["Probability"] = probability_frame["Probability"].map(lambda value: _format_pct(value, 2))
                show_table(probability_frame)
        extra = {key: value for key, value in payload.items() if key not in {"status", "error", "prediction", "predicted_value", "predicted_class", "cluster", "cluster_label", "model", "algorithm", "probabilities", "class_probabilities"}}
        if extra:
            with st.expander("Service response details"):
                st.json(extra)
        return
    metric_grid([{"label": title, "value": str(payload), "caption": "Configured service", "accent": TEAL}], columns=1)


def _service_prediction_error(error: Exception) -> None:
    callout(_friendly_service_error(error), "warn")


def render_regression() -> None:
    render_page_header(
        "Experiment / 02",
        "Regression Lab",
        "FIFA match-level performance modelling, with the notebook's ten-model benchmark and diagnostics kept separate from the live prediction boundary.",
        [("Target · performance_score", "blue"), ("52 numeric predictors", "violet"), ("10 algorithms", "teal")],
    )

    results = results_for("regression")
    baseline = results.get("baseline", pd.DataFrame()).copy()
    cv = results.get("cv", pd.DataFrame()).copy()
    tuned = results.get("tuned", pd.DataFrame()).copy()
    if not baseline.empty:
        baseline["Algorithm"] = baseline["Algorithm"].map(_clean_model_name)
    best = baseline.sort_values("R2", ascending=False).iloc[0] if not baseline.empty and "R2" in baseline.columns else None
    tuned_best = tuned.sort_values("R2", ascending=False).iloc[0] if not tuned.empty and "R2" in tuned.columns else None

    metric_grid(
        [
            {"label": "Best baseline R²", "value": _format_pct(best["R2"], 2) if best is not None else "—", "caption": str(best["Algorithm"]) if best is not None else "Notebook table unavailable", "accent": BLUE},
            {"label": "Baseline RMSE", "value": _format_num(best["RMSE"], 2) if best is not None else "—", "caption": "Lower is better · score points", "accent": TEAL},
            {"label": "Baseline MAE", "value": _format_num(best["MAE"], 2) if best is not None else "—", "caption": "Lower is better · score points", "accent": VIOLET},
            {"label": "Cross-validation", "value": "5-fold", "caption": "Top two baseline models", "accent": AMBER},
        ],
        columns=4,
    )

    section_header("Model comparison", "The notebook comparison is ranked by R²; RMSE and MAE remain visible because they measure different error behaviour.")
    if baseline.empty:
        empty_state("Regression comparison unavailable", "The dashboard could not read a comparison table from fifa-performance-regression.ipynb.")
    else:
        display_frame = baseline.copy()
        for column in ["R2", "RMSE", "MAE"]:
            if column in display_frame.columns:
                display_frame[column] = pd.to_numeric(display_frame[column], errors="coerce").round(4)
        show_table(display_frame, height=390)
        chart_cols = st.columns(2)
        with chart_cols[0]:
            render_figure(comparison_bar(baseline, "R2", title="Baseline R² by algorithm", color=BLUE), height=390, key="regression-r2")
        with chart_cols[1]:
            render_figure(comparison_bar(baseline, "RMSE", title="Baseline RMSE by algorithm", color=TEAL), height=390, key="regression-rmse")

    if best is not None:
        callout(
            f"Baseline selection: <strong>{best['Algorithm']}</strong> leads on R² at {_format_pct(best['R2'], 2)}. The notebook's tuned comparison separately selects <strong>{tuned_best['Algorithm'] if tuned_best is not None else 'the persisted tuned leader'}</strong>; these are reported as two evidence states rather than silently merged.",
            "teal",
        )

    section_header("Five-fold validation", "Cross-validation is shown for the two strongest baseline models, as required by the project guidelines.")
    if cv.empty:
        empty_state("Cross-validation table unavailable", "The notebook does not contain a readable five-fold output table.")
    else:
        show_table(cv.round(4), height=150)
        if "Mean CV R2" in cv.columns:
            render_figure(comparison_bar(cv, "Mean CV R2", title="Mean five-fold R²", color=VIOLET, height=290), height=290, key="regression-cv")

    section_header("Diagnostics", "Notebook-captured figures are rendered from the supplied output cells; no new model is trained by the UI.")
    diag_cols = st.columns(3)
    with diag_cols[0]:
        render_notebook_image(notebook_plot("regression_predicted_actual"), "Predicted versus actual · supplied notebook output")
    with diag_cols[1]:
        render_notebook_image(notebook_plot("regression_residuals"), "Residual analysis · supplied notebook output")
    with diag_cols[2]:
        render_notebook_image(notebook_plot("regression_importance"), "Top feature importance · supplied notebook output")
    source_note("The three diagnostic images are decoded from the executed fifa-performance-regression.ipynb outputs. They are not re-estimated on widget changes.")

    section_header("Interactive FIFA prediction", "The form sends the exact 52-feature contract to Agent 2's predict_regression(features) function.")
    if not service_available() or _service_function("predict_regression") is None:
        callout("Live regression predictions are unavailable because ml_services.py is not present. The feature form below is intentionally kept visible as the integration point and does not fabricate a score.", "warn")
    with st.expander("Open 52-feature prediction form", expanded=False):
        defaults = numeric_defaults("FIFA performance · preprocessed", REGRESSION_FEATURES, engineer="fifa")
        submitted, features = render_feature_form(
            form_key="regression_prediction_form",
            features=REGRESSION_FEATURES,
            defaults=defaults,
            title="FIFA performance input",
            help_text="Defaults are medians from the supplied preprocessed file. Replace them with a valid feature vector before submitting.",
            columns=4,
        )
        if submitted:
            if any(not math.isfinite(value) for value in features.values()):
                callout("Every input must be finite. Check the highlighted numeric fields and submit again.", "warn")
            else:
                try:
                    render_prediction_payload(call_service("predict_regression", features), title="Predicted performance_score")
                except Exception as error:
                    _service_prediction_error(error)


def _nutrition_classes() -> list[str]:
    frame = _dataset_or_empty("Nutrition · raw")
    if frame.empty or "category" not in frame.columns:
        return []
    top = frame["category"].value_counts().head(10).index
    return sorted(frame["category"].where(frame["category"].isin(top), "Other").dropna().unique().tolist())


def render_classification() -> None:
    render_page_header(
        "Experiment / 03",
        "Classification Lab",
        "Nutrition composition classification across all ten required algorithms, using the Review 2 consolidated metrics and the tuned Bagging comparison.",
        [("Target · category_group", "teal"), ("11 observed classes", "blue"), ("10 algorithms", "violet")],
    )

    frame = _classification_frame()
    cv = results_for("classification").get("cv", pd.DataFrame()).copy()
    tuned = results_for("classification").get("tuned", pd.DataFrame()).copy()
    classes = _nutrition_classes()
    best = None
    if not frame.empty and "F1-score (weighted)" in frame.columns:
        ranked = frame.assign(_f1=pd.to_numeric(frame["F1-score (weighted)"], errors="coerce")).sort_values("_f1", ascending=False)
        best = ranked.iloc[0]
    tuned_best = None
    if not tuned.empty and "F1-score (weighted)" in tuned.columns:
        tuned_best = tuned.assign(_f1=pd.to_numeric(tuned["F1-score (weighted)"], errors="coerce")).sort_values("_f1", ascending=False).iloc[0]

    metric_grid(
        [
            {"label": "Best baseline accuracy", "value": _format_pct(best["Accuracy"], 2) if best is not None else "—", "caption": str(best["Algorithm"]) if best is not None else "Comparison unavailable", "accent": BLUE},
            {"label": "Weighted F1", "value": _format_pct(best["F1-score (weighted)"], 2) if best is not None else "—", "caption": "Primary selection metric", "accent": TEAL},
            {"label": "ROC-AUC · OvR", "value": _format_pct(best["ROC-AUC (OvR)"], 2) if best is not None else "—", "caption": "Multiclass one-vs-rest", "accent": VIOLET},
            {"label": "Target classes", "value": str(len(classes) or 11), "caption": "Grouped nutrition categories", "accent": AMBER},
        ],
        columns=4,
    )

    section_header("Consolidated comparison", "All ten required algorithms are shown in one table. Weighted metrics account for the eleven-class target distribution.")
    if frame.empty:
        empty_state("Classification comparison unavailable", "The persisted classification_review2_comparison.csv could not be loaded.")
    else:
        display_frame = frame.copy()
        numeric_columns = ["Accuracy", "Precision", "Recall", "F1-score (weighted)", "ROC-AUC (OvR)"]
        for column in numeric_columns:
            if column in display_frame.columns:
                display_frame[column] = pd.to_numeric(display_frame[column], errors="coerce").round(4)
        show_table(display_frame, height=380)
        chart_cols = st.columns(2)
        with chart_cols[0]:
            render_figure(comparison_bar(frame, "F1-score (weighted)", title="Weighted F1 by algorithm", color=TEAL), height=390, key="classification-f1")
        with chart_cols[1]:
            render_figure(comparison_bar(frame, "ROC-AUC (OvR)", title="One-vs-rest ROC-AUC", color=VIOLET), height=390, key="classification-roc")

    if best is not None:
        callout(
            f"Best baseline justification: <strong>{best['Algorithm']}</strong> has the highest weighted F1 ({_format_pct(best['F1-score (weighted)'], 2)}) in the persisted ten-model comparison. The tuned Bagging result is shown separately below; no cross-task metric is used for this selection.",
            "teal",
        )

    section_header("Baseline versus tuned", "Review 2 tuned Bagging is compared with its own baseline so the uplift is attributable and easy to audit.")
    if tuned.empty:
        empty_state("Tuning comparison unavailable", "The supplied bagging_review2_tuning.csv could not be loaded.")
    else:
        show_table(tuned.round(4), height=140)
        if best is not None and tuned_best is not None:
            baseline_f1 = float(tuned.loc[tuned["Version"].astype(str).str.contains("Baseline", case=False, na=False), "F1-score (weighted)"].iloc[0]) if "Version" in tuned.columns and tuned["Version"].astype(str).str.contains("Baseline", case=False, na=False).any() else float(best["F1-score (weighted)"])
            delta = float(tuned_best["F1-score (weighted)"]) - baseline_f1
            metric_grid(
                [{"label": "Tuned Bagging uplift", "value": f"+{delta:.4f}", "caption": "Weighted F1 versus Bagging baseline", "accent": TEAL}],
                columns=1,
            )

    section_header("Confusion matrices", "Use the control to focus the evaluation view. Numeric matrices from the fitted service are preferred, with the supplied notebook image as a transparent fallback.")
    selected_model = st.selectbox("Model evaluation view", CLASSIFICATION_ALGORITHMS, index=8, key="classification_model_view")
    service_matrix = service_confusion_matrix(selected_model)
    if service_matrix is not None and not service_matrix.empty:
        render_figure(
            heatmap_chart(service_matrix, title=f"{selected_model} confusion matrix · live service", height=440),
            height=440,
            key="classification-service-cm",
        )
        source_note("Numeric confusion-matrix cells are read from the fitted classification service bundle.")
    elif selected_model == "Bagging":
        render_notebook_image(notebook_plot("classification_tuned_cm"), "Tuned Bagging confusion matrix · supplied notebook output")
        source_note("The tuned Bagging matrix is decoded from the executed nutrition-classification.ipynb output. Numeric matrix cells were not exported to a CSV.")
    elif selected_model in {"Random Forest", "AdaBoost", "Gradient Boosting", "MLP Classifier"}:
        part_b_order = ["Random Forest", "AdaBoost", "Gradient Boosting", "Bagging", "MLP Classifier"]
        occurrence = part_b_order.index(selected_model)
        matrix_images = _read_notebook_images("nutrition-classification.ipynb", 29)
        image = matrix_images[occurrence] if occurrence < len(matrix_images) else None
        render_notebook_image(image, f"{selected_model} confusion matrix · supplied notebook output")
        source_note("This matrix is decoded from the Review 2 Part-B notebook output. Numeric matrix cells were not exported to a CSV.")
    else:
        empty_state(
            f"Confusion matrix for {selected_model} is not persisted",
            "The source notebook contains plotted matrices for the Review 2 section, but the current workspace does not expose numeric matrix arrays or a service response for this model. No synthetic matrix is shown.",
        )

    if not cv.empty:
        section_header("Cross-validation check", "The two strongest baseline models are carried forward into the persisted three-fold weighted-F1 check.")
        show_table(cv.round(4), height=135)

    section_header("Interactive nutrition prediction", "The form sends the exact 25-feature engineered vector to Agent 2's predict_classification(features, model_name) function.")
    if not service_available() or _service_function("predict_classification") is None:
        callout("Live classification predictions are unavailable because ml_services.py is not present. The form remains an explicit integration point and will never substitute a fabricated class.", "warn")
    with st.expander("Open 25-feature prediction form", expanded=False):
        defaults = numeric_defaults("Nutrition · preprocessed", CLASSIFICATION_FEATURES, engineer="nutrition")
        submitted, features = render_feature_form(
            form_key="classification_prediction_form",
            features=CLASSIFICATION_FEATURES,
            defaults=defaults,
            title="Nutrition feature vector",
            help_text="Inputs use the same engineered names as the notebook: total macros plus the two ratio features are included.",
            columns=4,
        )
        if submitted:
            if any(not math.isfinite(value) for value in features.values()):
                callout("Every input must be finite. Check the numeric fields and submit again.", "warn")
            else:
                try:
                    render_prediction_payload(
                        call_service("predict_classification", features, selected_model),
                        title="Predicted nutrition category",
                    )
                except Exception as error:
                    _service_prediction_error(error)


def _build_cluster_visual_data_uncached() -> dict[str, Any]:
    """Recreate only the supplied notebook's 200-row visual diagnostics.

    This is a cached visualization fallback, not a prediction service.  The
    interactive customer assignment below remains exclusively delegated to
    Agent 2's fitted-pipeline service.
    """

    frame = _dataset_or_empty("Mall customers · raw")
    if frame.empty or not set(CLUSTER_FEATURES).issubset(frame.columns):
        return {}
    try:
        from scipy.cluster.hierarchy import linkage
        from sklearn.cluster import AgglomerativeClustering, KMeans
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {}

    work = frame.copy()
    x = work[CLUSTER_FEATURES].apply(pd.to_numeric, errors="coerce")
    if x.isna().any().any():
        x = x.fillna(x.median())
    if x.isna().any().any() or len(x) < 7:
        return {}
    x_scaled = StandardScaler().fit_transform(x)
    chosen_k = 6
    if len(x) <= chosen_k:
        return {}
    kmeans = KMeans(n_clusters=chosen_k, init="k-means++", n_init=10, random_state=42)
    kmeans_labels = kmeans.fit_predict(x_scaled)
    agg = AgglomerativeClustering(n_clusters=chosen_k, linkage="ward")
    agg_labels = agg.fit_predict(x_scaled)

    elbow_rows = []
    for k in range(2, min(10, len(x) - 1) + 1):
        model = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        labels = model.fit_predict(x_scaled)
        elbow_rows.append(
            {
                "k": k,
                "Inertia": model.inertia_,
                "Silhouette": silhouette_score(x_scaled, labels),
            }
        )

    pca_values = PCA(n_components=2, random_state=42).fit_transform(x_scaled)
    pca_frame = pd.DataFrame({"PC1": pca_values[:, 0], "PC2": pca_values[:, 1], "K-Means cluster": kmeans_labels.astype(str), "Agglomerative cluster": agg_labels.astype(str)})
    perplexity = min(30, max(2, (len(x) - 1) // 3))
    tsne_values = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca", learning_rate="auto").fit_transform(x_scaled)
    tsne_frame = pd.DataFrame({"t-SNE 1": tsne_values[:, 0], "t-SNE 2": tsne_values[:, 1], "K-Means cluster": kmeans_labels.astype(str)})

    profiles = []
    for algorithm, labels in [("K-Means", kmeans_labels), ("Agglomerative Hierarchical (Ward)", agg_labels)]:
        profile = x.copy()
        profile["Cluster"] = labels
        grouped = profile.groupby("Cluster", as_index=False)[CLUSTER_FEATURES].mean()
        grouped.insert(0, "Algorithm", algorithm)
        profiles.append(grouped)
    profile_frame = pd.concat(profiles, ignore_index=True)

    linkage_rows = []
    for method in ["ward", "complete", "average"]:
        labels = AgglomerativeClustering(n_clusters=chosen_k, linkage=method).fit_predict(x_scaled)
        linkage_rows.append(
            {
                "Linkage": method.title(),
                "Silhouette Score": silhouette_score(x_scaled, labels),
                "Davies-Bouldin Index": davies_bouldin_score(x_scaled, labels),
                "Calinski-Harabasz Index": calinski_harabasz_score(x_scaled, labels),
            }
        )

    return {
        "rows": work,
        "elbow": pd.DataFrame(elbow_rows),
        "pca": pca_frame,
        "tsne": tsne_frame,
        "profiles": profile_frame,
        "linkage": pd.DataFrame(linkage_rows),
        "dendrogram_matrix": linkage(x_scaled, method="ward"),
    }


@st.cache_data(show_spinner=False) if st is not None else (lambda fn: fn)
def build_cluster_visual_data() -> dict[str, Any]:
    """Return diagnostics or an empty payload so notebook fallbacks remain usable."""

    try:
        return _build_cluster_visual_data_uncached()
    except (ValueError, RuntimeError, TypeError, FloatingPointError):
        return {}


def _cluster_interpretation(profile: pd.DataFrame, cluster: int) -> str:
    if profile.empty or cluster not in profile["Cluster"].astype(int).tolist():
        return "No profile is available for this cluster."
    row = profile.loc[profile["Cluster"].astype(int) == cluster].iloc[0]
    means = profile[CLUSTER_FEATURES].mean(numeric_only=True)
    descriptors: list[str] = []
    for feature, short_name in [("Age", "age"), ("Annual Income (k$)", "income"), ("Spending Score (1-100)", "spending")]:
        value = float(row[feature])
        average = float(means[feature])
        if value >= average * 1.12:
            descriptors.append(f"higher {short_name}")
        elif value <= average * 0.88:
            descriptors.append(f"lower {short_name}")
        else:
            descriptors.append(f"mid-range {short_name}")
    return ", ".join(descriptors).capitalize() + "."


def render_clustering() -> None:
    render_page_header(
        "Experiment / 04",
        "Clustering Lab",
        "Customer segmentation with a strict feature boundary: CustomerID and Gender stay descriptive, while Age, income, and spending score form the scaled clustering matrix.",
        [("Feature set · 3 numeric fields", "violet"), ("Chosen k · 6", "blue"), ("No target label", "teal")],
    )

    results = results_for("clustering")
    metrics = _clustering_frame()
    profiles_saved = results.get("profiles", pd.DataFrame()).copy()
    visual_data = build_cluster_visual_data()
    kmeans_row = metrics.loc[metrics["Algorithm"].astype(str).str.contains("K-Means", case=False, na=False)].iloc[0] if not metrics.empty and metrics["Algorithm"].astype(str).str.contains("K-Means", case=False, na=False).any() else None
    agg_row = metrics.loc[~metrics["Algorithm"].astype(str).str.contains("K-Means", case=False, na=False)].iloc[0] if not metrics.empty and len(metrics) > 1 else None
    metric_grid(
        [
            {"label": "K-Means silhouette", "value": _format_num(kmeans_row["Silhouette Score"], 4) if kmeans_row is not None else "—", "caption": "Higher is better", "accent": BLUE},
            {"label": "K-Means DB index", "value": _format_num(kmeans_row["Davies-Bouldin Index"], 4) if kmeans_row is not None else "—", "caption": "Lower is better", "accent": TEAL},
            {"label": "Ward silhouette", "value": _format_num(agg_row["Silhouette Score"], 4) if agg_row is not None else "—", "caption": "Higher is better", "accent": VIOLET},
            {"label": "Clusters", "value": "6", "caption": "Fixed from the notebook configuration", "accent": AMBER},
        ],
        columns=4,
    )

    section_header("Feature boundary", "CustomerID is an identifier and Gender remains descriptive; neither enters the fitted three-dimensional clustering space.")
    feature_cols = st.columns(3)
    feature_details = [
        ("Age", "Numeric", "Customer age in years"),
        ("Annual Income (k$)", "Numeric", "Annual income in thousands"),
        ("Spending Score (1-100)", "Numeric", "Mall spending score"),
    ]
    for col, (name, dtype, detail) in zip(feature_cols, feature_details):
        with col:
            metric_grid([{"label": name, "value": dtype, "caption": detail, "accent": VIOLET}], columns=1)

    section_header("Elbow and silhouette analysis", "Cached diagnostics reproduce the supplied notebook configuration for k = 2…10 without running when a widget changes.")
    if visual_data:
        elbow_cols = st.columns(2)
        with elbow_cols[0]:
            render_figure(metric_line(visual_data["elbow"], "k", "Inertia", title="Elbow curve · inertia", color=BLUE), height=330, key="cluster-elbow")
        with elbow_cols[1]:
            render_figure(metric_line(visual_data["elbow"], "k", "Silhouette", title="Silhouette by k", color=TEAL), height=330, key="cluster-silhouette")
    else:
        render_notebook_image(notebook_plot("clustering_elbow"), "Elbow and silhouette output · supplied notebook")
        callout("The live visualization fallback needs scikit-learn and SciPy. The supplied notebook figure is shown when available.", "warn")

    section_header("Algorithm comparison and linkage", "Internal validation is reported with the project-required metrics; the three linkage variants are shown separately from the final Ward comparison.")
    if metrics.empty:
        empty_state("Clustering comparison unavailable", "The supplied clustering_review2_comparison.csv could not be loaded.")
    else:
        show_table(metrics.round(4), height=150)
    if visual_data and not visual_data["linkage"].empty:
        show_table(visual_data["linkage"].round(4), height=170)
    else:
        empty_state("Linkage comparison unavailable", "No computed linkage table is available in this runtime.")

    section_header("Cluster geometry", "PCA is a two-dimensional view of the three-feature scaled space. t-SNE is shown as an exploratory visual, not as a replacement for validation metrics.")
    if visual_data:
        plot_cols = st.columns(3)
        with plot_cols[0]:
            render_figure(scatter_chart(visual_data["pca"], "PC1", "PC2", color="K-Means cluster", title="PCA · K-Means", color_sequence=PALETTE), height=350, key="cluster-pca-kmeans")
        with plot_cols[1]:
            render_figure(scatter_chart(visual_data["pca"], "PC1", "PC2", color="Agglomerative cluster", title="PCA · Agglomerative Ward", color_sequence=PALETTE), height=350, key="cluster-pca-agg")
        with plot_cols[2]:
            render_figure(scatter_chart(visual_data["tsne"], "t-SNE 1", "t-SNE 2", color="K-Means cluster", title="t-SNE · K-Means", color_sequence=PALETTE), height=350, key="cluster-tsne")
    else:
        image_cols = st.columns(3)
        with image_cols[0]:
            render_notebook_image(notebook_plot("clustering_pca"), "PCA cluster visualisations · supplied notebook")
        with image_cols[1]:
            render_notebook_image(notebook_plot("clustering_tsne"), "t-SNE visualisation · supplied notebook")
        with image_cols[2]:
            render_notebook_image(notebook_plot("clustering_dendrogram"), "Ward dendrogram · supplied notebook")

    section_header("Dendrogram", "Ward linkage provides the hierarchical view used to inspect the final agglomerative solution.")
    render_notebook_image(notebook_plot("clustering_dendrogram"), "Ward linkage dendrogram · supplied notebook output")

    section_header("Cluster profiles", "Profiles are computed from the three clustering features and translated into concise, data-driven interpretations.")
    profile_source = visual_data.get("profiles", pd.DataFrame()) if visual_data else pd.DataFrame()
    if profile_source.empty and not profiles_saved.empty:
        # The persisted file contains side-by-side K-Means and Agglomerative columns.
        profile_source = profiles_saved
    if profile_source.empty:
        empty_state("Cluster profiles unavailable", "No profile table was returned by the local or notebook source.")
    else:
        algorithm_label = st.selectbox("Profile algorithm", ["K-Means", "Agglomerative Hierarchical (Ward)"], key="profile_algorithm")
        if set(["Algorithm", "Cluster", *CLUSTER_FEATURES]).issubset(profile_source.columns):
            selected_profile = profile_source.loc[profile_source["Algorithm"] == algorithm_label].copy()
            show_table(selected_profile.round(2), height=260)
            if not selected_profile.empty:
                interpretation_cols = st.columns(min(3, len(selected_profile)))
                for col, (_, row) in zip(interpretation_cols, selected_profile.iterrows()):
                    with col:
                        cluster_id = int(row["Cluster"])
                        metric_grid([{"label": f"Cluster {cluster_id}", "value": "Profile", "caption": _cluster_interpretation(selected_profile, cluster_id), "accent": VIOLET}], columns=1)
        else:
            show_table(profile_source.round(2), height=260)
            source_note("The persisted profile CSV stores both algorithms side by side; the interactive profile view uses the cached three-feature reconstruction when available.")

    section_header("Interactive customer segmentation", "New-customer assignment must use Agent 2's compatible fitted pipeline. K-Means is the deployable default; native AgglomerativeClustering does not expose predict().")
    if not service_available() or _service_function("assign_cluster") is None:
        callout("Live customer assignment is unavailable because ml_services.py is not present. No nearest-centroid substitute is used in the UI.", "warn")
    algorithm_labels = {
        "K-Means": "K-Means",
        "Agglomerative Hierarchical (Ward)": "Agglomerative Hierarchical (Ward)",
    }
    selected_algorithm_label = st.selectbox("Assignment algorithm", list(algorithm_labels), key="cluster_assignment_algorithm")
    selected_algorithm = algorithm_labels[selected_algorithm_label]
    defaults = numeric_defaults("Mall customers · raw", CLUSTER_FEATURES)
    with st.form("cluster_assignment_form", clear_on_submit=False):
        cluster_cols = st.columns(3)
        features: dict[str, float] = {}
        for col, feature in zip(cluster_cols, CLUSTER_FEATURES):
            with col:
                features[feature] = st.number_input(_pretty_feature_name(feature), value=defaults[feature], step=1.0, format="%.2f", key=f"cluster_input_{feature}")
        submitted = st.form_submit_button("Assign customer to cluster", type="primary", width="stretch")
    if submitted:
        try:
            render_prediction_payload(
                call_service("assign_cluster", features, algorithm_name=selected_algorithm),
                title="Assigned cluster",
            )
        except Exception as error:
            _service_prediction_error(error)


def render_comparison_workspace() -> None:
    render_page_header(
        "Synthesis / 05",
        "Model Comparison",
        "A unified comparison surface with separate task tabs. Each task keeps its own metric vocabulary and selection logic.",
        [("Sortable tables", "blue"), ("Metric-aware charts", "teal"), ("No cross-task ranking", "violet")],
    )

    regression_tab, classification_tab, clustering_tab = st.tabs(["Regression", "Classification", "Clustering"])
    with regression_tab:
        frame = results_for("regression").get("baseline", pd.DataFrame()).copy()
        if frame.empty:
            empty_state("Regression comparison unavailable", "The source notebook comparison table could not be read.")
        else:
            frame["Algorithm"] = frame["Algorithm"].map(_clean_model_name)
            metric_options = [column for column in ["R2", "RMSE", "MAE"] if column in frame.columns]
            c1, c2 = st.columns([1, 2])
            with c1:
                metric = st.selectbox("Regression metric", metric_options, format_func=lambda value: {"R2": "R² · higher is better", "RMSE": "RMSE · lower is better", "MAE": "MAE · lower is better"}.get(value, value), key="comparison_reg_metric")
            with c2:
                ascending = st.checkbox("Sort ascending", value=metric in {"RMSE", "MAE"}, key="comparison_reg_ascending")
            sortable = frame.sort_values(metric, ascending=ascending).reset_index(drop=True)
            show_table(sortable.round(4), height=390)
            render_figure(comparison_bar(frame, metric, title=f"Regression · {metric}", color=BLUE if metric == "R2" else TEAL), height=390, key="comparison-reg-chart")
            source_note("R², RMSE, and MAE describe the regression track only and are not combined with classification or clustering scores.")

    with classification_tab:
        frame = _classification_frame()
        if frame.empty:
            empty_state("Classification comparison unavailable", "The persisted Review 2 comparison table could not be loaded.")
        else:
            metric_options = [column for column in ["Accuracy", "Precision", "Recall", "F1-score (weighted)", "ROC-AUC (OvR)"] if column in frame.columns]
            c1, c2 = st.columns([1, 2])
            with c1:
                metric = st.selectbox("Classification metric", metric_options, format_func=lambda value: {"F1-score (weighted)": "Weighted F1", "ROC-AUC (OvR)": "ROC-AUC · OvR"}.get(value, value), key="comparison_cls_metric")
            with c2:
                ascending = st.checkbox("Sort ascending", value=False, key="comparison_cls_ascending")
            sortable = frame.sort_values(metric, ascending=ascending).reset_index(drop=True)
            show_table(sortable.round(4), height=390)
            render_figure(comparison_bar(frame, metric, title=f"Classification · {metric}", color=TEAL), height=390, key="comparison-cls-chart")
            source_note("All displayed classification metrics belong to the eleven-class nutrition target; ROC-AUC is one-vs-rest as required for multiclass evaluation.")

    with clustering_tab:
        frame = _clustering_frame()
        if frame.empty:
            empty_state("Clustering comparison unavailable", "The persisted Review 2 clustering table could not be loaded.")
        else:
            metric_options = [column for column in ["Silhouette Score", "Davies-Bouldin Index", "Calinski-Harabasz Index"] if column in frame.columns]
            c1, c2 = st.columns([1, 2])
            with c1:
                metric = st.selectbox("Clustering metric", metric_options, format_func=lambda value: {"Davies-Bouldin Index": "Davies-Bouldin · lower is better", "Calinski-Harabasz Index": "Calinski-Harabasz · higher is better"}.get(value, value), key="comparison_cluster_metric")
            with c2:
                ascending = st.checkbox("Sort ascending", value=metric == "Davies-Bouldin Index", key="comparison_cluster_ascending")
            sortable = frame.sort_values(metric, ascending=ascending).reset_index(drop=True)
            show_table(sortable.round(4), height=190)
            render_figure(comparison_bar(frame, metric, title=f"Clustering · {metric}", color=VIOLET), height=350, key="comparison-cluster-chart")
            source_note("Silhouette, Davies-Bouldin, and Calinski-Harabasz are internal clustering metrics. They are intentionally not ranked against predictive metrics.")


def _apply_dataset_filter(frame: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
    """Return a filtered copy for the explorer; the original CSV remains untouched."""

    if frame.empty:
        return frame
    filter_col, filter_value = st.session_state.get(f"explorer_filter_{dataset_key}", ("No filter", None))
    if filter_col == "No filter" or filter_col not in frame.columns:
        return frame
    if pd.api.types.is_numeric_dtype(frame[filter_col]):
        if isinstance(filter_value, (tuple, list)) and len(filter_value) == 2:
            low, high = filter_value
            numeric = pd.to_numeric(frame[filter_col], errors="coerce")
            return frame.loc[numeric.between(low, high, inclusive="both")]
    elif isinstance(filter_value, (list, tuple, set)):
        return frame.loc[frame[filter_col].isin(filter_value)]
    return frame


def _dataset_filter_controls(frame: pd.DataFrame, dataset_key: str) -> None:
    if frame.empty:
        return
    filter_options = ["No filter", *frame.columns.tolist()]
    filter_col = st.selectbox("Filter column", filter_options, key=f"explorer_filter_col_{dataset_key}")
    value: Any = None
    if filter_col != "No filter":
        if pd.api.types.is_numeric_dtype(frame[filter_col]):
            numeric = pd.to_numeric(frame[filter_col], errors="coerce").dropna()
            if not numeric.empty:
                low, high = float(numeric.min()), float(numeric.max())
                if math.isclose(low, high):
                    st.caption(f"Only one numeric value is present: {low:g}")
                    value = (low, high)
                else:
                    value = st.slider("Numeric range", min_value=low, max_value=high, value=(low, high), key=f"explorer_range_{dataset_key}")
        else:
            options = frame[filter_col].dropna().astype(str).sort_values().unique().tolist()
            value = st.multiselect("Allowed values", options, default=options, key=f"explorer_values_{dataset_key}")
    st.session_state[f"explorer_filter_{dataset_key}"] = (filter_col, value)


def _schema_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in frame.columns:
        rows.append(
            {
                "Column": column,
                "Data type": str(frame[column].dtype),
                "Missing": int(frame[column].isna().sum()),
                "Unique values": int(frame[column].nunique(dropna=True)),
                "Example": str(frame[column].dropna().iloc[0]) if not frame[column].dropna().empty else "—",
            }
        )
    return pd.DataFrame(rows)


def render_dataset_explorer() -> None:
    render_page_header(
        "Evidence / 06",
        "Dataset Explorer",
        "Inspect the supplied raw and preprocessed files without changing them: schemas, missingness, summary statistics, samples, filters, and distributions.",
        [("Read-only", "teal"), ("CSV-backed", "blue"), ("Filter locally", "violet")],
    )

    dataset_name = st.selectbox("Dataset", list(DATASET_FILES), key="explorer_dataset")
    frame = _dataset_or_empty(dataset_name)
    if frame.empty:
        empty_state("Dataset unavailable", f"The selected source file for {dataset_name} was not found or could not be parsed.")
        return
    service_preview = service_dataset_preview(dataset_name)
    key = re.sub(r"[^a-z0-9]+", "_", dataset_name.lower()).strip("_")
    numeric_columns = frame.select_dtypes(include=np.number).columns.tolist()
    metric_grid(
        [
            {"label": "Rows", "value": f"{len(frame):,}", "caption": dataset_name, "accent": BLUE},
            {"label": "Columns", "value": str(len(frame.columns)), "caption": "Original file shape", "accent": TEAL},
            {"label": "Missing cells", "value": f"{int(frame.isna().sum().sum()):,}", "caption": "Across all columns", "accent": VIOLET},
            {"label": "Numeric fields", "value": str(len(numeric_columns)), "caption": "Pandas dtype detection", "accent": AMBER},
        ],
        columns=4,
    )

    section_header("Filter and preview", "Filters apply only to the in-memory view; the CSV source remains unchanged.")
    _dataset_filter_controls(frame, key)
    filtered = _apply_dataset_filter(frame, key)
    sample_count = st.slider("Preview rows", min_value=5, max_value=min(100, max(5, len(filtered))), value=min(20, max(5, len(filtered))), key=f"explorer_sample_{key}")
    preview = filtered.head(sample_count)
    st.caption(f"Showing {len(preview):,} of {len(filtered):,} rows after filtering")
    show_table(preview, height=360)

    tab_preview, tab_schema, tab_summary, tab_distribution = st.tabs(["Preview", "Schema", "Summary", "Distribution"])
    with tab_preview:
        show_table(preview, height=360)
    with tab_schema:
        show_table(_schema_frame(frame), height=min(520, 80 + 42 * len(frame.columns)))
    with tab_summary:
        try:
            summary = frame.describe(include="all", datetime_is_numeric=True).transpose().reset_index().rename(columns={"index": "Column"})
        except TypeError:
            summary = frame.describe(include="all").transpose().reset_index().rename(columns={"index": "Column"})
        show_table(summary, height=420)
    with tab_distribution:
        if not numeric_columns:
            empty_state("No numeric distributions", "The selected dataset has no numeric columns for a histogram.")
        else:
            distribution_column = st.selectbox("Numeric column", numeric_columns, key=f"explorer_distribution_{key}")
            render_figure(histogram_chart(filtered, distribution_column, title=f"Distribution · {distribution_column}"), height=390, key=f"explorer_hist_{key}")
    if service_preview is not None:
        source_note("Agent 2's get_dataset_preview() is available for this dataset. Filtering and schema inspection remain backed by the local CSV so original project files stay read-only.")
    else:
        source_note("Dataset Explorer reads the supplied CSV each time the selected file changes. It does not call model training and does not overwrite original datasets.")


def render_page_safely(page_name: str, renderer: Any) -> None:
    """Keep one broken optional view from taking down the whole dashboard."""

    try:
        renderer()
    except Exception as error:  # pragma: no cover - exercised by malformed local assets
        callout(
            f"{page_name} could not render this view. The rest of the dashboard is still available; check the source files or service response and try again.",
            "warn",
        )
        with st.expander("Technical details"):
            st.code(f"{type(error).__name__}: {error}")


def main() -> None:
    _require_streamlit()
    st.set_page_config(
        page_title="ML Capstone · 23CSE301",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()
    page = render_sidebar()
    if page == "Project Overview":
        render_page_safely(page, render_overview)
    elif page == "Regression Lab":
        render_page_safely(page, render_regression)
    elif page == "Classification Lab":
        render_page_safely(page, render_classification)
    elif page == "Clustering Lab":
        render_page_safely(page, render_clustering)
    elif page == "Model Comparison":
        render_page_safely(page, render_comparison_workspace)
    elif page == "Dataset Explorer":
        render_page_safely(page, render_dataset_explorer)
    else:  # defensive fallback if Streamlit restores an unknown widget value
        render_page_safely("Project Overview", render_overview)
    st.markdown('<div class="footer-note">Machine Learning Capstone · 23CSE301 · Evidence-led local dashboard</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()

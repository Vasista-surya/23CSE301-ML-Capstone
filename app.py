from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------------------------------------------------------
# Configuration and theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ML Capstone | Analytics Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets"

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: #0b1020; color: #f2f5fb; }
    [data-testid="stHeader"] { background: rgba(11,16,32,.88); }
    [data-testid="stSidebar"] { background: #10172a; border-right: 1px solid #24304a; }
    [data-testid="stSidebar"] * { color: #e7edf8; }
    h1, h2, h3 { font-family: 'Manrope', sans-serif !important; letter-spacing: -0.035em; }
    h1 { font-weight: 800 !important; }
    .hero { padding: 1.6rem 1.8rem; border-radius: 22px;
        background: linear-gradient(120deg, #16213d 0%, #142b3d 52%, #18384a 100%);
        border: 1px solid #2c4865; margin-bottom: 1.25rem; }
    .eyebrow { color: #79e2d0; font-size: .75rem; text-transform: uppercase;
        letter-spacing: .18em; font-weight: 800; }
    .hero-sub { color: #bdcce0; font-size: 1rem; line-height: 1.65; max-width: 940px; }
    div[data-testid="stMetric"] { background: #121b30; border: 1px solid #273652;
        border-radius: 16px; padding: 15px 17px; }
    div[data-testid="stMetricLabel"] { color: #a8b8d1; }
    div[data-testid="stMetricValue"] { color: #f2f6ff; font-family: 'Manrope',sans-serif; }
    div[data-testid="stPlotlyChart"] { background: #10182b; border: 1px solid #202d47;
        border-radius: 15px; padding: 8px; }
    .pill { display:inline-block; padding:4px 9px; border-radius:30px; margin:2px 4px 2px 0;
        color:#9cf0df; background:#12352f; border:1px solid #256358; font-size:.76rem; }
    .footer { color:#8292ac; font-size:.78rem; border-top:1px solid #26334c;
        padding-top:1rem; margin-top:2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------
def find_dataset(filename: str) -> Path | None:
    """Locate a CSV in the expected repo layout or common launch directories."""
    candidates = [
        DATA_DIR / filename,
        ROOT / filename,
        Path.cwd() / "datasets" / filename,
        Path.cwd() / filename,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


@st.cache_data(show_spinner=False)
def read_csv(path_string: str) -> pd.DataFrame:
    return pd.read_csv(path_string, low_memory=False)


def optional_csv(filename: str) -> pd.DataFrame | None:
    path = find_dataset(filename)
    if path is None:
        return None
    try:
        return read_csv(str(path))
    except (OSError, UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        st.warning(f"Couldn't load `{filename}`: {exc}")
        return None


def numeric_columns(frame: pd.DataFrame, exclude: set[str] | None = None) -> list[str]:
    excluded = exclude or set()
    return [
        col for col in frame.select_dtypes(include=np.number).columns
        if col not in excluded and frame[col].nunique(dropna=True) > 1
    ]


def metric_cards(items: list[tuple[str, str, str | None]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value, help_text) in zip(cols, items):
        col.metric(label, value, help=help_text)


def chart_theme(fig, title: str | None = None):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color="#dce6f5"),
        title=dict(text=title or "", x=0.02, xanchor="left", font=dict(size=17)),
        margin=dict(l=18, r=18, t=55 if title else 20, b=22),
        colorway=["#79e2d0", "#7aa7ff", "#f6c66b", "#c79cff", "#ff8e9b", "#7fd39a"],
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="#26334b", zerolinecolor="#26334b")
    fig.update_yaxes(gridcolor="#26334b", zerolinecolor="#26334b")
    return fig


def missing_dataset_message(filename: str) -> None:
    st.warning(f"Couldn't find `{filename}`.")
    st.caption("Expected layout: keep the `datasets` folder next to `app.py` in the repository.")


# -----------------------------------------------------------------------------
# Overview
# -----------------------------------------------------------------------------
def show_overview() -> None:
    classification = optional_csv("classification_review2_comparison.csv")
    clustering = optional_csv("clustering_review2_comparison.csv")
    tuning = optional_csv("bagging_review2_tuning.csv")
    nutrition = optional_csv("nutrition.csv")
    fifa = optional_csv("fifa_preprocessed.csv")
    if fifa is None:
        fifa = optional_csv("fifa.csv")
    mall = optional_csv("mall_customers.csv")

    datasets = [("Nutrition", nutrition), ("FIFA", fifa), ("Mall Customers", mall)]
    total_rows = sum(len(frame) for _, frame in datasets if frame is not None)

    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Machine Learning Capstone · Analytics Studio</div>
          <h1 style="margin:.45rem 0 .35rem 0;">From data to decisions.</h1>
          <div class="hero-sub">A unified workspace for exploring datasets, comparing model performance,
          inspecting customer segments, and communicating results across regression, classification and clustering.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cards([
        ("Project workflows", "03", "Regression · Classification · Clustering"),
        ("Primary datasets found", f"{sum(frame is not None for _, frame in datasets)}/3", "Loaded from the repository datasets folder"),
        ("Combined dataset rows", f"{total_rows:,}" if total_rows else "—", "Across available primary datasets"),
        ("Classification models", str(len(classification)) if classification is not None else "—", "Saved model comparison results"),
    ])

    st.markdown("### Project workspace")
    cols = st.columns(3)
    summaries = {
        "Nutrition": "Food-category classification",
        "FIFA": "Player-performance regression",
        "Mall Customers": "Customer segmentation",
    }
    for col, (name, frame) in zip(cols, datasets):
        with col:
            st.markdown(f"#### {name}")
            st.caption(summaries[name])
            if frame is None:
                st.warning("Dataset not found.")
            else:
                a, b = st.columns(2)
                a.metric("Rows", f"{len(frame):,}")
                b.metric("Features", str(frame.shape[1]))
                st.caption(f"{frame.isna().sum().sum():,} missing cells")

    st.markdown("### Saved evaluation snapshot")
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("#### Classification leaderboard")
        if classification is not None and {"Algorithm", "Accuracy"}.issubset(classification.columns):
            ranking = classification.sort_values("Accuracy", ascending=True)
            fig = px.bar(
                ranking, x="Accuracy", y="Algorithm", orientation="h", color="Accuracy",
                range_x=[0, 1], labels={"Algorithm": "", "Accuracy": "Accuracy"},
                color_continuous_scale=["#315b75", "#79e2d0"],
            )
            fig.update_xaxes(tickformat=".0%")
            st.plotly_chart(chart_theme(fig), use_container_width=True)
            best = classification.loc[classification["Accuracy"].idxmax()]
            st.caption(f"Top saved test accuracy: **{best['Algorithm']}** · {best['Accuracy']:.2%}.")
        else:
            st.info("Add `datasets/classification_review2_comparison.csv` to display saved model metrics.")

    with right:
        st.markdown("#### Clustering comparison")
        if clustering is not None:
            st.dataframe(clustering, hide_index=True, use_container_width=True)
            st.caption("Higher Silhouette and Calinski–Harabasz and lower Davies–Bouldin values are generally preferred.")
        else:
            st.info("Add `datasets/clustering_review2_comparison.csv` to display clustering metrics.")

    if tuning is not None and {"Version", "Accuracy"}.issubset(tuning.columns):
        st.markdown("#### Bagging · baseline vs tuned")
        fig = px.bar(tuning, x="Version", y="Accuracy", color="Version", text="Accuracy",
                     labels={"Version": "", "Accuracy": "Accuracy"})
        fig.update_traces(texttemplate="%{text:.2%}", textposition="outside")
        fig.update_yaxes(tickformat=".0%", range=[0, min(1.0, float(tuning["Accuracy"].max()) + 0.08)])
        st.plotly_chart(chart_theme(fig), use_container_width=True)


# -----------------------------------------------------------------------------
# Dataset explorer
# -----------------------------------------------------------------------------
def show_dataset_explorer() -> None:
    st.markdown("## Dataset Explorer")
    st.caption("Inspect datasets, review data quality, filter records, and export a selected view.")
    options = {
        "Nutrition · source": "nutrition.csv",
        "Nutrition · preprocessed": "nutrition_preprocessed.csv",
        "FIFA · source": "fifa.csv",
        "FIFA · preprocessed": "fifa_preprocessed.csv",
        "Mall Customers · source": "mall_customers.csv",
        "Mall Customers · preprocessed": "mall_customers_preprocessed.csv",
    }
    choice = st.selectbox("Choose a dataset", list(options.keys()))
    path = find_dataset(options[choice])
    if path is None:
        missing_dataset_message(options[choice])
        return

    frame = read_csv(str(path))
    metric_cards([
        ("Rows", f"{len(frame):,}", None),
        ("Columns", str(frame.shape[1]), None),
        ("Missing cells", f"{frame.isna().sum().sum():,}", None),
        ("Duplicate rows", f"{frame.duplicated().sum():,}", None),
    ])
    with st.expander("Column quality profile", expanded=False):
        quality = pd.DataFrame({
            "Column": frame.columns,
            "Data type": [str(frame[col].dtype) for col in frame.columns],
            "Non-null": [int(frame[col].notna().sum()) for col in frame.columns],
            "Missing": [int(frame[col].isna().sum()) for col in frame.columns],
            "Unique values": [int(frame[col].nunique(dropna=True)) for col in frame.columns],
        })
        st.dataframe(quality, hide_index=True, use_container_width=True)

    selected_cols = st.multiselect(
        "Columns to display", frame.columns.tolist(),
        default=frame.columns.tolist()[: min(8, frame.shape[1])],
    )
    query = st.text_input("Search text columns", placeholder="e.g. Fruit, player name, Female")
    view = frame[selected_cols].copy() if selected_cols else frame.copy()
    if query.strip():
        text_cols = frame.select_dtypes(include=["object", "string"]).columns
        if len(text_cols):
            mask = frame[text_cols].astype(str).apply(
                lambda col: col.str.contains(query.strip(), case=False, na=False, regex=False)
            ).any(axis=1)
            view = view.loc[mask]
        else:
            st.caption("This dataset has no text columns to search.")

    st.dataframe(view.head(500), use_container_width=True, hide_index=True)
    st.caption(f"Showing up to 500 of {len(view):,} matching rows for a responsive preview.")
    st.download_button(
        "Download filtered data as CSV",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name=f"{Path(options[choice]).stem}_filtered.csv",
        mime="text/csv",
    )


# -----------------------------------------------------------------------------
# Classification dashboard
# -----------------------------------------------------------------------------
def show_classification() -> None:
    st.markdown("## Nutrition Classification")
    st.caption("Compare saved model metrics, inspect tuning and cross-validation results, and explore nutritional features.")
    comparison = optional_csv("classification_review2_comparison.csv")
    tuning = optional_csv("bagging_review2_tuning.csv")
    cv = optional_csv("classification_review2_cv_top2.csv")
    data = optional_csv("nutrition_preprocessed.csv")
    if data is None:
        data = optional_csv("nutrition.csv")

    if comparison is None or "Algorithm" not in comparison.columns:
        missing_dataset_message("classification_review2_comparison.csv")
    else:
        comp = comparison.copy()
        preferred = ["Accuracy", "F1-score (weighted)", "Precision", "Recall", "ROC-AUC (OvR)"]
        metrics = [col for col in preferred if col in comp.columns]
        if not metrics:
            metrics = comp.select_dtypes(include=np.number).columns.tolist()
        metric = st.selectbox("Metric to compare", metrics)
        ranked = comp.sort_values(metric, ascending=False)
        best = ranked.iloc[0]
        metric_cards([
            ("Top model", str(best["Algorithm"]), f"Ranked by {metric}"),
            ("Best score", f"{best[metric]:.2%}", metric),
            ("Models compared", str(len(comp)), "Saved test-set results"),
            ("Selected metric", metric, None),
        ])
        left, right = st.columns([1.2, 0.8])
        with left:
            ordered = ranked.sort_values(metric, ascending=True)
            fig = px.bar(ordered, x=metric, y="Algorithm", orientation="h", color=metric, text=metric,
                         color_continuous_scale=["#315b75", "#79e2d0"])
            fig.update_traces(texttemplate="%{text:.2%}", textposition="outside", cliponaxis=False)
            fig.update_xaxes(tickformat=".0%", range=[0, min(1.12, float(ordered[metric].max()) + 0.12)])
            st.plotly_chart(chart_theme(fig, f"Model leaderboard · {metric}"), use_container_width=True)
        with right:
            if {"Precision", "Recall"}.issubset(comp.columns):
                fig = px.scatter(
                    comp, x="Precision", y="Recall",
                    size="F1-score (weighted)" if "F1-score (weighted)" in comp.columns else None,
                    color="Algorithm", hover_name="Algorithm",
                )
                fig.update_xaxes(tickformat=".0%")
                fig.update_yaxes(tickformat=".0%")
                st.plotly_chart(chart_theme(fig, "Precision vs recall"), use_container_width=True)
        with st.expander("Full classification metrics", expanded=False):
            st.dataframe(ranked, hide_index=True, use_container_width=True)
            st.download_button("Download metrics CSV", comp.to_csv(index=False).encode("utf-8"),
                               "classification_metrics.csv", "text/csv")

    if tuning is not None and {"Version", "Accuracy"}.issubset(tuning.columns):
        st.markdown("### Hyperparameter tuning impact")
        st.dataframe(tuning, hide_index=True, use_container_width=True)
        accuracies = pd.to_numeric(tuning["Accuracy"], errors="coerce").dropna()
        if len(accuracies) >= 2:
            delta = (accuracies.max() - accuracies.iloc[0]) * 100
            st.metric("Accuracy change from baseline", f"{delta:+.2f} percentage points")

    if cv is not None:
        st.markdown("### Cross-validation · saved top models")
        st.dataframe(cv, hide_index=True, use_container_width=True)
        st.caption("Cross-validation summarizes performance across training folds; it is distinct from the held-out test score.")

    if data is not None:
        st.markdown("### Nutrition feature explorer")
        numeric = numeric_columns(data)
        if numeric:
            x_default = numeric.index("calories") if "calories" in numeric else 0
            y_default = numeric.index("protein_g") if "protein_g" in numeric else min(1, len(numeric) - 1)
            col1, col2 = st.columns(2)
            x = col1.selectbox("X-axis", numeric, index=x_default, key="nutrition_x")
            y = col2.selectbox("Y-axis", numeric, index=y_default, key="nutrition_y")
            fig = px.scatter(
                data, x=x, y=y,
                color="category" if "category" in data.columns else None,
                hover_name="food_name" if "food_name" in data.columns else None,
                opacity=0.72,
            )
            st.plotly_chart(chart_theme(fig, "Nutritional feature relationship"), use_container_width=True)
        if "category" in data.columns:
            distribution = data["category"].value_counts().rename_axis("Category").reset_index(name="Count")
            fig = px.bar(distribution, x="Category", y="Count", color="Count", text="Count")
            st.plotly_chart(chart_theme(fig, "Food category distribution"), use_container_width=True)


# -----------------------------------------------------------------------------
# Clustering dashboard
# -----------------------------------------------------------------------------
def show_clustering() -> None:
    st.markdown("## Mall Customer Segmentation")
    st.caption("Compare internal validation metrics, inspect cluster profiles, and explore customer behaviour.")
    comparison = optional_csv("clustering_review2_comparison.csv")
    profiles = optional_csv("clustering_cluster_profiles.csv")
    mall = optional_csv("mall_customers_preprocessed.csv")
    if mall is None:
        mall = optional_csv("mall_customers.csv")

    if comparison is not None and "Algorithm" in comparison.columns:
        st.markdown("### Internal validation metrics")
        metric_names = [
            col for col in ["Silhouette Score", "Davies-Bouldin Index", "Calinski-Harabasz Index"]
            if col in comparison.columns
        ]
        cards = st.columns(len(metric_names) + 1)
        for idx, metric in enumerate(metric_names):
            values = pd.to_numeric(comparison[metric], errors="coerce")
            best_idx = values.idxmin() if metric == "Davies-Bouldin Index" else values.idxmax()
            direction = "Lower is better" if metric == "Davies-Bouldin Index" else "Higher is better"
            cards[idx].metric(metric, f"{values.loc[best_idx]:.4f}", help=f"{comparison.loc[best_idx, 'Algorithm']} · {direction}")
        cards[-1].metric("Algorithms compared", str(len(comparison)))
        melted = comparison.melt(id_vars="Algorithm", value_vars=metric_names, var_name="Metric", value_name="Score")
        fig = px.bar(melted, x="Metric", y="Score", color="Algorithm", barmode="group", text_auto=".3f")
        st.plotly_chart(chart_theme(fig, "Clustering metrics by algorithm"), use_container_width=True)
        st.dataframe(comparison, hide_index=True, use_container_width=True)
        st.caption("Interpret metrics separately: higher Silhouette and Calinski–Harabasz and lower Davies–Bouldin are generally preferred.")
    else:
        st.info("Add `datasets/clustering_review2_comparison.csv` to display saved validation metrics.")

    if profiles is not None:
        st.markdown("### Cluster profiles")
        method = st.radio("Clustering method", ["K-Means", "Agglomerative Hierarchical"], horizontal=True)
        prefix = "KMeans_" if method == "K-Means" else "Agglomerative_"
        profile_options = {
            prefix + "Age": "Average age",
            prefix + "Annual Income (k$)": "Average annual income (k$)",
            prefix + "Spending Score (1-100)": "Average spending score",
        }
        available = {key: label for key, label in profile_options.items() if key in profiles.columns}
        if available:
            selected = st.selectbox("Profile measure", list(available), format_func=lambda key: available[key])
            measure = available[selected]
            plot_data = profiles[["Cluster", selected]].rename(columns={selected: measure})
            fig = px.bar(plot_data, x="Cluster", y=measure, color="Cluster", text_auto=".2f")
            st.plotly_chart(chart_theme(fig, f"{method} · {measure}"), use_container_width=True)
            with st.expander("Full cluster profile table", expanded=False):
                st.dataframe(profiles, hide_index=True, use_container_width=True)
                st.download_button("Download profiles CSV", profiles.to_csv(index=False).encode("utf-8"),
                                   "clustering_cluster_profiles.csv", "text/csv")

    if mall is not None:
        st.markdown("### Customer behaviour explorer")
        numeric = numeric_columns(mall, exclude={"CustomerID"})
        if len(numeric) >= 2:
            x_default = numeric.index("Annual Income (k$)") if "Annual Income (k$)" in numeric else 0
            y_default = numeric.index("Spending Score (1-100)") if "Spending Score (1-100)" in numeric else min(1, len(numeric) - 1)
            left, right = st.columns(2)
            x = left.selectbox("Horizontal axis", numeric, index=x_default, key="customer_x")
            y = right.selectbox("Vertical axis", numeric, index=y_default, key="customer_y")
            fig = px.scatter(
                mall, x=x, y=y,
                color="Gender" if "Gender" in mall.columns else None,
                hover_data=[c for c in ["Age", "Annual Income (k$)", "Spending Score (1-100)"] if c in mall.columns],
                opacity=0.8,
            )
            st.plotly_chart(chart_theme(fig, "Customer feature map"), use_container_width=True)
        if "Gender" in mall.columns:
            counts = mall["Gender"].value_counts().rename_axis("Gender").reset_index(name="Customers")
            fig = px.bar(counts, x="Gender", y="Customers", color="Gender", text="Customers")
            st.plotly_chart(chart_theme(fig, "Customer records by gender"), use_container_width=True)


# -----------------------------------------------------------------------------
# Regression dashboard
# -----------------------------------------------------------------------------
def show_regression() -> None:
    st.markdown("## FIFA Player-Performance Regression")
    st.caption("Explore FIFA player attributes, distributions, and relationships from the project dataset.")
    fifa = optional_csv("fifa_preprocessed.csv")
    if fifa is None:
        fifa = optional_csv("fifa.csv")
    if fifa is None:
        missing_dataset_message("fifa.csv or fifa_preprocessed.csv")
        return

    metric_cards([
        ("Player records", f"{len(fifa):,}", None),
        ("Columns", str(fifa.shape[1]), None),
        ("Missing cells", f"{fifa.isna().sum().sum():,}", None),
        ("Duplicate rows", f"{fifa.duplicated().sum():,}", None),
    ])
    numeric = numeric_columns(fifa)
    if len(numeric) < 2:
        st.info("Not enough numeric columns were detected for exploration.")
        return

    st.markdown("### Explore player attributes")
    col1, col2, col3 = st.columns(3)
    x = col1.selectbox("X-axis", numeric, index=0, key="fifa_x")
    y = col2.selectbox("Y-axis", numeric, index=min(1, len(numeric) - 1), key="fifa_y")
    upper = max(500, min(10000, len(fifa)))
    default_sample = min(3000, upper)
    sample_size = col3.slider("Rows to plot", min_value=500, max_value=upper, value=default_sample, step=500)
    plot_data = fifa[[x, y]].dropna()
    if len(plot_data) > sample_size:
        plot_data = plot_data.sample(sample_size, random_state=42)
    fig = px.scatter(plot_data, x=x, y=y, opacity=0.62)
    st.plotly_chart(chart_theme(fig, "Player attribute relationship"), use_container_width=True)

    candidates = [col for col in numeric if any(term in col.lower() for term in ["overall", "potential", "value", "wage", "rating"])]
    distribution_default = candidates[0] if candidates else numeric[-1]
    distribution_col = st.selectbox("Feature distribution", numeric, index=numeric.index(distribution_default), key="fifa_distribution")
    fig = px.histogram(fifa, x=distribution_col, nbins=40, marginal="box")
    st.plotly_chart(chart_theme(fig, f"Distribution · {distribution_col}"), use_container_width=True)

    with st.expander("Numeric correlation matrix", expanded=False):
        corr_cols = numeric[:20]
        corr = fifa[corr_cols].corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
        st.plotly_chart(chart_theme(fig, "Correlation matrix · first 20 numeric features"), use_container_width=True)

    st.info("This page currently focuses on exploratory analysis. Regression model scores are not fabricated; export the evaluation table from the regression notebook to a CSV to add a results leaderboard.")


# -----------------------------------------------------------------------------
# Navigation
# -----------------------------------------------------------------------------
def main() -> None:
    with st.sidebar:
        st.markdown("## ML Capstone")
        st.caption("Analytics Studio · Review dashboard")
        page = st.radio("Navigation", ["Overview", "Dataset Explorer", "Classification", "Clustering", "Regression"])
        st.markdown("---")
        st.markdown("**Project tracks**")
        st.markdown(
            '<span class="pill">Regression</span><span class="pill">Classification</span><span class="pill">Clustering</span>',
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.caption("Source: repository datasets and saved evaluation CSVs.")
        st.caption("Charts are interactive. Use the Plotly toolbar to zoom, pan, or download figures.")

    if page == "Overview":
        show_overview()
    elif page == "Dataset Explorer":
        show_dataset_explorer()
    elif page == "Classification":
        show_classification()
    elif page == "Clustering":
        show_clustering()
    else:
        show_regression()

    st.markdown(
        '<div class="footer">ML Capstone Analytics Studio · Saved metrics come from repository CSVs. '
        'Interpret all charts alongside the underlying project notebooks.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

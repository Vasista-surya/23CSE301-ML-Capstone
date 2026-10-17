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

# ML Capstone: Analytics Studio

A professional machine learning capstone project that brings together regression, classification, and clustering workflows in a single analytics dashboard. This repository explores real-world datasets, evaluates multiple model families, and presents the findings through interactive visualizations and reproducible notebook analysis.

## Project Overview

This project is designed as an end-to-end ML study with three main analytical tracks:

- Nutrition classification: Predict food category or class assignments using nutritional attributes and compare model performance.
- FIFA player-performance regression: Analyze player attributes and model performance using regression techniques.
- Customer segmentation: Identify behavioral patterns and cluster groups using customer data.

The project includes:

- Data exploration and preprocessing
- Model comparison and evaluation
- Notebook-based experimentation
- Interactive dashboard for dashboard-style reporting
- Visual analysis outputs for insight communication

## Key Features

- Streamlit-based analytics dashboard for quick exploration
- Dataset explorer for source and preprocessed data
- Model comparison views for classification and clustering results
- Regression analysis for FIFA-style player performance metrics
- Reusable CSV result tables for saved evaluation outputs
- Jupyter notebooks for detailed exploratory and model-building workflows

## Repository Structure

```text
23CSE301-ML-Capstone/
├── app.py                       # Streamlit application entry point
├── requirements.txt            # Python dependencies
├── datasets/                   # Project datasets and saved evaluation CSVs
│   ├── classification_review2_comparison.csv
│   ├── classification_review2_cv_top2.csv
│   ├── bagging_review2_tuning.csv
│   ├── clustering_cluster_profiles.csv
│   ├── clustering_review2_comparison.csv
│   ├── fifa.csv
│   ├── fifa_preprocessed.csv
│   ├── mall_customers.csv
│   ├── mall_customers_preprocessed.csv
│   ├── nutrition.csv
│   ├── nutrition_preprocessed.csv
│   └── nutrition_classification_review2_partB_comparison.csv
├── notebooks/                  # Jupyter notebooks for experimentation
│   ├── clustering.ipynb
│   ├── fifa-performance-regression.ipynb
│   └── nutrition_classification_partA_and_review2_partB.ipynb
├── plots/                      # Visual outputs generated during analysis
│   ├── clustering/
│   ├── fifa-performance-regression/
│   └── nutrition-classification/
└── README.md                   # Project documentation
```

## Dataset Coverage

The project uses a mix of raw and preprocessed datasets to support the three research tracks:

- Nutrition dataset: food features and classification analysis
- FIFA dataset: player attributes and performance modeling
- Mall customer dataset: segmentation and clustering behavior analysis

## Getting Started

### Prerequisites

Ensure Python 3.10+ is installed on your system.

### Installation

1. Clone the repository.
2. Navigate to the project folder.
3. Create a virtual environment:

```bash
python -m venv .venv
```

4. Activate the environment:

- Windows:

```bash
.venv\Scripts\activate
```

- macOS/Linux:

```bash
source .venv/bin/activate
```

5. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Dashboard

Start the Streamlit application from the project root:

```bash
streamlit run app.py
```

This opens the interactive analytics dashboard, which includes:

- Overview page
- Dataset explorer
- Classification dashboard
- Clustering dashboard
- Regression dashboard

## Notebook Workflow

The notebooks provide the deeper analytical workflow behind the dashboard. They contain:

- exploratory data analysis
- feature engineering
- preprocessing steps
- model training and comparison
- metric analysis and visualization

For a full understanding of the project, review the notebooks in the `notebooks/` directory alongside the dashboard results.

## Model and Analysis Areas

### 1. Classification

The classification track focuses on nutritional data, comparing multiple algorithms and validating performance across multiple metrics such as accuracy, precision, recall, and F1-score.

### 2. Regression

The regression analysis explores FIFA player attributes to understand which variables are most associated with overall performance metrics and model quality outcomes.

### 3. Clustering

The clustering track studies customer groups, applying unsupervised learning techniques to identify meaningful clusters and evaluate cluster quality using common metrics.

## Requirements

The project dependencies are listed in `requirements.txt`, including:

- Streamlit
- Pandas
- NumPy
- Plotly
- Dash

## Project Purpose

This capstone project demonstrates how machine learning can be used to analyze structured datasets, compare model performance, and communicate results through an interactive reporting interface. It is suitable for academic coursework, portfolio showcasing, and applied data-science demonstration work.

## Notes

- The dashboard loads saved CSV artifacts from the `datasets/` directory.
- Some visualizations are generated from project results and preprocessed datasets.
- The project is intended for educational and analytical use.

## License

This project does not currently declare a specific license. Please check with the repository owner or academic institution if you intend to reuse or distribute the work commercially or publicly.

## Contact / Academic Use

This repository is intended for academic and project-based exploration. For questions related to the project, contact the corresponding course or repository maintainer.

# Machine Learning Capstone Project

<p align="center">
  <strong>Regression • Classification • Clustering • Interactive Analytics Dashboard</strong><br>
  <em>A practical exploration of supervised and unsupervised machine learning using real-world datasets.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Jupyter-Notebooks-orange?logo=jupyter" alt="Jupyter">
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?logo=plotly" alt="Plotly">
  <img src="https://img.shields.io/badge/Project-Academic%20Capstone-informational" alt="Academic project">
</p>

## Overview

This repository contains the **23CSE301 Machine Learning Capstone Project**. It brings together three core machine learning tasks—**regression, classification, and clustering**—and an interactive dashboard for exploring project datasets and analytical outputs.

The project follows a practical workflow: preparing datasets, applying machine learning techniques, evaluating model performance, visualizing patterns, and presenting analysis through Jupyter notebooks and a Streamlit analytics interface.

## Project Objectives

- Apply regression techniques to estimate continuous outcomes from FIFA player-performance data.
- Build and compare classification models using nutrition-related data.
- Discover customer segments using clustering techniques.
- Explore data preprocessing, feature preparation, model comparison, and evaluation.
- Present datasets and analysis outputs through an interactive dashboard.

## Project Tracks

### 1. Regression — FIFA Player Performance

Uses FIFA player data to explore regression-based prediction of a continuous performance-related target. The notebook contains the analysis and model workflow for this track.

- **Dataset files:** [`fifa.csv`](datasets/fifa.csv), [`fifa_preprocessed.csv`](datasets/fifa_preprocessed.csv)
- **Notebook:** [`fifa-performance-regression.ipynb`](notebooks/fifa-performance-regression.ipynb)

### 2. Classification — Nutrition Data

Uses nutritional attributes to investigate classification and compare predictive models. The notebook covers data exploration, feature preparation, classifier training, and evaluation.

- **Dataset files:** [`nutrition.csv`](datasets/nutrition.csv), [`nutrition_preprocessed.csv`](datasets/nutrition_preprocessed.csv)
- **Notebook:** [`nutrition-classification.ipynb`](notebooks/nutrition-classification.ipynb)

### 3. Clustering — Customer Segmentation

Uses customer demographic and spending information to identify groups with similar characteristics. The analysis compares clustering approaches, including K-Means and Agglomerative Hierarchical Clustering, and uses cluster validation and visualizations to support interpretation.

- **Dataset files:** [`mall_customers.csv`](datasets/mall_customers.csv), [`mall_customers_preprocessed.csv`](datasets/mall_customers_preprocessed.csv)
- **Notebook:** [`clustering.ipynb`](notebooks/clustering.ipynb)

### 4. Interactive Analytics Dashboard

The Streamlit application provides a central interface for exploring project datasets and analytical outputs with interactive Plotly visualizations.

- **Application entry point:** [`app/app.py`](app/app.py)

## Repository Structure

```text
23CSE301-ML-Capstone/
├── app/
│   └── app.py
├── datasets/
│   ├── fifa.csv
│   ├── fifa_preprocessed.csv
│   ├── nutrition.csv
│   ├── nutrition_preprocessed.csv
│   ├── mall_customers.csv
│   ├── mall_customers_preprocessed.csv
│   ├── classification_review2_*.csv
│   ├── clustering_review2_*.csv
│   ├── clustering_cluster_profiles.csv
│   └── bagging_review2_tuning.csv
├── notebooks/
│   ├── fifa-performance-regression.ipynb
│   ├── nutrition-classification.ipynb
│   └── clustering.ipynb
├── requirements.txt
└── README.md
```

*The tree shows the main files. The `datasets/` directory also contains supporting comparison, tuning, and clustering-output CSV files.*

## Technology Stack

- **Python** — primary programming language
- **Jupyter Notebook** — interactive analysis and experimentation
- **Pandas and NumPy** — data manipulation and numerical operations
- **Scikit-learn** — machine learning workflows and evaluation in the notebooks
- **Matplotlib / Seaborn** — visualizations used where applicable in the notebooks
- **Streamlit** — interactive dashboard
- **Plotly** — interactive dashboard charts

The dashboard dependencies are listed in [`requirements.txt`](requirements.txt). The notebooks may require additional libraries depending on the cells executed.

## Getting Started

### Prerequisites

- Python 3.10 or later recommended
- Git
- Jupyter Notebook or JupyterLab
- A terminal or command prompt

### 1. Clone the Repository

```bash
git clone https://github.com/Vasista-surya/23CSE301-ML-Capstone.git
cd 23CSE301-ML-Capstone
```

### 2. Create and Activate a Virtual Environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For notebook execution, install Jupyter and any additional libraries required by the notebooks if they are not already available:

```bash
pip install jupyter scikit-learn matplotlib seaborn
```

### 4. Launch the Dashboard

Run this command from the repository root:

```bash
streamlit run app/app.py
```

Streamlit will print a local URL in the terminal, typically `http://localhost:8501`. Open that address in your browser.

### 5. Run the Notebooks

Start Jupyter from the repository root:

```bash
jupyter notebook
```

Open a notebook in the `notebooks/` folder and execute cells in order. Keeping the repository structure intact helps the notebooks and dashboard locate the dataset files.

## Model Evaluation

Each machine learning task needs evaluation measures suited to its objective. The table below describes common metrics for these task types; consult the relevant notebook for the exact metrics implemented in each experiment.

| Task | Common evaluation metrics | What they indicate |
|---|---|---|
| Regression | MAE, MSE/RMSE, R² | Prediction error and goodness of fit |
| Classification | Accuracy, precision, recall, F1-score, confusion matrix, cross-validation | Overall and class-wise predictive performance |
| Clustering | Silhouette score and other internal validation measures | How distinct and compact the discovered groups are |

No single metric should be interpreted in isolation. Evaluation results depend on the dataset, target definition, preprocessing, and experimental setup.

## Data and Reproducibility Notes

- Keep source and preprocessed CSV files in the `datasets/` directory.
- Run notebook cells from top to bottom so that preprocessing steps and variables are initialized in the correct order.
- Use the preprocessed datasets when a notebook expects them; otherwise follow the preprocessing workflow documented in that notebook.
- Results may vary if random seeds, train/test splits, preprocessing steps, or package versions are changed.
- Verify original dataset sources and their usage terms before redistributing datasets outside this repository.

## Troubleshooting

**The dashboard cannot find a CSV file**

- Run `streamlit run app/app.py` from the repository root.
- Confirm that the required CSV exists in `datasets/` and that `app/` and `datasets/` remain alongside one another.

**A notebook reports a missing package**

- Activate the project virtual environment and install the missing package with `pip install package-name`.
- Restart the notebook kernel after installation if the import still fails.

**Notebook results differ between runs**

- Check random-state settings, data splits, package versions, and the exact input dataset used.

## Academic Context

This repository was developed as part of the **23CSE301 Machine Learning Capstone** at **Amrita Vishwa Vidyapeetham, Coimbatore**.

## Contributors

| Team Member | Student ID | Contribution |
|---|---|---|
| **C. Vasista Surya** | `CB.SC.U4CSE24212` | Classification track — implementation and evaluation of classification models |
| **G. Hrithikesh Reddy** | `CB.SC.U4CSE24219` | Regression track — implementation of the first five regression models |
| **S. Lalitesh** | `CB.SC.U4CSE24250` | Regression track — implementation of the second five regression models |


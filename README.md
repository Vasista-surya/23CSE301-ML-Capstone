# Machine Learning Capstone Dashboard

Streamlit dashboard for the **23CSE301 Machine Learning** capstone, covering:

- FIFA performance regression
- Nutrition category classification
- Mall customer clustering
- Model comparison and dataset exploration

The dashboard reads the supplied CSV files and executed notebook outputs without
overwriting notebooks, datasets, or model artifacts. Live prediction controls
call the optional Agent 2 service contract in `ml_services.py` when it is
available; otherwise they remain visible as a clear integration state instead
of returning fabricated predictions.

## Run locally on Windows

From this directory:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Then open `http://localhost:8501`.

The UI uses the Python interpreter that runs `python`; keep installation and
launch commands on the same interpreter. The app resolves `datasets/` and
`notebooks/` relative to `app.py`, so it is safe to launch from another
working directory as well.

## Service integration

The UI detects these functions from `ml_services.py` when present:

```text
get_project_summary()
get_regression_results()
get_classification_results()
get_clustering_results()
predict_regression(features)
predict_classification(features, model_name)
assign_cluster(features, algorithm_name)
get_dataset_preview(dataset_name)
```

Result pages use persisted CSVs and notebook outputs as a fast evidence layer
on a clean checkout. If an Agent 2 model bundle is already available, service
results are normalized into the same tables. Prediction submissions always go
through the service layer.

## UI dependencies

The exact UI dependencies are declared in `requirements.txt`:

- Streamlit
- pandas and NumPy
- Plotly
- scikit-learn and SciPy for cached clustering visual diagnostics
- lxml for reading HTML result tables embedded in the regression notebook

## Checks

```powershell
python -m py_compile app.py app_components.py
python -m unittest discover -s tests -p "test_*.py" -v
```

The test suite covers the backend service contract, model inference boundaries,
regression metric-column normalization, and a Streamlit smoke test that visits
all six sidebar pages. The dashboard also catches page-level rendering errors
and presents a user-friendly diagnostic panel instead of taking down the whole
application.

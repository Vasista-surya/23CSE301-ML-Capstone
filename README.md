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
python -m streamlit run streamlit_app.py
```

Then open `http://localhost:8501`.

The UI uses the Python interpreter that runs `python`; keep installation and
launch commands on the same interpreter. The app resolves `datasets/` and
`notebooks/` relative to `streamlit_app.py`, so it is safe to launch from another
working directory as well.

## Deploy on Render

This repository includes `render.yaml` for a Render web service. In Render,
choose **New → Blueprint**, connect this repository, and select the `main`
branch. Render will install `requirements.txt`, start Streamlit on Render's
`$PORT`, and use `/_stcore/health` for the health check.

The free plan may sleep when idle, so the first request after inactivity can
take longer. The tested model bundles are committed under `models/` and are
validated by content fingerprint and scikit-learn version before use.

Vercel does not run the Streamlit `streamlit_app.py` process directly. This repository
therefore includes a separate static frontend and Python serverless adapter,
described below, so the same ML services can be deployed on Vercel without
pretending that Streamlit is a Vercel runtime.

## Deploy on Vercel

The repository also includes a separate Vercel-compatible presentation layer:

- `index.html` is a responsive static dashboard with the same overview,
  regression, classification, clustering, and dataset workflows.
- `api/index.py` is a Flask serverless adapter over `ml_services.py`.
- `vercel.json` rewrites `/api/*` requests to the Python function.

From the repository root, install the Vercel CLI and deploy:

```powershell
npm install -g vercel
vercel login
vercel --prod
```

When Vercel asks for the project directory, use this repository root. Do not
choose the Streamlit `streamlit_app.py` as the Vercel entrypoint; Vercel serves
`index.html` and the Python API instead. The Streamlit dashboard remains the
local/Render entrypoint.

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
python -m py_compile streamlit_app.py app_components.py
python -m unittest discover -s tests -p "test_*.py" -v
```

The test suite covers the backend service contract, model inference boundaries,
regression metric-column normalization, and a Streamlit smoke test that visits
all six sidebar pages. The dashboard also catches page-level rendering errors
and presents a user-friendly diagnostic panel instead of taking down the whole
application.

"""Vercel-compatible HTTP adapter for the capstone ML services.

The original dashboard remains available as Streamlit for local and Render
usage. Vercel runs this small Flask adapter as a Python serverless function and
serves the static dashboard from ``index.html``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import ml_services  # noqa: E402


app = Flask(__name__)


def _jsonable(value: Any) -> Any:
    """Convert pandas/NumPy service responses into safe JSON values."""

    if isinstance(value, pd.DataFrame):
        clean = value.copy()
        clean = clean.replace([np.inf, -np.inf], np.nan)
        return {
            "columns": [str(column) for column in clean.columns],
            "index": [str(index) for index in clean.index],
            "rows": [_jsonable(row) for row in clean.where(pd.notna(clean), None).to_dict("records")],
        }
    if isinstance(value, pd.Series):
        return _jsonable(value.to_frame().T)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _reply(payload: Any, status: int = 200):
    return jsonify(_jsonable(payload)), status


def _error(message: str, status: int = 400):
    return _reply({"status": "error", "error": message}, status)


def _body() -> dict[str, Any]:
    payload = request.get_json(silent=True)
    return payload if isinstance(payload, dict) else {}


def _dispatch(route: str):
    route = route.strip("/").removesuffix(".py")
    if route in {"", "health"}:
        return _reply({"status": "ok", "service": "23CSE301 ML Capstone API"})
    if route == "summary":
        return _reply(ml_services.get_project_summary())
    if route == "regression":
        return _reply(ml_services.get_regression_results())
    if route == "classification":
        return _reply(ml_services.get_classification_results())
    if route == "clustering":
        return _reply(ml_services.get_clustering_results())
    if route == "preview":
        dataset = request.args.get("dataset", "nutrition")
        try:
            n_rows = int(request.args.get("n_rows", "10"))
        except ValueError:
            return _error("n_rows must be an integer between 1 and 100")
        return _reply(ml_services.get_dataset_preview(dataset, n_rows))
    if route == "predict/regression":
        body = _body()
        features = body.get("features", body)
        return _reply(ml_services.predict_regression(features, body.get("model_name")))
    if route == "predict/classification":
        body = _body()
        features = body.get("features", body)
        return _reply(ml_services.predict_classification(features, body.get("model_name")))
    if route == "assign-cluster":
        body = _body()
        features = body.get("features", body)
        return _reply(ml_services.assign_cluster(features, body.get("algorithm_name", "K-Means")))
    return _error(f"Unknown API route: /api/{route}", 404)


@app.route("/api/<path:route>", methods=["GET", "POST", "OPTIONS"])
def api_router(route: str):
    if request.method == "OPTIONS":
        return _reply({"status": "ok"})
    # Vercel rewrites /api/<route> to this function and forwards the route in
    # the query string. Direct /api/index.py?route=summary calls work too.
    forwarded = request.args.get("route", route)
    return _dispatch(forwarded)


@app.after_request
def add_api_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


if __name__ == "__main__":  # pragma: no cover - convenient local adapter check
    app.run(host="0.0.0.0", port=5000, debug=True)

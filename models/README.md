# Generated model artifacts

`model_utils.py` generates the three `*.joblib` bundles lazily from the
unchanged source CSV files.  Generated artifacts are intentionally ignored by
Git because they are version- and environment-specific.  This directory is
kept in the project so the dashboard has a predictable local cache location.

The artifact stores the fitted estimator(s), feature order, train-only
imputation medians, IQR caps, scalers, class order, and computed result tables.
If the installed scikit-learn version differs from the version recorded in an
artifact, the service discards that cache and retrains reproducibly.

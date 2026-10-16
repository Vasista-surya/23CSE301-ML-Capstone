# Generated model artifacts

The three `*.joblib` bundles are deployment assets for the tested Python and
scikit-learn runtime. They are validated against the installed scikit-learn
version and a content fingerprint of the source CSV before loading. The
content-based fingerprint is stable across Git checkouts, so a hosted service
does not retrain the large FIFA SVR model just because file timestamps changed.

Regenerate them from the project root after changing source data, model code,
or dependency versions:

```powershell
python generate_model_artifacts.py
```

If a bundle is missing or incompatible, the dashboard still renders the
persisted evidence tables, but live prediction services may need to train a
replacement bundle on first use.

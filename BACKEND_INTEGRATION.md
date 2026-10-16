# ML backend integration guide

The frontend agent should import the public functions from `ml_services.py`.
The backend uses the extracted project directory as its root and resolves
datasets relative to that file, so it does not depend on the process working
directory.  No UI files are required by this backend.

## Common response contract

Every public service returns a dictionary with:

```python
{"status": "ok", ...}
```

or, for validation/training/data errors:

```python
{"status": "error", "error": "human-readable message", ...}
```

Tables in successful responses are pandas `DataFrame` objects.  The services
train once on first use and cache compatible bundles in `models/*.joblib`.
Those files are generated artifacts, not models that were present in the
uploaded source project.

## Public functions

### `get_project_summary()`

**Input:** none.

**Output:** a dictionary containing:

- `datasets`: discoverability, path, row count, and columns for the three raw
  datasets;
- `regression`: target (`performance_score`), exact feature list, split, and
  ten baseline model names;
- `classification`: target (`category_group`), exact 25-feature list, split,
  ten baseline model names, and selected tuned model;
- `clustering`: exact three-feature list, `customer_id_used_as_feature=False`,
  `gender_used_as_feature=False`, algorithms, and selected `n_clusters=6`;
- `artifacts`: local generated-cache status and directory.

### `get_regression_results()`

**Input:** none.

**Output keys:**

- `comparison`: ten rows with stable columns `Model`, `R2`, `RMSE`, `MAE`,
  `CV_R2_Mean`, and `CV_R2_Std`.  CV values are populated for the two best
  baseline models and are null/NaN for the other rows;
- `cv_results`: `Model`, canonical `CV_R2_Mean`/`CV_R2_Std`, UI-compatible
  `Mean CV R2`/`Std CV R2`, and raw five-fold `CV_R2_Scores`;
- `tuning_results`: actual held-out metrics for tuned Gradient Boosting and
  Random Forest, including `Best_Params`;
- `feature_names`, `target_name`, `selected_model`, `split`, and `n_rows`.

The ten baseline names are `Linear Regression`, `Ridge Regression`, `Lasso
Regression`, `ElasticNet Regression`, `Polynomial Regression`, `Decision Tree
Regressor`, `Random Forest Regressor`, `Gradient Boosting Regressor`, `Support
Vector Regressor`, and `K-Nearest Neighbors Regressor`.

The exact 52 model feature order is:

```text
age, jersey_number, height_cm, weight_kg, market_value_eur, goals_team,
goals_opponent, minutes_played, goals, assists, shots, shots_on_target,
expected_goals_xg, expected_assists_xa, key_passes, successful_passes,
total_passes, pass_accuracy, dribbles_attempted, successful_dribbles,
crosses, successful_crosses, tackles, interceptions, clearances, blocks,
aerial_duels_won, aerial_duels_lost, recoveries, defensive_actions,
fouls_committed, fouls_suffered, yellow_cards, red_cards, offsides, saves,
save_percentage, punches, clean_sheet, goals_conceded, penalty_saves,
distance_covered_km, sprint_distance_km, top_speed_kmh, accelerations,
decelerations, stamina_score, match_year, match_month,
goal_contribution_rate, work_rate_intensity, passing_efficiency
```

### `get_classification_results()`

**Input:** none.

**Output keys:**

- `comparison`: ten rows with `Model`, `Accuracy`, `Precision_weighted`,
  `Recall_weighted`, `F1_weighted`, and `ROC_AUC_OvR`;
- `confusion_matrices`: a dictionary from model name to an 11 x 11 pandas
  DataFrame.  Both its index and columns use `class_labels`;
- `class_labels`: class order used for every probability and matrix;
- `cv_results`: weighted-F1 cross-validation for the two strongest baseline
  models;
- `baseline_vs_tuned`: baseline and tuned Bagging metrics;
- `tuning`, `feature_names`, `target_name`, `selected_model`, `split`, and
  `target_counts`.

The target is not guessed: `category_group` is formed from the ten most
frequent raw `category` values; every other raw category becomes `Other`.  The
25 features, in order, are:

```text
calories, protein_g, carbohydrates_g, fat_g, fiber_g, sugars_g, calcium_mg,
iron_mg, sodium_mg, potassium_mg, magnesium_mg, phosphorus_mg, zinc_mg,
vitamin_a_iu, vitamin_c_mg, vitamin_b1_mg, vitamin_b2_mg, vitamin_b3_mg,
vitamin_b5_mg, vitamin_b6_mg, vitamin_e_mg, folate_mcg, total_macros_g,
fat_carb_ratio, protein_carb_ratio
```

The baseline names are the ten required classifiers.  `Tuned Bagging
Classifier` is also available for inference and is the selected final model.
The executable notebook uses **three-fold** stratified CV for the two best
models and three-fold tuning, despite a markdown heading that says five-fold;
the backend preserves the executable behavior rather than the stale heading.

### `get_clustering_results()`

**Input:** none.

**Output keys:**

- `comparison`: two rows with `Algorithm`, `Silhouette_Score`,
  `Davies_Bouldin_Index`, `Calinski_Harabasz_Index`, and `n_clusters`;
- `elbow`: `k`, `Inertia`, `Silhouette` for `k=2..10`;
- `linkage_comparison`: `Linkage`, `Silhouette`, `Davies_Bouldin`, and
  `Calinski_Harabasz` for Ward, complete, and average linkage;
- `assignments`: original customer columns plus both cluster labels and PCA /
  t-SNE coordinates;
- `profiles`: a dictionary from each algorithm to a cluster-profile DataFrame;
- `pca_coordinates`, `pca_explained_variance_ratio`, `tsne_coordinates`,
  `feature_names`, `selected_n_clusters`, and `n_rows`.

Clustering uses only `Age`, `Annual Income (k$)`, and `Spending Score (1-100)`
after a full-dataset `StandardScaler`, exactly as in the notebook.  CustomerID
and Gender are never fit into a cluster.  PCA and t-SNE are visualization
outputs only.

### `predict_regression(features, model_name=None)`

**Input:** a mapping or one-row DataFrame.  The mapping must contain the 52
model feature names above.  For convenience, a raw FIFA row can instead be
provided: `match_year` and `match_month` are derived from `match_date`, and
the three engineered fields are derived from the raw pass/movement/goal
fields.  `model_name` defaults to the tuned Random Forest and accepts any
baseline name or the two tuned names.

**Output:**

```python
{
    "status": "ok",
    "prediction": float,
    "predicted_value": float,
    "model": str,
    "target_name": "performance_score",
    "feature_order": list[str],
    "preprocessing": str,
}
```

Input data is ordered, numeric-coerced, filled with training medians, clipped
using training-only IQR bounds, scaled with the training-only scaler, and only
then passed to the estimator.  Polynomial Regression additionally applies
the notebook's degree-2 transform to its five source features.

### `predict_classification(features, model_name=None)`

**Input:** a mapping or one-row DataFrame with the 25 nutrition feature names.
Raw nutrition rows are accepted and the three engineered fields are derived.
`model_name` is optional; by default it selects tuned Bagging.  Baseline and
numbered notebook aliases are accepted.

**Output:**

```python
{
    "status": "ok",
    "predicted_class": str,
    "probabilities": {class_name: float} | None,
    "probability_supported": bool,
    "probability_message": str,
    "class_order": list[str],
    "model": str,
    "feature_order": list[str],
    "preprocessing": str,
}
```

Probabilities are aligned to `class_order`, not to an arbitrary UI ordering.
If a future estimator does not implement `predict_proba`, the service returns
`probabilities=None` and a clear message rather than fabricating probabilities.

### `assign_cluster(features, algorithm_name="K-Means")`

**Input:** a mapping or one-row DataFrame containing the three cluster feature
names.  Extra `CustomerID` and `Gender` fields are ignored.

**Output:** `algorithm`, integer `cluster`, `cluster_label`, `n_clusters`,
`feature_order`, and `assignment_strategy` in addition to `status`.

K-Means uses the fitted estimator's `predict()`.  Agglomerative clustering has
no supported sklearn `predict()` method; the backend explicitly assigns a new
row to the nearest centroid computed from its fitted Ward cluster
representation and states this strategy in the response.

### `get_dataset_preview(dataset_name, n_rows=10)`

**Input:** one of `fifa`, `nutrition`, `mall_customers`, or their `.csv` names;
the three `_preprocessed` names are also available. `n_rows` must be 1--100.

**Output:** `dataset`, `path`, `data` (DataFrame), `columns`, and full source
`shape`, plus `status`.

## Reproducibility and checks

- `SEED=42` is used wherever the notebook used a random seed.
- No original notebook or source CSV is modified.
- The generated artifacts include train-only preprocessing state and are
  invalidated when the installed scikit-learn version changes.
- Run the backend tests from this directory:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

To explicitly pre-generate all local artifacts before launching the dashboard,
run:

```powershell
python generate_model_artifacts.py
```

The first full test run can take longer because it trains the ten FIFA
regressors, ten nutrition classifiers, tuned models, and clustering models.

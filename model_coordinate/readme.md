# Coordinate Classification Project

## Mission & Success Criteria
- Label a target coordinate as `friend` or `foe` by learning spatial patterns from paired friend/foe coordinate lists.
- Deliver reproducible Python pipelines for training, evaluating, and persisting the best-performing model artifact.
- Provide an inference interface (CLI + callable helper) that loads the artifact, accepts raw coordinate lists, and emits class + probability.
- Produce statistical summaries (accuracy, precision/recall/F1, ROC-AUC, calibration) and map-based visuals to help interpret the model.

### Success metrics
- Reach ≥90% accuracy and ≥0.9 ROC-AUC on held-out test data (tunable when real data arrives).
- Maintain probability calibration error ≤0.05 so downstream consumers can trust confidence scores.
- Keep training pipeline deterministic: repeated runs with the same seed must yield identical metrics and artifacts.

## Problem Scope
- **In scope**: CSV ingestion, preprocessing utilities, feature engineering, scikit-learn models, visualization scripts, CLI tooling, test coverage.
- **Out of scope (for now)**: streaming ingestion, online learning, geofencing enforcement beyond basic validation, and deployment packaging (Docker/serverless).

## Data Model & CSV Specification
- Each row captures a full scenario snapshot:
	- `friend_pos_list`: JSON string representing an ordered array of `[longitude, latitude]` float pairs (WGS84 degrees, longitude first).
	- `foe_pos_list`: same encoding for opponent positions.
	- `target_coord`: JSON string `[longitude, latitude]` for the point to classify.
	- `label`: `friend` or `foe`. Required for training rows, omitted/blank for inference batches.
- Validation rules:
	- Longitude ∈ [-180, 180], latitude ∈ [-90, 90].
	- Minimum one coordinate in each list; pad or flag scenarios that violate this.
	- Lists should be pre-sorted chronologically or by observation order; we persist the provided ordering.
- Example CSV row:

```csv
friend_pos_list,foe_pos_list,target_coord,label
"[[-73.98,40.75],[-73.97,40.76]]","[[-74.02,40.71],[-74.03,40.70]]","[-73.99,40.74]",friend
```

## Repository Layout (planned)
```
model_coordinate/
├── configs/
│   └── train.yaml
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── visualization/
│   └── cli/
├── artifacts/
├── reports/
├── tests/
└── readme.md
```

## Dependencies
- Python 3.10+
- Core: `pandas`, `numpy`, `scikit-learn`, `joblib`, `pyyaml`.
- Neural experiments: `pytorch` (preferred) or `tensorflow` + `keras` for lightweight multilayer perceptrons.
- Visualization: `folium` for interactive HTML maps, `matplotlib` (optionally `cartopy`) for static figures.
- Utilities: `typer` (preferred) or `click` for CLI, `rich` for colored logging, `pytest` for tests.
- Optional tracking: `mlflow` for experiment logs, `geopandas` if advanced geospatial ops become necessary.

## Training Workflow & API
- CLI entry point: `python src/train.py --data data/raw/train.csv --config configs/train.yaml --output artifacts/model.joblib`.
- Config keys (example):
	- `split.seed`, `split.train_ratio`, `split.val_ratio`.
	- `features.distance_metrics` (e.g., centroid distance, nearest foe distance).
	- `model.type` (`knn`, `random_forest`, `gradient_boosting`) and hyperparameters.
	- `evaluation.metrics` list and `visualization.sample_ids` for map renders.
- Pipeline stages:
	1. **Ingestion**: load CSV, JSON-parse coordinate columns, perform validation, drop/flag bad rows.
	2. **Feature engineering**: derive centroids, pairwise distances, spatial densities, bounding box overlap, time-independent summary stats. Keep functions pure for reuse.
	3. **Model fitting**: wrap preprocessing + estimator inside `sklearn.pipeline.Pipeline`. Start with `KNeighborsClassifier`, `RandomForestClassifier`, and an `MLPClassifier`. In parallel, evaluate a compact PyTorch feed-forward network (e.g., two hidden layers, ReLU activations) trained on the same engineered features. Grid-search or Bayesian-tune hyperparameters via `GridSearchCV`, `RandomizedSearchCV`, or Optuna for the neural model.
	4. **Evaluation**: compute accuracy, precision, recall, F1, ROC-AUC, PR-AUC, Brier score; derive confusion matrix and calibration curves. Persist metrics to `reports/metrics.json` and plots to `reports/figures/`.
	5. **Persistence**: store the full pipeline via `joblib.dump`, plus metadata (`artifacts/model_meta.json`) capturing feature list, training data hash, and evaluation summary.
- Outputs:
	- `artifacts/model.joblib`: serialized pipeline (preprocess + estimator).
	- `artifacts/model_meta.json`: metadata for reproducibility.
	- `reports/metrics.json` and `reports/figures/*.png|html`.

## Inference Workflow & API
- CLI entry point: `python src/infer.py --model artifacts/model.joblib --friends examples/friends.json --foes examples/foes.json --target -73.99 40.74 --viz reports/inference_map.html`.
- Python helper (planned):

```python
from src.infer import classify_coordinate

prediction = classify_coordinate(
	model_path="artifacts/model.joblib",
	friend_pos_list=[(-73.98, 40.75)],
	foe_pos_list=[(-74.02, 40.71)],
	target_coord=(-73.99, 40.74),
)
```

- Return payload:
	- `prediction`: `friend` or `foe`.
	- `probabilities`: dict with normalized probabilities.
	- `features` (optional debug): engineered values for the request.
- Optional flags:
	- `--viz`: emit Folium map with friend (green), foe (red), target (blue star) markers.
	- `--explain`: run `shap` or permutation importance for interpretability when feasible.

## Visualization & Statistics
- Generate per-split summary CSV/Markdown showing counts, coordinate ranges, and class balance.
- Produce map overlays for a handful of representative scenarios (train and inference) highlighting friend/foe clusters and decision boundaries.
- Include confusion matrix heatmaps, ROC/PR curves, and calibration plots in `reports/figures/`.
- Track metric history per training run (file naming convention: `reports/run_<timestamp>/`).

## Task Breakdown (bite-sized)
1. **Scaffold project**: create directory tree, `pyproject.toml`/`requirements.txt`, and base config files.
2. **Data loader**: implement CSV reader + JSON parsing + validation utilities with unit tests.
3. **Feature module**: design reusable feature functions plus tests verifying numeric stability.
4. **Model training CLI**: wire ingestion + features + estimator selection, log metrics, persist artifacts.
5. **Evaluation & visualization**: script to build metrics report and map plots for selected scenarios.
6. **Neural model track**: implement PyTorch MLP training script/notebook, compare metrics vs classical models, and document selection criteria.
7. **Inference CLI/API**: load artifact, accept input lists, output prediction JSON, optional visualization.
8. **Testing & CI**: add pytest suites (unit + integration) and configure GitHub Actions or similar for lint/test runs.
9. **Documentation**: keep README + usage examples updated, add docstrings and sample notebooks if needed.

## Outstanding Questions
1. Confirm preferred visualization stack (Folium-only vs Folium + Matplotlib) before locking dependencies.
2. Are there geographic regions of interest requiring projection transforms (e.g., UTM) for better distance calculations?
3. Do we expect temporal ordering or motion vectors that should influence feature design?

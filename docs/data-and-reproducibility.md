# Data provenance and reproducibility

This guide explains where the project data comes from, how it is prepared, and what is required to reproduce a run.

## Data source behaviour

The ingestion code in `src/data/ingest.py` first attempts to download a Telco Customer Churn CSV from the IBM sample-data GitHub repository.

If that request fails, the code calls `generate_benchmark_churn_data()`, which creates a synthetic dataset with 7,043 rows, a similar set of columns and a seeded probability-based churn label generator.

That means a pipeline run can use either:
- A downloaded CSV, if the configured URL is reachable and the request succeeds; or
- Synthetic benchmark data, if the download fails.

The current pipeline does not write a separate provenance manifest that records which path was used for each run. A raw CSV already present on disk is also reused unless the ingestion function is forced to fetch again.

**Important when reporting metrics:** the committed test metrics do not, by themselves, prove whether the run used the downloaded CSV or synthetic fallback. Do not present the scores as verified performance on the original IBM data or real customer records unless you can identify the exact source file and run provenance.

## Data preparation

The current settings in `params.yaml` specify:

- Target: `Churn`
- Identifier removed from model features: `customerID`
- Numeric features: `tenure`, `MonthlyCharges`, `TotalCharges`
- Categorical features: 16 customer/service/contract/payment attributes
- Split proportions: 70% train, 15% validation, 15% test
- Random state: 42

The loader:
1. Validates required features and target.
2. Converts `TotalCharges` to numeric and imputes missing values using `MonthlyCharges * tenure` (with tenure clipped to at least one).
3. Converts `SeniorCitizen` to string for categorical processing.
4. Converts the `Churn` label to 0/1.
5. Performs stratified train/validation/test splits.
6. Saves the training split again as the reference distribution for drift analysis.

## Re-run the pipeline

From the repository root, activate the project's Python environment and run:

```bash
python scripts/run_pipeline.py
```

This runs ingestion, cleaning and splitting, candidate training, held-out evaluation and drift analysis.

If you want the pipeline to attempt a fresh download instead of reusing an existing raw CSV, remove or move the current raw CSV first, or call the ingestion function with `force_download=True` from a controlled script. Preserve the resulting raw CSV and record its source/checksum before using its metrics in a report.

## DVC workflow

The repository defines stages in `dvc.yaml`. Run:

```bash
dvc repro
```

The checked-in `.dvc/config` currently contains no remote configuration. DVC can therefore run/cache stages locally, but another machine cannot retrieve this account's DVC cache through a configured remote. Configure and document a remote (and commit only the non-secret remote configuration) if shared data/artifact reproduction is required.

## Model selection and evaluation

The pipeline trains three candidates and selects the one with the highest configured primary metric on the validation split. The current primary metric is ROC-AUC.

The `min_roc_auc_threshold` value in `params.yaml` is currently used to produce a warning. The code does not block champion artifact creation if the score is below that threshold.

After selection, the pipeline evaluates the champion on the held-out test split and saves:
- `artifacts/reports/evaluation_metrics.json`
- Confusion matrix, ROC curve and feature-importance plots
- `artifacts/models/champion_model.joblib`
- `artifacts/models/model_metadata.json`

Model artifacts and generated datasets may be ignored by Git and created locally or during the Docker build.

## What to retain for a defensible result

For each run, record:
- Whether the raw data was downloaded or synthetically generated.
- The source URL and retrieval date, or the synthetic-generator version and seed.
- A SHA-256 checksum of the raw CSV.
- Dataset row/column counts and the train/validation/test row counts.
- The split seed and relevant `params.yaml` version.
- Candidate metrics and selected model.
- Held-out test metrics and the exact model artifact.
- Python and dependency versions.
- MLflow run IDs and the commit SHA.

A future improvement would be to write this information automatically to a `run_manifest.json` beside the evaluation report.

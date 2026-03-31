# Databricks Demand Forecasting Pipeline

## Overview

This repo contains a working demo of a demand forecasting data pipeline built for Databricks.
It covers:
- Data pipeline setup with synthetic data and Delta Lake
- Feature engineering in PySpark
- Model training and experiment tracking with MLflow
- Delta Lake deployment of predictions
- Kafka-like streaming inference simulation

The code is organized as notebook-like Python scripts in `notebooks/`.

## Project Structure

- `notebooks/01_data_pipeline.py`  : Generate synthetic order data, clean, store as Delta table (`supply_chain_orders`).
- `notebooks/02_feature_engineering.py` : Build features and save to Delta table (`demand_forecast_features`).
- `notebooks/03_model_training.py` : Train and compare Linear Regression, GBT, Random Forest; log with MLflow.
- `notebooks/04_delta_deployment.py` : Train final model on full dataset, produce predictions table (`demand_forecast_predictions`), show Delta history.
- `notebooks/05_kafka_streaming.py` : Simulate streaming ingestion using `rate` source, generate features and score model in streaming mode.
- `PROJECT_CONTEXT.md`           : Project context notes.
- `TECHNICAL_DOCUMENTATION.md`   : Detailed project walkthrough (local only, ignored by Git).
- `.gitignore` includes `dbenv/` and `TECHNICAL_DOCUMENTATION.md`.

## Key Technologies

- Databricks / Apache Spark (PySpark)
- Delta Lake for ACID table storage and versioning
- MLflow for experiment tracking and model registry
- Structured Streaming (simulated via `rate` source)
- Scikit-Learn + Spark MLlib

## Setup

1. Ensure Python 3.9+ and Databricks-compatible environment.
2. Create and activate the virtual environment (already in `dbenv/` in this repo):
   ```powershell
   cd C:\Documentpc\databricks-demand-forecasting-pipeline
   .\dbenv\Scripts\Activate.ps1
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
   (If `requirements.txt` does not exist, install `pyspark`, `mlflow`, `pandas`, `scikit-learn`, `delta-spark`.)

4. Confirm `.gitignore` blocks local artifacts:
   - `dbenv/`
   - `TECHNICAL_DOCUMENTATION.md`

## Usage

### 1) Step 1 - Data pipeline

```powershell
python notebooks/01_data_pipeline.py
```

This generates synthetic orders, cleans them, and writes to Delta table `supply_chain_orders`.

### 2) Step 2 - Feature engineering

```powershell
python notebooks/02_feature_engineering.py
```

Builds lag/rolling stats, frequency features, popularity metrics, and saves table `demand_forecast_features`.

### 3) Step 3 - Model training + MLflow

```powershell
python notebooks/03_model_training.py
```

Trains 3 models (LinearRegression, Gradient Boosting, RandomForest) and logs metrics to MLflow. Best model chosen by RMSE.

### 4) Step 4 - Delta deployment

```powershell
python notebooks/04_delta_deployment.py
```

Retrains chosen model on full data, writes predictions table `demand_forecast_predictions`, and validates Delta history/time travel.

### 5) Step 5 - Streaming simulation

```powershell
python notebooks/05_kafka_streaming.py
```

Starts a micro-batch generated stream via `rate` and scores with a linear model, demonstrates continuous prediction output.

## Expected Table Names

- `supply_chain_orders`
- `demand_forecast_features`
- `demand_forecast_predictions`

## Notes

- Ensure you're running in a Databricks runtime or environment where `spark` is available.
- The notebooks use in-memory synthetic and streaming data for demo and testing.
- `TECHNICAL_DOCUMENTATION.md` is intentionally excluded from commit history.

## Validation

- `spark.read.table("<tbl>").count()` after each step to confirm persistence.
- `spark.sql("DESCRIBE HISTORY <tbl>").show()` for Delta version control.

## Troubleshooting

- If `git push` fails due to missing branch: initialize branch and commit as shown previously.
- For Spark errors, confirm cluster runtime and module versions.

---

### Quick command history

```powershell
# check local state
git status

# set branch name
git branch -M main

# initial commit/push
git add .
git commit -m "Add demand forecasting pipeline"
git push -u origin main
```

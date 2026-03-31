%python
import mlflow
mlflow.end_run()  # clear any previous active runs

from pyspark.sql.functions import col, lead
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow.sklearn
import pandas as pd
import numpy as np

print("Step 3: Model Training with MLflow")

df = spark.read.table("demand_forecast_features")
print(f"Loaded {df.count()} feature rows")

window_spec = Window.partitionBy("product_id").orderBy("order_date")
df = df.withColumn('target_quantity', lead("quantity", 1).over(window_spec))

df = df.dropna(subset=["target_quantity"])

print(f"After adding target: {df.count()} rows")

feature_cols = ["prev_day_quantity", "rolling_7day_avg", "orders_last_30days", 
                "avg_order_value_per_product", "total_product_quantity", "category_total_orders"]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_ml = assembler.transform(df)

train_df, test_df = df_ml.randomSplit([0.8, 0.2], seed=42)
print(f"\nTrain rows: {train_df.count()}, Test rows: {test_df.count()}")

train_pd = train_df.select(feature_cols + ["target_quantity"]).toPandas()
test_pd = test_df.select(feature_cols + ["target_quantity"]).toPandas()

X_train = train_pd[feature_cols]
y_train = train_pd["target_quantity"]
X_test = test_pd[feature_cols]
y_test = test_pd["target_quantity"]

print("\n--- TRAINING MODELS ---")

print("\nModel 1: Linear Regression")
mlflow.start_run(run_name="LinearRegression")
mlflow.set_tag("model_type", "Linear Regression")

lr = LinearRegression(featuresCol="features", labelCol="target_quantity", maxIter=10)
lr_model = lr.fit(train_df)
lr_pred = lr_model.transform(test_df)

rmse_evaluator = RegressionEvaluator(labelCol="target_quantity", predictionCol="prediction", metricName="rmse")
r2_evaluator = RegressionEvaluator(labelCol="target_quantity", predictionCol="prediction", metricName="r2")

lr_rmse = rmse_evaluator.evaluate(lr_pred)
lr_r2 = r2_evaluator.evaluate(lr_pred)

mlflow.log_param("max_iter", 10)
mlflow.log_metric("rmse", lr_rmse)
mlflow.log_metric("r2", lr_r2)

print(f"Linear Regression - RMSE: {lr_rmse:.2f}, R2: {lr_r2:.2f}")
mlflow.end_run()

print("\nModel 2: Gradient Boosting")
mlflow.start_run(run_name="GradientBoosting")
mlflow.set_tag("model_type", "Gradient Boosting")

gb = GBTRegressor(featuresCol="features", labelCol="target_quantity", maxIter=10)
gb_model = gb.fit(train_df)
gb_pred = gb_model.transform(test_df)

gb_rmse = rmse_evaluator.evaluate(gb_pred)
gb_r2 = r2_evaluator.evaluate(gb_pred)

mlflow.log_param("max_iter", 10)
mlflow.log_metric("rmse", gb_rmse)
mlflow.log_metric("r2", gb_r2)

print(f"Gradient Boosting - RMSE: {gb_rmse:.2f}, R2: {gb_r2:.2f}")
mlflow.end_run()

print("\nModel 3: Random Forest")
mlflow.start_run(run_name="RandomForest")
mlflow.set_tag("model_type", "Random Forest")

rf = RandomForestRegressor(n_estimators=10, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

mlflow.log_param("n_estimators", 10)
mlflow.log_metric("mae", rf_mae)
mlflow.log_metric("rmse", rf_rmse)
mlflow.log_metric("r2", rf_r2)

mlflow.sklearn.log_model(rf, "model")

print(f"Random Forest - RMSE: {rf_rmse:.2f}, R2: {rf_r2:.2f}")
mlflow.end_run()

print("\n--- MODEL COMPARISON ---")
print(f"Linear Regression RMSE: {lr_rmse:.2f}")
print(f"Gradient Boosting RMSE: {gb_rmse:.2f}")
print(f"Random Forest RMSE: {rf_rmse:.2f}")

models = {
    "LinearRegression": lr_rmse,
    "GradientBoosting": gb_rmse,
    "RandomForest": rf_rmse
}

best_model = min(models, key=models.get)
print(f"\n✓ Best model: {best_model} with RMSE: {models[best_model]:.2f}")


print("✓ Check MLflow UI to see all experiments")
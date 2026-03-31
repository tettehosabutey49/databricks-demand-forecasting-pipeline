from pyspark.sql.functions import col, current_timestamp, lit, abs as spark_abs, avg, lead
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql.window import Window
import mlflow.sklearn

print("Step 4: Delta Lake & Model Deployment")

df = spark.read.table("demand_forecast_features")

window_spec = Window.partitionBy("product_id").orderBy("order_date")
df = df.withColumn('target_quantity', lead("quantity", 1).over(window_spec))
df = df.dropna(subset=["target_quantity"])

print(f"Loaded {df.count()} rows")

feature_cols = ["prev_day_quantity", "rolling_7day_avg", "orders_last_30days", 
                "avg_order_value_per_product", "total_product_quantity", "category_total_orders"]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_ml = assembler.transform(df)

print("\nTraining best model (Linear Regression) on full dataset...")
lr = LinearRegression(featuresCol="features", labelCol="target_quantity", maxIter=10)
best_model = lr.fit(df_ml)

print("Generating predictions...")
predictions = best_model.transform(df_ml)

predictions_table = predictions.select(
    col("order_date"),
    col("product_id"),
    col("quantity").alias("actual_quantity"),
    col("target_quantity").alias("next_day_actual"),
    col("prediction").alias("predicted_quantity"),
    current_timestamp().alias("prediction_timestamp"),
    lit("LinearRegression").alias("model_name"),
    lit("v1.0").alias("model_version")
)

print("\n--- PREDICTIONS SAMPLE ---")
predictions_table.show(5)

predictions_table_name = "demand_forecast_predictions"
predictions_table.write.mode("overwrite").format("delta").saveAsTable(predictions_table_name)

print(f"\n✓ Saved {predictions_table.count()} predictions to {predictions_table_name}")

print("\n--- DELTA TABLE VERIFICATION ---")
df_pred = spark.read.table(predictions_table_name)
print(f"✓ Table loaded: {df_pred.count()} rows")
df_pred.show(3)

print("\n--- DELTA TABLE HISTORY ---")
spark.sql(f"DESCRIBE HISTORY {predictions_table_name}").show()

print(f"\n--- TABLE SCHEMA ---")
df_pred.printSchema()

print("\n--- PREDICTION QUALITY ---")
mae = df_pred.select(
    avg(spark_abs(col("predicted_quantity") - col("next_day_actual"))).alias("mean_absolute_error")
).collect()[0][0]

print(f"Mean Absolute Error: {mae:.2f}")

print("\n--- TIME TRAVEL EXAMPLE ---")
print(f"✓ You can query old versions of this table using:")
print(f"  SELECT * FROM {predictions_table_name} VERSION AS OF 0")


print(f"✓ Predictions stored in Delta table: {predictions_table_name}")
print(f"✓ Model version: v1.0")
print(f"✓ Ready for consumption and monitoring")
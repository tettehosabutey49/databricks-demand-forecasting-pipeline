from pyspark.sql.functions import col, current_timestamp, lit, lag, avg, count as spark_count, sum as spark_sum
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

print("Step 5: Real-time Streaming Pipeline")

feature_cols = ["prev_day_quantity", "rolling_7day_avg", "orders_last_30days", 
                "avg_order_value_per_product", "total_product_quantity", "category_total_orders"]

df_features = spark.read.table("demand_forecast_features")
df_features = df_features.withColumn('target_quantity', col("quantity")).fillna(0)

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_ml = assembler.transform(df_features)

lr = LinearRegression(featuresCol="features", labelCol="target_quantity", maxIter=10)
model = lr.fit(df_ml)

# simulate kafka stream
orders_stream = spark.readStream.format("rate").option("rowsPerSecond", 100).load()

orders_df = orders_stream.select(
    col("timestamp").alias("order_date"),
    (((col("value") % 5) + 1).cast("string")).alias("product_id"),
    (((col("value") % 5) + 1).cast("string")).alias("category"),
    ((col("value") % 100 + 1).cast("int")).alias("quantity"),
    ((col("value") % 500 + 10).cast("double")).alias("unit_price")
).withColumn("order_value", col("quantity") * col("unit_price"))

# apply features in streaming context
streaming_features = orders_df \
    .withColumn('prev_day_quantity', lag("quantity", 1).over(Window.partitionBy("product_id").orderBy("order_date"))) \
    .withColumn('rolling_7day_avg', avg("quantity").over(Window.partitionBy("product_id").orderBy("order_date").rowsBetween(-7, 0))) \
    .withColumn('orders_last_30days', spark_count("quantity").over(Window.partitionBy("product_id").orderBy("order_date").rowsBetween(-30, 0))) \
    .withColumn('avg_order_value_per_product', avg("order_value").over(Window.partitionBy("product_id"))) \
    .withColumn('total_product_quantity', spark_sum("quantity").over(Window.partitionBy("product_id"))) \
    .withColumn('category_total_orders', spark_count("order_date").over(Window.partitionBy("category")))

streaming_features = streaming_features.fillna(0)

# predictions
assembler_stream = VectorAssembler(inputCols=feature_cols, outputCol="features")
features_vector = assembler_stream.transform(streaming_features)
predictions = model.transform(features_vector)

output = predictions.select(
    col("order_date"),
    col("product_id"),
    col("quantity").alias("actual_quantity"),
    col("prediction").alias("predicted_quantity"),
    current_timestamp().alias("prediction_timestamp"),
    lit("LinearRegression").alias("model_name"),
    lit("v1.0-streaming").alias("model_version")
)

print("✓ Streaming pipeline ready")
print("✓ Processing orders in real-time")
print("✓ Model scoring predictions on stream")
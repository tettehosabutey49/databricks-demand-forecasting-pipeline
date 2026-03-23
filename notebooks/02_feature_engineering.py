%python
from pyspark.sql.functions import col, lag, sum as spark_sum, avg, count as spark_count
from pyspark.sql.window import Window

print("Step 2: Feature Engineering")

# reading the cleaned orders from step 1
df = spark.read.table("supply_chain_orders")
print(f"Loaded {df.count()} orders")

# sorting by product and date (needed for window functions)
df = df.sort("product_id", "order_date")

# feature 1: previous day's quantity (lag)
window_spec = Window.partitionBy("product_id").orderBy("order_date")
df = df.withColumn('prev_day_quantity', lag("quantity", 1).over(window_spec))

# feature 2: 7-day rolling average of quantity (using ROWS instead of RANGE)
df = df.withColumn('rolling_7day_avg', avg("quantity").over(
    Window.partitionBy("product_id").orderBy("order_date").rowsBetween(-7, 0)
))

# feature 3: total orders in the last 30 days (frequency)
df = df.withColumn('orders_last_30days', spark_count("quantity").over(
    Window.partitionBy("product_id").orderBy("order_date").rowsBetween(-30, 0)
))

# feature 4: average order value per product
df = df.withColumn('avg_order_value_per_product', avg("order_value").over(
    Window.partitionBy("product_id")
))

# feature 5: total quantity sold per product (product popularity)
df = df.withColumn('total_product_quantity', spark_sum("quantity").over(
    Window.partitionBy("product_id")
))

# feature 6: category popularity (total orders in category)
df = df.withColumn('category_total_orders', spark_count("order_date").over(
    Window.partitionBy("category")
))

# fill nulls with 0 (from lag function on first row)
df = df.fillna(0)

print("\n--- FEATURES CREATED ---")
df.select("order_date", "product_id", "quantity", "prev_day_quantity", 
          "rolling_7day_avg", "orders_last_30days", "avg_order_value_per_product").show(10)

print("\nFeature columns:")
df.printSchema()

# check if features make sense
print("\n--- FEATURE STATS ---")
df.select("prev_day_quantity", "rolling_7day_avg", "orders_last_30days").describe().show()

# save features to delta
feature_table = "demand_forecast_features"
df.write.mode("overwrite").format("delta").saveAsTable(feature_table)

print(f"\n✓ Saved {df.count()} rows to {feature_table}")

# verify
df_features = spark.read.table(feature_table)
print(f"✓ Verified: {df_features.count()} rows")

print("\n✓ STEP 2 COMPLETE")
from pyspark.sql.functions import col, min, max, desc
from datetime import datetime, timedelta
import random

print("Generating synthetic supply chain data...")

num_rows = 10000
start_date = datetime(2022, 1, 1)
products = ['ProductA', 'ProductB', 'ProductC', 'ProductD', 'ProductE']
categories = ['Electronics', 'Clothing', 'Home', 'Beauty', 'Sports']

# creating 10k fake orders with random dates, products, quantities, prices
data = []
for i in range(num_rows):
    days_offset = random.randint(0, 730)
    order_date = start_date + timedelta(days=days_offset)
    
    data.append({
        'order_date': order_date,
        'product_id': random.choice(products),
        'category': random.choice(categories),
        'quantity': random.randint(1, 100),
        'unit_price': round(random.uniform(10, 500), 2),
        'customer_id': f"CUST_{random.randint(1000, 9999)}"
    })

# converting to spark dataframe
df = spark.createDataFrame(data)
df = df.withColumn('order_value', col('quantity') * col('unit_price'))

print(f"✓ Generated {df.count()} orders")

print("\n--- RAW DATA ---")
df.show(5)
df.printSchema()

# fixing the date column and removing empty rows
print("\n--- CLEANING ---")
df = df.withColumn('order_date', col('order_date').cast('date'))
df = df.dropna()

print(f"✓ Cleaned: {df.count()} rows")

# checking what the data looks like
print("\n--- STATISTICS ---")
df.select('order_value').describe().show()

print("\nOrders by product:")
df.groupBy('product_id').count().orderBy(desc('count')).show()

# saving to delta
print("\n--- SAVING TO DELTA ---")
table_name = "supply_chain_orders"
df.write.mode("overwrite").format("delta").saveAsTable(table_name)

print(f"✓ Saved to {table_name}")

# verifying it actually saved
print("\n--- VERIFICATION ---")
df_delta = spark.read.table(table_name)
print(f"✓ Verified: {df_delta.count()} rows in delta table")
df_delta.show(3)


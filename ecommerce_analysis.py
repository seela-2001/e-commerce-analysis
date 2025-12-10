#!/usr/bin/env python3


from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as _sum, desc, to_timestamp, month, year
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

def main():
    spark = SparkSession.builder \
        .appName("EcommerceAnalysis") \
        .master("local[*]") \
        .getOrCreate()

    # Define schema to avoid type inference issues
    schema = StructType([
        StructField("InvoiceNo", StringType(), True),
        StructField("StockCode", StringType(), True),
        StructField("Description", StringType(), True),
        StructField("Quantity", IntegerType(), True),
        StructField("InvoiceDate", StringType(), True),
        StructField("UnitPrice", DoubleType(), True),
        StructField("CustomerID", StringType(), True),
        StructField("Country", StringType(), True)
    ])

    input_path = "online_retail.csv"  
    df = spark.read.csv(input_path, header=True, schema=schema)

    print("=== Initial schema and sample ===")
    df.printSchema()
    df.show(5, truncate=False)

    # Clean data
    df_clean = df.dropna(subset=["InvoiceNo", "Description", "Quantity", "UnitPrice", "CustomerID"])
    df_clean = df_clean.filter((col("Quantity") > 0) & (col("UnitPrice") > 0))

    df_clean = df_clean.withColumn("InvoiceTimestamp", to_timestamp(col("InvoiceDate"), "yyyy-MM-dd HH:mm"))

    df_clean = df_clean.withColumn("TotalPrice", col("Quantity") * col("UnitPrice"))

    df_clean.cache()

    total_revenue_row = df_clean.agg(_sum("TotalPrice").alias("TotalRevenue")).collect()
    total_revenue = total_revenue_row[0]["TotalRevenue"] if total_revenue_row else 0.0
    print(f"Total Revenue: {total_revenue}")

    top_products = df_clean.groupBy("Description") \
                    .agg(_sum("Quantity").alias("TotalQuantity"), _sum("TotalPrice").alias("Revenue")) \
                    .orderBy(desc("TotalQuantity"))

    print("=== Top 10 Products by Quantity ===")
    top_products.show(10, truncate=60)

    top_customers = df_clean.groupBy("CustomerID") \
                    .agg(_sum("TotalPrice").alias("CustomerRevenue"), _sum("Quantity").alias("TotalItems")) \
                    .orderBy(desc("CustomerRevenue"))

    print("=== Top 10 Customers by Revenue ===")
    top_customers.show(10, truncate=False)

    sales_country = df_clean.groupBy("Country") \
                    .agg(_sum("TotalPrice").alias("CountryRevenue"), _sum("Quantity").alias("TotalItems")) \
                    .orderBy(desc("CountryRevenue"))

    print("=== Sales by Country ===")
    sales_country.show(truncate=False)

    invoice_totals = df_clean.groupBy("InvoiceNo") \
                        .agg(_sum("TotalPrice").alias("InvoiceTotal"))
    avg_basket_row = invoice_totals.agg({"InvoiceTotal": "avg"}).collect()
    avg_basket = avg_basket_row[0][0] if avg_basket_row else 0.0
    print(f"Average Basket Value (average invoice total): {avg_basket}")

    df_time = df_clean.withColumn("Year", year(col("InvoiceTimestamp"))).withColumn("Month", month(col("InvoiceTimestamp")))
    monthly_sales = df_time.groupBy("Year", "Month").agg(_sum("TotalPrice").alias("MonthlyRevenue")).orderBy("Year", "Month")
    print("=== Monthly Sales ===")
    monthly_sales.show(truncate=False)

    out_dir = "ecommerce_results"
    top_products.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{out_dir}/top_products_by_quantity")
    top_customers.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{out_dir}/top_customers_by_revenue")
    sales_country.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{out_dir}/sales_by_country")
    invoice_totals.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{out_dir}/invoice_totals")
    monthly_sales.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{out_dir}/monthly_sales")

    summary_lines = []
    summary_lines.append("E-Commerce Analysis Summary")
    summary_lines.append(f"Total Revenue: {total_revenue}")
    summary_lines.append(f"Average Basket Value: {avg_basket}")
    summary_lines.append("Top 5 Products (by quantity):")

    top5 = top_products.limit(5).collect()
    for r in top5:
        desc = r['Description'] if r['Description'] else "N/A"
        summary_lines.append(f"- {desc[:80]}  | Quantity: {r['TotalQuantity']}  | Revenue: {r['Revenue']}")

    summary_lines.append("Top 5 Customers (by revenue):")
    top5c = top_customers.limit(5).collect()
    for r in top5c:
        summary_lines.append(f"- CustomerID: {r['CustomerID']}  | Revenue: {r['CustomerRevenue']}  | Items: {r['TotalItems']}")

    import os
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\\n".join(summary_lines))

    print("=== Saved results to ./ecommerce_results/ ===")
    print("You can download the CSVs and the summary.txt for inclusion in your report.")

    spark.stop()

if __name__ == "__main__":
    main()
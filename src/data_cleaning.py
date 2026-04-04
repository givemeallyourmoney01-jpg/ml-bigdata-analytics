from pyspark.sql import DataFrame
from pyspark.sql import functions as F

def clean_taxi_data(df: DataFrame) -> DataFrame:
    pickup_col = "tpep_pickup_datetime" if "tpep_pickup_datetime" in df.columns else "pickup_datetime"
    dropoff_col = "tpep_dropoff_datetime" if "tpep_dropoff_datetime" in df.columns else "dropoff_datetime"

    required = [pickup_col, dropoff_col]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Required column missing: {c}")

    df = df.dropna(subset=required)

    df = df.withColumn(
        "trip_duration_min",
        (F.unix_timestamp(F.col(dropoff_col)) - F.unix_timestamp(F.col(pickup_col))) / 60.0
    )

    df = df.filter((F.col("trip_duration_min") > 1) & (F.col("trip_duration_min") < 240))

    if "fare_amount" in df.columns:
        df = df.filter((F.col("fare_amount") > 0) & (F.col("fare_amount") < 500))

    if "passenger_count" in df.columns:
        df = df.filter((F.col("passenger_count") > 0) & (F.col("passenger_count") < 9))

    df = df.dropDuplicates()

    return df

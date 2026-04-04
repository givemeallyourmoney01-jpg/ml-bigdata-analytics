from pyspark.sql import DataFrame
from pyspark.sql import functions as F

def build_features(df: DataFrame) -> DataFrame:
    pickup_col = "tpep_pickup_datetime" if "tpep_pickup_datetime" in df.columns else "pickup_datetime"

    df = (
        df.withColumn("pickup_hour", F.hour(F.col(pickup_col)))
          .withColumn("pickup_dayofweek", F.dayofweek(F.col(pickup_col)))
          .withColumn("pickup_month", F.month(F.col(pickup_col)))
          .withColumn("is_weekend", F.when(F.col("pickup_dayofweek").isin([1, 7]), 1).otherwise(0))
          .withColumn("is_rush_hour", F.when(F.col("pickup_hour").isin([7, 8, 9, 16, 17, 18, 19]), 1).otherwise(0))
    )

    coord_cols = {"pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"}
    if coord_cols.issubset(set(df.columns)):
        df = df.withColumn(
            "manhattan_distance_proxy",
            F.abs(F.col("pickup_longitude") - F.col("dropoff_longitude")) +
            F.abs(F.col("pickup_latitude") - F.col("dropoff_latitude"))
        )
    else:
        df = df.withColumn("manhattan_distance_proxy", F.lit(0.0))

    return df

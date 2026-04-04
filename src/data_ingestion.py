from pyspark.sql import SparkSession, DataFrame
from src.config import Paths

def load_raw_data(spark: SparkSession) -> DataFrame:
    if not Paths().raw_csv.exists():
        raise FileNotFoundError(
            f"Raw file not found at {Paths().raw_csv}. "
            f"Please download Kaggle dataset and place/rename file."
        )
    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(str(Paths().raw_csv))
    )
    return df

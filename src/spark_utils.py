from pyspark.sql import SparkSession

def create_spark(app_name: str = "MLBigDataAnalytics") -> SparkSession:
    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
    return spark

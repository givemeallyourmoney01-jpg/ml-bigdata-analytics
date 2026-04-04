from src.spark_utils import create_spark
from src.data_ingestion import load_raw_data
from src.data_cleaning import clean_taxi_data
from src.features import build_features
from src.train import train_model
from src.evaluate import save_metrics
from src.save_artifacts import ensure_dirs
from src.config import Paths

def main():
    ensure_dirs()
    spark = create_spark("TaxiMLPipeline")

    print("Loading raw data...")
    raw_df = load_raw_data(spark)

    print("Cleaning data...")
    clean_df = clean_taxi_data(raw_df)
    clean_df.write.mode("overwrite").parquet(str(Paths().cleaned_parquet))

    print("Building features...")
    feat_df = build_features(clean_df)
    feat_df.write.mode("overwrite").parquet(str(Paths().features_parquet))

    print("Converting to pandas for sklearn training...")
    pdf = feat_df.toPandas()

    print("Training model...")
    _, metrics = train_model(pdf)

    print("Saving metrics...")
    save_metrics(metrics)

    print("Pipeline complete.")
    print(metrics)
    spark.stop()

if __name__ == "__main__":
    main()

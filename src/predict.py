<<<<<<< HEAD
import os
import json
import joblib
import numpy as np
import pandas as pd


INPUT_PATH = "data/raw/new_trips.csv"
MODEL_PATH = "artifacts/final_model.pkl"
FEATURES_PATH = "artifacts/train_feature_columns.json"
OUTPUT_PATH = "artifacts/predictions.csv"


def preprocess_for_inference(raw_df: pd.DataFrame) -> pd.DataFrame:
    d = raw_df.copy()

    d["tpep_pickup_datetime"] = pd.to_datetime(d["tpep_pickup_datetime"], errors="coerce")
    d["tpep_dropoff_datetime"] = pd.to_datetime(d["tpep_dropoff_datetime"], errors="coerce")

    d["pickup_hour"] = d["tpep_pickup_datetime"].dt.hour
    d["pickup_dayofweek"] = d["tpep_pickup_datetime"].dt.dayofweek
    d["trip_duration_min"] = (
        (d["tpep_dropoff_datetime"] - d["tpep_pickup_datetime"]).dt.total_seconds() / 60
    )

    d = d.drop(columns=["tpep_pickup_datetime", "tpep_dropoff_datetime"], errors="ignore")

    # Soft cleaning
    if "trip_distance" in d.columns:
        d.loc[d["trip_distance"] < 0, "trip_distance"] = np.nan
    if "fare_amount" in d.columns:
        d.loc[d["fare_amount"] < 0, "fare_amount"] = np.nan
    if "trip_duration_min" in d.columns:
        d.loc[d["trip_duration_min"] <= 0, "trip_duration_min"] = np.nan
    if "passenger_count" in d.columns:
        d.loc[d["passenger_count"] <= 0, "passenger_count"] = np.nan

    for col in d.select_dtypes(include=[np.number]).columns:
        d[col] = d[col].fillna(d[col].median())

    for col in d.select_dtypes(exclude=[np.number]).columns:
        if d[col].isnull().sum() > 0:
            mode_val = d[col].mode()
            d[col] = d[col].fillna(mode_val.iloc[0] if len(mode_val) > 0 else "unknown")

    d = pd.get_dummies(d, drop_first=True)
    return d


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Feature columns file not found: {FEATURES_PATH}")

    raw = pd.read_csv(INPUT_PATH)
    print("Input shape:", raw.shape)

    proc = preprocess_for_inference(raw)

    with open(FEATURES_PATH, "r") as f:
        train_feature_cols = json.load(f)

    proc_aligned = proc.reindex(columns=train_feature_cols, fill_value=0)

    model = joblib.load(MODEL_PATH)
    preds = model.predict(proc_aligned)

    out = raw.copy()
    out["predicted_total_amount"] = preds

    os.makedirs("artifacts", exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    print("Predictions generated:", len(preds))
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
=======
import joblib
import pandas as pd
from src.config import Paths

def load_model():
    if not Paths().sklearn_model_path.exists():
        raise FileNotFoundError("Model not found. Train first.")
    return joblib.load(Paths().sklearn_model_path)

def predict_one(payload: dict) -> float:
    model = load_model()
    df = pd.DataFrame([payload])
    pred = model.predict(df)[0]
    return float(pred)
>>>>>>> 8daa2ad (Update README with completed pipeline, metrics, and usage instructions)

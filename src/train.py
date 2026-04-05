import os
import json
import joblib
import numpy as np
import pandas as pd

from datetime import datetime, UTC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


DATA_PATH = "data/raw/yellow_tripdata_2015-01.csv"
ARTIFACT_DIR = "artifacts"
TARGET = "total_amount"
SAMPLE_SIZE = 300000
RANDOM_STATE = 42

# ONLY features available in Streamlit single prediction form
MODEL_FEATURES = [
    "passenger_count",
    "trip_distance",
    "pickup_hour",
    "pickup_dayofweek",
    "pickup_month",
    "VendorID",
]


def preprocess_training_data(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # Datetime parsing
    d["tpep_pickup_datetime"] = pd.to_datetime(d["tpep_pickup_datetime"], errors="coerce")
    d["tpep_dropoff_datetime"] = pd.to_datetime(d["tpep_dropoff_datetime"], errors="coerce")

    # Form-aligned time features
    d["pickup_hour"] = d["tpep_pickup_datetime"].dt.hour
    d["pickup_dayofweek"] = d["tpep_pickup_datetime"].dt.dayofweek
    d["pickup_month"] = d["tpep_pickup_datetime"].dt.month

    # Keep only needed columns + target
    keep_cols = MODEL_FEATURES + [TARGET]
    d = d[[c for c in keep_cols if c in d.columns]].copy()

    # Basic cleaning
    d = d.drop_duplicates()
    d = d.dropna(subset=[TARGET, "pickup_hour", "pickup_dayofweek", "pickup_month"])

    d = d[d["trip_distance"] >= 0]
    d = d[d[TARGET] >= 0]
    d = d[d["passenger_count"] > 0]

    # Fill numeric nulls with median
    for col in d.select_dtypes(include=[np.number]).columns:
        d[col] = d[col].fillna(d[col].median())

    # Ensure VendorID exists
    if "VendorID" not in d.columns:
        d["VendorID"] = 1
    d["VendorID"] = d["VendorID"].fillna(1).astype(int)

    # Outlier clipping on key continuous vars
    for c in ["trip_distance", TARGET]:
        if c in d.columns:
            q1 = d[c].quantile(0.01)
            q99 = d[c].quantile(0.99)
            d[c] = d[c].clip(q1, q99)

    return d


def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    print(f"Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print("Original shape:", df.shape)

    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).copy()
        print("Sampled shape:", df.shape)

    d = preprocess_training_data(df)
    print("Processed shape:", d.shape)

    # One-hot encode VendorID safely
    d_model = pd.get_dummies(d, columns=["VendorID"], drop_first=True)

    X = d_model.drop(columns=[TARGET])
    y = d_model[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=20,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = mean_squared_error(y_test, preds) ** 0.5
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"RMSE: {rmse:.6f}")
    print(f"MAE : {mae:.6f}")
    print(f"R²  : {r2:.6f}")

    # Save model
    model_path = os.path.join(ARTIFACT_DIR, "final_model.pkl")
    joblib.dump(model, model_path)

    # Save feature columns for inference alignment
    features_path = os.path.join(ARTIFACT_DIR, "train_feature_columns.json")
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(X.columns.tolist(), f, indent=2)

    # Save metadata
    metadata = {
        "target": TARGET,
        "model_type": "RandomForestRegressor",
        "model_purpose": "single_prediction_form_compatible",
        "feature_set": X.columns.tolist(),
        "base_form_features": MODEL_FEATURES,
        "params": {
            "n_estimators": 400,
            "max_depth": 20,
            "min_samples_leaf": 1
        },
        "metrics": {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)},
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z")
    }

    metadata_path = os.path.join(ARTIFACT_DIR, "final_model_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Saved:", model_path)
    print("Saved:", features_path)
    print("Saved:", metadata_path)


if __name__ == "__main__":
    main()

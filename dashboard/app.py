import json
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="NYC Taxi Fare Intelligence", page_icon="🚕", layout="wide")
st.title("🚕 NYC Taxi Fare Intelligence")
st.caption("ML dashboard for analytics, single prediction, and batch scoring")

BASE_DIR = Path(r"C:\Users\Mansoor Kachhi\OneDrive\Desktop\ml-bigdata-analytics")
model_path = BASE_DIR / "artifacts" / "dashboard_model.pkl"
feature_cols_path = BASE_DIR / "artifacts" / "dashboard_feature_columns.json"
metrics_path = BASE_DIR / "artifacts" / "dashboard_model_metadata.json"

st.caption(f"DEBUG model exists: {model_path.exists()} -> {model_path}")
st.caption(f"DEBUG features exists: {feature_cols_path.exists()} -> {feature_cols_path}")
st.caption(f"DEBUG metrics exists: {metrics_path.exists()} -> {metrics_path}")

def load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

model_ready = model_path.exists() and feature_cols_path.exists()
metrics = load_json(metrics_path) if metrics_path.exists() else {}

k1, k2, k3, k4 = st.columns(4)
k1.metric("Model Ready", "Yes ✅" if model_ready else "No ❌")
k2.metric("RMSE", f"{metrics.get('metrics', {}).get('rmse', 'N/A')}")
k3.metric("MAE", f"{metrics.get('metrics', {}).get('mae', 'N/A')}")
k4.metric("R²", f"{metrics.get('metrics', {}).get('r2', 'N/A')}")

if not model_ready:
    st.error("Missing dashboard artifacts. Run src/train.py first.")
    st.stop()

model = joblib.load(model_path)
feature_cols = load_json(feature_cols_path)

st.markdown("---")
st.subheader("🎯 Fare Estimator")

passenger_count = st.number_input("Passenger Count", 1, 8, 1)
trip_distance = st.number_input("Trip Distance (miles)", 0.1, 100.0, 2.0, 0.1)
pickup_hour = st.slider("Pickup Hour", 0, 23, 12)
pickup_dayofweek = st.slider("Pickup Weekday (0=Mon, 6=Sun)", 0, 6, datetime.now().weekday())
pickup_month = st.slider("Pickup Month", 1, 12, datetime.now().month)
vendor_id = st.selectbox("Vendor ID", [1, 2], index=0)

if st.button("Predict Fare", type="primary"):
    row = pd.DataFrame([{
        "passenger_count": float(passenger_count),
        "trip_distance": float(trip_distance),
        "pickup_hour": int(pickup_hour),
        "pickup_dayofweek": int(pickup_dayofweek),
        "pickup_month": int(pickup_month),
        "VendorID": int(vendor_id),
        "VendorID_2": 1 if int(vendor_id) == 2 else 0,
    }])

    for c in feature_cols:
        if c not in row.columns:
            row[c] = 0

    X = row[feature_cols]
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)

    raw_pred = float(model.predict(X)[0])
    pred = max(raw_pred, 0.0)

    st.success(f"Estimated Fare: ${pred:.2f}")
    st.write("Using model:", model.__class__.__name__)
    st.write("Aligned trip_distance:", X.iloc[0].get("trip_distance", "MISSING"))
    st.write("Raw pred:", raw_pred)

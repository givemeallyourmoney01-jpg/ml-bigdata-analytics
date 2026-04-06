import streamlit as st; st.error("RUNNING NEW APP.PY MARKER v999")
import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="NYC Taxi Fare Intelligence", page_icon="🚕", layout="wide")
st.title("🚕 NYC Taxi Fare Intelligence")

# HARD PATH (your exact repo root)
BASE_DIR = Path(r"C:\Users\Mansoor Kachhi\OneDrive\Desktop\ml-bigdata-analytics")

model_path = BASE_DIR / "artifacts" / "dashboard_model.pkl"
feature_cols_path = BASE_DIR / "artifacts" / "dashboard_feature_columns.json"
metrics_path = BASE_DIR / "artifacts" / "dashboard_model_metadata.json"

st.write("DEBUG model:", model_path, model_path.exists())
st.write("DEBUG features:", feature_cols_path, feature_cols_path.exists())
st.write("DEBUG metrics:", metrics_path, metrics_path.exists())

def load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

model_ready = model_path.exists() and feature_cols_path.exists()
metrics = load_json(metrics_path) if metrics_path.exists() else {}

c1, c2, c3, c4 = st.columns(4)
c1.metric("Model Ready", "Yes ✅" if model_ready else "No ❌")
c2.metric("RMSE", f"{metrics.get('metrics', {}).get('rmse', 'N/A')}")
c3.metric("MAE", f"{metrics.get('metrics', {}).get('mae', 'N/A')}")
c4.metric("R²", f"{metrics.get('metrics', {}).get('r2', 'N/A')}")

if not model_ready:
    st.error("Model artifacts not found at hard path above.")
    st.stop()

model = joblib.load(model_path)
feature_cols = load_json(feature_cols_path)

st.subheader("🎯 Single Prediction")
passenger_count = st.number_input("Passenger Count", 1, 8, 1)
trip_distance = st.number_input("Trip Distance (miles)", 0.1, 100.0, 2.0, 0.1)
pickup_hour = st.slider("Pickup Hour", 0, 23, 12)
pickup_dayofweek = st.slider("Pickup Weekday", 0, 6, 2)
pickup_month = st.slider("Pickup Month", 1, 12, 4)
vendor_id = st.selectbox("Vendor ID", [1, 2], index=0)

if st.button("Predict Fare"):
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

    raw = float(model.predict(X)[0])
    st.success(f"Estimated Fare: ${max(raw,0):.2f}")
    st.write("Using model:", model.__class__.__name__)
    st.write("Aligned trip_distance:", X.iloc[0].get("trip_distance"))
    st.write("Raw pred:", raw)

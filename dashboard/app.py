import json
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from src.config import Paths

st.set_page_config(page_title="ML Big Data Analytics Dashboard", layout="wide")
st.title("🚕 NYC Taxi ML + Analytics Dashboard")

paths = Paths()
metrics_path = paths.metrics_path
features_path = paths.features_parquet
model_path = Path("artifacts/final_model.pkl")
feature_cols_path = Path("artifacts/train_feature_columns.json")

# ---------- Top: Metrics + Data Snapshot ----------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Model Metrics")
    if Path(metrics_path).exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        st.json(metrics)
    else:
        st.info("Metrics file not found. Run pipeline first.")

with col2:
    st.subheader("Data Snapshot")
    if Path(features_path).exists():
        df = pd.read_parquet(features_path)
        st.write(df.head())
        st.write(f"Rows: {len(df):,} | Columns: {df.shape[1]}")
    else:
        st.info("Features parquet not found. Run pipeline first.")
        df = None

# ---------- Charts ----------
st.subheader("Pickup Hour Distribution")
if Path(features_path).exists():
    if "pickup_hour" in df.columns:
        fig = px.histogram(df, x="pickup_hour", nbins=24, title="Trip Count by Pickup Hour")
        st.plotly_chart(fig, use_container_width=True)

st.subheader("Duration Distribution")
if Path(features_path).exists():
    if "trip_duration_min" in df.columns:
        fig2 = px.histogram(df, x="trip_duration_min", nbins=80, title="Trip Duration (min)")
        st.plotly_chart(fig2, use_container_width=True)

# ---------- Prediction Panel ----------
st.markdown("---")
st.subheader("💰 Fare Prediction")

if not model_path.exists() or not feature_cols_path.exists():
    st.info("Prediction artifacts not found. Run: python src/train.py")
else:
    @st.cache_resource
    def load_model():
        return joblib.load(model_path)

    @st.cache_data
    def load_feature_cols():
        with open(feature_cols_path, "r", encoding="utf-8") as f:
            return json.load(f)

    model = load_model()
    feature_cols = load_feature_cols()

    c1, c2, c3 = st.columns(3)
    with c1:
        passenger_count = st.number_input("Passenger Count", min_value=1, max_value=8, value=1, step=1)
        trip_distance = st.number_input("Trip Distance (miles)", min_value=0.1, max_value=100.0, value=2.5, step=0.1)
    with c2:
        pickup_hour = st.slider("Pickup Hour", 0, 23, 14)
        pickup_weekday = st.slider("Pickup Weekday (0=Mon, 6=Sun)", 0, 6, 2)
    with c3:
        pickup_month = st.slider("Pickup Month", 1, 12, 1)
        vendor_id = st.selectbox("Vendor ID", [1, 2], index=0)

    input_dict = {
        "passenger_count": passenger_count,
        "trip_distance": trip_distance,
        "pickup_hour": pickup_hour,
        "pickup_weekday": pickup_weekday,
        "pickup_month": pickup_month,
        "VendorID": vendor_id,
    }

    X = pd.DataFrame([input_dict])

    # Align with training schema
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_cols]

    if st.button("Predict Fare"):
        try:
            pred = float(model.predict(X)[0])
            st.success(f"Estimated Fare: ${pred:.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

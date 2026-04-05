import json
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import Paths

st.set_page_config(page_title="NYC Taxi Fare Intelligence", page_icon="🚕", layout="wide")
st.title("🚕 NYC Taxi Fare Intelligence")
st.caption("ML dashboard for analytics, single prediction, and batch scoring")

paths = Paths()

metrics_path = Path(paths.metrics_path)
features_path = Path(paths.features_parquet)
model_path = Path("artifacts/final_model.pkl")
feature_cols_path = Path("artifacts/train_feature_columns.json")

# ---- Fallback candidates ----
metrics_candidates = [
    metrics_path,
    Path("artifacts/metrics.json"),
    Path("artifacts/final_model_metadata.json"),  # added
    Path("metrics.json"),
    Path("outputs/metrics.json"),
    Path("artifacts/model_metrics.json"),
]

features_candidates = [
    features_path,
    Path("data/processed/features.parquet"),
    Path("data/processed/day2_sample_clean.csv"),  # added
    Path("data/processed/day1_sample_clean.csv"),  # added
    Path("artifacts/features.parquet"),
    Path("features.parquet"),
    Path("outputs/features.parquet"),
]

def first_existing(candidates):
    for p in candidates:
        p = Path(p)
        if p.exists():
            return p
    return None

metrics_file = first_existing(metrics_candidates)
features_file = first_existing(features_candidates)

@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)

@st.cache_data
def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_dataset(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    return None

def align_to_training_schema(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    out = df.copy()
    for col in feature_cols:
        if col not in out.columns:
            out[col] = 0
    return out[feature_cols]

def build_input_row(passenger_count, trip_distance, pickup_hour, pickup_weekday, pickup_month, vendor_id):
    return pd.DataFrame([{
        "passenger_count": passenger_count,
        "trip_distance": trip_distance,
        "pickup_hour": pickup_hour,
        "pickup_weekday": pickup_weekday,
        "pickup_month": pickup_month,
        "VendorID": vendor_id,
    }])

def pick_metric(d: dict, keys: list, default="N/A"):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

# ---- Load metrics ----
metrics = {}
if metrics_file:
    try:
        metrics = load_json(metrics_file)
    except Exception:
        metrics = {}

# ---- Load dataset ----
df = None
if features_file:
    try:
        df = load_dataset(features_file)
    except Exception:
        df = None

# ---- Load model ----
model_ready = model_path.exists() and feature_cols_path.exists()
model = None
feature_cols = []

if model_ready:
    try:
        model = load_model(model_path)
        feature_cols = load_json(feature_cols_path)
    except Exception as e:
        st.error(f"Model artifacts found but failed to load: {e}")
        model_ready = False

# ---- KPI values with flexible key names ----
rmse_val = pick_metric(metrics, ["rmse", "test_rmse", "best_rmse", "val_rmse"])
mae_val = pick_metric(metrics, ["mae", "test_mae", "best_mae", "val_mae"])
r2_val = pick_metric(metrics, ["r2", "test_r2", "best_r2", "val_r2"])

k1, k2, k3, k4 = st.columns(4)
k1.metric("Model Ready", "Yes ✅" if model_ready else "No ❌")
k2.metric("RMSE", str(rmse_val))
k3.metric("MAE", str(mae_val))
k4.metric("R²", str(r2_val))

if not metrics:
    st.info("Metrics unavailable right now. Predictions can still run if model artifacts are present.")
if df is None:
    st.info("Feature dataset unavailable right now. Analytics charts may be limited.")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🎯 Single Prediction", "📦 Batch Prediction"])

with tab1:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Model Metrics")
        if metrics:
            st.json(metrics)
            st.caption(f"Source: {metrics_file}")
        else:
            st.write("No metrics JSON found in configured/fallback paths.")

    with c2:
        st.subheader("Data Snapshot")
        if df is not None:
            st.dataframe(df.head(10), use_container_width=True)
            st.caption(f"Rows: {len(df):,} | Columns: {df.shape[1]}")
            st.caption(f"Source: {features_file}")
        else:
            st.write("No dataset found in configured/fallback paths.")

    ch1, ch2 = st.columns(2)
    with ch1:
        st.subheader("Pickup Hour Distribution")
        if df is not None and "pickup_hour" in df.columns:
            fig = px.histogram(df, x="pickup_hour", nbins=24, title="Trip Count by Pickup Hour")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Chart unavailable (missing data/column).")

    with ch2:
        st.subheader("Duration Distribution")
        if df is not None and "trip_duration_min" in df.columns:
            fig2 = px.histogram(df, x="trip_duration_min", nbins=80, title="Trip Duration (min)")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.caption("Chart unavailable (missing data/column).")

with tab2:
    st.subheader("Fare Estimator")
    st.caption("Enter trip details and estimate fare.")

    if not model_ready:
        st.warning("Prediction unavailable. Ensure artifacts/final_model.pkl and train_feature_columns.json exist.")
    else:
        a, b, c = st.columns(3)

        with a:
            passenger_count = st.number_input("Passenger Count", min_value=1, max_value=8, value=1, step=1)
            trip_distance = st.number_input("Trip Distance (miles)", min_value=0.1, max_value=100.0, value=2.5, step=0.1)

        with b:
            pickup_hour = st.slider("Pickup Hour", min_value=0, max_value=23, value=14)
            pickup_weekday = st.slider("Pickup Weekday (0=Mon, 6=Sun)", min_value=0, max_value=6, value=datetime.now().weekday())

        with c:
            pickup_month = st.slider("Pickup Month", min_value=1, max_value=12, value=datetime.now().month)
            vendor_id = st.selectbox("Vendor ID", options=[1, 2], index=0)

        if st.button("Predict Fare", type="primary"):
            try:
                row = build_input_row(
                    passenger_count, trip_distance, pickup_hour, pickup_weekday, pickup_month, vendor_id
                )
                X = align_to_training_schema(row, feature_cols)
                pred = float(model.predict(X)[0])
                pred = max(pred, 0.0)
                st.success(f"Estimated Fare: ${pred:.2f}")
                st.caption("Estimate may vary due to traffic, tolls, route choice, and real-time conditions.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

with tab3:
    st.subheader("Batch Prediction (CSV)")
    st.caption("Upload a CSV, generate predictions, and download output.")

    if not model_ready:
        st.warning("Batch prediction unavailable. Missing model artifacts.")
    else:
        st.code("Required columns: passenger_count, trip_distance, pickup_hour, pickup_weekday, pickup_month, VendorID")
        uploaded = st.file_uploader("Upload CSV file", type=["csv"])

        if uploaded is not None:
            try:
                batch_df = pd.read_csv(uploaded)
                st.dataframe(batch_df.head(10), use_container_width=True)

                required = [
                    "passenger_count",
                    "trip_distance",
                    "pickup_hour",
                    "pickup_weekday",
                    "pickup_month",
                    "VendorID",
                ]
                missing = [c for c in required if c not in batch_df.columns]
                if missing:
                    st.error(f"Missing required columns: {missing}")
                else:
                    Xb = align_to_training_schema(batch_df, feature_cols)
                    preds = model.predict(Xb)
                    out = batch_df.copy()
                    out["predicted_fare"] = preds

                    st.success(f"Predictions generated for {len(out):,} rows.")
                    st.dataframe(out.head(20), use_container_width=True)

                    st.download_button(
                        label="⬇️ Download predictions CSV",
                        data=out.to_csv(index=False).encode("utf-8"),
                        file_name="predictions_output.csv",
                        mime="text/csv",
                    )
            except Exception as e:
                st.error(f"Failed to process uploaded file: {e}")

import json
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import Paths

# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(
    page_title="NYC Taxi Fare Intelligence",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🚕 NYC Taxi Fare Intelligence")
st.caption("Professional ML dashboard for analytics + fare prediction")

# -----------------------------
# Paths / Artifacts
# -----------------------------
paths = Paths()
metrics_path = Path(paths.metrics_path)
features_path = Path(paths.features_parquet)
model_path = Path("artifacts/final_model.pkl")
feature_cols_path = Path("artifacts/train_feature_columns.json")

# -----------------------------
# Helpers
# -----------------------------
@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)

@st.cache_data
def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_features(path: Path):
    return pd.read_parquet(path)

def build_input_row(
    passenger_count: int,
    trip_distance: float,
    pickup_hour: int,
    pickup_weekday: int,
    pickup_month: int,
    vendor_id: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        [{
            "passenger_count": passenger_count,
            "trip_distance": trip_distance,
            "pickup_hour": pickup_hour,
            "pickup_weekday": pickup_weekday,
            "pickup_month": pickup_month,
            "VendorID": vendor_id,
        }]
    )

def align_to_training_schema(df: pd.DataFrame, train_cols: list[str]) -> pd.DataFrame:
    aligned = df.copy()
    for col in train_cols:
        if col not in aligned.columns:
            aligned[col] = 0
    aligned = aligned[train_cols]
    return aligned

def prediction_quality_note(distance: float) -> str:
    if distance < 0.5:
        return "Low-distance trips can be noisier due to minimum fare effects."
    if distance > 30:
        return "Long-distance trips may have higher uncertainty due to route/toll variability."
    return "Prediction confidence is generally better in typical city trip ranges."

# -----------------------------
# Sidebar: System Health
# -----------------------------
st.sidebar.header("🛠️ System Health")
artifacts_ok = model_path.exists() and feature_cols_path.exists()
data_ok = features_path.exists()
metrics_ok = metrics_path.exists()

st.sidebar.write(f"Model artifact: {'✅' if model_path.exists() else '❌'}")
st.sidebar.write(f"Feature schema: {'✅' if feature_cols_path.exists() else '❌'}")
st.sidebar.write(f"Metrics file: {'✅' if metrics_ok else '❌'}")
st.sidebar.write(f"Features parquet: {'✅' if data_ok else '❌'}")

if not artifacts_ok:
    st.sidebar.warning("Run training first: `python src/train.py`")

# -----------------------------
# Load available resources
# -----------------------------
metrics = {}
if metrics_ok:
    try:
        metrics = load_json(metrics_path)
    except Exception:
        metrics = {}

df_features = None
if data_ok:
    try:
        df_features = load_features(features_path)
    except Exception:
        df_features = None

model = None
train_cols = []
if artifacts_ok:
    try:
        model = load_model(model_path)
        train_cols = load_json(feature_cols_path)
    except Exception as e:
        st.error(f"Failed to load model artifacts: {e}")
        artifacts_ok = False

# -----------------------------
# Top KPI Row
# -----------------------------
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Model Ready", "Yes" if artifacts_ok else "No")
with k2:
    rmse = metrics.get("rmse", "N/A")
    st.metric("RMSE", f"{rmse}")
with k3:
    mae = metrics.get("mae", "N/A")
    st.metric("MAE", f"{mae}")
with k4:
    r2 = metrics.get("r2", "N/A")
    st.metric("R²", f"{r2}")

st.markdown("---")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["📊 Analytics Dashboard", "🎯 Single Prediction", "📦 Batch Prediction"])

# -----------------------------
# Tab 1: Analytics
# -----------------------------
with tab1:
    st.subheader("Data & Model Overview")

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("#### Model Metrics")
        if metrics:
            st.json(metrics)
        else:
            st.info("No metrics found. Run training pipeline to generate metrics.")

    with c2:
        st.markdown("#### Data Snapshot")
        if df_features is not None:
            st.dataframe(df_features.head(10), use_container_width=True)
            st.caption(f"Rows: {len(df_features):,} | Columns: {df_features.shape[1]}")
        else:
            st.info("Features parquet not found or failed to load.")

    if df_features is not None:
        d1, d2 = st.columns(2)

        with d1:
            if "pickup_hour" in df_features.columns:
                fig_hour = px.histogram(
                    df_features,
                    x="pickup_hour",
                    nbins=24,
                    title="Trip Count by Pickup Hour",
                    color_discrete_sequence=["#1f77b4"],
                )
                st.plotly_chart(fig_hour, use_container_width=True)

        with d2:
            if "trip_duration_min" in df_features.columns:
                fig_duration = px.histogram(
                    df_features,
                    x="trip_duration_min",
                    nbins=80,
                    title="Trip Duration Distribution (minutes)",
                    color_discrete_sequence=["#2ca02c"],
                )
                st.plotly_chart(fig_duration, use_container_width=True)

# -----------------------------
# Tab 2: Single Prediction
# -----------------------------
with tab2:
    st.subheader("Predict Fare for One Trip")
    st.caption("Enter trip details to estimate taxi fare.")

    if not artifacts_ok:
        st.warning("Prediction unavailable. Missing model artifacts.")
    else:
        c1, c2, c3 = st.columns(3)

        with c1:
            passenger_count = st.number_input("Passenger Count", min_value=1, max_value=8, value=1, step=1)
            trip_distance = st.number_input("Trip Distance (miles)", min_value=0.1, max_value=100.0, value=2.5, step=0.1)

        with c2:
            pickup_hour = st.slider("Pickup Hour", 0, 23, 14)
            pickup_weekday = st.slider("Pickup Weekday (0=Mon, 6=Sun)", 0, 6, datetime.now().weekday())

        with c3:
            pickup_month = st.slider("Pickup Month", 1, 12, datetime.now().month)
            vendor_id = st.selectbox("Vendor ID", [1, 2], index=0)

        if st.button("Predict Fare", type="primary"):
            # Basic business validation
            if trip_distance <= 0:
                st.error("Trip distance must be greater than 0.")
            else:
                row = build_input_row(
                    passenger_count=passenger_count,
                    trip_distance=trip_distance,
                    pickup_hour=pickup_hour,
                    pickup_weekday=pickup_weekday,
                    pickup_month=pickup_month,
                    vendor_id=vendor_id,
                )

                try:
                    X = align_to_training_schema(row, train_cols)
                    pred = float(model.predict(X)[0])

                    # Guardrails for display only
                    pred_display = max(pred, 0.0)
                    low = max(pred_display * 0.9, 0.0)
                    high = pred_display * 1.1

                    st.success(f"Estimated Fare: **${pred_display:.2f}**")
                    st.info(f"Expected range: **${low:.2f} – ${high:.2f}**")
                    st.caption(prediction_quality_note(trip_distance))

                except Exception as e:
                    st.error(f"Prediction failed: {e}")

# -----------------------------
# Tab 3: Batch Prediction
# -----------------------------
with tab3:
    st.subheader("Batch Fare Prediction (CSV)")
    st.caption("Upload a CSV file, generate predictions, and download results.")

    if not artifacts_ok:
        st.warning("Batch prediction unavailable. Missing model artifacts.")
    else:
        st.markdown("#### Required input columns")
        st.code(
            "passenger_count, trip_distance, pickup_hour, pickup_weekday, pickup_month, VendorID",
            language="text",
        )

        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.write("Preview:", batch_df.head())

                required_cols = [
                    "passenger_count",
                    "trip_distance",
                    "pickup_hour",
                    "pickup_weekday",
                    "pickup_month",
                    "VendorID",
                ]
                missing = [c for c in required_cols if c not in batch_df.columns]

                if missing:
                    st.error(f"Missing required columns: {missing}")
                else:
                    X_batch = align_to_training_schema(batch_df, train_cols)
                    preds = model.predict(X_batch)

                    out_df = batch_df.copy()
                    out_df["predicted_fare"] = preds

                    st.success(f"Generated predictions for {len(out_df):,} rows.")
                    st.dataframe(out_df.head(20), use_container_width=True)

                    csv_data = out_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇️ Download Predictions CSV",
                        data=csv_data,
                        file_name="predictions_output.csv",
                        mime="text/csv",
                    )

            except Exception as e:
                st.error(f"Failed to process uploaded file: {e}")

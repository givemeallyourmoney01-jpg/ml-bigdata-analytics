import json
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import Paths

# -----------------------------
# Page config + lightweight styling
# -----------------------------
st.set_page_config(
    page_title="NYC Taxi Fare Intelligence",
    page_icon="🚕",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main > div {padding-top: 1.2rem;}
    .block-container {padding-top: 1rem; padding-bottom: 2rem;}
    .kpi-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 14px;
        padding: 14px 16px;
    }
    .kpi-title {font-size: 0.85rem; color: #94a3b8;}
    .kpi-value {font-size: 1.35rem; font-weight: 700; color: #f8fafc;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🚕 NYC Taxi Fare Intelligence")
st.caption("Professional ML dashboard for analytics, single prediction, and batch scoring")

# -----------------------------
# Paths and robust file discovery
# -----------------------------
paths = Paths()

# Primary paths from config
metrics_path = Path(paths.metrics_path)
features_path = Path(paths.features_parquet)

# Known artifacts
model_path = Path("artifacts/final_model.pkl")
feature_cols_path = Path("artifacts/train_feature_columns.json")

# Fallback candidates if config points to missing files
metrics_candidates = [
    metrics_path,
    Path("artifacts/metrics.json"),
    Path("metrics.json"),
    Path("outputs/metrics.json"),
    Path("artifacts/model_metrics.json"),
]

features_candidates = [
    features_path,
    Path("data/processed/features.parquet"),
    Path("artifacts/features.parquet"),
    Path("features.parquet"),
    Path("outputs/features.parquet"),
]

def first_existing(candidates):
    for p in candidates:
        if Path(p).exists():
            return Path(p)
    return None

metrics_found = first_existing(metrics_candidates)
features_found = first_existing(features_candidates)

# -----------------------------
# Cached loaders
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

def align_to_training_schema(df: pd.DataFrame, train_cols: list) -> pd.DataFrame:
    x = df.copy()
    for c in train_cols:
        if c not in x.columns:
            x[c] = 0
    return x[train_cols]

def build_input_row(passenger_count, trip_distance, pickup_hour, pickup_weekday, pickup_month, vendor_id):
    return pd.DataFrame([{
        "passenger_count": passenger_count,
        "trip_distance": trip_distance,
        "pickup_hour": pickup_hour,
        "pickup_weekday": pickup_weekday,
        "pickup_month": pickup_month,
        "VendorID": vendor_id,
    }])

# -----------------------------
# Load resources safely
# -----------------------------
model_ready = model_path.exists() and feature_cols_path.exists()
model = None
train_cols = []
if model_ready:
    try:
        model = load_model(model_path)
        train_cols = load_json(feature_cols_path)
    except Exception as e:
        st.error(f"Model artifacts found but failed to load: {e}")
        model_ready = False

metrics = {}
if metrics_found is not None:
    try:
        metrics = load_json(metrics_found)
    except Exception:
        metrics = {}

df_features = None
if features_found is not None:
    try:
        df_features = load_features(features_found)
    except Exception:
        df_features = None

# -----------------------------
# KPI row (professional cards)
# -----------------------------
c1, c2, c3, c4 = st.columns(4)

def kpi(col, title, value):
    col.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

kpi(c1, "Model Status", "READY ✅" if model_ready else "NOT READY ❌")
kpi(c2, "RMSE", str(metrics.get("rmse", "N/A")))
kpi(c3, "MAE", str(metrics.get("mae", "N/A")))
kpi(c4, "R²", str(metrics.get("r2", "N/A")))

st.markdown("")

# Friendly info instead of harsh "not found"
info_msgs = []
if not metrics:
    info_msgs.append("Metrics unavailable (run training to generate metrics JSON).")
if df_features is None:
    info_msgs.append("Feature dataset unavailable (run feature pipeline to enable charts).")
if info_msgs:
    st.info(" | ".join(info_msgs))

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🎯 Single Prediction", "📦 Batch Prediction"])

# -----------------------------
# TAB 1 Dashboard
# -----------------------------
with tab1:
    left, right = st.columns([1.1, 1])

    with left:
        st.subheader("Model Details")
        if metrics:
            st.json(metrics)
            if metrics_found:
                st.caption(f"Metrics source: `{metrics_found}`")
        else:
            st.write("No metrics loaded yet.")

    with right:
        st.subheader("Data Snapshot")
        if df_features is not None:
            st.dataframe(df_features.head(10), use_container_width=True)
            st.caption(f"Rows: {len(df_features):,} | Columns: {df_features.shape[1]}")
            if features_found:
                st.caption(f"Feature source: `{features_found}`")
        else:
            st.write("No features dataset loaded yet.")

    if df_features is not None:
        g1, g2 = st.columns(2)

        with g1:
            if "pickup_hour" in df_features.columns:
                fig1 = px.histogram(
                    df_features,
                    x="pickup_hour",
                    nbins=24,
                    title="Trip Count by Pickup Hour",
                    color_discrete_sequence=["#3b82f6"],
                )
                st.plotly_chart(fig1, use_container_width=True)

        with g2:
            if "trip_duration_min" in df_features.columns:
                fig2 = px.histogram(
                    df_features,
                    x="trip_duration_min",
                    nbins=80,
                    title="Trip Duration Distribution (minutes)",
                    color_discrete_sequence=["#10b981"],
                )
                st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# TAB 2 Single Prediction
# -----------------------------
with tab2:
    st.subheader("Fare Estimator")
    st.caption("Enter trip attributes to get a fare estimate.")

    if not model_ready:
        st.warning("Model not ready. Ensure artifacts/final_model.pkl and train_feature_columns.json exist.")
    else:
        a, b, c = st.columns(3)

        with a:
            passenger_count = st.number_input("Passenger Count", 1, 8, 1, 1)
            trip_distance = st.number_input("Trip Distance (miles)", 0.1, 100.0, 2.5, 0.1)

        with b:
            pickup_hour = st.slider("Pickup Hour", 0, 23, 14)
            pickup_weekday = st.slider("Pickup Weekday (0=Mon, 6=Sun)", 0, 6, datetime.now().weekday())

        with c:
            pickup_month = st.slider("Pickup Month", 1, 12, datetime.now().month)
            vendor_id = st.selectbox("Vendor ID", [1, 2], index=0)

        if st.button("Predict Fare", type="primary", use_container_width=True):
            try:
                row = build_input_row(
                    passenger_count,
                    trip_distance,
                    pickup_hour,
                    pickup_weekday,
                    pickup_month,
                    vendor_id,
                )
                X = align_to_training_schema(row, train_cols)
                pred = float(model.predict(X)[0])
                pred = max(pred, 0.0)

                st.success(f"Estimated Fare: **${pred:.2f}**")
                st.caption("Note: estimate excludes dynamic factors like tolls, traffic shocks, and surge-like effects.")
            except Exception as e:
                st.error(f"Prediction error: {e}")

# -----------------------------
# TAB 3 Batch Prediction
# -----------------------------
with tab3:
    st.subheader("Batch Scoring (CSV Upload)")
    st.caption("Upload a CSV and download predictions.")

    if not model_ready:
        st.warning("Batch scoring unavailable until model artifacts are present.")
    else:
        st.code("Required columns: passenger_count, trip_distance, pickup_hour, pickup_weekday, pickup_month, VendorID")

        file = st.file_uploader("Upload CSV", type=["csv"])

        if file is not None:
            try:
                raw = pd.read_csv(file)
                st.dataframe(raw.head(10), use_container_width=True)

                required = ["passenger_count", "trip_distance", "pickup_hour", "pickup_weekday", "pickup_month", "VendorID"]
                missing = [c for c in required if c not in raw.columns]
                if missing:
                    st.error(f"Missing required columns: {missing}")
                else:
                    Xb = align_to_training_schema(raw, train_cols)
                    preds = model.predict(Xb)

                    out = raw.copy()
                    out["predicted_fare"] = preds
                    st.success(f"Predictions generated for {len(out):,} rows.")
                    st.dataframe(out.head(20), use_container_width=True)

                    st.download_button(
                        "⬇️ Download predictions CSV",
                        data=out.to_csv(index=False).encode("utf-8"),
                        file_name="batch_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
            except Exception as e:
                st.error(f"Could not process file: {e}")

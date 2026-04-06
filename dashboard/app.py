import json
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="NYC Taxi Fare Intelligence",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------
# Premium Styling v2
# -----------------------------
st.markdown(
    """
    <style>
      :root {
        --bg: #070b14;
        --card: rgba(255,255,255,0.05);
        --border: rgba(255,255,255,0.12);
        --text: #f3f7ff;
        --muted: #c3cde0;
        --accent: #5cc8ff;
        --accent2: #7c5cff;
      }

      .stApp {
        background:
          radial-gradient(circle at 15% -10%, rgba(92,200,255,0.18), transparent 40%),
          radial-gradient(circle at 85% -20%, rgba(124,92,255,0.20), transparent 35%),
          linear-gradient(180deg, #05070d 0%, #0a1020 45%, #0b1222 100%);
      }

      .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2.2rem;
        max-width: 1220px;
      }

      .premium-hero {
        position: relative;
        overflow: hidden;
        border-radius: 22px;
        border: 1px solid var(--border);
        padding: 26px 26px 20px 26px;
        margin-bottom: 16px;
        background:
          linear-gradient(135deg, rgba(12,20,40,0.96), rgba(17,28,56,0.86)),
          radial-gradient(circle at 90% 0%, rgba(92,200,255,0.25), transparent 30%);
        box-shadow: 0 20px 50px rgba(0,0,0,0.35);
      }

      .premium-hero::after{
        content: "";
        position: absolute;
        right: -50px;
        top: -50px;
        width: 220px;
        height: 220px;
        border-radius: 999px;
        background: radial-gradient(circle, rgba(92,200,255,0.25), transparent 60%);
        pointer-events: none;
      }

      .hero-title {
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
        color: var(--text);
        letter-spacing: 0.2px;
      }

      .hero-sub {
        margin-top: 6px;
        color: var(--muted);
        font-size: 0.98rem;
      }

      .chip {
        display: inline-block;
        margin-top: 12px;
        margin-right: 8px;
        padding: 6px 11px;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 600;
        color: #dff4ff;
        border: 1px solid rgba(92,200,255,0.35);
        background: rgba(92,200,255,0.14);
      }

      div[data-testid="stMetric"] {
        background: linear-gradient(180deg, rgba(255,255,255,0.07), rgba(255,255,255,0.03));
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 14px 14px;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
      }

      div[data-testid="stMetricLabel"] {
        color: #c8d5ee !important;
      }

      div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 750 !important;
      }

      .section-card {
        border: 1px solid var(--border);
        background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
        border-radius: 16px;
        padding: 16px;
        box-shadow: 0 10px 24px rgba(0,0,0,0.18);
      }

      .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
      }

      .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 8px 14px;
      }

      .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, rgba(92,200,255,0.18), rgba(124,92,255,0.18)) !important;
        border: 1px solid rgba(124,92,255,0.5) !important;
      }

      .stButton > button, .stDownloadButton > button {
        border-radius: 12px !important;
        border: 1px solid rgba(92,200,255,0.45) !important;
        background: linear-gradient(90deg, #1b7cb8, #5b46d8) !important;
        color: white !important;
        font-weight: 700 !important;
      }

      .stButton > button:hover, .stDownloadButton > button:hover {
        filter: brightness(1.07);
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Premium Header v2
# -----------------------------
st.markdown(
    """
    <div class="premium-hero">
      <div class="hero-title">🚕 NYC Taxi Fare Intelligence</div>
      <div class="hero-sub">Premium analytics cockpit for live fare prediction and batch scoring.</div>
      <span class="chip">Live Inference</span>
      <span class="chip">Interactive Analytics</span>
      <span class="chip">Enterprise Batch Scoring</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">🚕 NYC Taxi Fare Intelligence — Premium Console</div>
        <div class="hero-sub">
            Analytics dashboard, instant fare estimation, and enterprise-grade batch scoring.
        </div>
        <div style="margin-top:10px;">
            <span class="pill">Live Model Inference</span>
            <span class="pill">Batch CSV Scoring</span>
            <span class="pill">Ops Ready</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(r"C:\Users\Mansoor Kachhi\OneDrive\Desktop\ml-bigdata-analytics")
model_path = BASE_DIR / "artifacts" / "dashboard_model.pkl"
feature_cols_path = BASE_DIR / "artifacts" / "dashboard_feature_columns.json"
metrics_path = BASE_DIR / "artifacts" / "dashboard_model_metadata.json"
features_file = BASE_DIR / "data" / "processed" / "day2_sample_clean.csv"

if not features_file.exists():
    for p in [
        BASE_DIR / "data" / "processed" / "day1_sample_clean.csv",
        BASE_DIR / "data" / "processed" / "features.parquet",
        BASE_DIR / "artifacts" / "features.parquet",
        BASE_DIR / "features.parquet",
        BASE_DIR / "outputs" / "features.parquet",
    ]:
        if p.exists():
            features_file = p
            break
    else:
        features_file = None

# -----------------------------
# Helpers
# -----------------------------
def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_dataset(path: Path):
    if path is None:
        return None
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return None

def pick_metric(d: dict, keys: list, default="N/A"):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

def align_to_training_schema(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    out = df.copy()

    if "VendorID_2" in feature_cols and "VendorID_2" not in out.columns:
        if "VendorID" in out.columns:
            out["VendorID_2"] = (
                pd.to_numeric(out["VendorID"], errors="coerce").fillna(1).astype(int) == 2
            ).astype(int)
        else:
            out["VendorID_2"] = 0

    for col in feature_cols:
        if col not in out.columns:
            out[col] = 0

    out = out[feature_cols]
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    return out

def build_input_row(passenger_count, trip_distance, pickup_hour, pickup_weekday, pickup_month, vendor_id):
    return pd.DataFrame([{
        "passenger_count": float(passenger_count),
        "trip_distance": float(trip_distance),
        "pickup_hour": int(pickup_hour),
        "pickup_dayofweek": int(pickup_weekday),
        "pickup_weekday": int(pickup_weekday),
        "pickup_month": int(pickup_month),
        "VendorID": int(vendor_id),
        "VendorID_2": 1 if int(vendor_id) == 2 else 0,
    }])

# -----------------------------
# Load resources
# -----------------------------
model_ready = model_path.exists() and feature_cols_path.exists()
model, feature_cols, metrics, df = None, [], {}, None

if metrics_path.exists():
    try:
        metrics = load_json(metrics_path)
    except Exception:
        metrics = {}

if features_file is not None:
    try:
        df = load_dataset(features_file)
    except Exception:
        df = None

if model_ready:
    try:
        model = joblib.load(model_path)
        feature_cols = load_json(feature_cols_path)
    except Exception as e:
        st.error(f"Artifacts found but failed to load: {e}")
        model_ready = False

metrics_block = metrics.get("metrics", {}) if isinstance(metrics, dict) else {}
rmse_val = pick_metric(metrics_block, ["rmse", "test_rmse", "best_rmse", "val_rmse"])
mae_val = pick_metric(metrics_block, ["mae", "test_mae", "best_mae", "val_mae"])
r2_val = pick_metric(metrics_block, ["r2", "test_r2", "best_r2", "val_r2"])

if rmse_val == "N/A":
    rmse_val = pick_metric(metrics, ["rmse", "test_rmse", "best_rmse", "val_rmse"])
if mae_val == "N/A":
    mae_val = pick_metric(metrics, ["mae", "test_mae", "best_mae", "val_mae"])
if r2_val == "N/A":
    r2_val = pick_metric(metrics, ["r2", "test_r2", "best_r2", "val_r2"])

# -----------------------------
# KPI Row
# -----------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Model Status", "Ready ✅" if model_ready else "Unavailable ❌")
k2.metric("RMSE", f"{float(rmse_val):.4f}" if rmse_val != "N/A" else "N/A")
k3.metric("MAE", f"{float(mae_val):.4f}" if mae_val != "N/A" else "N/A")
k4.metric("R² Score", f"{float(r2_val):.4f}" if r2_val != "N/A" else "N/A")

if not metrics:
    st.info("Metrics metadata unavailable. Inference can still run if artifacts exist.")
if df is None:
    st.info("Feature dataset unavailable; analytics visuals may be limited.")

st.markdown("---")
tab1, tab2, tab3 = st.tabs(["📊 Analytics Studio", "🎯 Live Fare Prediction", "📦 Batch Scoring"])

# -----------------------------
# Tab 1: Analytics
# -----------------------------
with tab1:
    left, right = st.columns([1, 1])

    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Model Metadata")
        if metrics:
            st.json(metrics)
            st.caption(f"Source: {metrics_path}")
        else:
            st.write("No metrics JSON found.")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Data Snapshot")
        if df is not None:
            st.dataframe(df.head(10), use_container_width=True)
            st.caption(f"Rows: {len(df):,} | Columns: {df.shape[1]}")
            st.caption(f"Source: {features_file}")
        else:
            st.write("No feature dataset found.")
        st.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Pickup Hour Distribution")
        if df is not None and "pickup_hour" in df.columns:
            fig = px.histogram(
                df, x="pickup_hour", nbins=24, title="Trip Count by Pickup Hour",
                color_discrete_sequence=["#38bdf8"]
            )
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Chart unavailable (missing pickup_hour).")

    with c2:
        st.subheader("Trip Duration Distribution")
        if df is not None and "trip_duration_min" in df.columns:
            fig2 = px.histogram(
                df, x="trip_duration_min", nbins=80, title="Trip Duration (minutes)",
                color_discrete_sequence=["#a78bfa"]
            )
            fig2.update_layout(template="plotly_dark")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.caption("Chart unavailable (missing trip_duration_min).")

# -----------------------------
# Tab 2: Single Prediction
# -----------------------------
with tab2:
    st.subheader("Live Fare Estimator")
    st.caption("Enter trip details and get instant estimated fare.")

    if not model_ready:
        st.warning("Prediction unavailable. Ensure dashboard artifacts exist in /artifacts.")
    else:
        with st.form("single_pred_form", clear_on_submit=False):
            a, b, c = st.columns(3)

            with a:
                passenger_count = st.number_input("Passenger Count", min_value=1, max_value=8, value=1, step=1)
                trip_distance = st.number_input("Trip Distance (miles)", min_value=0.1, max_value=100.0, value=2.0, step=0.1)

            with b:
                pickup_hour = st.slider("Pickup Hour", min_value=0, max_value=23, value=14)
                pickup_weekday = st.slider("Pickup Weekday (0=Mon, 6=Sun)", min_value=0, max_value=6, value=datetime.now().weekday())

            with c:
                pickup_month = st.slider("Pickup Month", min_value=1, max_value=12, value=datetime.now().month)
                vendor_id = st.selectbox("Vendor ID", options=[1, 2], index=0)

            submitted = st.form_submit_button("🚕 Predict Fare", use_container_width=True)

        if submitted:
            try:
                row = build_input_row(
                    passenger_count, trip_distance, pickup_hour, pickup_weekday, pickup_month, vendor_id
                )
                X = align_to_training_schema(row, feature_cols)
                raw_pred = float(model.predict(X)[0])
                pred = max(raw_pred, 0.0)

                st.success(f"Estimated Fare: **${pred:.2f}**")
                st.caption("This is an ML estimate; actual fare may vary with route, traffic, tolls, and demand.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# -----------------------------
# Tab 3: Batch Prediction
# -----------------------------
with tab3:
    st.subheader("Batch CSV Scoring")
    st.caption("Upload a CSV file and download predicted fares.")

    if not model_ready:
        st.warning("Batch scoring unavailable. Missing model artifacts.")
    else:
        st.code("Required columns: passenger_count, trip_distance, pickup_hour, pickup_weekday (or pickup_dayofweek), pickup_month, VendorID")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded is not None:
            try:
                batch_df = pd.read_csv(uploaded)
                st.dataframe(batch_df.head(10), use_container_width=True)

                if "pickup_dayofweek" not in batch_df.columns and "pickup_weekday" in batch_df.columns:
                    batch_df["pickup_dayofweek"] = batch_df["pickup_weekday"]

                required = ["passenger_count", "trip_distance", "pickup_hour", "pickup_month", "VendorID", "pickup_dayofweek"]
                missing = [c for c in required if c not in batch_df.columns]

                if missing:
                    st.error(f"Missing required columns: {missing}")
                else:
                    Xb = align_to_training_schema(batch_df, feature_cols)
                    preds = model.predict(Xb)

                    out = batch_df.copy()
                    out["predicted_fare"] = pd.to_numeric(preds, errors="coerce").fillna(0).clip(lower=0)

                    st.success(f"Generated predictions for {len(out):,} rows.")
                    st.dataframe(out.head(20), use_container_width=True)

                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.download_button(
                        label="⬇️ Download Predictions CSV",
                        data=out.to_csv(index=False).encode("utf-8"),
                        file_name=f"predictions_{ts}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
            except Exception as e:
                st.error(f"Failed to process uploaded file: {e}")

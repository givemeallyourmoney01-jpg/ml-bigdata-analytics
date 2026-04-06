import json
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="NYC Taxi Fare Intelligence", page_icon="🚕", layout="wide")
st.title("🚕 NYC Taxi Fare Intelligence")
st.caption("ML dashboard for analytics, single prediction, and batch scoring")

# -----------------------------
<<<<<<< HEAD
# Paths (hard-set to avoid path ambiguity)
=======
# Paths (hard-set for reliability)
>>>>>>> b2eee7f (Fix dashboard model loading and prediction pipeline; restore full 3-tab Streamlit UI)
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
# Loaders
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

<<<<<<< HEAD
    # derive VendorID_2 if model expects it
=======
>>>>>>> b2eee7f (Fix dashboard model loading and prediction pipeline; restore full 3-tab Streamlit UI)
    if "VendorID_2" in feature_cols and "VendorID_2" not in out.columns:
        if "VendorID" in out.columns:
            out["VendorID_2"] = (
                pd.to_numeric(out["VendorID"], errors="coerce").fillna(1).astype(int) == 2
            ).astype(int)
        else:
            out["VendorID_2"] = 0

<<<<<<< HEAD
    # ensure required cols exist
=======
>>>>>>> b2eee7f (Fix dashboard model loading and prediction pipeline; restore full 3-tab Streamlit UI)
    for col in feature_cols:
        if col not in out.columns:
            out[col] = 0

    out = out[feature_cols]

<<<<<<< HEAD
    # numeric coercion
=======
>>>>>>> b2eee7f (Fix dashboard model loading and prediction pipeline; restore full 3-tab Streamlit UI)
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    return out

def build_input_row(passenger_count, trip_distance, pickup_hour, pickup_weekday, pickup_month, vendor_id):
    return pd.DataFrame([{
        "passenger_count": float(passenger_count),
        "trip_distance": float(trip_distance),
        "pickup_hour": int(pickup_hour),
        "pickup_dayofweek": int(pickup_weekday),
<<<<<<< HEAD
        "pickup_weekday": int(pickup_weekday),  # compatibility
=======
        "pickup_weekday": int(pickup_weekday),
>>>>>>> b2eee7f (Fix dashboard model loading and prediction pipeline; restore full 3-tab Streamlit UI)
        "pickup_month": int(pickup_month),
        "VendorID": int(vendor_id),
        "VendorID_2": 1 if int(vendor_id) == 2 else 0,
    }])

# -----------------------------
# Load resources
# -----------------------------
model_ready = model_path.exists() and feature_cols_path.exists()
model = None
feature_cols = []
metrics = {}
df = None

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
        st.error(f"Model artifacts found but failed to load: {e}")
        model_ready = False

# metrics extraction
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
# KPI row
# -----------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Model Ready", "Yes ✅" if model_ready else "No ❌")
k2.metric("RMSE", f"{float(rmse_val):.4f}" if rmse_val != "N/A" else "N/A")
k3.metric("MAE", f"{float(mae_val):.4f}" if mae_val != "N/A" else "N/A")
k4.metric("R²", f"{float(r2_val):.4f}" if r2_val != "N/A" else "N/A")

if not metrics:
    st.info("Metrics unavailable right now. Predictions can still run if model artifacts are present.")
if df is None:
    st.info("Feature dataset unavailable right now. Analytics charts may be limited.")

st.markdown("---")
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🎯 Single Prediction", "📦 Batch Prediction"])

# -----------------------------
# Tab 1: Dashboard
# -----------------------------
with tab1:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Model Metrics")
        if metrics:
            st.json(metrics)
            st.caption(f"Source: {metrics_path}")
        else:
            st.write("No metrics JSON found.")

    with c2:
        st.subheader("Data Snapshot")
        if df is not None:
            st.dataframe(df.head(10), use_container_width=True)
            st.caption(f"Rows: {len(df):,} | Columns: {df.shape[1]}")
            st.caption(f"Source: {features_file}")
        else:
            st.write("No feature dataset found.")

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

# -----------------------------
# Tab 2: Single Prediction
# -----------------------------
with tab2:
    st.subheader("Fare Estimator")
    st.caption("Enter trip details and estimate fare.")

    if not model_ready:
        st.warning("Prediction unavailable. Ensure dashboard artifacts exist in /artifacts.")
    else:
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

        if st.button("Predict Fare", type="primary"):
            try:
                row = build_input_row(
                    passenger_count,
                    trip_distance,
                    pickup_hour,
                    pickup_weekday,
                    pickup_month,
                    vendor_id,
                )
                X = align_to_training_schema(row, feature_cols)

                raw_pred = float(model.predict(X)[0])
                pred = max(raw_pred, 0.0)

                st.success(f"Estimated Fare: ${pred:.2f}")
                st.caption("Estimate may vary due to traffic, tolls, route choice, and real-time conditions.")

<<<<<<< HEAD
                # concise debug lines (can remove later)
=======
                # compact debug (optional)
>>>>>>> b2eee7f (Fix dashboard model loading and prediction pipeline; restore full 3-tab Streamlit UI)
                st.write("Using model:", model.__class__.__name__)
                st.write("Aligned trip_distance:", X.iloc[0].get("trip_distance", "MISSING"))
                st.write("Raw pred:", raw_pred)

            except Exception as e:
                st.error(f"Prediction failed: {e}")

# -----------------------------
# Tab 3: Batch Prediction
# -----------------------------
with tab3:
    st.subheader("Batch Prediction (CSV)")
    st.caption("Upload a CSV, generate predictions, and download output.")

    if not model_ready:
        st.warning("Batch prediction unavailable. Missing model artifacts.")
    else:
        st.code("Required columns: passenger_count, trip_distance, pickup_hour, pickup_weekday (or pickup_dayofweek), pickup_month, VendorID")
        uploaded = st.file_uploader("Upload CSV file", type=["csv"])

        if uploaded is not None:
            try:
                batch_df = pd.read_csv(uploaded)
                st.dataframe(batch_df.head(10), use_container_width=True)

                if "pickup_dayofweek" not in batch_df.columns and "pickup_weekday" in batch_df.columns:
                    batch_df["pickup_dayofweek"] = batch_df["pickup_weekday"]

                required_any = ["passenger_count", "trip_distance", "pickup_hour", "pickup_month", "VendorID", "pickup_dayofweek"]
                missing = [c for c in required_any if c not in batch_df.columns]

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

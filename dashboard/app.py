import json
import time
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="NYC Taxi Fare Intelligence",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ===== LUXURY VISUAL LAYER (FORCED) =====
st.markdown("""
<style>
/* Force full-page luxury background */
.stApp {
  background: linear-gradient(180deg,#050505 0%,#0c0f14 100%) !important;
}

/* Make main app transparent so custom layer shows */
[data-testid="stAppViewContainer"] { background: transparent !important; }
[data-testid="stHeader"] { background: rgba(0,0,0,0) !important; }
[data-testid="stToolbar"] { right: 1rem; }

/* Glass cards */
div[data-testid="stMetric"]{
  background: rgba(255,255,255,0.06) !important;
  border: 1px solid rgba(255,193,7,0.35) !important;
  border-radius: 14px !important;
  box-shadow: 0 8px 24px rgba(0,0,0,0.35) !important;
}
</style>
""", unsafe_allow_html=True)

# Animated scene (always visible)
components.html("""
<div id="lux-bg">
  <div class="sky"></div>
  <div class="road"></div>
  <div class="lane lane1"></div>
  <div class="lane lane2"></div>
  <div class="taxi"></div>
</div>

<style>
  #lux-bg{
    position: fixed; inset: 0; z-index: 0; pointer-events:none; overflow:hidden;
    background:
      radial-gradient(circle at 15% -10%, rgba(255,193,7,.14), transparent 35%),
      radial-gradient(circle at 85% -10%, rgba(255,140,0,.10), transparent 30%),
      linear-gradient(180deg,#060606 0%,#0d1018 60%,#101522 100%);
  }
  .sky{
    position:absolute; left:0; right:0; bottom:30%; height:22%; opacity:.18;
    background: repeating-linear-gradient(to right, rgba(255,255,255,.16) 0 20px, rgba(255,255,255,.08) 20px 30px, transparent 30px 40px);
    mask-image: linear-gradient(to top, black 55%, transparent 100%);
  }
  .road{ position:absolute; left:0; right:0; bottom:0; height:30%; background:#0a0a0c; }
  .lane{
    position:absolute; left:-20%; width:140%; height:3px;
    background: repeating-linear-gradient(to right, transparent 0 35px, rgba(255,193,7,.8) 35px 70px);
    animation: move 2.6s linear infinite;
  }
  .lane1{ bottom:16%; }
  .lane2{ bottom:10%; opacity:.65; animation-duration:3.2s; }
  @keyframes move { from {transform:translateX(0)} to {transform:translateX(-220px)} }

  .taxi{
    position:absolute; bottom:12%; width:88px; height:34px; border-radius:10px;
    background: linear-gradient(180deg,#ffd54f,#ffb300);
    box-shadow: 0 0 22px rgba(255,193,7,.5);
    animation: drive 14s linear infinite;
  }
  .taxi:before{
    content:''; position:absolute; top:-11px; left:20px; width:46px; height:15px;
    background:#f3c141; border-radius:7px 7px 0 0;
  }
  @keyframes drive{
    0%{ transform: translateX(-140px); }
    100%{ transform: translateX(calc(100vw + 160px)); }
  }
</style>
""", height=0)

# Foreground container hint
st.markdown("<div style='position:relative; z-index:2'></div>", unsafe_allow_html=True)

# =============================
# LUXURY THEME CSS + BACKGROUND
# =============================
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

      :root{
        --bg0:#050505;
        --bg1:#0b0b0d;
        --glass:rgba(255,255,255,0.06);
        --glass2:rgba(255,255,255,0.03);
        --line:rgba(255,255,255,0.12);
        --txt:#f8f8f5;
        --muted:#d3d0c3;
        --gold:#f4c542;
        --amber:#ffb020;
        --taxi:#ffc107;
        --ok:#37d67a;
      }

      html, body, [class*="css"]  {
        font-family: 'Manrope', sans-serif !important;
      }

      .stApp {
        background:
          radial-gradient(1200px 500px at 20% -10%, rgba(255,193,7,0.12), transparent 50%),
          radial-gradient(900px 400px at 90% -20%, rgba(255,176,32,0.10), transparent 55%),
          linear-gradient(180deg, #040404 0%, #0a0a0c 40%, #0d0f13 100%);
        color: var(--txt);
      }

      .block-container{
        max-width: 1250px;
        padding-top: 1rem;
        padding-bottom: 2rem;
      }

      /* Animated background layer */
      .nyc-bg-wrap{
        position: fixed;
        inset: 0;
        z-index: -2;
        pointer-events: none;
        overflow: hidden;
      }
      .road{
        position: absolute;
        left: 0; right: 0; bottom: 0;
        height: 32%;
        background:
          linear-gradient(180deg, rgba(20,20,22,0.8) 0%, rgba(8,8,9,0.95) 100%);
      }
      .lane{
        position: absolute;
        left: -20%;
        width: 140%;
        height: 3px;
        background: repeating-linear-gradient(
          to right,
          rgba(255,193,7,0.0) 0 40px,
          rgba(255,193,7,0.65) 40px 80px
        );
        animation: laneMove 2.2s linear infinite;
      }
      .lane.l1{ bottom: 18%; }
      .lane.l2{ bottom: 12%; animation-duration: 2.8s; opacity: .7;}
      @keyframes laneMove{
        0%{ transform: translateX(0); }
        100%{ transform: translateX(-240px); }
      }

      .taxi{
        position:absolute;
        bottom: 14%;
        width: 86px;
        height: 36px;
        border-radius: 10px;
        background: linear-gradient(180deg, #ffd54f, #ffb300);
        box-shadow: 0 0 25px rgba(255,193,7,.45);
        animation: taxiDrive 16s linear infinite;
      }
      .taxi:before{
        content:'';
        position:absolute;
        top:-12px; left:20px;
        width:46px; height:16px;
        border-radius:7px 7px 0 0;
        background:#f2c037;
      }
      .taxi:after{
        content:'';
        position:absolute;
        left:8px; right:8px; bottom:-6px;
        height:6px;
        border-radius:6px;
        background:rgba(0,0,0,.35);
        filter: blur(2px);
      }
      @keyframes taxiDrive{
        0%{ transform: translateX(-120px); }
        100%{ transform: translateX(calc(100vw + 140px)); }
      }

      .skyline{
        position:absolute;
        left:0; right:0; bottom:27%;
        height:26%;
        background:
          linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,0)),
          repeating-linear-gradient(
            to right,
            rgba(200,200,200,.14) 0 22px,
            rgba(130,130,130,.08) 22px 32px,
            transparent 32px 38px
          );
        mask-image: linear-gradient(to top, black 55%, transparent 100%);
        opacity:.18;
      }

      .rain{
        position:absolute;
        inset:0;
        background-image: repeating-linear-gradient(
          110deg,
          rgba(255,255,255,0.0) 0 14px,
          rgba(255,255,255,0.08) 14px 15px,
          rgba(255,255,255,0.0) 15px 28px
        );
        animation: rainFall .7s linear infinite;
        opacity:.14;
      }
      @keyframes rainFall{
        from{ background-position: 0 0; }
        to{ background-position: -90px 140px; }
      }

      .hero{
        border: 1px solid rgba(255,193,7,0.28);
        background:
          linear-gradient(120deg, rgba(255,193,7,0.12), rgba(255,176,32,0.06)),
          rgba(255,255,255,0.03);
        border-radius: 18px;
        padding: 20px 24px;
        box-shadow: 0 12px 35px rgba(0,0,0,.35);
      }
      .hero h1{
        margin:0;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: .2px;
        color:#fff9e6;
      }
      .hero p{
        margin:.4rem 0 0;
        color:#f3e4b6;
      }

      .chip{
        display:inline-block;
        margin-top: 12px;
        margin-right: 8px;
        padding: 6px 11px;
        border-radius: 999px;
        background: rgba(255,193,7,0.13);
        border: 1px solid rgba(255,193,7,0.40);
        color:#ffe8a3;
        font-size:.75rem;
        font-weight:700;
      }

      div[data-testid="stMetric"]{
        background: linear-gradient(180deg, var(--glass), var(--glass2));
        border:1px solid var(--line);
        border-radius:14px;
        padding: 12px;
        box-shadow: 0 8px 20px rgba(0,0,0,.25);
      }

      .card{
        background: linear-gradient(180deg, var(--glass), var(--glass2));
        border:1px solid var(--line);
        border-radius:14px;
        padding:14px;
        box-shadow: 0 8px 20px rgba(0,0,0,.25);
      }

      .stTabs [data-baseweb="tab-list"]{ gap:8px; }
      .stTabs [data-baseweb="tab"]{
        border-radius: 10px;
        border:1px solid rgba(255,255,255,.12);
        background: rgba(255,255,255,.03);
      }
      .stTabs [aria-selected="true"]{
        border:1px solid rgba(255,193,7,.45) !important;
        background: linear-gradient(90deg, rgba(255,193,7,.22), rgba(255,176,32,.13)) !important;
      }

      .stButton > button, .stDownloadButton > button{
        border-radius:10px !important;
        border:1px solid rgba(255,193,7,.55) !important;
        background: linear-gradient(90deg, #cc8d00, #ffb020) !important;
        color:#121212 !important;
        font-weight:800 !important;
        transition: .2s ease;
      }
      .stButton > button:hover, .stDownloadButton > button:hover{
        transform: translateY(-1px);
        box-shadow: 0 8px 22px rgba(255,176,32,.28);
      }

      .small-muted{ color:#d2c8a6; font-size:.85rem; }

    </style>

    <div class="nyc-bg-wrap">
      <div class="skyline"></div>
      <div class="road"></div>
      <div class="lane l1"></div>
      <div class="lane l2"></div>
      <div class="taxi"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Rain mode toggle
night_rain = st.toggle("🌧️ Night / Rain Mode", value=False)
if night_rain:
    st.markdown('<div class="nyc-bg-wrap"><div class="rain"></div></div>', unsafe_allow_html=True)

# =============================
# HEADER
# =============================
st.markdown(
    """
    <div class="hero">
      <h1>🚕 NYC Taxi Fare Intelligence</h1>
      <p>Luxury analytics cockpit for intelligent fare forecasting, route insights, and batch scoring.</p>
      <span class="chip">Real-time Fare Intelligence</span>
      <span class="chip">Surge-Aware UX</span>
      <span class="chip">Premium SaaS Aesthetic</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# =============================
# PATHS
# =============================
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


# =============================
# HELPERS
# =============================
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

# =============================
# LOAD RESOURCES
# =============================
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

# =============================
# ANIMATED COUNTERS (micro-interaction)
# =============================
def animated_metric(label, value, suffix=""):
    placeholder = st.empty()
    try:
        target = float(value)
        steps = 20
        for i in range(1, steps + 1):
            val = target * i / steps
            if abs(target) < 10:
                display = f"{val:.2f}{suffix}"
            else:
                display = f"{val:.1f}{suffix}"
            placeholder.metric(label, display)
            time.sleep(0.01)
    except Exception:
        placeholder.metric(label, f"{value}{suffix}")

# KPI row
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("System", "Ready ✅" if model_ready else "Offline ❌")
with k2:
    animated_metric("RMSE", rmse_val if rmse_val != "N/A" else 0)
with k3:
    animated_metric("MAE", mae_val if mae_val != "N/A" else 0)
with k4:
    animated_metric("R² Score", r2_val if r2_val != "N/A" else 0)

if not metrics:
    st.info("Metrics metadata unavailable. Inference still works if artifacts are present.")
if df is None:
    st.info("Feature dataset unavailable; some analytics may be limited.")

st.markdown("---")
tab1, tab2, tab3 = st.tabs(["📊 Fare Insights", "🗺️ Live Map + Fare", "📦 Analytics & Batch"])

# =============================
# TAB 1: FARE INSIGHTS PANEL
# =============================
with tab1:
    a, b, c = st.columns([1.1, 1.1, 1])

    with a:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Real-Time Fare Insights")
        if metrics:
            st.json(metrics)
        else:
            st.write("No metrics available.")
        st.markdown("</div>", unsafe_allow_html=True)

    with b:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Surge Pricing Indicator")
        hr = datetime.now().hour
        surge = 1.0
        if hr in [7, 8, 9, 17, 18, 19]:
            surge = 1.35
        elif hr in [22, 23, 0, 1]:
            surge = 1.18
        color = "🟢" if surge <= 1.1 else ("🟠" if surge < 1.3 else "🔴")
        st.metric("Current Surge", f"{surge:.2f}x {color}")
        st.caption("Based on time-of-day heuristic (demo logic).")
        st.markdown("</div>", unsafe_allow_html=True)

    with c:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Distance & Time Breakdown")
        if df is not None and "trip_distance" in df.columns:
            st.metric("Avg Distance", f"{df['trip_distance'].mean():.2f} mi")
        else:
            st.metric("Avg Distance", "N/A")
        if df is not None and "trip_duration_min" in df.columns:
            st.metric("Avg Duration", f"{df['trip_duration_min'].mean():.1f} min")
        else:
            st.metric("Avg Duration", "N/A")
        st.markdown("</div>", unsafe_allow_html=True)

# =============================
# TAB 2: LIVE MAP SECTION + PREDICTION
# =============================
with tab2:
    st.subheader("Live Route Experience")
    st.caption("Pickup/drop simulation + live fare estimation")

    # Faux animated map block (premium visual if no Mapbox)
    components.html(
        """
        <div style="
          height:260px; border-radius:14px; border:1px solid rgba(255,255,255,.12);
          background:
            radial-gradient(circle at 20% 30%, rgba(255,193,7,.18), transparent 24%),
            radial-gradient(circle at 80% 70%, rgba(255,193,7,.12), transparent 20%),
            linear-gradient(120deg, #0f1726, #111827);
          position:relative; overflow:hidden;">
          <div style="position:absolute; left:8%; top:35%; width:84%; height:2px; background: repeating-linear-gradient(to right, #ffc107 0 12px, transparent 12px 22px); opacity:.8;"></div>
          <div style="position:absolute; left:8%; top:31%; color:#9dd9ff; font-size:12px;">Pickup ●</div>
          <div style="position:absolute; right:8%; top:38%; color:#ffcc80; font-size:12px;">● Drop-off</div>
          <div id="cab" style="position:absolute; left:8%; top:33%; width:22px; height:12px; border-radius:4px; background:#ffc107; box-shadow:0 0 10px rgba(255,193,7,.6);"></div>
          <script>
            const cab = document.getElementById('cab');
            let x = 8;
            setInterval(() => {
              x += 0.35;
              if (x > 84) x = 8;
              cab.style.left = x + '%';
            }, 60);
          </script>
        </div>
        """,
        height=270,
    )

    st.write("")
    if not model_ready:
        st.warning("Prediction unavailable. Ensure artifacts exist in /artifacts.")
    else:
        with st.form("single_pred_form", clear_on_submit=False):
            x1, x2, x3 = st.columns(3)
            with x1:
                passenger_count = st.number_input("Passenger Count", 1, 8, 1)
                trip_distance = st.number_input("Trip Distance (miles)", 0.1, 100.0, 2.0, 0.1)
            with x2:
                pickup_hour = st.slider("Pickup Hour", 0, 23, datetime.now().hour)
                pickup_weekday = st.slider("Pickup Weekday (0=Mon, 6=Sun)", 0, 6, datetime.now().weekday())
            with x3:
                pickup_month = st.slider("Pickup Month", 1, 12, datetime.now().month)
                vendor_id = st.selectbox("Vendor ID", [1, 2], index=0)

            submitted = st.form_submit_button("🚕 Calculate Premium Fare", use_container_width=True)

        if submitted:
            row = build_input_row(
                passenger_count, trip_distance, pickup_hour, pickup_weekday, pickup_month, vendor_id
            )
            X = align_to_training_schema(row, feature_cols)
            raw_pred = float(model.predict(X)[0])
            pred = max(raw_pred, 0.0)
            st.success(f"Estimated Fare: **${pred:.2f}**")
            st.caption("Fare may vary due to route, tolls, weather, and demand.")

# =============================
# TAB 3: ANALYTICS CARDS + BATCH
# =============================
with tab3:
    c1, c2, c3, c4 = st.columns(4)

    total_rides = len(df) if df is not None else 0
    avg_fare = df["total_amount"].mean() if (df is not None and "total_amount" in df.columns) else 0
    peak_hour = int(df["pickup_hour"].mode()[0]) if (df is not None and "pickup_hour" in df.columns and not df.empty) else 0

    with c1:
        st.metric("Total Rides", f"{total_rides:,}")
    with c2:
        st.metric("Avg Fare", f"${avg_fare:.2f}")
    with c3:
        st.metric("Peak Hour", f"{peak_hour}:00")
    with c4:
        st.metric("Demand Index", "High" if total_rides > 100000 else "Moderate")

    st.write("")
    left, right = st.columns(2)

    with left:
        st.subheader("Peak Hours Distribution")
        if df is not None and "pickup_hour" in df.columns:
            fig = px.histogram(df, x="pickup_hour", nbins=24, color_discrete_sequence=["#ffc107"])
            fig.update_layout(template="plotly_dark", height=330)
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Demand Heatmap (Hour vs Weekday)")
        if df is not None and {"pickup_hour", "pickup_dayofweek"}.issubset(df.columns):
            heat = df.groupby(["pickup_dayofweek", "pickup_hour"]).size().reset_index(name="rides")
            fig2 = px.density_heatmap(
                heat,
                x="pickup_hour",
                y="pickup_dayofweek",
                z="rides",
                color_continuous_scale="YlOrBr"
            )
            fig2.update_layout(template="plotly_dark", height=330)
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("Batch CSV Scoring")
    st.caption("Upload CSV to score fares at scale")

    if not model_ready:
        st.warning("Batch scoring unavailable. Missing model artifacts.")
    else:
        st.code("Required columns: passenger_count, trip_distance, pickup_hour, pickup_weekday (or pickup_dayofweek), pickup_month, VendorID")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded is not None:
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
                    "⬇️ Download Predictions CSV",
                    out.to_csv(index=False).encode("utf-8"),
                    file_name=f"luxury_taxi_predictions_{ts}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

st.markdown('<p class="small-muted">© NYC Taxi Fare Intelligence • Luxury Theme Pack</p>', unsafe_allow_html=True)

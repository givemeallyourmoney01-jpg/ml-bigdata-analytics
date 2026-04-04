import json
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
from src.config import Paths

st.set_page_config(page_title="ML Big Data Analytics Dashboard", layout="wide")
st.title("🚕 NYC Taxi ML + Analytics Dashboard")

metrics_path = Paths().metrics_path
features_path = Paths().features_parquet

col1, col2 = st.columns(2)

with col1:
    st.subheader("Model Metrics")
    if metrics_path.exists():
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

st.subheader("Pickup Hour Distribution")
if Path(features_path).exists():
    df = pd.read_parquet(features_path)
    if "pickup_hour" in df.columns:
        fig = px.histogram(df, x="pickup_hour", nbins=24, title="Trip Count by Pickup Hour")
        st.plotly_chart(fig, use_container_width=True)

st.subheader("Duration Distribution")
if Path(features_path).exists():
    if "trip_duration_min" in df.columns:
        fig2 = px.histogram(df, x="trip_duration_min", nbins=80, title="Trip Duration (min)")
        st.plotly_chart(fig2, use_container_width=True)

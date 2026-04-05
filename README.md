# ML Big Data Analytics

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Project-Active-success)

End-to-end **NYC Taxi Fare Prediction** project using **Python, Pandas, Scikit-learn**, with a **Streamlit dashboard** for analytics and inference.

## 📌 Objective
Build a robust regression pipeline to predict `total_amount` from trip-level taxi data, with:
- EDA + cleaning
- Feature engineering
- Model comparison
- Hyperparameter tuning
- Inference pipeline
- Interactive dashboard (KPIs, charts, single + batch prediction)

---

## 📸 Dashboard Preview

> Add screenshots to `assets/` with the filenames below.

### Main Dashboard
![Dashboard Main](assets/dashboard-main.png)

### Single Prediction
![Dashboard Single Prediction](assets/dashboard-single-prediction.png)

### Batch Prediction
![Dashboard Batch Prediction](assets/dashboard-batch-prediction.png)

---

## 🗂️ Project Structure

```text
ml-bigdata-analytics/
│── assets/
│   ├── dashboard-main.png
│   ├── dashboard-single-prediction.png
│   └── dashboard-batch-prediction.png
│── dashboard/
│   └── app.py                       # Streamlit dashboard UI
│── data/
│   ├── raw/                         # input datasets
│   └── processed/                   # cleaned outputs
│── notebooks/
│   └── eda.ipynb                    # full EDA + modeling workflow
│── src/
│   ├── train.py                     # train + evaluate + save artifacts
│   ├── predict.py                   # batch inference pipeline
│   ├── config.py                    # centralized paths/config
│   └── predict_service.py           # optional legacy name (if retained)
│── artifacts/
│   ├── day3_model_comparison.csv
│   ├── day4_rf_tuning_results.csv
│   ├── final_model.pkl
│   ├── train_feature_columns.json
│   ├── final_model_metadata.json
│   └── predictions.csv
│── requirements.txt
│── .gitignore
└── README.md
```

---

## ⚙️ Setup

### 1) Clone
```bash
git clone https://github.com/givemeallyourmoney01-jpg/ml-bigdata-analytics.git
cd ml-bigdata-analytics
```

### 2) Create virtual environment

**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Mac/Linux**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🧪 Model Development Summary

### Day 1 (Baseline RF)
- RMSE: **0.8107**
- MAE: **0.0305**
- R²: **0.9955**

### Day 2 (Feature engineering + outlier handling)
- RMSE: **0.3575**
- MAE: **0.0298**
- R²: **0.9990**

### Day 3 (Model comparison)
- RandomForestRegressor: RMSE **0.357460**, MAE **0.029818**, R² **0.999008**
- HistGradientBoostingRegressor: RMSE **0.367301**, MAE **0.078947**, R² **0.998953**

### Day 4 (RF tuning - best model)
Best parameters:
- `n_estimators=400`
- `max_depth=20`
- `min_samples_leaf=1`

Final metrics:
- RMSE: **0.351503**
- MAE: **0.026806**
- R²: **0.999041**

### Day 5
- Built inference flow to load model and generate `predicted_total_amount`
- Exports predictions to `artifacts/predictions.csv`

### Day 6 (Final polish)
- Updated metadata timestamp generation to timezone-aware UTC:
  - `datetime.now(UTC).isoformat().replace("+00:00", "Z")`
- Re-ran training successfully on `yellow_tripdata_2015-01.csv`
- Verified inference:
  - `python src/predict.py` generates `artifacts/predictions.csv`

---

## 📊 Streamlit Dashboard

Dashboard entrypoint:
- `dashboard/app.py`

### Features
- KPI cards for:
  - Model readiness
  - RMSE / MAE / R² (auto-loaded from metadata JSON)
- Dataset snapshot preview
- Analytics charts:
  - Pickup hour distribution
  - Trip duration distribution (if present in dataset)
- Single prediction form
- Batch CSV prediction + downloadable output

### Run dashboard locally
```bash
streamlit run dashboard/app.py
```

### Dashboard data/artifact expectations
- Model: `artifacts/final_model.pkl`
- Feature schema: `artifacts/train_feature_columns.json`
- Metrics: `artifacts/final_model_metadata.json`
- Analytics sample data (preferred): `data/processed/day2_sample_clean.csv`

---

## 🔁 Reproducibility

```bash
python src/train.py
python src/predict.py
streamlit run dashboard/app.py
```

---

## ▶️ Script Usage

### Train
1. Place dataset in `data/raw/`
2. Update data path/config in `src/train.py` or `src/config.py`
3. Run:

```bash
python src/train.py
```

Generated:
- `artifacts/final_model.pkl`
- `artifacts/train_feature_columns.json`
- `artifacts/final_model_metadata.json`

### Predict
1. Place inference file at `data/raw/new_trips.csv`
2. Run:

```bash
python src/predict.py
```

Output:
- `artifacts/predictions.csv`

---

## 🧠 Tech Stack
- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib / Seaborn
- Plotly
- Streamlit
- PySpark (project-ready structure)

---

## 👤 Authors
- **Jafar Kachhi**
- **Janvi Patel**

GitHub: [givemeallyourmoney01-jpg](https://github.com/givemeallyourmoney01-jpg)

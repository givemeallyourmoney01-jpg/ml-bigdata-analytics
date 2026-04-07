<p align="center">
  <img src="https://img.shields.io/badge/NYC-Taxi%20Fare%20Intelligence-111111?style=for-the-badge&logo=googlemaps&logoColor=F4C542" />
  <img src="https://img.shields.io/badge/Luxury%20Theme-Pack-FFB020?style=for-the-badge&logo=streamlit&logoColor=111111" />
  <img src="https://img.shields.io/badge/ML-Powered-2E7D32?style=for-the-badge&logo=scikitlearn&logoColor=white" />
</p>

<h1 align="center">🚕 NYC Taxi Fare Intelligence</h1>
<p align="center">
  Premium ML dashboard for fare forecasting, live route simulation, surge awareness, and batch analytics.
</p>

<p align="center">
  <a href="https://github.com/givemeallyourmoney01-jpg/ml-bigdata-analytics">
    <img src="https://img.shields.io/badge/View%20Repository-000000?style=flat-square&logo=github&logoColor=white" />
  </a>
</p>

---

# ML Big Data Analytics

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Project-Active-success)

End-to-end **NYC Taxi Fare Prediction** project using **Python, Pandas, Scikit-learn**, featuring a premium **Streamlit dashboard** with analytics, live fare estimation, and batch scoring.

---

## 📌 Objective

Build a robust regression pipeline to predict taxi fare (`total_amount`) from trip-level data with:

- EDA + cleaning
- Feature engineering
- Model comparison
- Hyperparameter tuning
- Inference pipeline
- Interactive premium dashboard (KPIs, charts, single + batch prediction)

---

## 🌟 Luxury Theme Pack (UI Upgrade)

The dashboard now includes a premium “Luxury Taxi Theme Pack”:

- Animated NYC-inspired background (moving taxi + lane motion)
- Dark luxury styling (black, amber, gold accents)
- Glassmorphism KPI and panel cards
- Surge pricing indicator (time-based heuristic)
- Live route simulation section
- Night/Rain mode toggle
- Improved visual hierarchy and tabbed UX

Main title: **NYC Taxi Fare Intelligence**

---

## 📸 Dashboard Preview

> Add screenshots to `assets/` with the filenames below.

### Main Dashboard
![Dashboard Main](assets/dashboard-main.png)

### Fare Insights
![Dashboard Fare Insights](assets/dashboard-fare-insights.png)

### Live Map + Fare
![Dashboard Live Map](assets/dashboard-live-map.png)

### Analytics + Batch
![Dashboard Batch Prediction](assets/dashboard-batch-prediction.png)

---

## 🗂️ Project Structure

```text
ml-bigdata-analytics/
│── assets/
│   ├── dashboard-main.png
│   ├── dashboard-fare-insights.png
│   ├── dashboard-live-map.png
│   └── dashboard-batch-prediction.png
│── dashboard/
│   └── app.py
│── data/
│   ├── raw/
│   └── processed/
│── notebooks/
│   └── eda.ipynb
│── src/
│   ├── train.py
│   ├── predict.py
│   ├── config.py
│   └── predict_service.py
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
- Built inference flow to generate `predicted_total_amount`
- Exports predictions to `artifacts/predictions.csv`

### Day 6
- UTC-safe metadata timestamp polishing
- Retraining + prediction flow validated end-to-end

### Day 7 (Dashboard Premium Upgrade)
- Added Luxury Theme Pack UI
- Added animated route section and premium tabs
- Added Night/Rain visual mode
- Preserved single + batch prediction functionality

---

## 📊 Streamlit Dashboard

Entrypoint:

- `dashboard/app.py`

### Features

- **Fare Insights tab**
  - Model metadata
  - Surge pricing indicator
  - Distance & duration snapshot

- **Live Map + Fare tab**
  - Animated route visualization
  - Single trip fare prediction form

- **Analytics & Batch tab**
  - Ride and demand KPI cards
  - Hourly distribution chart
  - Demand heatmap
  - Batch CSV upload + downloadable predictions

### Run dashboard locally

```bash
streamlit run dashboard/app.py
```

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

```bash
python src/train.py
```

Generated artifacts:
- `artifacts/final_model.pkl`
- `artifacts/train_feature_columns.json`
- `artifacts/final_model_metadata.json`

### Predict

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

## ⚠️ Known Limitations

- **Data scope:** Current training/evaluation is based on sampled/processed NYC taxi data and may not fully represent all seasonal or borough-level behavior.
- **Feature coverage:** Fare drivers like tolls, exact route geometry, weather severity, and special events are simplified or not fully modeled.
- **Surge logic in UI:** Surge indicator is currently a **time-based heuristic** for UX demonstration, not a real-time market feed.
- **Model consistency:** Experimental notebooks may include multiple model variants; dashboard inference depends on the artifact currently loaded in `artifacts/`.
- **Map realism:** Live route section is a visual simulation unless integrated with a real map/traffic API.
- **Operational robustness:** No deployed API, auth, request throttling, or centralized observability stack yet.
- **Drift handling:** Automated retraining and drift alerts are not yet enabled.

---

## 🚀 Future Work

- Integrate **real-time map and traffic layers** (Mapbox/Leaflet + traffic API).
- Add **weather/event features** to improve robustness during abnormal demand periods.
- Serve model via **FastAPI** with versioned endpoints and structured logging.
- Add **ML monitoring** (data drift, prediction drift, performance decay alerts).
- Compare advanced models (e.g., **XGBoost/LightGBM/CatBoost**) with explainability (SHAP).
- Add CI/CD for model + dashboard deployment (GitHub Actions).
- Introduce role-based dashboard views for Ops, Product, and Finance users.
- Build scheduled retraining + artifact registry workflow.

---

## 🙏 Acknowledgements

- NYC TLC trip-record public datasets for enabling real-world fare modeling experiments.
- Open-source Python ecosystem: **Pandas, NumPy, Scikit-learn, Plotly, Streamlit**.
- Community documentation and examples that accelerated iterative dashboard and ML pipeline development.

---

## 📚 How to Cite

If you use this project, please cite it as:

```bibtex
@misc{kachhi2026nyctaxi,
  author       = {Jafar Kachhi and Janvi Patel},
  title        = {NYC Taxi Fare Intelligence: ML Big Data Analytics},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/givemeallyourmoney01-jpg/ml-bigdata-analytics}}
}
```

Or:

> Kachhi, J., & Patel, J. (2026). *NYC Taxi Fare Intelligence: ML Big Data Analytics*.  
> GitHub repository: https://github.com/givemeallyourmoney01-jpg/ml-bigdata-analytics

---

## 👤 Authors

- **Jafar Kachhi**
- **Janvi Patel**

GitHub: [givemeallyourmoney01-jpg](https://github.com/givemeallyourmoney01-jpg)

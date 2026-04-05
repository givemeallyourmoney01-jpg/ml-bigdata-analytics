# ML Big Data Analytics

End-to-end **Taxi Fare Prediction** project using **Python, Pandas, Scikit-learn, and PySpark-ready structure**.

## 📌 Objective
Build a robust regression pipeline to predict `total_amount` from trip-level taxi data, with:
- EDA + cleaning
- feature engineering
- model comparison
- tuning
- inference pipeline

---

## 🗂️ Project Structure

```text
ml-bigdata-analytics/
│── data/
│   ├── raw/                         # input datasets
│   └── processed/                   # cleaned outputs
│── notebooks/
│   └── eda.ipynb                    # full EDA + modeling workflow
│── src/
│   ├── train.py                     # train + evaluate + save artifacts
│   ├── predict.py                   # batch inference pipeline
│   └── predict_service.py           # optional legacy name (if retained)
│── artifacts/
│   ├── day3_model_comparison.csv
│   ├── day4_rf_tuning_results.csv
│   ├── final_model.pkl              # local artifact (optional in git)
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

### 2) Create venv

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

---

## ▶️ Usage

### Notebook workflow
```bash
code .
```
Open `notebooks/eda.ipynb`, select `.venv` kernel, run all cells.

### Script workflow

#### Train
1. Place dataset in `data/raw/`
2. Update `DATA_PATH` inside `src/train.py`
3. Run:

```bash
python src/train.py
```

Generated:
- `artifacts/final_model.pkl`
- `artifacts/train_feature_columns.json`
- `artifacts/final_model_metadata.json`

#### Predict
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
- PySpark (project-ready structure)

---

## 👤 Authors
- **Jafar Kachhi**
- **Janvi Patel**

GitHub: [givemeallyourmoney01-jpg](https://github.com/givemeallyourmoney01-jpg)

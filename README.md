# ML Big Data Analytics

An end-to-end Machine Learning + Big Data analytics project using **Python, Pandas, Scikit-learn, and PySpark**.

## 📌 Project Goals
- Perform exploratory data analysis (EDA)
- Clean and preprocess real-world data
- Train a baseline ML model
- Evaluate model performance
- Prepare project structure for scalable big-data workflows (Spark)

---

## 🗂️ Project Structure
```text
ml-bigdata-analytics/
│── data/
│   ├── raw/                # original dataset(s)
│   └── processed/          # cleaned/transformed data
│── notebooks/
│   └── eda.ipynb           # EDA + baseline modeling notebook
│── src/
│   └── main.py             # python script entry point
│── models/                 # saved ML models (optional)
│── artifacts/              # plots/reports/figures
│── requirements.txt
│── .gitignore
└── README.md
```

---

## ⚙️ Setup Instructions

### 1) Clone repo
```bash
git clone https://github.com/givemeallyourmoney01-jpg/ml-bigdata-analytics.git
cd ml-bigdata-analytics
```

### 2) Create and activate virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Mac/Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

### Notebook workflow
```bash
code .
```
Open `notebooks/eda.ipynb`, select `.venv` kernel, run all cells.

### Python script workflow
```bash
python src/main.py
```

---

## 📊 Current Progress
- [x] Repository setup complete
- [x] Virtual environment + dependencies configured
- [x] EDA notebook scaffold created
- [x] Initial notebook analysis committed
- [ ] Final target selection and baseline metric logging
- [ ] Feature engineering + model tuning
- [ ] Spark pipeline extension

---

## 🧠 Tech Stack
- Python
- NumPy
- Pandas
- Matplotlib / Seaborn
- Scikit-learn
- PySpark

---

## 🚀 Next Milestones
1. Finalize target variable and task type (classification/regression)
2. Add feature engineering pipeline
3. Compare multiple models (RF, XGBoost, Logistic/Linear)
4. Add Spark-based preprocessing for larger datasets
5. Export model + reproducible results report

---

## 👤 Author
**Jafar Kachhi**

**Janvi Patel**

GitHub: [givemeallyourmoney01-jpg](https://github.com/givemeallyourmoney01-jpg)

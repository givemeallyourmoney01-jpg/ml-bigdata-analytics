import joblib
import pandas as pd
from src.config import Paths

def load_model():
    if not Paths().sklearn_model_path.exists():
        raise FileNotFoundError("Model not found. Train first.")
    return joblib.load(Paths().sklearn_model_path)

def predict_one(payload: dict) -> float:
    model = load_model()
    df = pd.DataFrame([payload])
    pred = model.predict(df)[0]
    return float(pred)

from fastapi import FastAPI
from pydantic import BaseModel
from src.predict_service import predict_one

app = FastAPI(title="NYC Taxi Duration Predictor")

class PredictRequest(BaseModel):
    pickup_hour: int
    pickup_dayofweek: int
    pickup_month: int
    is_weekend: int
    is_rush_hour: int
    manhattan_distance_proxy: float
    passenger_count: int = 1
    PULocationID: int = 0
    DOLocationID: int = 0

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    pred = predict_one(req.model_dump())
    return {"predicted_trip_duration_min": pred}

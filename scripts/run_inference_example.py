from src.predict_service import predict_one

if __name__ == "__main__":
    sample = {
        "pickup_hour": 8,
        "pickup_dayofweek": 2,
        "pickup_month": 5,
        "is_weekend": 0,
        "is_rush_hour": 1,
        "manhattan_distance_proxy": 0.08,
        "passenger_count": 1,
        "PULocationID": 161,
        "DOLocationID": 236
    }
    pred = predict_one(sample)
    print(f"Predicted trip duration (min): {pred:.2f}")

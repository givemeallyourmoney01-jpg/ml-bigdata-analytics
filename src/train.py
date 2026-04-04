import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from src.config import Settings, Paths

def train_model(pdf: pd.DataFrame):
    target = Settings.target_col
    if target not in pdf.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe")

    candidate_features = [
        "pickup_hour", "pickup_dayofweek", "pickup_month", "is_weekend",
        "is_rush_hour", "manhattan_distance_proxy", "passenger_count",
        "PULocationID", "DOLocationID"
    ]
    features = [c for c in candidate_features if c in pdf.columns]

    X = pdf[features].copy()
    y = pdf[target].copy()

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    model = RandomForestRegressor(
        n_estimators=180,
        max_depth=18,
        random_state=Settings.random_state,
        n_jobs=-1
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=Settings.test_size, random_state=Settings.random_state
    )

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(mean_squared_error(y_test, preds) ** 0.5),
        "r2": float(r2_score(y_test, preds)),
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
    }

    Paths().models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, Paths().sklearn_model_path)

    return pipeline, metrics

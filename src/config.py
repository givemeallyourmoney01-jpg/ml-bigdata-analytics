from dataclasses import dataclass
from pathlib import Path

@dataclass
class Paths:
    project_root: Path = Path(__file__).resolve().parent.parent
    raw_dir: Path = project_root / "data" / "raw"
    processed_dir: Path = project_root / "data" / "processed"
    models_dir: Path = project_root / "models"
    artifacts_dir: Path = project_root / "artifacts"

    raw_csv: Path = raw_dir / "yellow_tripdata.csv"
    cleaned_parquet: Path = processed_dir / "cleaned.parquet"
    features_parquet: Path = processed_dir / "features.parquet"

    sklearn_model_path: Path = models_dir / "duration_model.joblib"
    metrics_path: Path = artifacts_dir / "metrics.json"

class Settings:
    target_col = "trip_duration_min"
    random_state = 42
    test_size = 0.2

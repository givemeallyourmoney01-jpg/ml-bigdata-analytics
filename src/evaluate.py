import json
from src.config import Paths

def save_metrics(metrics: dict):
    Paths().artifacts_dir.mkdir(parents=True, exist_ok=True)
    with open(Paths().metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

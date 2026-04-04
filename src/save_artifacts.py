from src.config import Paths

def ensure_dirs():
    p = Paths()
    p.raw_dir.mkdir(parents=True, exist_ok=True)
    p.processed_dir.mkdir(parents=True, exist_ok=True)
    p.models_dir.mkdir(parents=True, exist_ok=True)
    p.artifacts_dir.mkdir(parents=True, exist_ok=True)

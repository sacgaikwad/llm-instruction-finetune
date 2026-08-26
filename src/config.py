from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
V1_MODEL_DIR = MODEL_DIR / "v1"
V2_MODEL_DIR = MODEL_DIR / "v2"
V2_TRAINING_DATA = DATA_DIR / "v2" / "training_data_v2.json"
V2_VALIDATION_DATA = DATA_DIR / "v2" / "validation_data_v2.json"
V2_TEST_DATA = DATA_DIR / "v2" / "test_data.json"

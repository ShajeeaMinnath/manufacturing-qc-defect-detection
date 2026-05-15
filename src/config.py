# src/config.py
from pathlib import Path

# This finds the 'defect-detection-project' folder
BASE_DIR = Path(__file__).resolve().parent.parent

# This points to where the images are
DATA_ROOT = BASE_DIR / "data" / "mvtec_anomaly_detection"
CATEGORY = 'bottle'
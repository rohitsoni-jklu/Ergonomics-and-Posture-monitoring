"""Central settings for the whole project. Change values here, not inside the scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
SPLITS_CSV = ROOT / "data" / "splits.csv"
FEATURES_CSV = ROOT / "data" / "features.csv"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

# Posture classes. Each one is a folder name inside data/raw/
CLASS_NAMES = ["good", "forward_head", "slouching", "leaning"]
GOOD_CLASS = "good"

# Image / model settings
IMG_SIZE = 224          # Model 2 (CNN) input size
SEED = 42
VAL_FRACTION = 0.15     # share of PEOPLE used for validation
TEST_FRACTION = 0.15    # share of PEOPLE used for testing

# Webcam settings (must be the same for collecting data and for live use)
CAMERA_INDEX = 0
MIRROR = True           # flip frame like a mirror

# Live monitoring settings
SMOOTH_WINDOW = 10      # average the last N predictions
ALERT_SECONDS = 10      # alert when bad posture lasts this long
ALERT_COOLDOWN = 60     # seconds between two alerts

# Pose feature names (order matters, used for the CSV and Model 1)
FEATURE_NAMES = [
    "shoulder_tilt", "head_tilt", "neck_angle", "neck_vert_ratio",
    "nose_vert_ratio", "head_offset_x", "head_depth", "shoulder_width",
]

for _d in (RAW_DIR, MODELS_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

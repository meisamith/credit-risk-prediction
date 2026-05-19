"""
Central configuration. One source of truth for paths and constants.

Why this file exists:
In a real project, hardcoding "data/raw/loans.csv" in 5 different places
is how bugs are born. Change the path once, here, and everything follows.
"""
from pathlib import Path

# Project root = the folder this repo lives in (resolved automatically)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- Data paths ---
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RAW_LOANS_CSV = DATA_RAW / "loans.csv"
CLEAN_LOANS_PARQUET = DATA_PROCESSED / "loans_clean.parquet"

# --- Model paths ---
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "credit_risk_model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"

# --- Reports ---
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# --- Modeling constants ---
TARGET_COL = "default"          # 1 = defaulted, 0 = repaid
RANDOM_STATE = 42               # reproducibility: same split every run
TEST_SIZE = 0.20                # 80/20 train-test split

# Risk band thresholds (probability of default)
# These map a model probability to the High/Medium/Low your demo shows.
RISK_BANDS = {
    "Low": 0.15,        # P(default) < 0.15  -> Low risk
    "Medium": 0.40,     # 0.15 <= P < 0.40   -> Medium risk
    # >= 0.40 -> High risk
}

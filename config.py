"""
Global configuration for the Temporal Fusion Transformer FX project.

All project-wide constants and paths should live here.
"""

from pathlib import Path
import torch

# =====================================================
# DEFAULT TIME RANGE
# (used by ingest_macro_data.py)
# =====================================================

START_DATE = "2024-01-01"

END_DATE = "2026-06-01"

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

MACRO_DIR = PROJECT_ROOT / "macro_data"
MACRO_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_DIR = PROJECT_ROOT / "features"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

FX_DIR = PROJECT_ROOT / "data"
FX_DIR.mkdir(parents=True, exist_ok=True)
FX_CACHE_DIRS = [
    FX_DIR,
    PROJECT_ROOT.parent / "Neural-Spot-FX-Alpha-Model" / "data"
]

FX_SOURCE_DIRS = FX_CACHE_DIRS

# ============================================================
# LOCATION OF RAW FX FILES
# ============================================================

# Existing Neural Spot FX project
FX_RAW_DIR = (
        PROJECT_ROOT.parent
        / "Neural-Spot-FX-Alpha-Model"
        / "data"
)

# Local cached consolidated FX file
FX_CACHE = DATA_DIR / "fx.parquet"

# Resampled 1-minute bars
FX_1MIN_CACHE = DATA_DIR / "fx_1min.parquet"

# ============================================================
# MACRO DATA
# ============================================================

MACRO_PATTERN = "macro_*.parquet"

# ============================================================
# TRAINING DATA
# ============================================================

FEATURE_DATASET = FEATURE_DIR / "training_features.parquet"

# ============================================================
# MODEL FILES
# ============================================================

MODEL_FILE = MODEL_DIR / "tft_alpha.pt"

SCALER_FILE = MODEL_DIR / "feature_scaler.pkl"

# ============================================================
# TRAINING PARAMETERS
# ============================================================

SEQUENCE_LENGTH = 120

PREDICTION_HORIZON = 15

TRAIN_BATCH_SIZE = 256

VALID_BATCH_SIZE = 1024

NUM_WORKERS = 4

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

EPOCHS = 20

# ============================================================
# DATA REDUCTION
# ============================================================

# Convert ticks to 1-minute bars
BAR_FREQUENCY = "1min"

# Maximum number of rows loaded simultaneously
MAX_ROWS_IN_MEMORY = 500_000

# ============================================================
# MODEL
# ============================================================

D_MODEL = 64

NUM_HEADS = 4

NUM_ENCODER_LAYERS = 3

DROPOUT = 0.1

# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# ============================================================
# YAHOO FINANCE
# ============================================================

YAHOO_TICKERS = {
    "spx": "^GSPC",
    "eustoxx": "^STOXX50E",
    "vix": "^VIX",
    "dxy": "DX-Y.NYB",
}

# ============================================================
# FRED
# ============================================================

FRED_SERIES = {
    "us2y": "DGS2",
    "yield_curve": "T10Y2Y",
}

# ============================================================
# SIGNAL THRESHOLDS
# ============================================================

LONG_THRESHOLD = 1.0

SHORT_THRESHOLD = -1.0

TARGET_PORTFOLIO_VOL = 0.10

# ============================================================
# RANDOMNESS
# ============================================================

RANDOM_SEED = 42

# ============================================================
# HELPERS
# ============================================================

def latest_macro_file():
    """
    Returns the newest macro parquet file.

    Returns
    -------
    pathlib.Path

    Raises
    ------
    FileNotFoundError
    """

    files = list(MACRO_DIR.glob(MACRO_PATTERN))

    if not files:
        raise FileNotFoundError(
            f"No macro files found in {MACRO_DIR}"
        )

    return max(
        files,
        key=lambda f: f.stat().st_mtime,
    )


def ensure_directories():
    """
    Ensure every project directory exists.
    """

    for directory in (
            DATA_DIR,
            MACRO_DIR,
            MODEL_DIR,
            FEATURE_DIR,
            LOG_DIR,
            CACHE_DIR,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# Ensure directories are present on import
ensure_directories()
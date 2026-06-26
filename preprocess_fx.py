"""
preprocess_fx.py

Memory-safe preprocessing pipeline for FX + macro TFT training.

Key goals:
- Avoid loading full 40M+ rows into memory at once
- Downsample FX ticks safely
- Align macro data using merge_asof
- Produce compact training dataset
"""

from pathlib import Path
import numpy as np
import pandas as pd

from config import DATA_DIR, MACRO_DIR

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

FX_PATH = DATA_DIR / "fx.parquet"


# You can tune this to control memory usage
DEFAULT_SAMPLE_FRACTION = 0.02   # 2% of ticks
DEFAULT_SEQ_LEN = 120


# ---------------------------------------------------------
# SAFE LOADING
# ---------------------------------------------------------

def load_fx(sample_fraction: float = DEFAULT_SAMPLE_FRACTION) -> pd.DataFrame:
    """
    Loads FX parquet safely with optional downsampling.
    """

    if not FX_PATH.exists():
        raise FileNotFoundError(f"Missing {FX_PATH}")

    print(f"[FX] Loading {FX_PATH}")

    df = pd.read_parquet(FX_PATH)

    # Ensure timestamp is usable for merge_asof
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    df = df.dropna(subset=["timestamp"])

    df = df.sort_values("timestamp")

    # -----------------------------------------------------
    # Downsampling to prevent memory explosion
    # -----------------------------------------------------

    if sample_fraction < 1.0:

        print(f"[FX] Downsampling to {sample_fraction:.2%}")

        df = df.sample(
            frac=sample_fraction,
            random_state=42,
        ).sort_values("timestamp")

    return df


# ---------------------------------------------------------
# LOAD MACRO
# ---------------------------------------------------------

def load_macro() -> pd.DataFrame:
    """
    Loads latest macro parquet.
    """

    files = sorted(MACRO_DIR.glob("macro_*.parquet"))

    if not files:
        raise FileNotFoundError(
            f"No macro files in {MACRO_DIR}"
        )

    macro_file = files[-1]

    print(f"[MACRO] Loading {macro_file}")

    macro = pd.read_parquet(macro_file)

    macro["timestamp"] = pd.to_datetime(
        macro["timestamp"],
        utc=True,
        errors="coerce",
    )

    macro = macro.dropna(subset=["timestamp"])

    macro = macro.sort_values("timestamp")

    return macro


# ---------------------------------------------------------
# ALIGNMENT PREP (no merge yet)
# ---------------------------------------------------------

def prepare_datasets(sample_fraction: float = DEFAULT_SAMPLE_FRACTION):
    """
    Loads FX + macro datasets separately.
    Merge happens in Part 2.
    """

    fx = load_fx(sample_fraction)
    macro = load_macro()

    print(
        f"[PREP] FX rows: {len(fx):,} | Macro rows: {len(macro):,}"
    )

    return fx, macro

# ---------------------------------------------------------
# SAFE MERGE (FX + MACRO)
# ---------------------------------------------------------

def align_fx_macro(
        fx: pd.DataFrame,
        macro: pd.DataFrame,
) -> pd.DataFrame:
    """
    Time-align FX and macro using merge_asof safely.

    Fixes:
    - NaN merge keys crash
    - timezone mismatch
    - unsorted input issues
    """

    print("[ALIGN] Preparing merge_asof")

    # -----------------------------------------------------
    # Clean timestamps (CRITICAL FIX)
    # -----------------------------------------------------

    fx = fx.dropna(subset=["timestamp"])
    macro = macro.dropna(subset=["timestamp"])

    fx = fx.sort_values("timestamp")
    macro = macro.sort_values("timestamp")

    # Ensure same dtype + timezone
    fx["timestamp"] = pd.to_datetime(
        fx["timestamp"],
        utc=True,
        errors="coerce",
    )

    macro["timestamp"] = pd.to_datetime(
        macro["timestamp"],
        utc=True,
        errors="coerce",
    )

    fx = fx.dropna(subset=["timestamp"])
    macro = macro.dropna(subset=["timestamp"])

    # -----------------------------------------------------
    # Reduce macro columns (important for memory)
    # -----------------------------------------------------

    macro_cols = [
        "timestamp",
        "spx",
        "eustoxx",
        "vix",
        "dxy",
        "ust2y",
        "yield_curve",
    ]

    macro = macro[
        [c for c in macro_cols if c in macro.columns]
    ]

    # -----------------------------------------------------
    # Merge (safe)
    # -----------------------------------------------------

    merged = pd.merge_asof(
        fx.sort_values("timestamp"),
        macro.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
        allow_exact_matches=True,
    )

    print(
        f"[ALIGN] Merged shape: {merged.shape}"
    )

    return merged


# ---------------------------------------------------------
# FEATURE BUILDING (LIGHTWEIGHT VERSION)
# ---------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lightweight feature engineering for TFT training.
    """

    print("[FEATURES] Creating features")

    df = df.copy()

    # -----------------------------------------------------
    # FX mid + returns
    # -----------------------------------------------------

    if "bid" in df.columns and "ask" in df.columns:
        df["mid"] = (df["bid"] + df["ask"]) / 2
    elif "close" in df.columns:
        df["mid"] = df["close"]
    else:
        raise ValueError("No FX price columns found")

    df["fx_ret"] = df["mid"].pct_change()

    # -----------------------------------------------------
    # Macro returns (safe guards added)
    # -----------------------------------------------------

    if "spx" in df.columns:
        df["spx_ret"] = df["spx"].pct_change()

    if "eustoxx" in df.columns:
        df["eustoxx_ret"] = df["eustoxx"].pct_change()

    if "vix" in df.columns:
        df["vix_change"] = df["vix"].pct_change()

    if "dxy" in df.columns:
        df["dxy_ret"] = df["dxy"].pct_change()

    # -----------------------------------------------------
    # Volatility (rolling, safe)
    # -----------------------------------------------------

    df["fx_vol_20"] = (
        df["fx_ret"]
        .rolling(20)
        .std()
    )

    df["fx_vol_100"] = (
        df["fx_ret"]
        .rolling(100)
        .std()
    )

    # -----------------------------------------------------
    # Target (15-min forward return proxy)
    # -----------------------------------------------------

    df["target"] = (
            df["mid"].shift(-15) / df["mid"] - 1
    )

    # Vol-adjusted target (stabilises training)
    df["target_vol"] = (
            df["target"] / (df["fx_vol_20"] + 1e-6)
    )

    # -----------------------------------------------------
    # Cleanup
    # -----------------------------------------------------

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    print(
        f"[FEATURES] Final dataset size: {len(df):,}"
    )

    return df

# ---------------------------------------------------------
# FINAL DATASET BUILDER
# ---------------------------------------------------------

def build_training_dataset(
        sample_fraction: float = DEFAULT_SAMPLE_FRACTION,
        max_rows: int = 500_000,
):
    """
    Full pipeline:

    FX + macro load
        → align
        → feature engineering
        → downsample/truncate
        → return training-ready dataframe

    This prevents memory crashes.
    """

    fx, macro = prepare_datasets(
        sample_fraction=sample_fraction
    )

    df = align_fx_macro(fx, macro)

    df = build_features(df)

    # -----------------------------------------------------
    # HARD MEMORY LIMIT (CRITICAL FIX FOR SIGKILL)
    # -----------------------------------------------------

    if len(df) > max_rows:

        print(
            f"[TRIM] Reducing dataset from {len(df):,} → {max_rows:,}"
        )

        df = df.iloc[-max_rows:].reset_index(drop=True)

    # -----------------------------------------------------
    # FINAL CLEANUP
    # -----------------------------------------------------

    df = df.dropna().reset_index(drop=True)

    print(
        f"[DATASET] Final training size: {len(df):,}"
    )

    return df


# ---------------------------------------------------------
# OPTIONAL: SAVE TRAINING DATASET
# ---------------------------------------------------------

def save_training_dataset(
        out_path: Path = DATA_DIR / "training.parquet",
        **kwargs,
):
    """
    Builds dataset and saves it for TFT training reuse.
    """

    df = build_training_dataset(**kwargs)

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        out_path,
        index=False,
    )

    print(
        f"[SAVED] Training dataset → {out_path}"
    )

    return out_path


# ---------------------------------------------------------
# MAIN ENTRY (standalone execution)
# ---------------------------------------------------------

if __name__ == "__main__":

    df = build_training_dataset()

    print(df.head())
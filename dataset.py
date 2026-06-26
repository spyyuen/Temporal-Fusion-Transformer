from pathlib import Path
import numpy as np
import pandas as pd
import torch


# =====================================================
# SAFE PARQUET LOADER (chunked / memory-safe)
# =====================================================

def load_parquet_safe(path: str, columns=None):
    """
    Loads parquet safely without exploding memory.
    """
    df = pd.read_parquet(path, columns=columns)

    if "timestamp" not in df.columns:
        raise ValueError(f"{path} missing timestamp column")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp")

    return df


# =====================================================
# MERGE FX + MACRO (SAFE VERSION)
# =====================================================

def merge_fx_macro(fx_path: str, macro_path: str):
    """
    Memory-safe merge using merge_asof (fixes null key crash).
    """

    fx = load_parquet_safe(fx_path)
    macro = load_parquet_safe(macro_path)

    # remove bad timestamps
    fx = fx.dropna(subset=["timestamp"])
    macro = macro.dropna(subset=["timestamp"])

    fx = fx.sort_values("timestamp")
    macro = macro.sort_values("timestamp")

    # IMPORTANT: remove duplicates (prevents merge crash)
    fx = fx.drop_duplicates("timestamp")
    macro = macro.drop_duplicates("timestamp")

    # ensure numeric only for merge stability
    macro = macro.replace([np.inf, -np.inf], np.nan).ffill()
    fx = fx.replace([np.inf, -np.inf], np.nan).ffill()

    # SAFE MERGE
    df = pd.merge_asof(
        fx,
        macro,
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta("2D")  # prevents huge backfill explosions.  If the nearest previous row is more than 2 days earlier, the result will contain NaN for the columns from right.
    )

    df = df.dropna()

    return df


# =====================================================
# MEMORY SAFE FEATURE BUILDER (STREAMING STYLE)
# =====================================================

def build_training_frame(df: pd.DataFrame):
    """
    Reduces memory usage before TFT training.
    Keeps only required columns + downcasts.
    """

    df = df.copy()

    # downcast floats
    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].astype(np.float32)

    # remove unused columns aggressively
    keep_cols = [c for c in df.columns if c != "index"]
    df = df[keep_cols]

    return df


# =====================================================
# SLIDING WINDOW DATASET (NO FULL TENSOR IN RAM)
# =====================================================

class StreamingDataset(torch.utils.data.Dataset):
    """
    Does NOT materialize all sequences in memory.
    Fixes SIGKILL crash issue.
    """

    def __init__(self, df, feature_cols, target_col, seq_len=120):

        self.seq_len = seq_len

        self.features = df[feature_cols].values.astype(np.float32)
        self.targets = df[target_col].values.astype(np.float32)

        self.length = len(df)

    def __len__(self):
        return max(0, self.length - self.seq_len)

    def __getitem__(self, idx):

        x = self.features[idx:idx + self.seq_len]
        y = self.targets[idx + self.seq_len]

        return torch.tensor(x), torch.tensor(y)


# =====================================================
# FEATURE COLUMN EXTRACTOR
# =====================================================

def get_feature_columns(df: pd.DataFrame):
    """
    Automatically selects numeric features.
    """

    exclude = {"timestamp", "target"}

    cols = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]

    return cols


# =====================================================
# MAIN DATA BUILDER
# =====================================================

def build_dataset(fx_path: str, macro_path: str):
    """
    Full pipeline:
    FX + macro merge -> clean -> feature select -> dataset
    """

    df = merge_fx_macro(fx_path, macro_path)

    df = build_training_frame(df)

    feature_cols = get_feature_columns(df)

    if "target" not in df.columns:
        raise ValueError("Missing target column (run feature engineering first)")

    dataset = StreamingDataset(
        df,
        feature_cols=feature_cols,
        target_col="target",
        seq_len=120
    )

    return dataset, feature_cols
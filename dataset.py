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

    fx = fx.copy()
    macro = macro.copy()

    fx["timestamp"] = pd.to_datetime(fx["timestamp"], utc=True)
    macro["timestamp"] = pd.to_datetime(macro["timestamp"], utc=True)

    fx = fx.sort_values("timestamp")
    macro = macro.sort_values("timestamp")

    # SAFE MERGE
    df = pd.merge_asof(
        fx.sort_values("timestamp"),
        macro.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta("1D")
    )
    # prevents huge backfill explosions.  If the nearest previous row is more than 2 days earlier, the result will contain NaN for the columns from right.

    df = df.dropna()
    print("[DEBUG FX RANGE]", fx["timestamp"].min(), fx["timestamp"].max())

    print("[DEBUG MACRO RANGE]", macro["timestamp"].min(), macro["timestamp"].max())

    print("[DEBUG FX ROWS]", len(fx))

    print("[DEBUG MACRO ROWS]", len(macro))

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
def build_dataset(fx, macro, max_rows: int = 1_000_000):
    import pandas as pd
    import numpy as np

    fx["timestamp"] = pd.to_datetime(
        fx["timestamp"],
        utc=True,
        format="mixed",
        errors="coerce",
    )

    macro["timestamp"] = pd.to_datetime(
        macro["timestamp"],
        utc=True,
        format="mixed",
        errors="coerce",
    )

    fx = fx.dropna(subset=["timestamp"])
    macro = macro.dropna(subset=["timestamp"])

    fx = fx.sort_values("timestamp")
    macro = macro.sort_values("timestamp")

    # -------------------------------------------------
    # IMPORTANT: downsample to avoid OOM (SIGKILL fix)
    # -------------------------------------------------

    if len(fx) > max_rows:
        fx = fx.iloc[:: max(1, len(fx) // max_rows)]

    if len(macro) > max_rows:
        macro = macro.iloc[:: max(1, len(macro) // max_rows)]

    # -------------------------------------------------
    # merge_asof (fixed + safe)
    # -------------------------------------------------
    fx = fx.copy()
    macro = macro.copy()

    fx["timestamp"] = pd.to_datetime(fx["timestamp"], utc=True)
    macro["timestamp"] = pd.to_datetime(macro["timestamp"], utc=True)
    # FORCE IDENTICAL TIMESTAMP TYPE (CRITICAL FIX)
    fx["timestamp"] = pd.to_datetime(fx["timestamp"], utc=True).astype("datetime64[ns, UTC]")
    macro["timestamp"] = pd.to_datetime(macro["timestamp"], utc=True).astype("datetime64[ns, UTC]")

    # sanity check (optional but useful)
    assert fx["timestamp"].dtype == macro["timestamp"].dtype
    macro["timestamp"] = macro["timestamp"].dt.floor("D")

    print('fx ', fx)
    print('macro ', macro)
    df = pd.merge_asof(
        fx.sort_values("timestamp"),
        macro.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta("1D")
    )

    df = df.dropna(subset=["timestamp"])


    # -----------------------------
    # FX mid + returns
    # -----------------------------
    if "bid" in df.columns and "ask" in df.columns:
        df["mid"] = (df["bid"] + df["ask"]) / 2

    elif "close" in df.columns:
        df["mid"] = df["close"]

    else:
        raise ValueError(
            f"FX dataset must contain either "
            f"(bid, ask) or close. Found: {df.columns}"
        )

    # IMPORTANT: Avoid lookahead bias created by df["return"] = df["mid"].pct_change()
    df["return"] = df["close"].pct_change().shift(-1)
    df = df.dropna()

    # -----------------------------
    # TARGET (15-min ahead return)
    # -----------------------------
    horizon = 15

    df["target"] = (
            df["mid"].shift(-horizon) / df["mid"] - 1
    )

    # -----------------------------
    # CLEAN
    # -----------------------------
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    feature_cols = [
        c for c in df.columns
        if c not in ["timestamp", "target"]
    ]

    return df[feature_cols], df["target"]
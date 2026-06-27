from dataset import build_dataset
from model import train_model, generate_signals
import pandas as pd


# =====================================================
# VALIDATION (prevents KeyError crashes like bund2y)
# =====================================================

REQUIRED_MACRO_COLUMNS = [
    "spx",
    "eustoxx",
    "vix",
    "dxy",
    "us2y",
    "yield_curve"
]


def validate_dataset(df: pd.DataFrame):

    missing = [c for c in REQUIRED_MACRO_COLUMNS if c not in df.columns]

    if missing:
        raise ValueError(
            f"""
Missing required macro columns: {missing}

This usually means:
1. FRED download failed
2. macro dataset is stale
3. merge failed silently
"""
        )


# =====================================================
# MAIN TRAINING PIPELINE
# =====================================================

def run_training(fx_path, macro_path, seq_len=120, epochs=10):

    import pandas as pd
    from dataset import build_dataset

    fx = pd.read_parquet(fx_path)
    macro = pd.read_parquet(macro_path)

    print("[RUN] Building dataset...")

    X, y = build_dataset(fx, macro)

    print(f"[RUN] Dataset size: {len(X):,} rows")

    # optional safety cap (VERY IMPORTANT for your RAM issue)
    max_rows = 1_000_000
    if len(X) > max_rows:
        print(f"[WARN] Downsampling to {max_rows}")
        X = X.tail(max_rows)
        y = y.tail(max_rows)

    model = train_model(X, y, seq_len=seq_len, epochs=epochs)

    return model

# =====================================================
# OPTIONAL ENTRYPOINT
# =====================================================

if __name__ == "__main__":

    model, result = run_training(
        fx_path="data/fx.parquet",
        macro_path="macro_data/macro_2026-01-01_2026-06-01.parquet",
        epochs=5
    )

    print(result.tail())
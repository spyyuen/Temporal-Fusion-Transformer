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

def run_training(
        fx_path: str,
        macro_path: str,
        seq_len: int = 120,
        batch_size: int = 256,
        epochs: int = 5,
        device: str = None
):

    print("[RUN] Loading dataset...")

    dataset, feature_cols = build_dataset(
        fx_path=fx_path,
        macro_path=macro_path
    )

    print(f"[RUN] Features: {len(feature_cols)}")

    # =================================================
    # TRAIN MODEL
    # =================================================

    print("[RUN] Training model...")

    model = train_model(
        dataset=dataset,
        input_dim=len(feature_cols),
        batch_size=batch_size,
        epochs=epochs,
        device=device
    )

    # =================================================
    # SIGNAL GENERATION
    # =================================================

    print("[RUN] Generating signals...")

    signals = generate_signals(
        model,
        dataset
    )

    result = pd.DataFrame({
        "signal": signals
    })

    print("[RUN] Done")

    return model, result


# =====================================================
# OPTIONAL ENTRYPOINT
# =====================================================

if __name__ == "__main__":

    model, result = run_training(
        fx_path="data/fx.parquet",
        macro_path="macro_data/macro_2024-01-01_2026-06-01.parquet",
        epochs=5
    )

    print(result.tail())
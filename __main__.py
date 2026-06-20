import argparse
import subprocess
from pathlib import Path
from pathlib import Path
from ingest_macro_data import build_macro_dataset as ingest_macro
from temporal_fusion_transformer import run as train_tft

# =====================================================
# PATHS / CACHE
# =====================================================

MACRO_DIR = Path("macro_data")
FX_DIR = Path("data")

DEFAULT_MACRO_FILE = None


# =====================================================
# CACHE CHECKS
# =====================================================

def macro_cache_exists():
    return any(MACRO_DIR.glob("macro_*.parquet"))


def fx_cache_exists():
    return FX_DIR.exists() and any(FX_DIR.glob("*.parquet"))



# =====================================================
# OPTIONAL CLEAN BACKFILL
# =====================================================

def wipe_cache():
    """
    Deletes cached macro + FX data.
    """

    print("[BACKFILL] Clearing cache...")

    for path in MACRO_DIR.glob("*.parquet"):
        path.unlink()

    for path in FX_DIR.glob("*.parquet"):
        path.unlink()


# =====================================================
# MAIN
# =====================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Force full regeneration of all cached datasets"
    )

    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip macro ingestion step"
    )

    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip model training step"
    )

    args = parser.parse_args()

    # -------------------------------------------------
    # BACKFILL MODE
    # -------------------------------------------------

    if args.backfill:
        wipe_cache()

    # -------------------------------------------------
    # INGEST STEP
    # -------------------------------------------------

    macro_path = (
        "macro_data/macro_2024-01-01_2026-06-01.parquet"
    )

    if not args.skip_ingest:

        if args.backfill or not Path(macro_path).exists():

            print(
                "[INGEST] Building macro dataset..."
            )

            build_macro_dataset(
                start="2024-01-01",
                end="2026-06-01"
            )

        else:

            print(
                f"[CACHE HIT] {macro_path}"
            )

    # -------------------------------------------------
    # TRAINING STEP
    # -------------------------------------------------

    if not args.skip_train:

        print(
            "[TRAIN] Running TFT model..."
        )

        model, result = train_tft(
            fx_path="data/fx.parquet",
            macro_path=macro_path,
            seq_len=120,
            epochs=20
        )

        print(result.tail())

# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":
    main()
import argparse
from pathlib import Path

from ingest_macro_data import build_macro_dataset, build_fx_dataset
from temporal_fusion_transformer import run as train_tft

# =====================================================
# PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent
MACRO_DIR = PROJECT_ROOT / "macro_data"
FX_DIR = PROJECT_ROOT / "data"


# =====================================================
# CACHE HELPERS
# =====================================================

def macro_exists():
    return any(MACRO_DIR.glob("macro_*.parquet"))


def fx_exists():
    return (FX_DIR / "fx.parquet").exists()


# =====================================================
# CLEAN CACHE
# =====================================================

def wipe_cache():
    print("[BACKFILL] Clearing cache...")

    for p in MACRO_DIR.glob("*.parquet"):
        p.unlink()

    for p in FX_DIR.glob("*.parquet"):
        p.unlink()


# =====================================================
# MAIN
# =====================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--backfill", action="store_true")
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--skip-train", action="store_true")

    args = parser.parse_args()

    if args.backfill:
        wipe_cache()

    # =================================================
    # INGEST STEP (SOURCE OF TRUTH)
    # =================================================

    macro_path = None
    fx_path = None

    if not args.skip_ingest:

        if args.backfill or not macro_exists():
            print("[INGEST] Building macro dataset...")
            macro_path = build_macro_dataset(
                start="2024-01-01",
                end="2026-06-01"
            )
        else:
            macro_path = max(
                MACRO_DIR.glob("macro_*.parquet"),
                key=lambda x: x.stat().st_mtime
            )
            print(f"[CACHE HIT] {macro_path}")

        if args.backfill or not fx_exists():
            fx_path = build_fx_dataset()
        else:
            fx_path = FX_DIR / "fx.parquet"
            print(f"[CACHE HIT] {fx_path}")

    # =================================================
    # TRAIN STEP
    # =================================================

    if not args.skip_train:

        print("[TRAIN] Running TFT model...")

        model, result = train_tft(
            fx_path=str(fx_path),
            macro_path=str(macro_path),
            seq_len=120,
            epochs=20
        )

        print(result.tail())


# =====================================================
# ENTRY
# =====================================================

if __name__ == "__main__":
    main()
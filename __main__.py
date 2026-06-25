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


# Ensure directories always exist (fixes your crash chain)
MACRO_DIR.mkdir(parents=True, exist_ok=True)
FX_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# CACHE RESOLUTION
# =====================================================

def find_latest_parquet(directory: Path, pattern: str):

    files = list(directory.glob(pattern))

    if not files:
        return None

    return max(files, key=lambda f: f.stat().st_mtime)


def macro_cache():
    return find_latest_parquet(MACRO_DIR, "macro_*.parquet")


def fx_cache():
    return find_latest_parquet(FX_DIR, "fx*.parquet")


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
# MAIN PIPELINE
# =====================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--backfill", action="store_true")
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--skip-train", action="store_true")

    args = parser.parse_args()

    # -------------------------------------------------
    # BACKFILL MODE
    # -------------------------------------------------

    if args.backfill:
        wipe_cache()

    # -------------------------------------------------
    # INGEST STEP
    # -------------------------------------------------

    if not args.skip_ingest:

        macro_file = macro_cache()

        if args.backfill or macro_file is None:

            print("[INGEST] Building macro dataset...")

            macro_file = build_macro_dataset(
                start="2024-01-01",
                end="2026-06-01",
            )

        else:

            print(f"[CACHE HIT] {macro_file}")

        fx_file = fx_cache()

        if args.backfill or fx_file is None:

            fx_file = build_fx_dataset()

        else:

            print(f"[CACHE HIT] {fx_file}")

    else:

        macro_file = macro_cache()
        fx_file = fx_cache()

    # -------------------------------------------------
    # VALIDATION SAFETY
    # -------------------------------------------------

    if macro_file is None:
        raise FileNotFoundError(
            "Macro dataset missing. Run ingestion first."
        )

    if fx_file is None:
        raise FileNotFoundError(
            "FX dataset missing. Run ingestion first."
        )

    # -------------------------------------------------
    # TRAINING STEP
    # -------------------------------------------------

    if not args.skip_train:

        print("[TRAIN] Running TFT model...")

        model, result = train_tft(
            fx_path=str(fx_file),
            macro_path=str(macro_file),
            seq_len=120,
            epochs=20,
        )

        print("[TRAIN] Done")
        print(result.tail())


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":
    main()
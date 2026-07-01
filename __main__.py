import argparse
from pathlib import Path

from ingest_macro_data import build_macro_dataset, build_fx_dataset
from temporal_fusion_transformer import train_tft
from datetime import datetime, timedelta
from backtest import backtest_pipeline
from report import generate_html_report, metrics_html


# -----------------------------
# DEFAULT CONFIG
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MACRO_DIR = PROJECT_ROOT / "macro_data"
FX_DIR = PROJECT_ROOT / "data"


def find_latest_parquet(directory: Path, pattern: str):
    files = list(directory.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda f: f.stat().st_mtime)


# -----------------------------
# MAIN
# -----------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--backfill", action="store_true")
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--skip-train", action="store_true")

    # NEW: date arguments
    parser.add_argument("--start", type=str, required=True, help="Start date of format YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=(datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"), help="End date in format YYYY-MM-DD (default: yesterday)")

    args = parser.parse_args()

    start = args.start
    end = args.end

    # -----------------------------
    # INGEST
    # -----------------------------
    macro_file = MACRO_DIR / f"macro_{start}_{end}.parquet"
    fx_file = FX_DIR / "fx.parquet"

    if not args.skip_ingest:

        if args.backfill or not macro_file.exists():
            print("[INGEST] Building macro dataset...")
            build_macro_dataset(start=start, end=end, force=args.backfill)
        else:
            print(f"[CACHE HIT] {macro_file}")

        if args.backfill or not fx_file.exists():
            print("[INGEST] Building FX dataset...")
            build_fx_dataset(start=start, end=end, force=args.backfill)
        else:
            print(f"[CACHE HIT] {fx_file}")

    # -----------------------------
    # TRAIN
    # -----------------------------
    if not args.skip_train:
        print("[TRAIN] Running TFT model...")

        model, result = train_tft(
            fx_path=str(fx_file),
            macro_path=str(macro_file),
            seq_len=120,
            epochs=20
        )


    # -----------------------------
    # Backtest
    # -----------------------------
    preds = result["predictions"]
    future_returns = result["returns"]
    bt_df, metrics = backtest_pipeline(preds, future_returns)


    generate_html_report(bt_df, metrics, output="backtest_report.html")

    # Metrics table HTML
    with open("metrics.html", "w") as f:
        f.write(metrics_html(metrics))

if __name__ == "__main__":
    main()
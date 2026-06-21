from pathlib import Path
import numpy as np
import pandas as pd
import requests
import yfinance as yf
import time
import os

# =====================================================
# CONFIG
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "macro_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

FX_LOCAL_DIR = PROJECT_ROOT / "data"
FX_LOCAL_DIR.mkdir(parents=True, exist_ok=True)

# fallback external repo (your old dataset location)
FX_EXTERNAL_DIR = (
        PROJECT_ROOT.parent
        / "Neural-Spot-FX-Alpha-Model"
        / "data"
)

YF_TICKERS = {
    "spx": "^GSPC",
    "eustoxx": "^STOXX50E",
    "vix": "^VIX",
    "dxy": "DX-Y.NYB",
}

FRED_SERIES = {
    "us2y": "DGS2",
    "yield_curve": "T10Y2Y",
}


# =====================================================
# YAHOO DOWNLOAD
# =====================================================

def download_yahoo(ticker: str, start: str, end: str) -> pd.DataFrame:

    print(f"Downloading {ticker}")

    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    ts_col = "Datetime" if "Datetime" in df.columns else "Date"

    df = df.rename(columns={
        ts_col: "timestamp",
        "Close": "close"
    })

    return df[["timestamp", "close"]]


# =====================================================
# FRED DOWNLOAD
# =====================================================

def download_fred(series_id: str, start: str, end: str) -> pd.DataFrame:

    print(f"Downloading FRED {series_id}")

    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise ValueError("Missing FRED_API_KEY environment variable")

    url = "https://api.stlouisfed.org/fred/series/observations"

    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start,
        "observation_end": end,
    }

    last_error = None

    for attempt in range(5):

        try:
            r = requests.get(url, params=params, timeout=120)
            r.raise_for_status()

            data = r.json()
            obs = data["observations"]

            df = pd.DataFrame(obs)
            if df.empty:
                raise ValueError(f"No data for {series_id}")

            df = df[["date", "value"]]
            df.columns = ["timestamp", "value"]

            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")

            return df

        except Exception as e:
            last_error = e
            print(f"Retry {attempt+1}/5 failed: {e}")
            time.sleep(5)

    raise RuntimeError(f"FRED download failed: {series_id}: {last_error}")


# =====================================================
# MACRO DATASET
# =====================================================

def build_macro_dataset(start: str, end: str) -> str:

    all_dfs = []

    # -------------------------
    # Yahoo
    # -------------------------
    for name, ticker in YF_TICKERS.items():

        df = download_yahoo(ticker, start, end)
        df = df.rename(columns={"close": name})
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.dropna(subset=["timestamp"])

        all_dfs.append(df.set_index("timestamp"))

    # -------------------------
    # FRED
    # -------------------------
    for name, series in FRED_SERIES.items():

        df = download_fred(series, start, end)
        df = df.rename(columns={"value": name})
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.dropna(subset=["timestamp"])

        all_dfs.append(df.set_index("timestamp"))

    # -------------------------
    # MERGE
    # -------------------------
    print("Combining datasets...")

    merged = (
        pd.concat(all_dfs, axis=1, sort=True)
        .sort_index()
        .ffill()
        .reset_index()
    )

    merged = merged.sort_values("timestamp")

    # -------------------------
    # FEATURES
    # -------------------------
    print("Creating macro features...")

    merged["spx_ret"] = merged["spx"].pct_change()
    merged["eustoxx_ret"] = merged["eustoxx"].pct_change()
    merged["equity_relative"] = merged["eustoxx_ret"] - merged["spx_ret"]

    merged["vix_change"] = merged["vix"].pct_change()
    merged["dxy_ret"] = merged["dxy"].pct_change()

    merged["yield_spread"] = merged["yield_curve"]

    merged["risk_regime"] = (
            merged["vix"] > merged["vix"].rolling(50).mean()
    ).astype(int)

    merged["spx_vix_corr"] = (
        merged["spx_ret"].rolling(20).corr(merged["vix_change"])
    )

    merged["spx_momentum_20"] = merged["spx"].pct_change(20)
    merged["eustoxx_momentum_20"] = merged["eustoxx"].pct_change(20)
    merged["dxy_momentum_20"] = merged["dxy"].pct_change(20)

    merged["vix_zscore"] = (
            (merged["vix"] - merged["vix"].rolling(100).mean())
            / merged["vix"].rolling(100).std()
    )

    merged = merged.replace([np.inf, -np.inf], np.nan).ffill().dropna()

    # -------------------------
    # SAVE
    # -------------------------
    outfile = DATA_DIR / f"macro_{start}_{end}.parquet"

    merged.to_parquet(outfile, index=False)

    print(f"[MACRO CREATED] {outfile}")
    print(f"Rows: {len(merged):,}")

    return str(outfile)


def _normalize_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"])
    return df

# =====================================================
# FX DATASET (ROBUST + CROSS PROJECT)
# =====================================================

def build_fx_dataset() -> str:

    print("Building FX dataset...")

    search_dirs = [
        FX_LOCAL_DIR,
        FX_EXTERNAL_DIR
    ]

    files = []

    for d in search_dirs:

        if not d.exists():
            continue

        matches = list(d.glob("EURUSD_*.parquet"))

        if matches:
            print(f"Found {len(matches)} files in {d}")
            files.extend(matches)

    if not files:
        raise RuntimeError(
            "No EURUSD parquet files found in any known directory"
        )

    dfs = []

    for f in sorted(files):

        print(f"Loading {f}")
        dfs.append(
            _normalize_timestamp(pd.read_parquet(file))
        )

    fx = (
        pd.concat(dfs, ignore_index=True)
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
    )

    out = FX_LOCAL_DIR / "fx.parquet"

    fx.to_parquet(out, index=False)

    print(f"[FX CACHE CREATED] {out}")
    print(f"Rows: {len(fx):,}")

    return str(out)


# =====================================================
# ENTRY
# =====================================================

if __name__ == "__main__":

    build_macro_dataset(
        start="2024-01-01",
        end="2026-06-01"
    )
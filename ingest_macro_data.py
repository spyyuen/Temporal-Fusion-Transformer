"""
ingest_macro_data.py

Downloads and caches:

    • SPX
    • EuroStoxx
    • VIX
    • DXY
    • US 2Y Treasury
    • Yield Curve

Also builds a single FX cache by combining all
EURUSD parquet files from either

    ./data

or

    ../Neural-Spot-FX-Alpha-Model/data

Outputs

    macro_data/macro_*.parquet
    data/fx.parquet
"""

from pathlib import Path
import os
import time

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from config import (
    PROJECT_ROOT,
    DATA_DIR,
    MACRO_DIR,
    FX_SOURCE_DIRS,
    START_DATE,
    END_DATE,
)

# ---------------------------------------------------------
# Yahoo tickers
# ---------------------------------------------------------

YF_TICKERS = {
    "spx": "^GSPC",
    "eustoxx": "^STOXX50E",
    "vix": "^VIX",
    "dxy": "DX-Y.NYB",
}

# ---------------------------------------------------------
# FRED series
# ---------------------------------------------------------

FRED_SERIES = {
    "ust2y": "DGS2",
    "yield_curve": "T10Y2Y",
}

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def ensure_directories():

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MACRO_DIR.mkdir(parents=True, exist_ok=True)


def latest_macro_cache():

    files = sorted(
        MACRO_DIR.glob("macro_*.parquet")
    )

    if not files:
        return None

    return max(
        files,
        key=lambda p: p.stat().st_mtime
    )


def latest_fx_cache():

    file = DATA_DIR / "fx.parquet"

    if file.exists():
        return file

    return None


# ---------------------------------------------------------
# Yahoo download
# ---------------------------------------------------------

def download_yahoo(
        ticker: str,
        start: str,
        end: str,
):

    print(f"[Yahoo] {ticker}")

    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if df.empty:
        raise RuntimeError(
            f"No data returned for {ticker}"
        )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    ts_col = (
        "Datetime"
        if "Datetime" in df.columns
        else "Date"
    )

    df = df.rename(
        columns={
            ts_col: "timestamp",
            "Close": "close",
        }
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    return df[
        [
            "timestamp",
            "close",
        ]
    ]


# ---------------------------------------------------------
# FRED JSON download
# ---------------------------------------------------------

def download_fred(
        series_id: str,
        start: str,
        end: str,
):

    print(f"[FRED] {series_id}")

    api_key = os.getenv(
        "FRED_API_KEY"
    )

    if api_key is None:
        raise RuntimeError(
            "FRED_API_KEY not set."
        )

    url = (
        "https://api.stlouisfed.org/"
        "fred/series/observations"
    )

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

            r = requests.get(
                url,
                params=params,
                timeout=120,
            )

            r.raise_for_status()

            js = r.json()

            obs = js["observations"]

            df = pd.DataFrame(obs)

            if df.empty:
                raise RuntimeError(
                    f"No observations for {series_id}"
                )

            df = df[
                [
                    "date",
                    "value",
                ]
            ]

            df.columns = [
                "timestamp",
                "value",
            ]

            df["timestamp"] = pd.to_datetime(
                df["timestamp"],
                utc=True,
            )

            df["value"] = pd.to_numeric(
                df["value"],
                errors="coerce",
            )

            return df

        except Exception as e:

            last_error = e

            print(
                f"Retry {attempt+1}/5 ({series_id})"
            )

            time.sleep(5)

    raise RuntimeError(last_error)

# ---------------------------------------------------------
# Build macro dataset
# ---------------------------------------------------------

def build_macro_dataset(
        start: str = START_DATE,
        end: str = END_DATE,
        force: bool = False,
):
    """
    Downloads all macro series, engineers features,
    caches the result and returns the parquet path.
    """

    ensure_directories()

    outfile = (
            MACRO_DIR
            / f"macro_{start}_{end}.parquet"
    )

    if outfile.exists() and not force:
        print(f"[CACHE HIT] {outfile}")
        return outfile

    frames = []

    # -------------------------------------------------
    # Yahoo assets
    # -------------------------------------------------

    for name, ticker in YF_TICKERS.items():

        df = download_yahoo(
            ticker,
            start,
            end,
        )

        df = df.rename(
            columns={
                "close": name
            }
        )

        frames.append(
            df.set_index("timestamp")
        )

    # -------------------------------------------------
    # FRED assets
    # -------------------------------------------------

    for name, series in FRED_SERIES.items():

        df = download_fred(
            series,
            start,
            end,
        )

        df = df.rename(
            columns={
                "value": name
            }
        )

        frames.append(
            df.set_index("timestamp")
        )

    print("Combining datasets...")

    merged = (
        pd.concat(
            frames,
            axis=1,
            sort=True,
        )
        .sort_index()
        .ffill()
        .reset_index()
    )

    # -------------------------------------------------
    # Feature engineering
    # -------------------------------------------------

    print("Creating macro features...")

    merged["spx_ret"] = (
        merged["spx"]
        .pct_change()
    )

    merged["eustoxx_ret"] = (
        merged["eustoxx"]
        .pct_change()
    )

    merged["equity_relative"] = (
            merged["eustoxx_ret"]
            - merged["spx_ret"]
    )

    merged["vix_change"] = (
        merged["vix"]
        .pct_change()
    )

    merged["dxy_ret"] = (
        merged["dxy"]
        .pct_change()
    )

    merged["risk_regime"] = (
            merged["vix"]
            >
            merged["vix"]
            .rolling(50)
            .mean()
    ).astype(int)

    merged["spx_vix_corr"] = (
        merged["spx_ret"]
        .rolling(20)
        .corr(
            merged["vix_change"]
        )
    )

    merged["spx_momentum_20"] = (
            merged["spx"]
            / merged["spx"].shift(20)
            - 1
    )

    merged["eustoxx_momentum_20"] = (
            merged["eustoxx"]
            / merged["eustoxx"].shift(20)
            - 1
    )

    merged["dxy_momentum_20"] = (
            merged["dxy"]
            / merged["dxy"].shift(20)
            - 1
    )

    merged["vix_zscore"] = (
            (
                    merged["vix"]
                    -
                    merged["vix"]
                    .rolling(100)
                    .mean()
            )
            /
            merged["vix"]
            .rolling(100)
            .std()
    )

    # Use yield curve if German 2Y isn't available

    merged["yield_spread"] = (
        merged["yield_curve"]
    )

    # -------------------------------------------------
    # Cleanup
    # -------------------------------------------------

    pd.set_option('display.max_columns', None)

    #Fix for rolling stats that are nan
    merged = merged.dropna(subset=["spx", "vix", "dxy"])
    merged = merged.ffill()
    merged = merged.fillna(0)

    merged.to_parquet(
        outfile,
        index=False,
    )

    print()
    print(f"[MACRO CREATED] {outfile}")
    print(f"Rows: {len(merged):,}")

    return outfile

# ---------------------------------------------------------
# Build FX dataset
# ---------------------------------------------------------
def download_eurusd(start: str, end: str) -> pd.DataFrame:
    print("[FX] Backfilling EURUSD from Yahoo")

    df = yf.download(
        "EURUSD=X",
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if df.empty:
        raise RuntimeError("Failed to download EURUSD data")

    df = df.reset_index()

    ts_col = "Datetime" if "Datetime" in df.columns else "Date"

    df = df.rename(
        columns={
            ts_col: "timestamp",
            "Close": "close",
        }
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    return df[["timestamp", "close"]]

def build_fx_dataset(
        start: str = START_DATE,
        end: str = END_DATE,
        force: bool = False
):
    """
    Time-windowed FX dataset builder.
    Cache key is fully deterministic on (start, end).
    """

    ensure_directories()

    versioned_file = DATA_DIR / f"fx_{start}_{end}.parquet"
    latest_file = DATA_DIR / "fx.parquet"
    # -------------------------------------------------
    # cache hit
    # -------------------------------------------------
    if versioned_file.exists() and not force:
        print(f"[CACHE HIT] {versioned_file}")

        # Keep latest alias in sync
        if not latest_file.exists():
            pd.read_parquet(versioned_file).to_parquet(latest_file, index=False)

        return latest_file

    # -------------------------------------------------
    # gather raw files
    # -------------------------------------------------

    files = []

    for directory in FX_SOURCE_DIRS:

        directory = Path(directory)

        if not directory.exists():
            continue

        matches = sorted([
            f for f in directory.glob("*.parquet")
            if "EURUSD" in f.name.upper()
        ])

        if matches:
            print(f"Found {len(matches)} files in {directory}")
            files.extend(matches)

    if not files:
        print("[FX] No cached parquet found → backfilling from source")

        fx = download_eurusd(start, end)

        raw_file = DATA_DIR / "EURUSD_1.parquet"
        raw_file.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(fx.columns, pd.MultiIndex):
            fx.columns = fx.columns.get_level_values(0)

        fx = fx.reset_index()

        ts_col = "Datetime" if "Datetime" in fx.columns else "Date"

        fx = fx.rename(
            columns={
                ts_col: "timestamp",
                "Close": "close",
            }
        )

        fx = fx[["timestamp", "close"]]
        fx.to_parquet(raw_file, index=False)

        files = [raw_file]

    # -------------------------------------------------
    # load + combine
    # -------------------------------------------------

    dfs = []

    for file in files:
        print(f"Loading {file.name}")

        df = pd.read_parquet(file).reset_index()

        print('df ', df)
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            utc=True,
            errors="coerce",
        )

        df = df.dropna(subset=["timestamp"])

        dfs.append(df)

    fx = (
        pd.concat(dfs, ignore_index=True)
        .sort_values("timestamp")
        .drop_duplicates(subset="timestamp")
    )

    fx["timestamp"] = pd.to_datetime(fx["timestamp"], utc=True, errors="coerce")

    # -------------------------------------------------
    # STRICT TIME FILTER (this is now authoritative)
    # -------------------------------------------------

    start_ts = pd.to_datetime(start, utc=True)
    end_ts = pd.to_datetime(end, utc=True)

    fx = fx.loc[
        (fx["timestamp"] >= start_ts) &
        (fx["timestamp"] <= end_ts)
        ].reset_index(drop=True)

    # -------------------------------------------------
    # cache write
    # -------------------------------------------------
    versioned_file = DATA_DIR / f"fx_{start}_{end}.parquet"
    latest_file = DATA_DIR / "fx.parquet"

    fx.to_parquet(versioned_file, index=False)
    fx.to_parquet(latest_file, index=False)

    print()
    print(f"[FX CACHE CREATED] {versioned_file}")
    print(f"[FX LATEST] {latest_file}")
    print(f"Rows: {len(fx):,}")

    return latest_file

    fx.to_parquet(output_file, index=False)

    print()
    print(f"[FX CACHE CREATED] {output_file}")
    print(f"Rows: {len(fx):,}")

    return output_file

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":

    start = START_DATE
    end = END_DATE

    macro_path = build_macro_dataset(
        start=start,
        end=end,
        force=False,
    )

    fx_path = build_fx_dataset(
        start=start,
        end=end,
        force=False,
    )
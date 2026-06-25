"""
ingest_macro_data.py

Data ingestion pipeline for the Temporal Fusion Transformer project.

Responsibilities
----------------
1. Download macro data
    - Yahoo Finance
    - FRED JSON API

2. Build macro feature dataset

3. Build consolidated FX cache

4. Cache management

Everything is timezone-normalized to UTC.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import requests
import yfinance as yf


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "macro_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

FX_CACHE_DIR = (
        PROJECT_ROOT.parent
        / "Neural-Spot-FX-Alpha-Model"
        / "data"

)

MACRO_DIR = PROJECT_ROOT / "macro_data"
MACRO_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_FX_DIR = PROJECT_ROOT / "data"
LOCAL_FX_DIR.mkdir(parents=True, exist_ok=True)

# Existing FX project
EXTERNAL_FX_DIR = (
        PROJECT_ROOT.parent
        / "Neural-Spot-FX-Alpha-Model"
        / "data"
)

SEARCH_DIRS = [
    LOCAL_FX_DIR,
    EXTERNAL_FX_DIR,
]


# ============================================================
# MARKET DATA CONFIG
# ============================================================

YF_TICKERS = {
    "spx": "^GSPC",
    "eustoxx": "^STOXX50E",
    "vix": "^VIX",
    "dxy": "DX-Y.NYB",
}

#
# Germany 2Y is intentionally omitted because it is not
# consistently available through FRED.
#
FRED_SERIES = {
    "us2y": "DGS2",
    "yield_curve": "T10Y2Y",
}


# ============================================================
# HELPERS
# ============================================================

def retry(
        attempts: int = 5,
        delay: int = 5,
):
    """
    Retry decorator for transient HTTP failures.
    """

    def decorator(func):

        def wrapper(*args, **kwargs):

            last_exception = None

            for i in range(attempts):

                try:
                    return func(*args, **kwargs)

                except Exception as e:

                    last_exception = e

                    print(
                        f"[Retry {i+1}/{attempts}] {e}"
                    )

                    if i < attempts - 1:
                        time.sleep(delay)

            raise last_exception

        return wrapper

    return decorator


# ============================================================
# TIMESTAMP NORMALIZATION
# ============================================================

def normalize_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures timestamps are:

    * datetime64[ns, UTC]
    * sorted
    * unique
    * non-null
    """

    if "timestamp" not in df.columns:
        raise ValueError(
            "DataFrame has no timestamp column."
        )

    df = df.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        format="mixed",
        utc=True,
    )

    df = df.dropna(
        subset=["timestamp"]
    )

    df = (
        df.sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# YAHOO DOWNLOAD
# ============================================================

@retry()
def download_yahoo(
        ticker: str,
        start: str,
        end: str,
) -> pd.DataFrame:

    print(f"[Yahoo] {ticker}")

    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if df.empty:
        raise RuntimeError(
            f"No data for {ticker}"
        )

    #
    # Flatten MultiIndex if Yahoo decides to use one.
    #
    if isinstance(
            df.columns,
            pd.MultiIndex,
    ):
        df.columns = (
            df.columns.get_level_values(0)
        )

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

    df = df[
        [
            "timestamp",
            "close",
        ]
    ]

    return normalize_timestamp(df)


# ============================================================
# FRED DOWNLOAD
# ============================================================

@retry()
def download_fred(
        series_id: str,
        start: str,
        end: str,
) -> pd.DataFrame:

    api_key = os.environ.get(
        "FRED_API_KEY"
    )

    if api_key is None:
        raise RuntimeError(
            "FRED_API_KEY not defined."
        )

    print(
        f"[FRED] {series_id}"
    )

    url = (
        "https://api.stlouisfed.org/"
        "fred/series/observations"
    )

    response = requests.get(
        url,
        params={
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": start,
            "observation_end": end,
        },
        timeout=120,
    )

    response.raise_for_status()

    data = response.json()

    observations = data.get(
        "observations",
        []
    )

    if len(observations) == 0:
        raise RuntimeError(
            f"No observations for {series_id}"
        )

    df = pd.DataFrame(
        observations
    )[["date", "value"]]

    df.columns = [
        "timestamp",
        "value",
    ]

    df["value"] = pd.to_numeric(
        df["value"],
        errors="coerce",
    )

    df = normalize_timestamp(df)

    return df


# ============================================================
# CACHE HELPERS
# ============================================================

def latest_macro_file():

    files = sorted(
        MACRO_DIR.glob(
            "macro_*.parquet"
        )
    )

    if not files:
        return None

    return max(
        files,
        key=lambda f: f.stat().st_mtime,
    )


def latest_fx_cache():

    file = LOCAL_FX_DIR / "fx.parquet"

    if file.exists():
        return file

    return None

# =====================================================
# BUILD MACRO DATASET
# =====================================================

def build_macro_dataset(
        start: str,
        end: str,
) -> Path:

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    outfile = DATA_DIR / f"macro_{start}_{end}.parquet"

    if outfile.exists():
        print(f"[CACHE HIT] {outfile}")
        return outfile

    all_dfs = []

    # ---------------------------------------------
    # Yahoo Finance
    # ---------------------------------------------

    for name, ticker in YF_TICKERS.items():

        df = download_yahoo(
            ticker=ticker,
            start=start,
            end=end,
        )

        df = df.rename(
            columns={
                "close": name
            }
        )

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            utc=True,
            format="mixed",
        )

        all_dfs.append(
            df.set_index("timestamp")
        )

    # ---------------------------------------------
    # FRED
    # ---------------------------------------------

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

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            utc=True,
            format="mixed",
        )

        all_dfs.append(
            df.set_index("timestamp")
        )

    print("Combining datasets...")

    merged = (
        pd.concat(
            all_dfs,
            axis=1,
            sort=True,
        )
        .sort_index()
        .ffill()
        .reset_index()
    )

    # ---------------------------------------------
    # Feature Engineering
    # ---------------------------------------------

    print("Creating macro features...")

    merged["spx_ret"] = merged["spx"].pct_change()

    merged["eustoxx_ret"] = merged["eustoxx"].pct_change()

    merged["equity_relative"] = (
            merged["eustoxx_ret"]
            - merged["spx_ret"]
    )

    merged["vix_change"] = merged["vix"].pct_change()

    merged["dxy_ret"] = merged["dxy"].pct_change()

    # We only have US rates right now
    merged["ust2y"] = merged["us2y"]

    merged["yield_spread"] = merged["yield_curve"]

    merged["risk_regime"] = (
            merged["vix"]
            >
            merged["vix"].rolling(50).mean()
    ).astype(int)

    merged["spx_vix_corr"] = (
        merged["spx_ret"]
        .rolling(20)
        .corr(merged["vix_change"])
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
                    - merged["vix"].rolling(100).mean()
            )
            /
            merged["vix"].rolling(100).std()
    )

    merged = (
        merged
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .dropna()
    )

    merged.to_parquet(
        outfile,
        index=False,
    )

    print(f"[MACRO CREATED] {outfile}")
    print(f"Rows: {len(merged):,}")

    return outfile

# =====================================================
# BUILD FX DATASET
# =====================================================

def build_fx_dataset(
        force: bool = False,
) -> Path:

    local_data_dir = PROJECT_ROOT / "data"
    local_data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = local_data_dir / "fx.parquet"

    # -----------------------------------------
    # Cache hit
    # -----------------------------------------

    if output_file.exists() and not force:

        print(f"[CACHE HIT] {output_file}")

        return output_file

    # -----------------------------------------
    # Search locations
    # -----------------------------------------

    search_dirs = [
        local_data_dir,
        FX_CACHE_DIR,
    ]

    eurusd_files = []

    for directory in search_dirs:

        if not directory.exists():
            continue

        matches = sorted(
            directory.glob("EURUSD_*.parquet")
        )

        if matches:

            print(
                f"Found {len(matches)} files in {directory}"
            )

            eurusd_files.extend(matches)

    if not eurusd_files:

        raise RuntimeError(
            f"""
No EURUSD parquet files found.

Searched:

{local_data_dir}

{FX_CACHE_DIR}

Run your Neural-Spot-FX-Alpha-Model downloader first.
"""
        )

    # -----------------------------------------
    # Load
    # -----------------------------------------

    dfs = []

    for file in eurusd_files:

        print(f"Loading {file.name}")

        df = pd.read_parquet(file)

        dfs.append(df)

    fx = pd.concat(
        dfs,
        ignore_index=True,
    )

    # -----------------------------------------
    # Timestamp cleanup
    # -----------------------------------------

    fx["timestamp"] = pd.to_datetime(
        fx["timestamp"],
        utc=True,
        format="mixed",
        errors="coerce",
    )

    fx = fx.dropna(
        subset=["timestamp"]
    )

    fx = (
        fx
        .sort_values("timestamp")
        .drop_duplicates(
            subset="timestamp",
            keep="last",
        )
        .reset_index(drop=True)
    )

    # -----------------------------------------
    # Save cache
    # -----------------------------------------

    fx.to_parquet(
        output_file,
        index=False,
    )

    print()
    print(f"[FX CACHE CREATED] {output_file}")
    print(f"Rows: {len(fx):,}")

    return output_file
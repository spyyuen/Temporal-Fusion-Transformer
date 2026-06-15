from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
import requests
import yfinance as yf


# =====================================================
# CONFIG
# =====================================================

DATA_DIR = Path("macro_data")
DATA_DIR.mkdir(exist_ok=True)

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

def download_yahoo(
        ticker: str,
        start: str,
        end: str
) -> pd.DataFrame:

    print(f"Downloading {ticker}")

    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        raise ValueError(
            f"No data returned for {ticker}"
        )

    # Flatten MultiIndex columns if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    timestamp_col = (
        "Datetime"
        if "Datetime" in df.columns
        else "Date"
    )

    df = df.rename(
        columns={
            timestamp_col: "timestamp",
            "Close": "close",
        }
    )

    return df[
        ["timestamp", "close"]
    ]


# =====================================================
# FRED DOWNLOAD
# =====================================================

def download_fred(
        series_id: str,
        start: str,
        end: str
) -> pd.DataFrame:

    print(
        f"Downloading FRED {series_id}"
    )

    url = (
        "https://fred.stlouisfed.org/"
        f"graph/fredgraph.csv?id={series_id}"
    )

    response = requests.get(
        url,
        timeout=30
    )

    response.raise_for_status()

    df = pd.read_csv(
        StringIO(response.text)
    )

    df.columns = [
        "timestamp",
        "value"
    ]

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    df = df[
        (df["timestamp"] >= start_ts)
        &
        (df["timestamp"] <= end_ts)
        ]

    return df


# =====================================================
# BUILD DATASET
# =====================================================

def build_macro_dataset(
        start: str,
        end: str
) -> pd.DataFrame:

    all_dfs = []

    # -----------------------------------------
    # Yahoo Assets
    # -----------------------------------------

    for name, ticker in YF_TICKERS.items():

        df = download_yahoo(
            ticker,
            start,
            end
        )

        df = df.rename(
            columns={
                "close": name
            }
        )

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            utc=True
        )

        all_dfs.append(
            df.set_index("timestamp")
        )

    # -----------------------------------------
    # FRED Assets
    # -----------------------------------------

    for name, series in FRED_SERIES.items():

        df = download_fred(
            series,
            start,
            end
        )

        df = df.rename(
            columns={
                "value": name
            }
        )

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            utc=True
        )

        all_dfs.append(
            df.set_index("timestamp")
        )

    # -----------------------------------------
    # Merge
    # -----------------------------------------

    print("Combining datasets...")

    merged = (
        pd.concat(
            all_dfs,
            axis=1,
            sort=True
        )
        .sort_index()
        .reset_index()
    )

    merged = merged.sort_values(
        "timestamp"
    )

    merged = merged.ffill()

    # -----------------------------------------
    # Derived Features
    # -----------------------------------------

    print(
        "Creating macro features..."
    )

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

    merged["yield_spread"] = (
        merged["yield_curve"]
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
            /
            merged["spx"].shift(20)
            - 1
    )

    merged["eustoxx_momentum_20"] = (
            merged["eustoxx"]
            /
            merged["eustoxx"].shift(20)
            - 1
    )

    merged["dxy_momentum_20"] = (
            merged["dxy"]
            /
            merged["dxy"].shift(20)
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

    # -----------------------------------------
    # Cleanup
    # -----------------------------------------

    merged = merged.replace(
        [np.inf, -np.inf],
        np.nan
    )

    merged = merged.ffill()

    merged = merged.dropna()

    # -----------------------------------------
    # Save
    # -----------------------------------------

    outfile = (
            DATA_DIR
            /
            f"macro_{start}_{end}.parquet"
    )

    merged.to_parquet(
        outfile,
        index=False
    )

    print()
    print(f"Saved: {outfile}")
    print(f"Rows: {len(merged):,}")
    print(f"Columns: {len(merged.columns)}")

    return merged


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    build_macro_dataset(
        start="2024-01-01",
        end="2026-06-01"
    )
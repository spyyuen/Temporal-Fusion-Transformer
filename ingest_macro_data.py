import os
from pathlib import Path

import pandas as pd
import yfinance as yf

from fredapi import Fred


DATA_DIR = Path("macro_data")
DATA_DIR.mkdir(exist_ok=True)


# -----------------------------------
# Yahoo Finance
# -----------------------------------

YF_TICKERS = {
    "spx": "^GSPC",
    "eustoxx": "^STOXX50E",
    "vix": "^VIX",
    "dxy": "DX-Y.NYB",
}


# -----------------------------------
# FRED
# -----------------------------------

FRED_SERIES = {
    "us2y": "DGS2",
    "yield_curve": "T10Y2Y",
}


# -----------------------------------
# Download Yahoo
# -----------------------------------

def download_yahoo(
        ticker,
        start,
        end
):

    print(f"Downloading {ticker}")

    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )

    if len(df) == 0:
        raise ValueError(
            f"No data for {ticker}"
        )

    df = df.reset_index()

    df = df.rename(
        columns={
            "Date": "timestamp",
            "Datetime": "timestamp",
            "Close": "close",
        }
    )

    return df[[
        "timestamp",
        "close"
    ]]


# -----------------------------------
# Download FRED
# -----------------------------------

def download_fred(
        fred,
        series_id,
        start,
        end
):

    print(
        f"Downloading FRED {series_id}"
    )

    s = fred.get_series(
        series_id,
        observation_start=start,
        observation_end=end
    )

    df = pd.DataFrame({
        "timestamp": s.index,
        "value": s.values
    })

    return df


# -----------------------------------
# Main
# -----------------------------------

def build_macro_dataset(
        start,
        end
):

    fred = Fred(
        api_key=os.environ[
            "FRED_API_KEY"
        ]
    )

    merged = None

    # ----------------------------
    # Yahoo assets
    # ----------------------------

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

        if merged is None:

            merged = df

        else:

            merged = pd.merge(
                merged,
                df,
                on="timestamp",
                how="outer"
            )

    # ----------------------------
    # FRED assets
    # ----------------------------

    for name, series in FRED_SERIES.items():

        df = download_fred(
            fred,
            series,
            start,
            end
        )

        df = df.rename(
            columns={
                "value": name
            }
        )

        merged = pd.merge(
            merged,
            df,
            on="timestamp",
            how="outer"
        )

    # ----------------------------
    # Clean
    # ----------------------------

    merged["timestamp"] = pd.to_datetime(
        merged["timestamp"],
        utc=True
    )

    merged = merged.sort_values(
        "timestamp"
    )

    merged = merged.ffill()

    # ----------------------------
    # Derived Macro Features
    # ----------------------------

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

    # strongest macro signal

    merged["yield_spread"] = (
        merged["yield_curve"]
    )

    # additional useful feature

    merged["risk_regime"] = (
        (
                merged["vix"]
                > merged["vix"]
                .rolling(50)
                .mean()
        )
        .astype(int)
    )

    # ----------------------------
    # Save
    # ----------------------------

    outfile = (
            DATA_DIR /
            f"macro_{start}_{end}.parquet"
    )

    merged.to_parquet(
        outfile,
        index=False
    )

    print(
        f"Saved {outfile}"
    )

    return merged


if __name__ == "__main__":

    build_macro_dataset(
        start="2024-01-01",
        end="2026-06-01"
    )
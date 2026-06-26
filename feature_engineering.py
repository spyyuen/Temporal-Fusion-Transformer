"""
feature_engineering.py

Centralized feature engineering for FX + macro TFT pipeline.

Design goals:
- Safe with missing columns
- No hard dependencies on bund2y / ust2y
- Reusable across ingestion + training
- Lightweight enough for 500k–1M row datasets
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# CORE FX FEATURES
# ---------------------------------------------------------

def add_fx_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ----------------------------
    # Mid price
    # ----------------------------

    if "bid" in df.columns and "ask" in df.columns:
        df["mid"] = (df["bid"] + df["ask"]) / 2
    elif "close" in df.columns:
        df["mid"] = df["close"]
    else:
        raise ValueError("No price columns found (bid/ask or close)")

    # ----------------------------
    # Returns
    # ----------------------------

    df["fx_ret"] = df["mid"].pct_change()

    # ----------------------------
    # Volatility
    # ----------------------------

    df["fx_vol_20"] = df["fx_ret"].rolling(20).std()
    df["fx_vol_100"] = df["fx_ret"].rolling(100).std()

    # ----------------------------
    # Momentum
    # ----------------------------

    for lag in [1, 2, 5, 10]:
        df[f"fx_ret_lag_{lag}"] = df["fx_ret"].shift(lag)

    return df


# ---------------------------------------------------------
# MACRO FEATURES (SAFE VERSION)
# ---------------------------------------------------------

def add_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ----------------------------
    # Equity returns
    # ----------------------------

    if "spx" in df.columns:
        df["spx_ret"] = df["spx"].pct_change()

    if "eustoxx" in df.columns:
        df["eustoxx_ret"] = df["eustoxx"].pct_change()

    # relative equity pressure
    if "spx_ret" in df.columns and "eustoxx_ret" in df.columns:
        df["equity_relative"] = df["eustoxx_ret"] - df["spx_ret"]

    # ----------------------------
    # Volatility index
    # ----------------------------

    if "vix" in df.columns:
        df["vix_change"] = df["vix"].pct_change()

        df["vix_zscore"] = (
                                   df["vix"]
                                   - df["vix"].rolling(100).mean()
                           ) / (df["vix"].rolling(100).std() + 1e-6)

    # ----------------------------
    # Dollar index
    # ----------------------------

    if "dxy" in df.columns:
        df["dxy_ret"] = df["dxy"].pct_change()

    # ----------------------------
    # Rates (SAFE FIX for bund2y crash)
    # ----------------------------

    # We never assume bund2y exists anymore
    if "ust2y" in df.columns:
        df["rate_level"] = df["ust2y"]

    if "yield_curve" in df.columns:
        df["yield_spread"] = df["yield_curve"]

    # fallback safe feature if both missing
    if "ust2y" not in df.columns and "yield_curve" not in df.columns:
        df["yield_spread"] = 0.0

    return df


# ---------------------------------------------------------
# REGIME FEATURES
# ---------------------------------------------------------

def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "vix" in df.columns:

        df["risk_regime"] = (
                df["vix"] > df["vix"].rolling(50).mean()
        ).astype(int)

        df["vix_spike"] = (
                df["vix_change"] > df["vix_change"].rolling(20).mean()
        ).astype(int)

    return df


# ---------------------------------------------------------
# FINAL TARGET
# ---------------------------------------------------------

def add_target(df: pd.DataFrame, horizon: int = 15) -> pd.DataFrame:
    df = df.copy()

    df["target"] = (
            df["mid"].shift(-horizon) / df["mid"] - 1
    )

    df["target_vol_adj"] = df["target"] / (
            df["fx_vol_20"] + 1e-6
    )

    return df


# ---------------------------------------------------------
# MASTER PIPELINE
# ---------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature pipeline used by preprocessing + training.
    """

    df = add_fx_features(df)
    df = add_macro_features(df)
    df = add_regime_features(df)
    df = add_target(df)

    # cleanup
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    return df
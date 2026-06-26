# Temporal Fusion Transformer FX + Macro Model

This project builds a **macro-driven FX trading model** using a simplified **Temporal Fusion Transformer (TFT)-inspired architecture**.

It integrates:

- FX data (EURUSD)
- Global equities (SPX, EuroStoxx)
- Volatility (VIX)
- Dollar index (DXY)
- Interest rates (US 2Y, German 2Y proxy)
- Macro regime features
- Deep learning sequence model (Transformer-based)

---

# ⚠️ Important Note

This is a **research / experimental system**, not production trading advice.

It is optimized for:

- learning time-series ML
- macro feature engineering
- large-scale dataset handling
- backtesting signal logic

---

# 🧠 Model Overview

The model predicts:

> Risk-adjusted 15-minute EURUSD returns

Then converts predictions into trading signals:

- `+1` → Long EURUSD
- `-1` → Short EURUSD
- `0` → Flat

---

# 📊 Pipeline

## 1. Data Ingestion

### Macro data
- Yahoo Finance (SPX, VIX, STOXX50E, DXY)
- FRED (US 2Y, yield curve)

### FX data
- EURUSD parquet files (external source)

---

## 2. Feature Engineering

Creates:

- FX momentum (lags 1–10)
- Rolling volatility (20, 100)
- Equity relative strength
- Yield spreads
- Dollar regime signals
- VIX regime indicators
- Time-of-day cyclical encoding

---

## 3. Model

A lightweight Transformer:

- Input projection layer
- Transformer encoder
- Fully connected head
- Outputs scalar risk-adjusted return

---

## 4. Training

- Huber / SmoothL1 loss
- Batch training via DataLoader
- Gradient clipping
- Streaming dataset (no full RAM load)

---

## 5. Backtesting

Includes:

- Transaction costs
- Equity curve
- Sharpe ratio
- Drawdown
- Win rate

---

# 🚀 How to Run

## 1. Install dependencies

```bash
pip install pandas numpy torch yfinance requests pyarrow

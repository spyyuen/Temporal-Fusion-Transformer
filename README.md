# Temporal Fusion Transformer

A research framework for building machine learning-based alpha signals in the FX market using cross-asset information from equities, rates, volatility, and macroeconomic indicators.

This project combines:

- EURUSD tick or bar data
- Equity indices
- Interest rate data
- Volatility indices
- Transformer-based forecasting models
- Walk-forward backtesting
- Volatility-targeted portfolio construction

The objective is to predict risk-adjusted future EURUSD returns and generate systematic trading signals.

---

# Installation

pip install yfinance pandas pyarrow fredapi requests
Get a free FRED API key from: https://fred.stlouisfed.org/docs/api/api_key.html
Bash: export FRED_API_KEY="YOUR_KEY"

---

# Motivation

Traditional FX models often rely exclusively on lagged price action and technical indicators.

However, FX markets are heavily influenced by:

- Relative economic growth
- Monetary policy divergence
- Risk-on / risk-off sentiment
- Equity market performance
- Yield spreads

This project attempts to incorporate these macroeconomic drivers into a unified machine learning framework.

---

# Research Foundation

The forecasting architecture is inspired by:

**Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting**

Bryan Lim et al.

https://arxiv.org/abs/1912.09363

Key ideas adopted from the paper:

- Multi-source time series inputs
- Sequence modeling
- Attention mechanisms
- Dynamic feature importance
- Multi-horizon forecasting

The implementation in this repository is a simplified TFT-inspired architecture built using PyTorch's TransformerEncoder.

---

# Data Sources

## FX Data

EURUSD bid/ask data

Required fields:

| Column | Description |
|----------|-------------|
| timestamp | UTC timestamp |
| bid | bid price |
| ask | ask price |

---

## Market Data

Downloaded automatically from Yahoo Finance.

| Asset | Symbol |
|---------|---------|
| S&P 500 | ^GSPC |
| Euro Stoxx 50 | ^STOXX50E |
| VIX | ^VIX |
| US Dollar Index | DX-Y.NYB |

---

## Macroeconomic Data

Downloaded from FRED.

| Series | FRED Code |
|----------|----------|
| US 2Y Treasury Yield | DGS2 |
| 10Y-2Y Yield Curve | T10Y2Y |

---

# Project Structure

```text
.
├── data/
│   ├── fx/
│   └── macro/
│
├── ingest_macro_data.py
├── features.py
├── models.py
├── train.py
├── backtest.py
├── strategy.py
├── requirements.txt
└── README.md
```

---

# Installation

Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

or

```bash
pip install pandas numpy torch yfinance fredapi pyarrow scikit-learn
```

---

# FRED API Key

Create a free API key:

https://fred.stlouisfed.org/docs/api/api_key.html

Export the key:

```bash
export FRED_API_KEY="YOUR_KEY"
```

---

# Data Ingestion

Download macro data:

```bash
python ingest_macro_data.py
```

This will download:

- SPX
- EuroStoxx
- VIX
- DXY
- Treasury data

and save a cached parquet dataset.

Output:

```text
macro_data/
└── macro_2024-01-01_2026-06-01.parquet
```

---

# Feature Engineering

The model generates several categories of features.

## FX Features

- Lagged returns
- Momentum
- Volatility
- Spread dynamics
- Rolling z-scores

---

## Equity Features

- SPX returns
- EuroStoxx returns
- Relative equity performance

```text
equity_relative =
EuroStoxx return
-
SPX return
```

---

## Macro Features

- Yield spread
- Yield curve slope
- DXY momentum
- VIX changes

---

## Regime Features

- Risk regime
- Volatility regime
- Session effects

---

# Target Construction

The model predicts:

```text
15-minute future return
/
future volatility
```

This creates a risk-adjusted prediction target rather than predicting raw returns.

---

# Model

The default architecture is a Transformer-based sequence model.

Input:

```text
120 timestep sequence
```

Features:

```text
FX
+
Equities
+
Rates
+
Volatility
```

Output:

```text
Forecast risk-adjusted return
```

---

# Training

Train the model:

```bash
python train.py
```

The model uses:

- Huber loss
- AdamW optimizer
- Sequence inputs
- Walk-forward evaluation

---

# Signal Generation

Signals are generated from model predictions.

```text
prediction > threshold
→ Long EURUSD

prediction < -threshold
→ Short EURUSD

otherwise
→ Flat
```

Example:

```python
signal = np.where(
    pred > 1,
    1,
    np.where(
        pred < -1,
        -1,
        0
    )
)
```

---

# Position Sizing

The strategy uses volatility targeting.

```text
Position Size =
Target Volatility
/
Realized Volatility
```

This reduces exposure during unstable market regimes.

---

# Backtesting

The framework supports walk-forward testing.

Example:

```text
Train:
Jan 2024 → Dec 2025

Test:
Jan 2026

Retrain

Train:
Jan 2024 → Jan 2026

Test:
Feb 2026
```

This avoids look-ahead bias and provides realistic out-of-sample evaluation.

---

# Performance Metrics

The following metrics should be evaluated:

- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Hit Rate
- Profit Factor
- Annualized Return
- Turnover

---

# Future Improvements

Potential extensions:

- Full Temporal Fusion Transformer implementation
- Graph Neural Networks
- Multi-currency portfolio
- Reinforcement Learning execution layer
- Triple-barrier labeling
- LightGBM ensemble
- Transaction cost modeling
- Dynamic position sizing

---

# Disclaimer

This repository is intended for research and educational purposes only.

Past performance does not guarantee future results. Trading foreign exchange and leveraged products involves substantial risk and may result in losses exceeding initial capital.

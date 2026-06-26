import numpy as np
import pandas as pd


# =====================================================
# BASIC BACKTEST ENGINE (FX STYLE)
# =====================================================

def run_backtest(
        signals: np.ndarray,
        returns: np.ndarray,
        transaction_cost: float = 0.00002
):
    """
    Simple vectorized backtest.

    signals:
        -1 short
         0 flat
         1 long

    returns:
        future returns aligned with signals
    """

    signals = np.asarray(signals).astype(np.float32)
    returns = np.asarray(returns).astype(np.float32)

    if len(signals) != len(returns):
        raise ValueError("signals and returns must match length")

    # -------------------------------------------------
    # POSITION CHANGES (for costs)
    # -------------------------------------------------

    position_change = np.abs(np.diff(np.insert(signals, 0, 0)))

    costs = position_change * transaction_cost

    # -------------------------------------------------
    # STRATEGY RETURNS
    # -------------------------------------------------

    strategy_returns = signals * returns

    strategy_returns = strategy_returns - costs

    equity_curve = np.cumsum(strategy_returns)

    return strategy_returns, equity_curve


# =====================================================
# METRICS
# =====================================================

def compute_metrics(strategy_returns: np.ndarray):

    strategy_returns = np.asarray(strategy_returns)

    mean = np.mean(strategy_returns)
    std = np.std(strategy_returns) + 1e-12

    sharpe = mean / std * np.sqrt(252 * 96)  # approx 15-min bars

    equity = np.cumsum(strategy_returns)

    drawdown = equity - np.maximum.accumulate(equity)

    max_dd = np.min(drawdown)

    win_rate = np.mean(strategy_returns > 0)

    return {
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "win_rate": float(win_rate),
        "total_return": float(equity[-1])
    }


# =====================================================
# POSITION SIZING WRAPPER (OPTIONAL)
# =====================================================

def apply_position_sizing(
        signals: np.ndarray,
        realized_vol: np.ndarray,
        target_vol: float = 0.1
):
    """
    Volatility-targeted position sizing.
    """

    signals = np.asarray(signals).astype(np.float32)
    realized_vol = np.asarray(realized_vol).astype(np.float32)

    scaled = signals * (target_vol / (realized_vol + 1e-8))

    # clip extreme leverage
    scaled = np.clip(scaled, -5, 5)

    return scaled


# =====================================================
# FULL BACKTEST PIPELINE
# =====================================================

def backtest_pipeline(
        signals,
        future_returns,
        realized_vol=None,
        use_position_sizing=False
):

    signals = np.asarray(signals)
    future_returns = np.asarray(future_returns)

    if use_position_sizing and realized_vol is not None:
        signals = apply_position_sizing(signals, realized_vol)

    strat_ret, equity = run_backtest(signals, future_returns)

    metrics = compute_metrics(strat_ret)

    df = pd.DataFrame({
        "strategy_return": strat_ret,
        "equity_curve": equity,
        "signal": signals
    })

    print("\n=== BACKTEST RESULTS ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")

    return df, metrics
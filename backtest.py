import numpy as np
import pandas as pd


# =====================================================
# SIGNAL PROCESSING (NO LOOKAHEAD SAFE)
# =====================================================

def prepare_signals(preds: np.ndarray, threshold: float = 0.0):
    """
    Converts model predictions into trading signals safely.

    IMPORTANT:
    - shift signals by 1 to avoid lookahead bias
    """

    preds = np.asarray(preds).flatten()

    signals = np.where(preds > threshold, 1, -1)

    # shift to avoid using same-bar info
    signals = np.roll(signals, 1)
    signals[0] = 0

    return signals.astype(np.float32)


# =====================================================
# BACKTEST ENGINE (SAFE)
# =====================================================

def run_backtest(signals, returns, transaction_cost=0.00002):

    signals = np.asarray(signals, dtype=np.float32)
    returns = np.asarray(returns, dtype=np.float32)

    if len(signals) != len(returns):
        n = min(len(signals), len(returns))
        signals = signals[:n]
        returns = returns[:n]

    # position changes → cost
    position_change = np.abs(np.diff(np.insert(signals, 0, 0)))
    costs = position_change * transaction_cost

    strat_returns = signals * returns - costs
    equity = np.cumsum(strat_returns)

    return strat_returns, equity


# =====================================================
# METRICS
# =====================================================

def compute_metrics(strategy_returns):

    r = np.asarray(strategy_returns)

    mean = np.mean(r)
    std = np.std(r) + 1e-12

    sharpe = (mean / std) * np.sqrt(252)

    equity = np.cumsum(r)
    peak = np.maximum.accumulate(equity)
    drawdown = equity - peak

    return {
        "sharpe": float(sharpe),
        "total_return": float(equity[-1]),
        "max_drawdown": float(np.min(drawdown)),
        "win_rate": float(np.mean(r > 0))
    }


# =====================================================
# FULL PIPELINE
# =====================================================

def backtest_pipeline(preds, future_returns):

    signals = prepare_signals(preds)
    strat_ret, equity = run_backtest(signals, future_returns)
    metrics = compute_metrics(strat_ret)

    df = pd.DataFrame({
        "strategy_return": strat_ret,
        "equity_curve": equity,
        "signal": signals
    })

    return df, metrics

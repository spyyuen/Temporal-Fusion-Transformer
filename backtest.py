import numpy as np
import pandas as pd


def run_backtest(signals, returns, transaction_cost=0.00002):

    signals = np.asarray(signals, dtype=np.float32)
    returns = np.asarray(returns, dtype=np.float32)

    position = signals  # explicit alias (IMPORTANT FIX)

    position_change = np.abs(np.diff(np.insert(position, 0, 0)))
    costs = position_change * transaction_cost

    strategy_returns = position * returns - costs
    equity_curve = np.cumsum(strategy_returns)

    return strategy_returns, equity_curve, position


def compute_metrics(strategy_returns):

    r = np.asarray(strategy_returns)

    mean = np.mean(r)
    std = np.std(r) + 1e-12

    sharpe = mean / std * np.sqrt(252 * 96)

    downside = r[r < 0]
    downside_std = np.std(downside) + 1e-12

    sortino = mean / downside_std * np.sqrt(252 * 96)

    equity = np.cumsum(r)

    drawdown = equity - np.maximum.accumulate(equity)
    max_dd = np.min(drawdown)

    calmar = (equity[-1] / (abs(max_dd) + 1e-12))

    win_rate = np.mean(r > 0)

    return {
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "calmar": float(calmar),
        "max_drawdown": float(max_dd),
        "win_rate": float(win_rate),
        "total_return": float(equity[-1])
    }


def backtest_pipeline(signals, future_returns):

    strat_ret, equity, position = run_backtest(signals, future_returns)
    metrics = compute_metrics(strat_ret)

    df = pd.DataFrame({
        "strategy_return": strat_ret,
        "equity_curve": equity,
        "signal": signals,
        "position": position,
        "returns": future_returns
    })

    return df, metrics

def metrics_html(metrics: dict):

    rows = "".join([
        f"<tr><td>{k}</td><td>{v:.6f}</td></tr>"
        for k, v in metrics.items()
    ])

    return f"""
    <html>
    <head>
        <title>Metrics Report</title>
    </head>
    <body>
        <h1>Performance Metrics</h1>
        <table border="1">
            {rows}
        </table>
    </body>
    </html>
    """
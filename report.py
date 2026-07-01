import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio


# =====================================================
# MAIN REPORT GENERATOR (HTML DASHBOARD)
# =====================================================

def generate_html_report(bt_df: pd.DataFrame, metrics: dict, output="report.html"):

    bt_df = bt_df.copy()

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=(
            "Equity Curve",
            "Strategy Returns",
            "Positions",
            "Drawdown Proxy"
        )
    )

    # -------------------------------------------------
    # Equity curve
    # -------------------------------------------------

    fig.add_trace(
        go.Scatter(
            y=bt_df["equity_curve"],
            name="Equity",
            line=dict(width=2)
        ),
        row=1, col=1
    )

    # -------------------------------------------------
    # Strategy returns
    # -------------------------------------------------

    fig.add_trace(
        go.Scatter(
            y=bt_df["strategy_return"],
            name="Returns",
            line=dict(width=1)
        ),
        row=2, col=1
    )

    # -------------------------------------------------
    # Positions
    # -------------------------------------------------

    fig.add_trace(
        go.Scatter(
            y=bt_df["position"],
            name="Position",
            line=dict(width=1)
        ),
        row=3, col=1
    )

    # -------------------------------------------------
    # Drawdown
    # -------------------------------------------------

    equity = bt_df["equity_curve"].values
    peak = pd.Series(equity).cummax().values
    drawdown = equity - peak

    fig.add_trace(
        go.Scatter(
            y=drawdown,
            name="Drawdown",
            line=dict(width=1, color="red")
        ),
        row=4, col=1
    )

    # -------------------------------------------------
    # Layout
    # -------------------------------------------------

    fig.update_layout(
        title=(
            f"Backtest Report | "
            f"Sharpe: {metrics['sharpe']:.2f} | "
            f"Sortino: {metrics['sortino']:.2f} | "
            f"Max DD: {metrics['max_drawdown']:.4f}"
        ),
        height=1000,
        template="plotly_dark",
        showlegend=False
    )

    fig.write_html(output, include_plotlyjs="cdn")

    print(f"[REPORT SAVED] {output}")

def metrics_html(metrics: dict):

    rows = "".join([
        f"<tr><td>{k}</td><td>{v:.6f}</td></tr>"
        for k, v in metrics.items()
    ])

    return f"""
    <html>
    <head>
        <style>
            body {{
                font-family: Arial;
                background: #111;
                color: #eee;
                padding: 20px;
            }}
            table {{
                border-collapse: collapse;
                width: 400px;
            }}
            td {{
                border: 1px solid #333;
                padding: 8px;
            }}
        </style>
    </head>
    <body>
        <h2>Performance Metrics</h2>
        <table>
            {rows}
        </table>
    </body>
    </html>
    """
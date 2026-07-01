import os
import base64
import pandas as pd
import matplotlib.pyplot as plt


# =====================================================
# PLOT EQUITY CURVE
# =====================================================

def plot_equity(df):
    plt.figure()
    plt.plot(df["equity_curve"])
    plt.title("Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Return")

    path = "equity.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()

    return path


# =====================================================
# CONVERT IMAGE TO BASE64
# =====================================================

def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# =====================================================
# GENERATE HTML REPORT
# =====================================================

def generate_report(df, metrics, output_file="backtest_report.html"):

    img_path = plot_equity(df)
    img_b64 = img_to_base64(img_path)

    html = f"""
    <html>
    <head>
        <title>Backtest Report</title>
        <style>
            body {{
                font-family: Arial;
                margin: 40px;
            }}
            .metric {{
                margin: 10px 0;
                font-size: 16px;
            }}
            img {{
                width: 900px;
                border: 1px solid #ddd;
            }}
        </style>
    </head>

    <body>

        <h1>Trading Strategy Backtest Report</h1>

        <h2>Metrics</h2>
        <div class="metric">Sharpe: {metrics['sharpe']:.3f}</div>
        <div class="metric">Total Return: {metrics['total_return']:.3f}</div>
        <div class="metric">Max Drawdown: {metrics['max_drawdown']:.3f}</div>
        <div class="metric">Win Rate: {metrics['win_rate']:.3f}</div>

        <h2>Equity Curve</h2>
        <img src="data:image/png;base64,{img_b64}" />

    </body>
    </html>
    """

    with open(output_file, "w") as f:
        f.write(html)

    print(f"[REPORT SAVED] {output_file}")
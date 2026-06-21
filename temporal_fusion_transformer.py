"""
Instead of predicting next tick return, pretdict 15-minute risk-adjusted EURUSD return using
FX, SPX, EuroStoxx, VIX, US 2Y yield, German 2Y yield, DXY, session information
Then,
Long: predicted_sharpe > +1
Short: predicted_sharpe < -1
Flat: otherwise
Required data:
FX:
* EURUSD bid/ask
Equities:
* SPX (^GSPC)
* EuroStoxx (^STOXX50E)
Vol:
* VIX (^VIX)
Dollar:
* DXY (DX-Y.NYB)
Rates:
* US 2Y Treasury
* German 2Y Bund
Based on https://arxiv.org/abs/1912.09363 Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

def clean_time_index(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"])
    return df.sort_values("timestamp")

# ======================================
# FEATURE ENGINEERING
# ======================================

def create_features(df):

    df = df.copy()

    df["mid"] = (df["bid"] + df["ask"]) / 2

    df["return"] = df["mid"].pct_change()

    feature_cols = []

    # FX momentum
    for lag in [1,2,3,5,10]:
        col = f"ret_lag_{lag}"
        df[col] = df["return"].shift(lag)
        feature_cols.append(col)

    # volatility
    df["vol_20"] = df["return"].rolling(20).std()
    df["vol_100"] = df["return"].rolling(100).std()

    feature_cols += [
        "vol_20",
        "vol_100"
    ]

    # spread
    df["spread"] = df["ask"] - df["bid"]

    feature_cols.append("spread")

    # equities

    df["spx_ret"] = df["spx"].pct_change()
    df["eu_ret"] = df["eustoxx"].pct_change()

    df["equity_relative"] = (
            df["eu_ret"] - df["spx_ret"]
    )

    feature_cols += [
        "spx_ret",
        "eu_ret",
        "equity_relative"
    ]

    # volatility index

    df["vix_change"] = df["vix"].pct_change()

    feature_cols.append("vix_change")

    # rates

    df["yield_spread"] = (
            df["bund2y"] - df["ust2y"]
    )

    feature_cols.append(
        "yield_spread"
    )

    # dollar index

    df["dxy_ret"] = df["dxy"].pct_change()

    feature_cols.append(
        "dxy_ret"
    )

    # session features

    hour = pd.to_datetime(
        df["timestamp"]
    ).dt.hour

    df["hour_sin"] = np.sin(
        2*np.pi*hour/24
    )

    df["hour_cos"] = np.cos(
        2*np.pi*hour/24
    )

    feature_cols += [
        "hour_sin",
        "hour_cos"
    ]

    # ======================================
    # TARGET
    # ======================================

    future_return = (
            df["mid"].shift(-15)
            / df["mid"]
            - 1
    )

    future_vol = (
        df["return"]
        .rolling(60)
        .std()
    )

    df["target"] = (
            future_return / future_vol
    )

    dataset = (
        df[feature_cols + ["target"]]
        .replace([np.inf,-np.inf],np.nan)
        .dropna()
        .reset_index(drop=True)
    )

    X = dataset[feature_cols]

    y = dataset["target"]

    return X, y


# ======================================
# SEQUENCE BUILDER
# ======================================

def build_sequences(X, y, seq_len=120):

    xs = []
    ys = []

    values = X.values.astype(np.float32)

    target = y.values.astype(np.float32)

    for i in range(seq_len, len(X)):

        xs.append(
            values[i-seq_len:i]
        )

        ys.append(
            target[i]
        )

    return (
        np.array(xs),
        np.array(ys)
    )


# ======================================
# TFT-INSPIRED MODEL
# ======================================

class AlphaTransformer(nn.Module):

    def __init__(
            self,
            input_dim,
            d_model=64,
            nhead=4,
            layers=3
    ):

        super().__init__()

        self.input_proj = nn.Linear(
            input_dim,
            d_model
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                batch_first=True
            )
        )

        self.encoder = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=layers
            )
        )

        self.fc = nn.Sequential(
            nn.Linear(d_model,32),
            nn.ReLU(),
            nn.Linear(32,1)
        )

    def forward(self,x):

        x = self.input_proj(x)

        x = self.encoder(x)

        x = x[:,-1,:]

        return self.fc(x)


# ======================================
# TRAINING
# ======================================

def train_model(X,y):

    X_seq,y_seq = build_sequences(
        X,
        y,
        seq_len=120
    )

    X_tensor = torch.tensor(
        X_seq,
        dtype=torch.float32
    )

    y_tensor = torch.tensor(
        y_seq.reshape(-1,1),
        dtype=torch.float32
    )

    model = AlphaTransformer(
        input_dim=X.shape[1]
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4
    )

    loss_fn = nn.HuberLoss()

    for epoch in range(20):

        optimizer.zero_grad()

        preds = model(X_tensor)

        loss = loss_fn(
            preds,
            y_tensor
        )

        loss.backward()

        optimizer.step()

        print(
            epoch,
            loss.item()
        )

    return model


# ======================================
# SIGNAL GENERATION
# ======================================

def generate_signals(
        model,
        X,
        seq_len=120
):

    X_seq,_ = build_sequences(
        X,
        pd.Series(np.zeros(len(X))),
        seq_len
    )

    with torch.no_grad():

        pred = model(
            torch.tensor(
                X_seq,
                dtype=torch.float32
            )
        )

    pred = pred.numpy().flatten()

    signal = np.where(
        pred > 1,
        1,
        np.where(
            pred < -1,
            -1,
            0
        )
    )

    return signal


# ======================================
# POSITION SIZING
# ======================================

def position_size(
        signal,
        realized_vol
):

    target_vol = 0.10

    return (
            signal
            * target_vol
            / np.maximum(
        realized_vol,
        1e-6
    )
    )

def run(
        fx_path: str = "data/fx.parquet",
        macro_path: str = "macro_data/macro_2024-01-01_2026-06-01.parquet",
        seq_len: int = 120,
        epochs: int = 20,
        lr: float = 1e-4,
):
    """
    End-to-end training + signal generation pipeline.
    Designed to be called from __main__.py
    """

    print("[RUN] Loading datasets...")

    fx = pd.read_parquet(fx_path)
    macro = pd.read_parquet(macro_path)

    # -------------------------------------------------
    # ALIGN DATA
    # -------------------------------------------------

    fx["timestamp"] = pd.to_datetime(fx["timestamp"], format="ISO8601", errors="coerce", utc=True)
    macro["timestamp"] = pd.to_datetime(macro["timestamp"], format="ISO8601", errors="coerce", utc=True)

    fx = fx.sort_values("timestamp")
    macro = macro.sort_values("timestamp")
    fx = clean_time_index(fx)
    macro = clean_time_index(macro)

    df = pd.merge_asof(
        fx,
        macro,
        on="timestamp",
        direction="backward"
    )

    df = df.dropna()

    # -------------------------------------------------
    # FEATURE ENGINEERING
    # -------------------------------------------------

    print("[RUN] Creating features...")

    X, y = create_features(df)

    print(f"[RUN] Dataset size: {len(X):,} rows")

    # -------------------------------------------------
    # TRAIN MODEL
    # -------------------------------------------------

    print("[RUN] Training model...")

    X_seq, y_seq = build_sequences(
        X,
        y,
        seq_len=seq_len
    )

    X_tensor = torch.tensor(
        X_seq,
        dtype=torch.float32
    )

    y_tensor = torch.tensor(
        y_seq.reshape(-1, 1),
        dtype=torch.float32
    )

    model = AlphaTransformer(
        input_dim=X.shape[1]
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr
    )

    loss_fn = nn.HuberLoss()

    for epoch in range(epochs):

        model.train()

        optimizer.zero_grad()

        preds = model(X_tensor)

        loss = loss_fn(preds, y_tensor)

        loss.backward()

        optimizer.step()

        print(f"[EPOCH {epoch}] loss={loss.item():.6f}")

    # -------------------------------------------------
    # SIGNAL GENERATION
    # -------------------------------------------------

    print("[RUN] Generating signals...")

    signals = generate_signals(
        model,
        X,
        seq_len=seq_len
    )

    # -------------------------------------------------
    # REALIZED VOL FOR POSITION SIZING
    # -------------------------------------------------

    realized_vol = (
        df["mid"]
        .pct_change()
        .rolling(60)
        .std()
        .fillna(1e-6)
        .values[-len(signals):]
    )

    positions = position_size(
        signals,
        realized_vol
    )

    # -------------------------------------------------
    # OUTPUT
    # -------------------------------------------------

    result = pd.DataFrame({
        "signal": signals,
        "position": positions
    })

    print("[RUN] Done.")

    return model, result

if __name__ == "__main__":
    print("Use run() from __main__.py instead of executing directly")
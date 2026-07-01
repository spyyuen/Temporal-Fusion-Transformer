import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# =====================================================
# TIME NORMALISATION (CRITICAL FIX)
# =====================================================

def _normalize_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce"
    )

    # FORCE CONSISTENT RESOLUTION (fix ms vs us crash)
    df["timestamp"] = df["timestamp"].dt.floor("min")

    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp")

    return df


# =====================================================
# SAFE DATA LOADING
# =====================================================

def load_data(fx_path: str, macro_path: str, max_rows: int = 2_000_000):

    fx = pd.read_parquet(fx_path)
    macro = pd.read_parquet(macro_path)

    fx = _normalize_timestamp(fx)
    macro = _normalize_timestamp(macro)

    # -------------------------------------------------
    # Downsample for safety
    # -------------------------------------------------
    if len(fx) > max_rows:
        fx = fx.iloc[:: max(1, len(fx) // max_rows)]

    if len(macro) > max_rows:
        macro = macro.iloc[:: max(1, len(macro) // max_rows)]

    # -------------------------------------------------
    # CRITICAL: ensure identical dtype before merge_asof
    # -------------------------------------------------
    fx["timestamp"] = fx["timestamp"].astype("datetime64[ns, UTC]")
    macro["timestamp"] = macro["timestamp"].astype("datetime64[ns, UTC]")

    # -------------------------------------------------
    # MERGE
    # -------------------------------------------------
    df = pd.merge_asof(
        fx.sort_values("timestamp"),
        macro.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta("1D"),
    )

    df = df.dropna(subset=["timestamp"])

    return df


# =====================================================
# FEATURE ENGINEERING
# =====================================================

def create_features(df: pd.DataFrame):

    df = df.copy()

    # -----------------------------
    # MID PRICE (robust)
    # -----------------------------
    if "bid" in df.columns and "ask" in df.columns:
        df["mid"] = (df["bid"] + df["ask"]) / 2
    elif "close" in df.columns:
        df["mid"] = df["close"]
    else:
        raise ValueError("No price column found")

    df["return"] = df["mid"].pct_change()

    feature_cols = []

    # -----------------------------
    # LAGS
    # -----------------------------
    for lag in [1, 2, 3, 5, 10]:
        col = f"ret_lag_{lag}"
        df[col] = df["return"].shift(lag)
        feature_cols.append(col)

    # -----------------------------
    # VOLATILITY
    # -----------------------------
    df["vol_20"] = df["return"].rolling(20).std()
    df["vol_100"] = df["return"].rolling(100).std()
    feature_cols += ["vol_20", "vol_100"]

    # -----------------------------
    # SPREAD
    # -----------------------------
    if "bid" in df.columns and "ask" in df.columns:
        df["spread"] = df["ask"] - df["bid"]
    else:
        df["spread"] = 0

    feature_cols.append("spread")

    # -----------------------------
    # EQUITIES SAFE
    # -----------------------------
    df["spx_ret"] = df.get("spx", pd.Series(0)).pct_change().fillna(0)
    df["eu_ret"] = df.get("eustoxx", pd.Series(0)).pct_change().fillna(0)

    df["equity_relative"] = df["eu_ret"] - df["spx_ret"]

    feature_cols += ["spx_ret", "eu_ret", "equity_relative"]

    # -----------------------------
    # VIX SAFE
    # -----------------------------
    df["vix_change"] = df.get("vix", pd.Series(0)).pct_change().fillna(0)
    feature_cols.append("vix_change")

    # -----------------------------
    # RATES SAFE
    # -----------------------------
    df["yield_spread"] = df.get("yield_curve", 0)
    feature_cols.append("yield_spread")

    # -----------------------------
    # DXY SAFE
    # -----------------------------
    df["dxy_ret"] = df.get("dxy", pd.Series(0)).pct_change().fillna(0)
    feature_cols.append("dxy_ret")

    # -----------------------------
    # TIME FEATURES
    # -----------------------------
    hour = df["timestamp"].dt.hour.fillna(0)

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    feature_cols += ["hour_sin", "hour_cos"]

    # =================================================
    # TARGET
    # =================================================
    future_return = df["mid"].shift(-15) / df["mid"] - 1
    vol = df["return"].rolling(60).std()

    df["target"] = future_return / (vol + 1e-6)

    # =================================================
    # FINAL CLEANING (CRITICAL FIX)
    # =================================================
    dataset = df[feature_cols + ["target"]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    if len(dataset) == 0:
        raise ValueError(
            "Dataset collapsed to 0 rows. "
            "Check macro coverage or merge alignment."
        )

    return dataset[feature_cols], dataset["target"]


# =====================================================
# SEQUENCES
# =====================================================

def build_sequences(X, y, seq_len=120):

    X = X.values.astype(np.float32)
    y = y.values.astype(np.float32)

    xs, ys = [], []

    for i in range(seq_len, len(X)):
        xs.append(X[i - seq_len:i])
        ys.append(y[i])

    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


# =====================================================
# MODEL
# =====================================================

class AlphaTransformer(nn.Module):

    def __init__(self, input_dim, d_model=64, nhead=4, layers=2):

        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers
        )

        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.encoder(x)
        x = x[:, -1, :]
        return self.head(x)


# =====================================================
# TRAIN
# =====================================================

def train_model(X, y, seq_len=120, epochs=20):

    X_seq, y_seq = build_sequences(X, y, seq_len)

    if len(X_seq) == 0:
        raise ValueError("No sequences generated")

    X_tensor = torch.tensor(X_seq, dtype=torch.float32)
    y_tensor = torch.tensor(y_seq.reshape(-1, 1), dtype=torch.float32)

    model = AlphaTransformer(input_dim=X.shape[1])

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.HuberLoss()

    for epoch in range(epochs):

        opt.zero_grad()

        preds = model(X_tensor)
        loss = loss_fn(preds, y_tensor)

        loss.backward()
        opt.step()

        print(epoch, loss.item())

    return model


# =====================================================
# SIGNALS
# =====================================================

def generate_signals(model, X, seq_len=120):

    X_seq, _ = build_sequences(X, pd.Series(np.zeros(len(X))), seq_len)

    with torch.no_grad():
        preds = model(torch.tensor(X_seq)).numpy().flatten()

    return np.where(preds > 1, 1, np.where(preds < -1, -1, 0))


# =====================================================
# ENTRYPOINT
# =====================================================

def train_tft(fx_path, macro_path, seq_len=120, epochs=20):

    print("[RUN] Loading datasets...")

    df = load_data(fx_path, macro_path)

    print("[RUN] Building features + target...")

    X, y = create_features(df)

    print(f"[RUN] Dataset size: {len(X):,}")

    if len(X) == 0:
        raise ValueError("Empty dataset after feature engineering")

    print("[RUN] Training model...")

    print("[DEBUG] X shape:", X.shape)
    print("[DEBUG] y shape:", y.shape)

    model = train_model(X, y, seq_len=seq_len, epochs=epochs)

    # after training
    model = train_model(X, y, seq_len=seq_len, epochs=epochs)

    # -----------------------------
    # generate in-sample predictions
    # -----------------------------
    model.eval()

    X_seq, _ = build_sequences(X, y, seq_len=seq_len)

    X_tensor = torch.tensor(X_seq, dtype=torch.float32)

    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy().flatten()

    # align lengths (important for backtest)
    aligned_returns = y.iloc[-len(preds):].values
    aligned_features = X.iloc[-len(preds):]

    return model, {
        "predictions": preds,
        "returns": aligned_returns,
        "features": aligned_features
    }

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# =====================================================
# SAFE DATA LOADING
# =====================================================

def load_data(fx_path: str, macro_path: str, max_rows: int = 2_000_000):

    fx = pd.read_parquet(fx_path)
    fx = (
        fx.set_index("timestamp")
        .resample("1min")
        .agg({
            "bid": "last",
            "ask": "last"
        })
        .dropna()
        .reset_index()
    )
    # Keep every 100th tick
    #fx = fx.iloc[::100].reset_index(drop=True)

    macro = pd.read_parquet(macro_path)

    # -------------------------------------------------
    # Fix timestamps safely (your crash fix)
    # -------------------------------------------------

    fx["timestamp"] = pd.to_datetime(
        fx["timestamp"],
        utc=True,
        format="mixed",
        errors="coerce",
    )

    macro["timestamp"] = pd.to_datetime(
        macro["timestamp"],
        utc=True,
        format="mixed",
        errors="coerce",
    )

    fx = fx.dropna(subset=["timestamp"])
    macro = macro.dropna(subset=["timestamp"])

    fx = fx.sort_values("timestamp")
    macro = macro.sort_values("timestamp")

    # -------------------------------------------------
    # IMPORTANT: downsample to avoid OOM (SIGKILL fix)
    # -------------------------------------------------

    if len(fx) > max_rows:
        fx = fx.iloc[:: max(1, len(fx) // max_rows)]

    if len(macro) > max_rows:
        macro = macro.iloc[:: max(1, len(macro) // max_rows)]

    # -------------------------------------------------
    # merge_asof (fixed + safe)
    # -------------------------------------------------

    df = pd.merge_asof(
        fx,
        macro,
        on="timestamp",
        direction="backward",
        allow_exact_matches=True,
    )

    df = df.dropna(subset=["timestamp"])

    return df


# =====================================================
# FEATURE ENGINEERING (FIXED SAFE VERSION)
# =====================================================

def create_features(df: pd.DataFrame):

    df = df.copy()

    # -------------------------------------------------
    # FX mid
    # -------------------------------------------------

    if "bid" in df.columns and "ask" in df.columns:
        df["mid"] = (df["bid"] + df["ask"]) / 2
    else:
        df["mid"] = df.get("close", 0)

    df["return"] = df["mid"].pct_change()

    feature_cols = []

    # -------------------------------------------------
    # FX LAGS
    # -------------------------------------------------

    for lag in [1, 2, 3, 5, 10]:
        col = f"ret_lag_{lag}"
        df[col] = df["return"].shift(lag)
        feature_cols.append(col)

    # -------------------------------------------------
    # VOL
    # -------------------------------------------------

    df["vol_20"] = df["return"].rolling(20).std()
    df["vol_100"] = df["return"].rolling(100).std()

    feature_cols += ["vol_20", "vol_100"]

    # -------------------------------------------------
    # SPREAD (safe)
    # -------------------------------------------------

    df["spread"] = df.get("ask", 0) - df.get("bid", 0)
    feature_cols.append("spread")

    # -------------------------------------------------
    # EQUITIES (SAFE GUARDS)
    # -------------------------------------------------

    if "spx" in df.columns:
        df["spx_ret"] = df["spx"].pct_change()
    else:
        df["spx_ret"] = 0

    if "eustoxx" in df.columns:
        df["eu_ret"] = df["eustoxx"].pct_change()
    else:
        df["eu_ret"] = 0

    df["equity_relative"] = df["eu_ret"] - df["spx_ret"]

    feature_cols += ["spx_ret", "eu_ret", "equity_relative"]

    # -------------------------------------------------
    # VIX
    # -------------------------------------------------

    df["vix_change"] = df.get("vix", pd.Series(0)).pct_change()
    feature_cols.append("vix_change")

    # -------------------------------------------------
    # RATES (FIXED bund2y crash)
    # -------------------------------------------------

    # SAFE fallback: only use what exists
    df["ust2y"] = df.get("us2y", 0)

    df["yield_spread"] = df.get("yield_curve", 0)
    feature_cols.append("yield_spread")

    # -------------------------------------------------
    # DXY
    # -------------------------------------------------

    df["dxy_ret"] = df.get("dxy", pd.Series(0)).pct_change()
    feature_cols.append("dxy_ret")

    # -------------------------------------------------
    # TIME FEATURES (FIXED timestamp handling)
    # -------------------------------------------------

    hour = pd.to_datetime(df["timestamp"], errors="coerce").dt.hour.fillna(0)

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    feature_cols += ["hour_sin", "hour_cos"]

    # =================================================
    # TARGET (STABLE)
    # =================================================

    future_return = df["mid"].shift(-15) / df["mid"] - 1
    future_vol = df["return"].rolling(60).std()

    df["target"] = future_return / (future_vol + 1e-6)

    dataset = df[feature_cols + ["target"]].replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    return dataset[feature_cols], dataset["target"]


# =====================================================
# SEQUENCES (MEMORY SAFE)
# =====================================================

def build_sequences(X, y, seq_len=120, max_samples=200_000):

    values = X.values.astype(np.float32)
    target = y.values.astype(np.float32)

    xs, ys = [], []

    start = max(0, len(X) - max_samples)

    for i in range(start + seq_len, len(X)):

        xs.append(values[i - seq_len:i])
        ys.append(target[i])

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
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
        )

        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):

        x = self.input_proj(x)
        x = self.encoder(x)
        x = x[:, -1, :]
        return self.head(x)


# =====================================================
# TRAIN
# =====================================================

def train_model(X, y, epochs=10):

    X_seq, y_seq = build_sequences(X, y)

    X_tensor = torch.tensor(X_seq)
    y_tensor = torch.tensor(y_seq).unsqueeze(-1)

    model = AlphaTransformer(X.shape[1])

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.HuberLoss()

    model.train()

    for epoch in range(epochs):

        opt.zero_grad()

        pred = model(X_tensor)

        loss = loss_fn(pred, y_tensor)

        loss.backward()
        opt.step()

        print(f"epoch={epoch} loss={loss.item():.6f}")

    return model


# =====================================================
# SIGNALS
# =====================================================

def generate_signals(model, X, seq_len=120):

    X_seq, _ = build_sequences(
        X,
        pd.Series(np.zeros(len(X))),
        seq_len,
    )

    with torch.no_grad():
        pred = model(torch.tensor(X_seq)).numpy().flatten()

    return np.where(pred > 1, 1, np.where(pred < -1, -1, 0))


# =====================================================
# RUN ENTRYPOINT (USED BY __main__.py)
# =====================================================
def run(fx_path, macro_path, seq_len=120, epochs=20):

    import pandas as pd
    from dataset import build_dataset
    from model import train_model   # or wherever your model is

    print("[RUN] Loading datasets...")

    fx = pd.read_parquet(fx_path)
    macro = pd.read_parquet(macro_path)

    print("[RUN] Building features + target...")

    X, y = build_dataset(fx, macro)

    print(f"[RUN] Dataset size: {len(X):,} rows")

    # -----------------------------
    # SAFETY LIMIT (IMPORTANT)
    # -----------------------------
    MAX_ROWS = 1_000_000

    if len(X) > MAX_ROWS:
        print(f"[WARN] Downsampling from {len(X):,} → {MAX_ROWS:,}")
        X = X.iloc[-MAX_ROWS:]
        y = y.iloc[-MAX_ROWS:]

    print("[RUN] Training model...")

    model = train_model(X, y, seq_len=seq_len, epochs=epochs)

    return model, X
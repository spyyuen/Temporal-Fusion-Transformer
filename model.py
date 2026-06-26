import torch
import torch.nn as nn


# =====================================================
# SMALL TFT-STYLE ENCODER (STABLE VERSION)
# =====================================================

class AlphaTransformer(nn.Module):
    """
    Lightweight TFT-inspired model designed for:
    - large datasets
    - streaming batches
    - stable GPU memory usage
    """

    def __init__(
            self,
            input_dim: int,
            d_model: int = 64,
            nhead: int = 4,
            num_layers: int = 2,
            dropout: float = 0.1
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        """
        x: [batch, seq_len, features]
        """

        x = self.input_proj(x)

        x = self.encoder(x)

        # take last timestep (causal assumption)
        x = x[:, -1, :]

        return self.head(x)


# =====================================================
# SAFE TRAIN LOOP (BATCHED - NO MEMORY BLOWUPS)
# =====================================================

def train_model(
        dataset,
        input_dim: int,
        batch_size: int = 256,
        epochs: int = 5,
        lr: float = 1e-4,
        device: str = None
):

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # IMPORTANT: avoids RAM explosion on Mac
        pin_memory=False
    )

    model = AlphaTransformer(input_dim=input_dim).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()

    model.train()

    for epoch in range(epochs):

        total_loss = 0.0

        for x, y in loader:

            x = x.to(device)
            y = y.to(device).unsqueeze(-1)

            optimizer.zero_grad()

            preds = model(x)

            loss = loss_fn(preds, y)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        print(f"[EPOCH {epoch+1}] loss={avg_loss:.6f}")

    return model


# =====================================================
# INFERENCE / SIGNAL GENERATION
# =====================================================

def generate_signals(model, dataset, batch_size=512, device=None):

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    model.eval()

    preds_all = []

    with torch.no_grad():

        for x, _ in loader:

            x = x.to(device)

            preds = model(x)

            preds_all.append(preds.cpu())

    preds = torch.cat(preds_all).numpy().flatten()

    # trading rule
    signal = (preds > 1).astype(int) - (preds < -1).astype(int)

    return signal
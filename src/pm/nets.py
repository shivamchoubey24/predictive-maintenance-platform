"""Small PyTorch models (CPU friendly): GRU for RUL and an MLP autoencoder."""
import copy

import numpy as np
import torch
from torch import nn

from .config import SEED
from .numpy_ae import ae_error, ae_forward  # noqa: F401  (re-exported)


def seed_all(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


class GRURegressor(nn.Module):
    def __init__(self, n_features: int, hidden: int = 32):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def predict_in_batches(model, X: np.ndarray, bs: int = 2048) -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            outs.append(model(torch.from_numpy(X[i:i + bs])).numpy())
    return np.concatenate(outs)


def train_gru(Xtr, ytr, Xva, yva, epochs=30, lr=3e-3, bs=256, patience=6, hidden=32):
    """Targets must be scaled to ~[0,1]. Early-stops on validation MSE, returns best model."""
    seed_all()
    torch.set_num_threads(2)
    model = GRURegressor(Xtr.shape[2], hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr.astype(np.float32))
    best, best_state, bad, hist = np.inf, None, 0, []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(Xt[b]), yt[b])
            loss.backward()
            opt.step()
        val = float(np.mean((predict_in_batches(model, Xva) - yva) ** 2))
        hist.append(val)
        if val < best - 1e-6:
            best, best_state, bad = val, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, hist


def train_autoencoder(X: np.ndarray, epochs=80, lr=3e-3, bs=256, dims=(14, 8, 4)):
    """MLP autoencoder trained on HEALTHY data only. Returns numpy weights so the
    serving layer needs no torch."""
    seed_all()
    torch.set_num_threads(2)
    d_in, d_h, d_z = X.shape[1], dims[1], dims[2]
    net = nn.Sequential(nn.Linear(d_in, d_h), nn.ReLU(), nn.Linear(d_h, d_z), nn.ReLU(),
                        nn.Linear(d_z, d_h), nn.ReLU(), nn.Linear(d_h, d_in))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Xt = torch.from_numpy(X.astype(np.float32))
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            b = Xt[perm[i:i + bs]]
            opt.zero_grad()
            nn.functional.mse_loss(net(b), b).backward()
            opt.step()
    layers = [m for m in net if isinstance(m, nn.Linear)]
    return [(m.weight.detach().numpy().copy(), m.bias.detach().numpy().copy()) for m in layers]

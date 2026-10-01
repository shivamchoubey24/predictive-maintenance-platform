"""Feature engineering. Every feature at cycle t uses ONLY cycles <= t (no look-ahead)."""
import numpy as np
import pandas as pd

from .config import DELTA, ROLL, SENSORS, WINDOW


def feature_names() -> list:
    names = ["cycle"]
    for s in SENSORS:
        names += [s, f"{s}_mean{ROLL}", f"{s}_std{ROLL}", f"{s}_d{DELTA}"]
    return names


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-engine rolling mean/std and lagged delta for each informative sensor."""
    parts = []
    for unit, g in df.groupby("unit", sort=True):
        g = g.sort_values("cycle")
        cols = {"unit": np.full(len(g), unit), "cycle": g["cycle"].to_numpy()}
        for s in SENSORS:
            x = g[s].astype(float)
            cols[s] = x.to_numpy()
            cols[f"{s}_mean{ROLL}"] = x.rolling(ROLL, min_periods=1).mean().to_numpy()
            cols[f"{s}_std{ROLL}"] = x.rolling(ROLL, min_periods=2).std().fillna(0).to_numpy()
            cols[f"{s}_d{DELTA}"] = x.diff(DELTA).fillna(0).to_numpy()
        parts.append(pd.DataFrame(cols, index=g.index))
    return pd.concat(parts).sort_index()


def fit_minmax(df: pd.DataFrame):
    lo = df[SENSORS].min().to_numpy(float)
    hi = df[SENSORS].max().to_numpy(float)
    return lo, np.where(hi - lo == 0, 1.0, hi - lo)


def scale(df: pd.DataFrame, lo, rng) -> np.ndarray:
    return ((df[SENSORS].to_numpy(float) - lo) / rng).astype(np.float32)


def make_windows(df: pd.DataFrame, lo, rng, window: int = WINDOW, last_only: bool = False):
    """Sliding windows (n, window, n_sensors); early cycles are padded by repeating
    the first reading. Returns (X, row_index_of_df)."""
    X_all, idx_all = [], []
    for _, g in df.groupby("unit", sort=True):
        g = g.sort_values("cycle")
        x = scale(g, lo, rng)
        padded = np.vstack([np.repeat(x[:1], window - 1, axis=0), x])
        w = np.lib.stride_tricks.sliding_window_view(padded, window, axis=0)  # (n, f, window)
        w = np.transpose(w, (0, 2, 1)).copy()
        if last_only:
            w, ids = w[-1:], g.index.to_numpy()[-1:]
        else:
            ids = g.index.to_numpy()
        X_all.append(w)
        idx_all.append(ids)
    return np.concatenate(X_all), np.concatenate(idx_all)

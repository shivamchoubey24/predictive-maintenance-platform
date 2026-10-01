"""Data loading for C-MAPSS (FD001) and AI4I 2020."""
import numpy as np
import pandas as pd

from .config import RAW, RUL_CAP, SUBSET

CMAPSS_COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]


def load_cmapss(subset: str = SUBSET):
    """Return (train, test). Both carry `rul` (true) and `rul_capped` columns.

    Train: run-to-failure, RUL = last_cycle - cycle.
    Test: trajectories are truncated before failure; the official RUL file gives the
    remaining life at the *last* observed cycle of every engine.
    """
    kw = dict(sep=r"\s+", header=None, names=CMAPSS_COLS)
    train = pd.read_csv(RAW / f"train_{subset}.txt", **kw).reset_index(drop=True)
    test = pd.read_csv(RAW / f"test_{subset}.txt", **kw).reset_index(drop=True)
    rul_last = pd.read_csv(RAW / f"RUL_{subset}.txt", header=None)[0].to_numpy()

    train["rul"] = train.groupby("unit")["cycle"].transform("max") - train["cycle"]
    last = test.groupby("unit")["cycle"].transform("max")
    offset = test["unit"].map(lambda u: rul_last[u - 1])
    test["rul"] = offset + (last - test["cycle"])
    for df in (train, test):
        df["rul_capped"] = df["rul"].clip(upper=RUL_CAP)
    return train, test


def split_units(units, val_frac: float = 0.2, seed: int = 42):
    """Split by ENGINE (never by row) to avoid leakage between train and validation."""
    rng = np.random.RandomState(seed)
    units = np.array(sorted(units))
    val = rng.choice(units, size=int(round(len(units) * val_frac)), replace=False)
    tr = np.setdiff1d(units, val)
    return tr, np.sort(val)


def last_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Last observed cycle of every engine (the official C-MAPSS test snapshot)."""
    return df.sort_values(["unit", "cycle"]).groupby("unit", as_index=False).tail(1)


AI4I_FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]


def load_ai4i() -> pd.DataFrame:
    """AI4I 2020 with clean column names.

    The five failure-mode flags (TWF/HDF/PWF/OSF/RNF) DEFINE the target
    (Machine failure = OR of them) - they are kept only so a leakage demo can be run.
    """
    df = pd.read_csv(RAW / "ai4i2020.csv")
    df = df.rename(columns={
        "Type": "type", "Air temperature [K]": "air_temp",
        "Process temperature [K]": "process_temp", "Rotational speed [rpm]": "rot_speed",
        "Torque [Nm]": "torque", "Tool wear [min]": "tool_wear", "Machine failure": "failure",
    })
    return df.drop(columns=["UDI", "Product ID"])


def ai4i_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw + physically motivated features (no target leakage)."""
    out = pd.DataFrame({
        "air_temp": df["air_temp"], "process_temp": df["process_temp"],
        "rot_speed": df["rot_speed"], "torque": df["torque"], "tool_wear": df["tool_wear"],
    })
    out["temp_diff"] = out["process_temp"] - out["air_temp"]           # heat dissipation
    out["power_w"] = out["torque"] * out["rot_speed"] * 2 * np.pi / 60  # mechanical power
    out["wear_x_torque"] = out["tool_wear"] * out["torque"]            # overstrain proxy
    for t in ("L", "M", "H"):
        out[f"type_{t}"] = (df["type"] == t).astype(int)
    return out

"""Phase 4a: unsupervised anomaly detection on sensor streams (Isolation Forest + autoencoder)."""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

from .config import HEALTHY_CYCLES, HORIZON, SEED, SENSORS
from .data import last_rows
from .features import fit_minmax, scale
from .nets import train_autoencoder
from .numpy_ae import ae_error

SMOOTH = 5
CONSEC = 3


def _smooth(df, score):
    s = pd.Series(score, index=df.index)
    return s.groupby(df["unit"]).transform(lambda x: x.rolling(SMOOTH, min_periods=1).mean())


def first_alert_cycle(cycles, flags, consec=CONSEC):
    run = 0
    for c, f in zip(cycles, flags):
        run = run + 1 if f else 0
        if run >= consec:
            return int(c)
    return None


def run_anomaly(train, test, tr_units, va_units):
    healthy = train[train["unit"].isin(tr_units) & (train["cycle"] <= HEALTHY_CYCLES)]
    lo, rng = fit_minmax(healthy)
    Xh = scale(healthy, lo, rng)

    iforest = IsolationForest(n_estimators=200, random_state=SEED, n_jobs=2).fit(Xh)
    ae_w = train_autoencoder(Xh)

    def raw_scores(df):
        X = scale(df, lo, rng)
        return {"iforest": -iforest.score_samples(X), "autoencoder": ae_error(ae_w, X)}

    hs = raw_scores(healthy)
    thr = {k: float(np.percentile(v, 99)) for k, v in hs.items()}   # 99th pct of healthy

    val = train[train["unit"].isin(va_units)]
    te_last = last_rows(test)
    metrics = {}
    for name in ("iforest", "autoencoder"):
        sv = _smooth(val, raw_scores(val)[name] / thr[name])
        leads, early_false = [], 0
        for u, g in val.groupby("unit"):
            flags = (sv.loc[g.index] > 1).to_numpy()
            c = first_alert_cycle(g["cycle"], flags)
            if c is None:
                leads.append(np.nan)
            else:
                leads.append(int(g["cycle"].max() - c))
                early_false += int(c <= HEALTHY_CYCLES)
        leads = np.array(leads, float)
        st = _smooth(test, raw_scores(test)[name] / thr[name])
        y = (te_last["rul"] <= HORIZON).astype(int).to_numpy()
        auc = float(roc_auc_score(y, st.loc[te_last.index].to_numpy()))
        metrics[name] = dict(
            val_engines_alerted_pct=float(100 * np.mean(~np.isnan(leads))),
            val_median_lead_cycles=float(np.nanmedian(leads)) if np.any(~np.isnan(leads)) else None,
            val_false_alarms_in_healthy_phase=int(early_false),
            test_auc_vs_rul_le_30=auc)
    arts = dict(lo=lo, rng=rng, iforest=iforest, ae_weights=ae_w, thr=thr)
    return arts, metrics

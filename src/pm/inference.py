"""Serving layer: loads trained artifacts and scores engine histories / AI4I machines.
No torch needed: the autoencoder is evaluated with numpy, explanations use the
native TreeSHAP contributions built into LightGBM / XGBoost."""
import json

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from .config import ALERT_LEVELS, ARTIFACTS, RUL_CAP, SENSORS
from .data import ai4i_features
from .features import build_features, feature_names


def alert_level(rul: float, fail_prob: float = 0.0, fail_thr: float = 1.0) -> str:
    for name, bound in ALERT_LEVELS:
        if rul <= bound:
            return name
    return "Watch" if fail_prob >= fail_thr else "Healthy"


def group_contribs(contribs: np.ndarray, names: list, top: int = 5) -> list:
    """Sum per-feature TreeSHAP contributions into per-sensor impacts for one row."""
    agg = {}
    for c, n in zip(contribs[:-1], names):
        key = "cycle" if n == "cycle" else n.split("_")[0]
        agg[key] = agg.get(key, 0.0) + float(c)
    items = sorted(agg.items(), key=lambda kv: -abs(kv[1]))[:top]
    return [{"sensor": k, "impact": round(v, 3)} for k, v in items]


class Predictor:
    def __init__(self, art_dir=ARTIFACTS):
        self.meta = json.loads((art_dir / "meta.json").read_text())
        self.rul_model = joblib.load(art_dir / "rul_lgbm.joblib")
        self.clf = joblib.load(art_dir / "failure_xgb.joblib")
        self.anom = joblib.load(art_dir / "anomaly.joblib")
        self.ai4i = joblib.load(art_dir / "ai4i_xgb.joblib")
        self.names = feature_names()
        self.fail_thr = self.meta["failure_threshold"]

    # ---------- turbofan ----------
    def anomaly_index(self, feat: pd.DataFrame, smooth: int = 5) -> np.ndarray:
        a = self.anom
        X = (feat[SENSORS].to_numpy(float) - a["lo"]) / a["rng"]
        raw = -a["iforest"].score_samples(X) / a["thr"]["iforest"]
        s = pd.Series(raw, index=feat.index).groupby(feat["unit"]).transform(
            lambda x: x.rolling(smooth, min_periods=1).mean())
        return s.to_numpy()

    def score_features(self, feat: pd.DataFrame) -> pd.DataFrame:
        X = feat[self.names]
        rul = np.clip(self.rul_model.predict(X), 0, RUL_CAP)
        prob = self.clf.predict_proba(X)[:, 1]
        out = pd.DataFrame({"unit": feat["unit"].to_numpy(), "cycle": feat["cycle"].to_numpy(),
                            "pred_rul": rul, "fail_prob": prob,
                            "anomaly_index": self.anomaly_index(feat)}, index=feat.index)
        out["health_score"] = 100 * np.clip(out["pred_rul"], 0, RUL_CAP) / RUL_CAP
        out["alert"] = [alert_level(r, p, self.fail_thr) for r, p in zip(out["pred_rul"], prob)]
        return out

    def explain_rows(self, feat: pd.DataFrame, top: int = 5) -> list:
        X = feat[self.names]
        contribs = self.rul_model.predict(X, pred_contrib=True)   # (n, n_features + 1)
        return [group_contribs(c, self.names, top) for c in contribs]

    def score_history(self, history: pd.DataFrame, explain: bool = True) -> dict:
        """history: one engine, columns cycle + SENSORS. Scores the latest cycle."""
        h = history.copy()
        h["unit"] = 1
        h = h.sort_values("cycle").reset_index(drop=True)
        feat = build_features(h)
        scored = self.score_features(feat).iloc[-1]
        res = dict(cycle=int(scored["cycle"]), predicted_rul_cycles=float(scored["pred_rul"]),
                   health_score=float(scored["health_score"]), alert_level=scored["alert"],
                   failure_probability=float(scored["fail_prob"]),
                   failure_within_horizon=bool(scored["fail_prob"] >= self.fail_thr),
                   horizon_cycles=self.meta["horizon"],
                   anomaly_index=float(scored["anomaly_index"]),
                   anomaly_flag=bool(scored["anomaly_index"] > 1.0),
                   history_cycles_used=int(len(h)))
        if explain:
            res["top_drivers"] = self.explain_rows(feat.iloc[[-1]])[0]
        return res

    # ---------- AI4I machines ----------
    def score_machine(self, record: dict, explain: bool = True) -> dict:
        df = pd.DataFrame([record])
        X = ai4i_features(df)[self.meta["ai4i_features"]]
        p = float(self.ai4i.predict_proba(X)[:, 1][0])
        res = dict(failure_probability=p, threshold=self.meta["ai4i_threshold"],
                   predicted_failure=bool(p >= self.meta["ai4i_threshold"]))
        if explain:
            c = self.ai4i.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[0]
            items = sorted(zip(self.meta["ai4i_features"], c[:-1]), key=lambda kv: -abs(kv[1]))[:5]
            res["top_drivers"] = [{"feature": k, "impact": round(float(v), 3)} for k, v in items]
        return res

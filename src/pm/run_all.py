"""End-to-end pipeline: python -m pm.run_all   (about 1-2 minutes on CPU)."""
import json
import platform
import time
import warnings

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
import torch
import xgboost

from . import __version__, report
from .config import (ALERT_LEVELS, ARTIFACTS, COSTS, FIGURES, HEALTHY_CYCLES, HORIZON, REPORTS,
                     RUL_CAP, SENSORS, SEED, WINDOW, ensure_dirs)
from .cost import optimise
from .data import load_cmapss, split_units
from .features import feature_names
from .inference import group_contribs
from .numpy_ae import ae_error
from .train_anomaly import run_anomaly
from .train_failure import run_ai4i, run_fd001_failure
from .train_rul import run_rul

warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    t0 = time.time()
    ensure_dirs()
    train, test = load_cmapss()
    tr_units, va_units = split_units(train["unit"].unique(), 0.2, SEED)
    print(f"engines: train={len(tr_units)} val={len(va_units)} test={test['unit'].nunique()}")

    print("[1/5] RUL regression (ridge / LightGBM / XGBoost / GRU) ...")
    rul_arts, rul_res = run_rul(train, test, tr_units, va_units)
    print("[2/5] anomaly detection ...")
    an_arts, an_res = run_anomaly(train, test, tr_units, va_units)
    print("[3/5] failure classification (FD001) ...")
    fail_arts, fail_res = run_fd001_failure(train, test, tr_units, va_units,
                                            rul_arts["Ftr_all"], rul_arts["Fte_all"])
    print("[4/5] AI4I 2020 classification ...")
    ai_arts, ai_res = run_ai4i()
    print("[5/5] cost simulation, explainability, figures ...")
    cost_res = optimise(rul_arts["oof"], COSTS)

    # sensor-level SHAP importance on validation engines (native TreeSHAP of the served model)
    names = feature_names()
    Xva = rul_arts["Ftr_all"].loc[train["unit"].isin(va_units), names]
    contribs = rul_arts["lgbm"].predict(Xva, pred_contrib=True)
    per_sensor = {}
    for c in contribs:
        for d in group_contribs(c, names, top=len(names)):
            per_sensor.setdefault(d["sensor"], []).append(abs(d["impact"]))
    imp = pd.Series({k: float(np.mean(v)) for k, v in per_sensor.items()})

    # anomaly index by RUL bin (validation engines)
    va = train[train["unit"].isin(va_units)]
    from .features import scale
    Xs = scale(va, an_arts["lo"], an_arts["rng"])
    sc = pd.DataFrame({"rul": va["rul"].to_numpy(),
                       "iforest": -an_arts["iforest"].score_samples(Xs) / an_arts["thr"]["iforest"],
                       "autoencoder": ae_error(an_arts["ae_weights"], Xs) / an_arts["thr"]["autoencoder"]})

    report.fig_models(rul_res)
    report.fig_pred_vs_true(rul_arts["y_te_last"], rul_arts["test_last_preds"])
    report.fig_trajectories(rul_arts["oof"])
    report.fig_importance(imp)
    report.fig_anomaly(sc)
    report.fig_threshold(fail_res["val_cost_curve"], fail_res["cost_optimal_threshold"])
    report.fig_cost(cost_res)
    report.fig_ai4i(ai_arts["test_probs"])
    try:  # optional SHAP beeswarm using the shap library
        import matplotlib.pyplot as plt
        import shap
        sample = Xva.sample(min(800, len(Xva)), random_state=SEED)
        sv = shap.TreeExplainer(rul_arts["lgbm"]).shap_values(sample)
        shap.summary_plot(sv, sample, show=False, max_display=15)
        plt.tight_layout(); plt.savefig(FIGURES / "shap_summary.png", dpi=120); plt.close()
    except Exception as e:  # pragma: no cover
        print("  (shap beeswarm skipped:", type(e).__name__, ")")

    # ---- persist artifacts ----
    sensor_std = train[SENSORS].std().to_dict()
    joblib.dump(rul_arts["lgbm"], ARTIFACTS / "rul_lgbm.joblib", compress=3)
    joblib.dump(fail_arts["model"], ARTIFACTS / "failure_xgb.joblib", compress=3)
    joblib.dump(an_arts, ARTIFACTS / "anomaly.joblib", compress=3)
    joblib.dump(ai_arts["model"], ARTIFACTS / "ai4i_xgb.joblib", compress=3)
    torch.save({"state_dict": rul_arts["gru"].state_dict(), "lo": rul_arts["gru_lo"],
                "rng": rul_arts["gru_rng"], "window": WINDOW}, ARTIFACTS / "gru_rul.pt")
    rul_arts["oof"].to_csv(REPORTS / "oof_predictions.csv", index=False)
    rul_arts["test_pred_all"].to_csv(REPORTS / "test_predictions.csv", index=False)
    ai_arts["test_probs"].to_csv(REPORTS / "ai4i_test_probs.csv", index=False)
    meta = dict(version=__version__, subset="FD001", rul_cap=RUL_CAP, horizon=HORIZON,
                sensors=SENSORS, feature_names=names, failure_threshold=fail_arts["threshold"],
                ai4i_threshold=ai_arts["threshold"], ai4i_features=ai_arts["features"],
                sensor_std=sensor_std, alert_levels=ALERT_LEVELS, costs=COSTS)
    (ARTIFACTS / "meta.json").write_text(json.dumps(meta, indent=2))

    cost_out = {k: v for k, v in cost_res.items()}
    metrics = dict(
        protocol=dict(
            split="80/20 by ENGINE on the 100 C-MAPSS FD001 training engines; official 100-engine test set "
                   "used once for final scoring; test truth = official RUL_FD001.txt (uncapped).",
            rul_cap=RUL_CAP, window=WINDOW, healthy_cycles=HEALTHY_CYCLES, seed=SEED,
            train_engines=int(len(tr_units)), val_engines=int(len(va_units)),
            test_engines=int(test["unit"].nunique())),
        rul=rul_res, anomaly=an_res, failure_fd001=fail_res, ai4i=ai_res, cost=cost_out,
        sensor_importance=imp.sort_values(ascending=False).to_dict(),
        environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                         sklearn=sklearn.__version__, lightgbm=lightgbm.__version__,
                         xgboost=xgboost.__version__, torch=torch.__version__),
        runtime_seconds=round(time.time() - t0, 1))
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(f"done in {time.time() - t0:.0f}s -> artifacts/, reports/")


if __name__ == "__main__":
    main()

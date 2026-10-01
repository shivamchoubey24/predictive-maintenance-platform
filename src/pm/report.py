"""Figures for the README / dashboard."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from .config import FIGURES, HORIZON, RUL_CAP

plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / name)
    plt.close(fig)


def fig_models(rul_res):
    names = [k for k in rul_res if not k.startswith("_")]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].barh(names, [rul_res[n]["test_rmse_last"] for n in names], color="#4c78a8")
    ax[0].set_title("Test RMSE (last cycle, 100 engines) - lower is better"); ax[0].invert_yaxis()
    ax[1].barh(names, [rul_res[n]["test_nasa_last"] for n in names], color="#f58518")
    ax[1].set_title("NASA asymmetric score - lower is better"); ax[1].invert_yaxis()
    ax[1].set_yticklabels([])
    _save(fig, "model_comparison.png")


def fig_pred_vs_true(y, preds):
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    for a, k in zip(ax, ("LightGBM", "GRU (30-cycle window)")):
        a.scatter(y, preds[k], s=14, alpha=0.7)
        a.plot([0, 150], [0, 150], "k--", lw=1)
        a.set_title(k); a.set_xlabel("True RUL (cycles)"); a.set_ylabel("Predicted RUL")
    _save(fig, "rul_pred_vs_true.png")


def fig_trajectories(oof):
    units = sorted(oof["unit"].unique())[:4]
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    for a, u in zip(ax, units):
        g = oof[oof["unit"] == u]
        a.plot(g["cycle"], g["rul_capped"], "k", label="true (capped)")
        a.plot(g["cycle"], g["pred_rul"], color="#e45756", alpha=0.8, label="out-of-fold LightGBM")
        a.set_title(f"Engine {u}"); a.set_xlabel("cycle")
    ax[0].set_ylabel("RUL"); ax[0].legend(fontsize=7)
    _save(fig, "rul_trajectories.png")


def fig_importance(imp: pd.Series):
    imp = imp.sort_values()
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.barh(imp.index, imp.values, color="#54a24b")
    ax.set_title("Mean |SHAP| per sensor (cycles of predicted RUL), validation engines")
    _save(fig, "sensor_importance.png")


def fig_anomaly(val_scores: pd.DataFrame):
    bins = pd.cut(val_scores["rul"], [-1, 15, 30, 60, 100, 1000],
                  labels=["0-15", "16-30", "31-60", "61-100", ">100"])
    m = val_scores.groupby(bins, observed=True)[["iforest", "autoencoder"]].mean()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    m.plot(kind="bar", ax=ax, color=["#4c78a8", "#f58518"])
    ax.axhline(1, color="k", ls="--", lw=1); ax.set_xlabel("True RUL bin (cycles)")
    ax.set_ylabel("Anomaly index (1 = healthy 99th pct)"); ax.set_title("Anomaly index rises as failure nears")
    _save(fig, "anomaly_by_rul.png")


def fig_threshold(curve, thr):
    c = np.array(curve)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(c[:, 0], c[:, 1] / 1000, color="#4c78a8")
    ax.axvline(thr, color="r", ls="--", label=f"chosen threshold {thr:.2f}")
    ax.set_xlabel("Decision threshold on P(fail within %d cycles)" % HORIZON)
    ax.set_ylabel("Validation cost (k USD)"); ax.legend(); ax.set_title("Cost-sensitive threshold selection")
    _save(fig, "threshold_cost_curve.png")


def fig_cost(cost):
    labels = ["Run to failure", f"Best fixed schedule\n(every {cost['best_fixed_interval']} cycles)",
              f"Predictive\n(RUL <= {cost['best_trigger']})"]
    vals = [cost["reactive"]["per_engine"], cost["fixed"]["per_engine"], cost["predictive"]["per_engine"]]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(labels, vals, color=["#e45756", "#f58518", "#54a24b"])
    for i, v in enumerate(vals):
        ax.text(i, v, f"${v:,.0f}", ha="center", va="bottom")
    ax.set_ylabel("Avg cost per engine lifetime (USD)"); ax.set_title("Simulated maintenance cost (out-of-fold RUL)")
    _save(fig, "cost_comparison.png")


def fig_ai4i(probs: pd.DataFrame):
    p, r, _ = precision_recall_curve(probs["y"], probs["p"])
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.plot(r, p); ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("AI4I 2020: PR curve (test)")
    _save(fig, "ai4i_pr_curve.png")

"""Evaluation metrics, including the asymmetric NASA scoring function."""
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score


def rmse(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sqrt(np.mean((p - y) ** 2)))


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(p, float) - np.asarray(y, float))))


def nasa_score(y, p) -> float:
    """PHM08 score: late predictions (p > y) are penalised more than early ones."""
    d = np.asarray(p, float) - np.asarray(y, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)))


def threshold_cost(y, prob, thr, c_fn, c_fp) -> float:
    pred = np.asarray(prob) >= thr
    y = np.asarray(y).astype(bool)
    return float(((~pred) & y).sum() * c_fn + (pred & ~y).sum() * c_fp)


def best_threshold(y, prob, c_fn, c_fp):
    grid = np.unique(np.concatenate([np.linspace(0.01, 0.99, 99), np.asarray(prob)]))
    grid = grid[(grid > 0) & (grid < 1)]
    costs = np.array([threshold_cost(y, prob, t, c_fn, c_fp) for t in grid])
    i = int(np.argmin(costs))   # first minimum -> lowest threshold among ties
    return float(grid[i]), float(costs[i])


def classification_report_dict(y, prob, thr, c_fn, c_fp) -> dict:
    y = np.asarray(y).astype(int)
    pred = (np.asarray(prob) >= thr).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    out = dict(n=int(len(y)), positives=int(y.sum()), threshold=float(thr),
               precision=float(p), recall=float(r), f1=float(f),
               tp=tp, fp=fp, fn=fn, tn=tn, cost=float(fn * c_fn + fp * c_fp))
    if 0 < y.sum() < len(y):
        out["pr_auc"] = float(average_precision_score(y, prob))
        out["roc_auc"] = float(roc_auc_score(y, prob))
    return out

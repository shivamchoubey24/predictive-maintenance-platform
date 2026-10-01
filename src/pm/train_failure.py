"""Phase 4b: failure classification with imbalance handling and cost-sensitive thresholds.
(1) FD001: "fails within the next 30 cycles?"   (2) AI4I 2020: machine failure (3.4% positives)."""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import COSTS, HORIZON, SEED
from .data import ai4i_features, last_rows, load_ai4i, AI4I_FAILURE_MODES
from .features import feature_names
from .metrics import best_threshold, classification_report_dict, threshold_cost

XGB_KW = dict(n_estimators=2000, learning_rate=0.05, max_depth=4, subsample=0.8,
              colsample_bytree=0.8, min_child_weight=2, reg_lambda=1.0, tree_method="hist",
              early_stopping_rounds=60, random_state=SEED, n_jobs=2, eval_metric="aucpr")


def oversample(X, y, seed=SEED):
    """Random oversampling of the positive class to a 1:1 ratio (train data only)."""
    rng = np.random.RandomState(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    extra = rng.choice(pos, size=max(0, len(neg) - len(pos)), replace=True)
    idx = np.concatenate([np.arange(len(y)), extra])
    return X.iloc[idx], y[idx]


def fit_xgb(Xtr, ytr, Xva, yva, strategy):
    kw = dict(XGB_KW)
    if strategy == "class_weight":
        kw["scale_pos_weight"] = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    if strategy == "oversample":
        Xtr, ytr = oversample(Xtr, ytr)
    m = xgb.XGBClassifier(**kw)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m


def compare_strategies(Xtr, ytr, Xva, yva, include_rf=False):
    """Returns (dict name->(model, val_probs), table)."""
    from sklearn.metrics import average_precision_score
    fitted = {}
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    lr.fit(Xtr, ytr)
    fitted["Logistic Regression (baseline, balanced)"] = (lr, lr.predict_proba(Xva)[:, 1])
    if include_rf:
        rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample",
                                    random_state=SEED, n_jobs=2).fit(Xtr, ytr)
        fitted["Random Forest (balanced)"] = (rf, rf.predict_proba(Xva)[:, 1])
    for s, label in (("none", "XGBoost (no imbalance handling)"),
                     ("class_weight", "XGBoost (class weights)"),
                     ("oversample", "XGBoost (random oversampling)")):
        m = fit_xgb(Xtr, ytr, Xva, yva, s)
        fitted[label] = (m, m.predict_proba(Xva)[:, 1])
    table = {k: float(average_precision_score(yva, p)) for k, (m, p) in fitted.items()}
    return fitted, table


def run_fd001_failure(train, test, tr_units, va_units, Ftr_all, Fte_all):
    feats = feature_names()
    y_all = (train["rul"] <= HORIZON).astype(int).to_numpy()
    m_tr, m_va = train["unit"].isin(tr_units).to_numpy(), train["unit"].isin(va_units).to_numpy()
    Xtr, Xva = Ftr_all.loc[m_tr, feats], Ftr_all.loc[m_va, feats]
    ytr, yva = y_all[m_tr], y_all[m_va]
    fitted, val_pr = compare_strategies(Xtr, ytr, Xva, yva)

    xgb_names = [k for k in fitted if k.startswith("XGBoost")]
    best = max(xgb_names, key=lambda k: val_pr[k])
    model, pva = fitted[best]
    thr, val_cost = best_threshold(yva, pva, COSTS["fn"], COSTS["fp"])

    te_last = last_rows(test)
    Xte_last = Fte_all.loc[te_last.index, feats]
    y_last = (te_last["rul"] <= HORIZON).astype(int).to_numpy()
    p_last = model.predict_proba(Xte_last)[:, 1]
    y_all_te = (test["rul"] <= HORIZON).astype(int).to_numpy()
    p_all_te = model.predict_proba(Fte_all[feats])[:, 1]

    c_fn, c_fp = COSTS["fn"], COSTS["fp"]
    metrics = dict(
        horizon_cycles=HORIZON, positive_rate_train=float(ytr.mean()),
        val_pr_auc_by_strategy=val_pr, selected_model=best, cost_optimal_threshold=thr,
        test_last_cycle_cost_optimal=classification_report_dict(y_last, p_last, thr, c_fn, c_fp),
        test_last_cycle_default_0_5=classification_report_dict(y_last, p_last, 0.5, c_fn, c_fp),
        test_all_rows_cost_optimal=classification_report_dict(y_all_te, p_all_te, thr, c_fn, c_fp),
        val_cost_curve=[[float(t), threshold_cost(yva, pva, t, c_fn, c_fp)] for t in np.linspace(0.02, 0.98, 49)],
    )
    return dict(model=model, threshold=thr, val_y=yva, val_p=pva), metrics


def run_ai4i():
    df = load_ai4i()
    X, y = ai4i_features(df), df["failure"].to_numpy()
    Xtv, Xte, ytv, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    Xtr, Xva, ytr, yva = train_test_split(Xtv, ytv, test_size=0.2, stratify=ytv, random_state=SEED)
    fitted, val_pr = compare_strategies(Xtr, ytr, Xva, yva, include_rf=True)
    xgb_names = [k for k in fitted if k.startswith("XGBoost")]
    best = max(xgb_names, key=lambda k: val_pr[k])
    model, pva = fitted[best]
    c_fn, c_fp = COSTS["fn"], COSTS["fp"]
    thr, _ = best_threshold(yva, pva, c_fn, c_fp)
    pte = model.predict_proba(Xte)[:, 1]

    # Unsupervised baseline: Isolation Forest (no labels used)
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import average_precision_score
    iso = IsolationForest(n_estimators=200, random_state=SEED, n_jobs=2).fit(Xtr[ytr == 0])
    iso_pr = float(average_precision_score(yte, -iso.score_samples(Xte)))

    # Leakage demonstration: the failure-mode flags define the target
    leaky = df[AI4I_FAILURE_MODES]
    Xl = pd.concat([X, leaky], axis=1)
    Xl_tr, Xl_te, yl_tr, yl_te = train_test_split(Xl, y, test_size=0.2, stratify=y, random_state=SEED)
    leak_m = xgb.XGBClassifier(n_estimators=100, max_depth=3, tree_method="hist", n_jobs=2).fit(Xl_tr, yl_tr)
    leak_pr = float(average_precision_score(yl_te, leak_m.predict_proba(Xl_te)[:, 1]))

    test_rows = classification_report_dict(yte, pte, thr, c_fn, c_fp)
    test_rows["default_0_5"] = classification_report_dict(yte, pte, 0.5, c_fn, c_fp)
    metrics = dict(n_rows=int(len(df)), positive_rate=float(y.mean()),
                   val_pr_auc_by_strategy=val_pr, selected_model=best,
                   cost_optimal_threshold=thr, test=test_rows,
                   isolation_forest_unsupervised_test_pr_auc=iso_pr,
                   leakage_demo_pr_auc_with_failure_mode_flags=leak_pr,
                   test_cost_all_negative_baseline=float(yte.sum() * c_fn))
    probs = pd.DataFrame({"y": yte, "p": pte})
    return dict(model=model, threshold=thr, features=list(X.columns), test_probs=probs), metrics

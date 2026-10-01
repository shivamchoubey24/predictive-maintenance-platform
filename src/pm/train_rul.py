"""Phases 2-3: Remaining-Useful-Life regression (baseline, boosted trees, GRU) + OOF predictions."""
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from .config import RUL_CAP, SEED
from .data import last_rows
from .features import build_features, feature_names, fit_minmax, make_windows
from .metrics import mae, nasa_score, rmse
from .nets import predict_in_batches, train_gru

warnings.filterwarnings("ignore", message=".*eval_set.*")

LGB_PARAMS = dict(n_estimators=3000, learning_rate=0.03, num_leaves=31, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.7, min_child_samples=40,
                  reg_lambda=1.0, random_state=SEED, n_jobs=2, verbose=-1)


def score_block(y_true_last, p_last, y_true_all, p_all) -> dict:
    return dict(test_rmse_last=rmse(y_true_last, p_last), test_mae_last=mae(y_true_last, p_last),
                test_nasa_last=nasa_score(y_true_last, p_last),
                test_rmse_all_rows=rmse(y_true_all, p_all))


def run_rul(train, test, tr_units, va_units):
    """Returns (artifacts dict, metrics dict). Test set is only touched for final scoring."""
    feats = feature_names()
    Ftr_all, Fte_all = build_features(train), build_features(test)
    m_tr, m_va = train["unit"].isin(tr_units), train["unit"].isin(va_units)
    Xtr, Xva = Ftr_all.loc[m_tr, feats], Ftr_all.loc[m_va, feats]
    ytr, yva = train.loc[m_tr, "rul_capped"], train.loc[m_va, "rul_capped"]

    te_last = last_rows(test).index
    Xte, Xte_last = Fte_all[feats], Fte_all.loc[te_last, feats]
    y_te_last = test.loc[te_last, "rul"].to_numpy()          # official (uncapped) RUL
    y_te_all = test["rul_capped"].to_numpy()                 # per-row comparison uses capped truth
    clip = lambda p: np.clip(p, 0, RUL_CAP)

    results, preds_last = {}, {}

    # 1) Ridge baseline
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(Xtr, ytr)
    p_all, p_last = clip(ridge.predict(Xte)), clip(ridge.predict(Xte_last))
    results["Ridge (baseline)"] = dict(val_rmse=rmse(yva, clip(ridge.predict(Xva))),
                                       **score_block(y_te_last, p_last, y_te_all, p_all))
    preds_last["Ridge (baseline)"] = p_last

    # 2) LightGBM (served model)
    lgbm = lgb.LGBMRegressor(**LGB_PARAMS)
    lgbm.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="rmse",
             callbacks=[lgb.early_stopping(100, verbose=False)])
    p_all, p_last = clip(lgbm.predict(Xte)), clip(lgbm.predict(Xte_last))
    results["LightGBM"] = dict(val_rmse=rmse(yva, clip(lgbm.predict(Xva))),
                               best_iteration=int(lgbm.best_iteration_),
                               **score_block(y_te_last, p_last, y_te_all, p_all))
    preds_last["LightGBM"] = p_last

    # 3) XGBoost
    xg = xgb.XGBRegressor(n_estimators=3000, learning_rate=0.03, max_depth=5, subsample=0.8,
                          colsample_bytree=0.7, min_child_weight=5, reg_lambda=1.0,
                          early_stopping_rounds=100, random_state=SEED, n_jobs=2, tree_method="hist")
    xg.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    p_all, p_last = clip(xg.predict(Xte)), clip(xg.predict(Xte_last))
    results["XGBoost"] = dict(val_rmse=rmse(yva, clip(xg.predict(Xva))),
                              best_iteration=int(xg.best_iteration),
                              **score_block(y_te_last, p_last, y_te_all, p_all))
    preds_last["XGBoost"] = p_last

    # 4) GRU on 30-cycle windows
    lo, rng = fit_minmax(train.loc[m_tr])
    Wtr, itr = make_windows(train.loc[m_tr], lo, rng)
    Wva, iva = make_windows(train.loc[m_va], lo, rng)
    Wte_all, ite = make_windows(test, lo, rng)
    Wte_last, ilast = make_windows(test, lo, rng, last_only=True)
    gru, hist = train_gru(Wtr, train.loc[itr, "rul_capped"].to_numpy() / RUL_CAP,
                          Wva, train.loc[iva, "rul_capped"].to_numpy() / RUL_CAP)
    g = lambda W: clip(predict_in_batches(gru, W) * RUL_CAP)
    p_all_gru = pd.Series(g(Wte_all), index=ite).reindex(test.index).to_numpy()
    p_last = pd.Series(g(Wte_last), index=ilast).reindex(te_last).to_numpy()
    results["GRU (30-cycle window)"] = dict(val_rmse=rmse(train.loc[iva, "rul_capped"], g(Wva)),
                                            epochs_run=len(hist),
                                            **score_block(y_te_last, p_last, y_te_all, p_all_gru))
    preds_last["GRU (30-cycle window)"] = p_last

    # 5) Simple blend of the two best families (tree + GRU)
    p_blend_last = 0.5 * (preds_last["LightGBM"] + preds_last["GRU (30-cycle window)"])
    p_blend_all = 0.5 * (clip(lgbm.predict(Xte)) + p_all_gru)
    results["Blend (LightGBM + GRU)"] = dict(
        val_rmse=None, **score_block(y_te_last, p_blend_last, y_te_all, p_blend_all))
    preds_last["Blend (LightGBM + GRU)"] = p_blend_last
    test_pred_all = pd.DataFrame({"unit": test["unit"], "cycle": test["cycle"], "rul_true": test["rul"],
                                  "pred_lgbm": clip(lgbm.predict(Xte)), "pred_gru": p_all_gru})

    # Out-of-fold LightGBM predictions on ALL 100 training engines (for cost simulation)
    full_X = Ftr_all[feats]
    n_iter = max(50, int(lgbm.best_iteration_))
    oof = np.zeros(len(train))
    for k, (a, b) in enumerate(GroupKFold(n_splits=5).split(full_X, groups=train["unit"])):
        mdl = lgb.LGBMRegressor(**{**LGB_PARAMS, "n_estimators": n_iter})
        mdl.fit(full_X.iloc[a], train["rul_capped"].iloc[a])
        oof[b] = clip(mdl.predict(full_X.iloc[b]))
    oof_df = pd.DataFrame({"unit": train["unit"], "cycle": train["cycle"],
                           "rul": train["rul"], "rul_capped": train["rul_capped"], "pred_rul": oof})
    results["_oof_rmse_lightgbm"] = rmse(train["rul_capped"], oof)

    arts = dict(test_pred_all=test_pred_all, lgbm=lgbm, gru=gru, gru_lo=lo, gru_rng=rng, oof=oof_df,
                test_last_preds=preds_last, y_te_last=y_te_last,
                Ftr_all=Ftr_all, Fte_all=Fte_all)
    return arts, results

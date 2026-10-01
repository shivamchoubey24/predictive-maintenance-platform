"""Streamlit dashboard.  Run: streamlit run src/pm/dashboard.py"""
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pm.config import ARTIFACTS, COSTS, FIGURES, HORIZON, REPORTS, RUL_CAP, SENSORS
from pm.cost import optimise, simulate_fixed, simulate_predictive, simulate_reactive
from pm.data import last_rows, load_cmapss
from pm.features import build_features
from pm.inference import Predictor

st.set_page_config(page_title="Predictive Maintenance Platform", layout="wide", page_icon="🛠️")
LEVEL_COLORS = {"Critical": "#d62728", "Warning": "#ff7f0e", "Watch": "#e6c229", "Healthy": "#2ca02c"}


@st.cache_resource
def get_predictor() -> Predictor:
    return Predictor()


@st.cache_data
def load_fleet():
    """Test-set engines scored cycle by cycle with the served models."""
    pred = get_predictor()
    _, test = load_cmapss()
    feat = build_features(test)
    scored = pred.score_features(feat)
    scored["rul_true"] = test["rul"].to_numpy()
    extra = pd.read_csv(REPORTS / "test_predictions.csv")[["unit", "cycle", "pred_gru"]]
    scored = scored.merge(extra, on=["unit", "cycle"], how="left")
    last = last_rows(test)
    drivers = pred.explain_rows(feat.loc[last.index], top=3)
    snap = scored.merge(last[["unit", "cycle"]], on=["unit", "cycle"])
    snap = snap.sort_values("unit").reset_index(drop=True)
    snap["top_drivers"] = [", ".join(f"{d['sensor']} ({d['impact']:+.1f})" for d in drivers[i])
                           for i in last.sort_values("unit").reset_index(drop=True).index]
    return test, feat, scored, snap


@st.cache_data
def load_metrics() -> dict:
    return json.loads((REPORTS / "metrics.json").read_text())


@st.cache_data
def load_oof() -> pd.DataFrame:
    return pd.read_csv(REPORTS / "oof_predictions.csv")


@st.cache_data
def cached_optimise(unplanned, planned, wasted):
    return optimise(load_oof(), dict(COSTS, unplanned=unplanned, planned=planned, wasted_per_cycle=wasted))


def money(x: float) -> str:
    return f"${x:,.0f}"


if not (ARTIFACTS / "meta.json").exists() or not (REPORTS / "metrics.json").exists():
    st.error("Artifacts not found. Run `python -m pm.run_all` first.")
    st.stop()

pred = get_predictor()
test, feat, scored, snap = load_fleet()
M = load_metrics()
meta = pred.meta

st.title("🛠️ Predictive Maintenance Platform")
st.caption("NASA C-MAPSS FD001 turbofan engines (100 held-out test engines) + AI4I 2020 machine failures. "
           "'True RUL' is shown only because this is an offline benchmark; in production it is unknown.")

tab_fleet, tab_engine, tab_cost, tab_models, tab_ai4i = st.tabs(
    ["Fleet overview", "Engine detail & what-if", "Cost savings", "Models & evaluation", "AI4I machine failure"])

# ------------------------------------------------------------------ Fleet
with tab_fleet:
    counts = snap["alert"].value_counts()
    c = st.columns(5)
    c[0].metric("Engines monitored", len(snap))
    c[1].metric("Critical (RUL ≤ 15)", int(counts.get("Critical", 0)))
    c[2].metric("Warning (RUL ≤ 30)", int(counts.get("Warning", 0)))
    c[3].metric("Watch (RUL ≤ 60)", int(counts.get("Watch", 0)))
    c[4].metric("Fleet avg health", f"{snap['health_score'].mean():.0f} / 100")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Machine health scores")
        s = snap.sort_values("health_score")
        fig = go.Figure(go.Bar(x=s["unit"].astype(str), y=s["health_score"],
                               marker_color=[LEVEL_COLORS[a] for a in s["alert"]],
                               hovertext=[f"Engine {u}: RUL≈{r:.0f} cycles" for u, r in zip(s["unit"], s["pred_rul"])]))
        fig.update_layout(height=330, margin=dict(t=10, b=10), xaxis_title="Engine (sorted by health)",
                          yaxis_title="Health score (0-100)", xaxis_showticklabels=False)
        st.plotly_chart(fig)
    with right:
        st.subheader("Alert distribution")
        order = ["Critical", "Warning", "Watch", "Healthy"]
        fig = go.Figure(go.Bar(x=order, y=[int(counts.get(o, 0)) for o in order],
                               marker_color=[LEVEL_COLORS[o] for o in order]))
        fig.update_layout(height=330, margin=dict(t=10, b=10), yaxis_title="Engines")
        st.plotly_chart(fig)

    st.subheader("Alert feed (most urgent first)")
    feed = snap[snap["alert"] != "Healthy"].sort_values("pred_rul")
    feed = feed.assign(**{"RUL countdown (cycles)": feed["pred_rul"].round(0).astype(int),
                          "P(fail ≤ 30 cycles)": (feed["fail_prob"] * 100).round(0).astype(int).astype(str) + "%",
                          "Anomaly index": feed["anomaly_index"].round(2),
                          "Why (top sensors, Δ cycles)": feed["top_drivers"]})
    st.dataframe(feed[["unit", "alert", "RUL countdown (cycles)", "P(fail ≤ 30 cycles)", "Anomaly index",
                       "Why (top sensors, Δ cycles)", "cycle"]].rename(columns={"unit": "Engine", "alert": "Alert",
                                                                              "cycle": "Last cycle"}),
                 hide_index=True)

# ------------------------------------------------------------------ Engine
with tab_engine:
    units = sorted(snap["unit"].tolist())
    default_unit = int(snap.sort_values("pred_rul")["unit"].iloc[0])
    c1, c2 = st.columns([1, 3])
    unit = c1.selectbox("Engine", units, index=units.index(default_unit))
    hist_all = test[test["unit"] == unit]
    max_cycle = int(hist_all["cycle"].max())
    cyc = c2.slider("Replay up to cycle (simulates live monitoring)", 1, max_cycle, max_cycle)
    sc = scored[(scored["unit"] == unit) & (scored["cycle"] <= cyc)]
    cur = sc.iloc[-1]

    m = st.columns(5)
    m[0].metric("Predicted RUL", f"{cur['pred_rul']:.0f} cycles")
    m[1].metric("Health score", f"{cur['health_score']:.0f} / 100")
    m[2].metric("P(fail ≤ 30 cycles)", f"{100 * cur['fail_prob']:.0f}%")
    m[3].metric("Anomaly index", f"{cur['anomaly_index']:.2f}", help="> 1 means beyond the healthy 99th percentile")
    m[4].markdown(f"**Alert level**<br><span style='font-size:1.8em;color:{LEVEL_COLORS[cur['alert']]}'>"
                  f"{cur['alert']}</span>", unsafe_allow_html=True)

    g1, g2 = st.columns(2)
    with g1:
        fig = go.Figure()
        fig.add_scatter(x=sc["cycle"], y=sc["rul_true"].clip(upper=RUL_CAP), name="True RUL (capped, benchmark only)",
                        line=dict(color="black", dash="dot"))
        fig.add_scatter(x=sc["cycle"], y=sc["pred_rul"], name="LightGBM (served)", line=dict(color="#d62728"))
        fig.add_scatter(x=sc["cycle"], y=sc["pred_gru"], name="GRU (offline, more accurate)", line=dict(color="#1f77b4"))
        fig.add_hline(y=HORIZON, line_dash="dash", line_color="orange", annotation_text="30-cycle horizon")
        fig.update_layout(title="RUL countdown", height=340, xaxis_title="Cycle", yaxis_title="RUL (cycles)",
                          margin=dict(t=40, b=10), legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig)
    with g2:
        fig = go.Figure()
        fig.add_scatter(x=sc["cycle"], y=sc["anomaly_index"], name="Anomaly index", line=dict(color="#9467bd"))
        fig.add_hline(y=1, line_dash="dash", line_color="red", annotation_text="healthy 99th pct")
        fig.update_layout(title="Anomaly detection (Isolation Forest)", height=340, xaxis_title="Cycle",
                          margin=dict(t=40, b=10), yaxis_title="Index")
        st.plotly_chart(fig)

    row = feat[(feat["unit"] == unit) & (feat["cycle"] == cyc)]
    drivers = pred.explain_rows(row, top=8)[0]
    st.subheader("Which sensors drive this prediction? (TreeSHAP)")
    d = pd.DataFrame(drivers).sort_values("impact")
    fig = go.Figure(go.Bar(x=d["impact"], y=d["sensor"], orientation="h",
                           marker_color=["#d62728" if v < 0 else "#2ca02c" for v in d["impact"]]))
    fig.update_layout(height=300, margin=dict(t=10, b=10),
                      xaxis_title="Impact on predicted RUL (cycles). Red = shortens remaining life")
    st.plotly_chart(fig)

    st.subheader("What-if: shift sensor readings")
    st.caption("Applies a sustained shift (in training standard deviations) to the most recent readings of the "
               "selected engine and re-scores it. Useful to explore sensitivity, not a causal simulation.")
    names_sorted = [x["sensor"] for x in pred.explain_rows(row, top=len(SENSORS) + 1)[0] if x["sensor"] in SENSORS]
    keys = [f"wi_{s}" for s in SENSORS]
    for k in keys:
        st.session_state.setdefault(k, 0.0)

    def _reset():
        for k in keys:
            st.session_state[k] = 0.0

    last_n = st.slider("Apply the shift to the last N cycles", 1, 30, 10)
    cols = st.columns(4)
    for i, s in enumerate(names_sorted):
        cols[i % 4].slider(f"{s} (σ)", -3.0, 3.0, step=0.25, key=f"wi_{s}")
    st.button("Reset sliders", on_click=_reset)
    h = hist_all[hist_all["cycle"] <= cyc][["cycle"] + SENSORS].copy().reset_index(drop=True)
    tail = h.index[-last_n:]
    for s in SENSORS:
        h.loc[tail, s] = h.loc[tail, s] + st.session_state[f"wi_{s}"] * meta["sensor_std"][s]
    res = pred.score_history(h, explain=False)
    w = st.columns(3)
    w[0].metric("What-if RUL", f"{res['predicted_rul_cycles']:.0f} cycles",
                delta=f"{res['predicted_rul_cycles'] - cur['pred_rul']:+.1f} vs actual")
    w[1].metric("What-if P(fail ≤ 30)", f"{100 * res['failure_probability']:.0f}%",
                delta=f"{100 * (res['failure_probability'] - cur['fail_prob']):+.0f} pts", delta_color="inverse")
    w[2].metric("What-if alert", res["alert_level"])

# ------------------------------------------------------------------ Cost
with tab_cost:
    st.subheader("Maintenance cost-savings estimate")
    st.caption("Simulation on the 100 training engines using out-of-fold RUL predictions (the model never saw "
               "the engine it is scored on). All dollar amounts are editable assumptions, not real data.")
    a, b, c3 = st.columns(3)
    unpl = a.number_input("Cost of an in-service failure ($)", 1000, 1_000_000, COSTS["unplanned"], 1000)
    plan = b.number_input("Cost of a planned maintenance ($)", 500, 500_000, COSTS["planned"], 500)
    waste = c3.number_input("Cost per wasted cycle of remaining life ($)", 0, 5000, COSTS["wasted_per_cycle"], 5)
    costs = dict(COSTS, unplanned=unpl, planned=plan, wasted_per_cycle=waste)
    opt = cached_optimise(unpl, plan, waste)
    trig = st.slider("Predictive policy: maintain when smoothed predicted RUL ≤ (cycles)", 5, 100,
                     int(opt["best_trigger"]), 5)
    oof = load_oof()
    chosen = simulate_predictive(oof, trig, costs)
    rows = {"Run to failure": simulate_reactive(oof, costs),
            f"Best fixed schedule (every {opt['best_fixed_interval']} cycles)": opt["fixed"],
            f"Predictive (RUL ≤ {trig})": chosen}
    tbl = pd.DataFrame({"Avg cost / engine": [money(v["per_engine"]) for v in rows.values()],
                        "In-service failures (of 100)": [v["unplanned_failures"] for v in rows.values()],
                        "Avg life wasted (cycles)": [round(v["avg_wasted_cycles"], 1) for v in rows.values()]},
                       index=list(rows))
    st.dataframe(tbl)
    base = rows["Run to failure"]["per_engine"]
    sav = 100 * (1 - chosen["per_engine"] / base)
    st.success(f"Predictive maintenance saves ≈ {money(base - chosen['per_engine'])} per engine "
               f"({sav:.0f}%) vs run-to-failure, and ≈ "
               f"{100 * (1 - chosen['per_engine'] / opt['fixed']['per_engine']):.0f}% vs the best fixed schedule "
               f"under these assumptions.")
    fig = go.Figure(go.Scatter(x=[t for t, _ in opt["trigger_curve"]], y=[v for _, v in opt["trigger_curve"]],
                               mode="lines+markers"))
    fig.update_layout(height=300, margin=dict(t=30, b=10), title="Cost vs. maintenance trigger",
                      xaxis_title="Trigger (predicted RUL, cycles)", yaxis_title="Avg cost / engine ($)")
    st.plotly_chart(fig)
    st.caption("Caveat: the trigger and fixed interval are tuned on the same 100 engines, so savings are somewhat "
               "optimistic. The best fixed interval is also an oracle-tuned baseline, which makes the comparison "
               "conservative for the predictive policy.")

# ------------------------------------------------------------------ Models
with tab_models:
    st.subheader("RUL models: held-out test set (100 engines, last observed cycle)")
    rows = []
    for name, v in M["rul"].items():
        if name.startswith("_"):
            continue
        rows.append({"Model": name, "Val RMSE": None if v.get("val_rmse") is None else round(v["val_rmse"], 2),
                     "Test RMSE": round(v["test_rmse_last"], 2), "Test MAE": round(v["test_mae_last"], 2),
                     "NASA score": round(v["test_nasa_last"], 0)})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.image(str(FIGURES / "model_comparison.png"))
    c1, c2 = st.columns(2)
    c1.image(str(FIGURES / "rul_pred_vs_true.png"), caption="Predicted vs true RUL (test)")
    c2.image(str(FIGURES / "rul_trajectories.png"), caption="Out-of-fold trajectories (training engines)")

    st.subheader("Failure classification: 'fails within 30 cycles?' with a cost-sensitive threshold")
    f = M["failure_fd001"]
    st.write(f"Selected model: **{f['selected_model']}**. Cost-optimal threshold **{f['cost_optimal_threshold']:.2f}** "
             f"(missed failure = {money(COSTS['fn'])}, false alarm = {money(COSTS['fp'])}).")
    # live threshold slider on the 100 test snapshots
    thr = st.slider("Decision threshold", 0.01, 0.99, float(round(f["cost_optimal_threshold"], 2)), 0.01)
    y = (snap["rul_true"] <= HORIZON)
    p = snap["fail_prob"]
    tp = int(((p >= thr) & y).sum()); fp = int(((p >= thr) & ~y).sum()); fn = int(((p < thr) & y).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    cc = st.columns(4)
    cc[0].metric("Precision", f"{prec:.2f}"); cc[1].metric("Recall", f"{rec:.2f}")
    cc[2].metric("Missed / false alarms", f"{fn} / {fp}")
    cc[3].metric("Cost", money(fn * COSTS["fn"] + fp * COSTS["fp"]))
    st.image(str(FIGURES / "threshold_cost_curve.png"))

    st.subheader("Anomaly detection")
    st.dataframe(pd.DataFrame(M["anomaly"]).T.rename(index={"iforest": "Isolation Forest", "autoencoder": "Autoencoder"}))
    st.image(str(FIGURES / "anomaly_by_rul.png"))

    st.subheader("Explainability")
    c1, c2 = st.columns(2)
    c1.image(str(FIGURES / "sensor_importance.png"))
    if (FIGURES / "shap_summary.png").exists():
        c2.image(str(FIGURES / "shap_summary.png"), caption="SHAP beeswarm (LightGBM RUL model)")

# ------------------------------------------------------------------ AI4I
with tab_ai4i:
    A = M["ai4i"]
    st.subheader("AI4I 2020: machine failure (3.4% of rows fail)")
    st.write(f"Selected: **{A['selected_model']}** · test PR-AUC **{A['test']['pr_auc']:.3f}** · "
             f"ROC-AUC {A['test']['roc_auc']:.3f}. Failure-mode flags (TWF/HDF/PWF/OSF/RNF) are excluded because they "
             f"define the target (using them would give a misleadingly easy problem).")
    with st.form("machine"):
        c = st.columns(5)
        mtype = c[0].selectbox("Product type", ["L", "M", "H"])
        air = c[1].number_input("Air temp [K]", 290.0, 310.0, 300.0, 0.1)
        proc = c[2].number_input("Process temp [K]", 300.0, 320.0, 310.0, 0.1)
        rpm = c[3].number_input("Rotational speed [rpm]", 1000, 3000, 1500, 10)
        tq = c[4].number_input("Torque [Nm]", 3.0, 80.0, 40.0, 0.5)
        wear = st.slider("Tool wear [min]", 0, 260, 100)
        go_btn = st.form_submit_button("Score machine")
    out = pred.score_machine(dict(type=mtype, air_temp=air, process_temp=proc, rot_speed=rpm, torque=tq,
                                  tool_wear=wear))
    r = st.columns(3)
    r[0].metric("Failure probability", f"{100 * out['failure_probability']:.1f}%")
    r[1].metric("Decision threshold", f"{out['threshold']:.2f}")
    r[2].metric("Prediction", "FAILURE RISK" if out["predicted_failure"] else "OK")
    st.write("Top drivers (log-odds contribution):",
             ", ".join(f"`{d['feature']}` {d['impact']:+.2f}" for d in out["top_drivers"]))
    probs = pd.read_csv(REPORTS / "ai4i_test_probs.csv")
    t2 = st.slider("Threshold for the test-set trade-off", 0.01, 0.99, float(round(A["cost_optimal_threshold"], 2)), 0.01,
                   key="ai4i_thr")
    yy, pp = probs["y"].astype(bool), probs["p"]
    tp = int(((pp >= t2) & yy).sum()); fp = int(((pp >= t2) & ~yy).sum()); fn = int(((pp < t2) & yy).sum())
    q = st.columns(4)
    q[0].metric("Precision", f"{tp / max(tp + fp, 1):.2f}"); q[1].metric("Recall", f"{tp / max(tp + fn, 1):.2f}")
    q[2].metric("Missed / false alarms", f"{fn} / {fp}"); q[3].metric("Cost", money(fn * COSTS["fn"] + fp * COSTS["fp"]))
    comp = pd.DataFrame({"Validation PR-AUC": A["val_pr_auc_by_strategy"]}).round(3)
    st.dataframe(comp)

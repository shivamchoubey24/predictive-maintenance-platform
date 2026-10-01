import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from pm.api import app
from pm.config import RUL_CAP, SENSORS
from pm.cost import optimise, simulate_predictive, simulate_reactive
from pm.data import last_rows, load_cmapss, split_units
from pm.features import build_features, feature_names, make_windows, fit_minmax
from pm.inference import Predictor, alert_level
from pm.metrics import best_threshold, nasa_score, rmse


@pytest.fixture(scope="module")
def data():
    return load_cmapss()


def test_nasa_score_is_asymmetric():
    assert nasa_score([50], [60]) > nasa_score([50], [40])   # late is worse than early
    assert nasa_score([50], [50]) == 0
    assert rmse([1, 2], [1, 2]) == 0


def test_rul_labels(data):
    train, test = data
    g = train[train.unit == 1]
    assert g["rul"].iloc[-1] == 0 and g["rul"].iloc[0] == len(g) - 1
    assert train["rul_capped"].max() == RUL_CAP
    official = pd.read_csv("data/raw/RUL_FD001.txt", header=None)[0].to_numpy()
    assert np.array_equal(last_rows(test).sort_values("unit")["rul"].to_numpy(), official)


def test_split_is_by_engine_and_disjoint(data):
    tr, va = split_units(data[0]["unit"].unique())
    assert set(tr).isdisjoint(va) and len(tr) + len(va) == 100


def test_features_have_no_lookahead(data):
    train, _ = data
    full = build_features(train)
    cut = train[(train.unit == 3) & (train.cycle <= 80)]
    part = build_features(cut)
    assert np.allclose(full.loc[cut.index, feature_names()].to_numpy(), part[feature_names()].to_numpy())


def test_windows_shape(data):
    train, _ = data
    lo, rng = fit_minmax(train)
    X, idx = make_windows(train, lo, rng)
    assert X.shape == (len(train), 30, len(SENSORS)) and len(idx) == len(train)


def test_threshold_prefers_recall_when_misses_are_costly():
    y = np.array([0] * 90 + [1] * 10)
    p = np.concatenate([np.linspace(0, 0.6, 90), np.linspace(0.3, 1, 10)])
    low, _ = best_threshold(y, p, c_fn=50, c_fp=1)
    high, _ = best_threshold(y, p, c_fn=1, c_fp=50)
    assert low < high


def test_alert_levels():
    assert alert_level(10) == "Critical" and alert_level(25) == "Warning"
    assert alert_level(50) == "Watch" and alert_level(120) == "Healthy"


def test_cost_simulation_predictive_beats_reactive():
    oof = pd.read_csv("reports/oof_predictions.csv")
    r = optimise(oof)
    assert r["predictive"]["total"] < r["reactive"]["total"]
    assert simulate_reactive(oof)["unplanned_failures"] == oof["unit"].nunique()
    assert simulate_predictive(oof, 0)["unplanned_failures"] >= 0


def test_predictor_on_real_engines(data):
    _, test = data
    pred = Predictor()
    unit = test[test.unit == 1][["cycle"] + SENSORS]
    out = pred.score_history(unit)
    assert 0 <= out["predicted_rul_cycles"] <= RUL_CAP and 0 <= out["failure_probability"] <= 1
    assert len(out["top_drivers"]) == 5
    # degraded engine should look worse than a fresh one
    train, _ = data
    fresh = train[(train.unit == 1) & (train.cycle <= 20)][["cycle"] + SENSORS]
    dead = train[(train.unit == 1)][["cycle"] + SENSORS]
    assert pred.score_history(dead)["predicted_rul_cycles"] < pred.score_history(fresh)["predicted_rul_cycles"]


def test_api_end_to_end(data):
    _, test = data
    hist = test[test.unit == 5][["cycle"] + SENSORS].to_dict("records")
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "ok"
        assert "sensors" in c.get("/model-info").json()
        r = c.post("/predict/rul", json={"engine_id": "e5", "history": hist})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["engine_id"] == "e5" and body["alert_level"] in {"Healthy", "Watch", "Warning", "Critical"}
        assert c.post("/predict/rul", json={"history": []}).status_code == 422
        bad = dict(hist[0]); bad.pop("s2")
        assert c.post("/predict/rul", json={"history": [bad]}).status_code == 422
        dup = c.post("/predict/rul", json={"history": [hist[0], hist[0]]})
        assert dup.status_code == 422
        m = c.post("/predict/machine-failure", json={"type": "L", "air_temp": 298.1, "process_temp": 308.6,
                                                     "rot_speed": 1551, "torque": 42.8, "tool_wear": 0})
        assert m.status_code == 200 and 0 <= m.json()["failure_probability"] <= 1
        worn = c.post("/predict/machine-failure", json={"type": "L", "air_temp": 300, "process_temp": 309,
                                                        "rot_speed": 1300, "torque": 68, "tool_wear": 230})
        assert worn.json()["failure_probability"] > m.json()["failure_probability"]
        assert c.post("/predict/machine-failure", json={"type": "X", "air_temp": 300, "process_temp": 309,
                                                        "rot_speed": 1300, "torque": 68, "tool_wear": 2}).status_code == 422

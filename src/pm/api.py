"""FastAPI service.  Run: uvicorn pm.api:app --port 8000   (docs at /docs)"""
from contextlib import asynccontextmanager
from typing import List, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, create_model

from . import __version__
from .config import SENSORS
from .inference import Predictor

Reading = create_model(
    "Reading",
    cycle=(int, Field(..., ge=1, description="Operating cycle number")),
    **{s: (float, Field(..., description=f"Sensor {s}")) for s in SENSORS},
)


class EngineRequest(BaseModel):
    engine_id: str = Field("engine-1", description="Free-text identifier, echoed back")
    history: List[Reading] = Field(..., min_length=1, description="Cycle-by-cycle sensor history (oldest first)")
    explain: bool = True


class Driver(BaseModel):
    sensor: str
    impact: float = Field(..., description="Contribution to predicted RUL in cycles (negative = shortens life)")


class EngineResponse(BaseModel):
    engine_id: str
    cycle: int
    predicted_rul_cycles: float
    health_score: float
    alert_level: Literal["Healthy", "Watch", "Warning", "Critical"]
    failure_probability: float
    failure_within_horizon: bool
    horizon_cycles: int
    anomaly_index: float
    anomaly_flag: bool
    history_cycles_used: int
    top_drivers: List[Driver] = []


class MachineRequest(BaseModel):
    type: Literal["L", "M", "H"]
    air_temp: float = Field(..., ge=250, le=350, description="Kelvin")
    process_temp: float = Field(..., ge=250, le=400, description="Kelvin")
    rot_speed: float = Field(..., gt=0, le=5000, description="rpm")
    torque: float = Field(..., ge=0, le=200, description="Nm")
    tool_wear: float = Field(..., ge=0, le=500, description="minutes")
    explain: bool = True


class FeatureDriver(BaseModel):
    feature: str
    impact: float


class MachineResponse(BaseModel):
    failure_probability: float
    threshold: float
    predicted_failure: bool
    top_drivers: List[FeatureDriver] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = Predictor()
    yield


app = FastAPI(title="Predictive Maintenance API", version=__version__, lifespan=lifespan,
              description="RUL, failure probability, anomaly index and per-sensor explanations "
                          "for turbofan engines (C-MAPSS FD001) plus machine-failure scoring (AI4I 2020).")


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__}


@app.get("/model-info")
def model_info():
    m = app.state.predictor.meta
    return {k: m[k] for k in ("version", "subset", "rul_cap", "horizon", "sensors",
                              "failure_threshold", "ai4i_threshold", "alert_levels")}


@app.post("/predict/rul", response_model=EngineResponse)
def predict_rul(req: EngineRequest):
    hist = pd.DataFrame([r.model_dump() for r in req.history])
    if hist["cycle"].duplicated().any():
        raise HTTPException(422, "Duplicate cycle numbers in history.")
    res = app.state.predictor.score_history(hist, explain=req.explain)
    return EngineResponse(engine_id=req.engine_id, **res)


@app.post("/predict/machine-failure", response_model=MachineResponse)
def predict_machine(req: MachineRequest):
    rec = req.model_dump(exclude={"explain"})
    return MachineResponse(**app.state.predictor.score_machine(rec, explain=req.explain))

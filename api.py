
"""FastAPI endpoints for the simulated boring-bar vibration-control demo."""
import math
import os
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pathlib import Path
from stable_baselines3 import SAC
from functools import lru_cache
from experiment_engine import simulate_experiment, PLANT_PRESETS
from policy_registry import list_policies, get_policy
from training_jobs import start_job, list_jobs, get_job, cancel_job
from experiment_engine import resolve_plant_preset, plant_to_dict
from deployment_service import (
    DEFAULT_MODEL_PATH,
    compare_controllers,
    get_model_status,
    run_simulation,
)

app = FastAPI(
    title="Boring-Bar Vibration Control API",
    description=(
        "Educational API for a seven-mode simulated vibration-control model. "
        "This API does not control physical machinery."
    ),
    version="1.0.0",
)

MODEL_PATH = os.getenv("SAC_MODEL_PATH", DEFAULT_MODEL_PATH)


class SimulationRequest(BaseModel):
    controller: Literal["No control", "PID", "SAC"] = "SAC"
    amplitude_n: float = Field(default=1.0, ge=0.0, le=10.0)
    frequency_hz: float = Field(default=174.0, ge=0.0, le=1000.0)
    phase_rad: float = Field(default=0.0, ge=-100.0, le=100.0)
    duration_s: float = Field(default=0.2, gt=0.0, le=2.0)


class ExperimentRequest(BaseModel):
    controller: Literal["No control", "PID", "SAC"] = "SAC"
    policy_id: str | None = None
    plant_preset: str = "paper_nominal"
    variations: dict = Field(default_factory=dict)
    cutting_force: dict = Field(default_factory=lambda: {"kind":"sinusoidal", "amplitude_n":1.0,"frequency_hz":174.0,"phase_rad":0.0})
    noise: dict = Field(default_factory=lambda: {"process_std_n":0.0,"sensor_std_m":0.0})
    duration_s: float = Field(default=0.2, gt=0.0, le=2.0)
    seed: int = 42

class TrainingRequest(BaseModel):
    name: str = ""
    training_mode: Literal["individual", "generalized"] = "individual"
    total_timesteps: int = Field(default=50000, ge=1, le=5000000)
    seed: int = 42
    plant_variation_pct: float = Field(default=10.0, ge=0.0, le=30.0)
    plant_preset: str = "paper_nominal"
    variations: dict = Field(default_factory=dict)
    cutting_force: dict = Field(default_factory=lambda: {"kind":"sinusoidal", "amplitude_n":1.0,"frequency_hz":174.0,"phase_rad":0.0})
    noise: dict = Field(default_factory=lambda: {"process_std_n":0.0,"sensor_std_m":0.0})

class ComparisonRequest(BaseModel):
    amplitude_n: float = Field(default=1.0, ge=0.0, le=10.0)
    frequency_hz: float = Field(default=174.0, ge=0.0, le=1000.0)
    phase_rad: float = Field(default=0.0, ge=-100.0, le=100.0)
    duration_s: float = Field(default=0.2, gt=0.0, le=2.0)


@app.get("/")
def root():
    return {
        "service": "Boring-Bar Vibration Control API",
        "simulation_only": True,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    status = get_model_status(MODEL_PATH)
    return {
        "status": "ok",
        "simulation_only": True,
        "sac_model": status,
    }


@app.get("/model")
def model_status():
    return get_model_status(MODEL_PATH)


@app.post("/simulate")
def simulate(request: SimulationRequest):
    try:
        return run_simulation(
            controller_name=request.controller,
            amplitude_n=request.amplitude_n,
            frequency_hz=request.frequency_hz,
            phase_rad=request.phase_rad,
            duration_s=request.duration_s,
            model_path=MODEL_PATH,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        # Avoid returning internal exception details to API clients.
        raise HTTPException(
            status_code=500,
            detail="Simulation failed. Check the server terminal logs.",
        ) from exc


@app.post("/compare")
def compare(request: ComparisonRequest):
    try:
        return {
            "simulation_only": True,
            "results": compare_controllers(
                amplitude_n=request.amplitude_n,
                frequency_hz=request.frequency_hz,
                phase_rad=request.phase_rad,
                duration_s=request.duration_s,
                model_path=MODEL_PATH,
            ),
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Controller comparison failed. Check server logs.",
        ) from exc


@lru_cache(maxsize=16)
def _load_registered_policy(path):
    return SAC.load(path)

@app.get("/plant-presets")
def plant_presets():
    return {"presets": [{"id": k, "label": v["label"]} for k,v in PLANT_PRESETS.items()],
            "nominal_plant": plant_to_dict()}

@app.get("/policies")
def policies():
    return {"policies": list_policies()}

@app.post("/experiment/simulate")
def experiment_simulate(request: ExperimentRequest):
    try:
        plant_cfg, merged_variations = resolve_plant_preset(request.plant_preset, request.variations)
        payload=request.model_dump() if hasattr(request,"model_dump") else request.dict()
        payload["plant"]=plant_to_dict()
        payload["variations"]=merged_variations
        selected=None; warning=None
        if request.controller == "SAC":
            if not request.policy_id: raise ValueError("Choose a registered SAC policy before running SAC.")
            selected=get_policy(request.policy_id)
            if not selected.get("model_exists"): raise FileNotFoundError("The selected policy checkpoint is missing.")
            # A mismatch is permitted, but is explicitly reported.
            changed_variations = any(float(vv or 0) != 0 for mode_values in merged_variations.values() for vv in mode_values.values())
            train_force = selected.get("cutting_force", {})
            noise_mismatch = any(float(v or 0) != 0 for v in request.noise.values()) if selected.get("legacy_checkpoint") else selected.get("noise", {}) != request.noise
            if selected.get("legacy_checkpoint"):
                force_mismatch = request.cutting_force.get("kind", "sinusoidal") != "sinusoidal" or float(request.cutting_force.get("noise_std_n", 0)) != 0
                plant_mismatch = request.plant_preset != "paper_nominal" or changed_variations
            else:
                structural_keys = ("kind", "harmonics", "modulation_depth", "modulation_frequency_hz", "noise_std_n")
                force_mismatch = any(train_force.get(k) != request.cutting_force.get(k) for k in structural_keys)
                if selected.get("training_mode") == "individual" and request.cutting_force.get("kind") != "sinusoidal":
                    force_mismatch = force_mismatch or any(train_force.get(k) != request.cutting_force.get(k) for k in ("amplitude_n", "frequency_hz", "phase_rad"))
                plant_mismatch = (selected.get("plant_preset", "paper_nominal") != request.plant_preset
                                  or selected.get("training_variations", {}) != merged_variations)
            if force_mismatch or plant_mismatch or noise_mismatch:
                warning="Evaluation configuration differs from the policy's recorded training configuration; results are an out-of-distribution test."
            payload["controller"]="SAC"
            model=_load_registered_policy(selected["model_path"])
        else: model=None
        result=simulate_experiment(payload,model=model)
        result["policy"]={"id":selected.get("id"),"name":selected.get("name"),"training_mode":selected.get("training_mode")} if selected else None
        result["warning"]=warning
        return result
    except KeyError as exc: raise HTTPException(status_code=404,detail=str(exc)) from exc
    except FileNotFoundError as exc: raise HTTPException(status_code=503,detail=str(exc)) from exc
    except ValueError as exc: raise HTTPException(status_code=422,detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500,detail=f"Experiment failed: {exc}") from exc

@app.post("/training/jobs")
def create_training_job(request: TrainingRequest):
    try:
        payload=request.model_dump() if hasattr(request,"model_dump") else request.dict()
        return start_job(payload)
    except ValueError as exc: raise HTTPException(status_code=422,detail=str(exc)) from exc

@app.get("/training/jobs")
def training_job_list(): return {"jobs": list_jobs()}

@app.get("/training/jobs/{job_id}")
def training_job_status(job_id: str):
    try: return get_job(job_id)
    except KeyError as exc: raise HTTPException(status_code=404,detail="Training job not found") from exc

@app.post("/training/jobs/{job_id}/cancel")
def training_job_cancel(job_id: str):
    try: return cancel_job(job_id)
    except KeyError as exc: raise HTTPException(status_code=404,detail="Training job not found") from exc

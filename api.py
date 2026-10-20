"""FastAPI endpoints for the simulated boring-bar vibration-control demo."""

import os
import shutil
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from sac_training import train_sac

from deployment_service import (
    _load_sac_model,
    DEFAULT_MODEL_PATH,
    MAX_PLANT_VARIATION_PERCENT,
    MAX_PROCESS_NOISE_STD_N,
    MAX_CUTTING_FORCE_NOISE_STD_N,
    MAX_HARMONICS,
    compare_controllers,
    get_model_status,
    run_simulation,
    run_robust_evaluation,
    ROBUST_AMPLITUDES_N,
    ROBUST_FREQUENCIES_HZ,
    ROBUST_PHASES_RAD,
    ROBUST_DEFAULT_DURATION_S,
)


app = FastAPI(
    title="Boring-Bar Vibration Control API",
    description=(
        "Educational API for a seven-mode simulated vibration-control model. "
        "This API does not control physical machinery."
    ),
    version="1.0.0",
)

MODEL_PATH = os.getenv(
    "SAC_MODEL_PATH",
    DEFAULT_MODEL_PATH,
)


class SimulationRequest(BaseModel):
    model_path: str | None = None
    controller: Literal[
        "No control",
        "PID",
        "SAC",
    ] = "SAC"

    amplitude_n: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
    )

    frequency_hz: float = Field(
        default=174.0,
        ge=0.0,
        le=1000.0,
    )

    phase_rad: float = Field(
        default=0.0,
        ge=-100.0,
        le=100.0,
    )

    duration_s: float = Field(
        default=0.2,
        gt=0.0,
        le=2.0,
    )

    pid_gains: dict[str, float] = Field(
        default_factory=lambda: {
            "kp": 180000.0,
            "ki": 0.0,
            "kd": 0.0,
            "integral_limit_m_s": 1.0e-3,
        },
        description="PID gains and integral limit. Only relevant to PID.",
    )

    plant_variations: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            f"Physical plant parameter changes in percent. "
            f"Allowed magnitude: +/-{MAX_PLANT_VARIATION_PERCENT:.0f}%."
        ),
    )

    process_noise_std_n: float = Field(
        default=0.0,
        ge=0.0,
        le=MAX_PROCESS_NOISE_STD_N,
    )

    process_noise_seed: int = Field(
        default=42,
        ge=0,
    )

    harmonic_amplitudes_ratio: list[float] = Field(
        default_factory=list,
        description="Relative amplitudes of harmonics 2, 3, ... as fractions of the fundamental.",
    )

    harmonic_phase_rad: list[float] = Field(
        default_factory=list,
        description="Absolute phase of harmonics 2, 3, ... in radians.",
    )

    cutting_force_noise_std_n: float = Field(
        default=0.0,
        ge=0.0,
        le=MAX_CUTTING_FORCE_NOISE_STD_N,
    )

    cutting_force_noise_seed: int = Field(
        default=42,
        ge=0,
    )


class ComparisonRequest(BaseModel):
    model_path: str | None = None
    amplitude_n: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
    )

    frequency_hz: float = Field(
        default=174.0,
        ge=0.0,
        le=1000.0,
    )

    phase_rad: float = Field(
        default=0.0,
        ge=-100.0,
        le=100.0,
    )

    duration_s: float = Field(
        default=0.2,
        gt=0.0,
        le=2.0,
    )

    pid_gains: dict[str, float] = Field(
        default_factory=lambda: {
            "kp": 180000.0,
            "ki": 0.0,
            "kd": 0.0,
            "integral_limit_m_s": 1.0e-3,
        },
        description="PID gains and integral limit. Only relevant to PID.",
    )

    plant_variations: dict[str, Any] = Field(
        default_factory=dict,
    )

    process_noise_std_n: float = Field(
        default=0.0,
        ge=0.0,
        le=MAX_PROCESS_NOISE_STD_N,
    )

    process_noise_seed: int = Field(
        default=42,
        ge=0,
    )

    harmonic_amplitudes_ratio: list[float] = Field(
        default_factory=list,
        description="Relative amplitudes of harmonics 2, 3, ... as fractions of the fundamental.",
    )

    harmonic_phase_rad: list[float] = Field(
        default_factory=list,
        description="Absolute phase of harmonics 2, 3, ... in radians.",
    )

    cutting_force_noise_std_n: float = Field(
        default=0.0,
        ge=0.0,
        le=MAX_CUTTING_FORCE_NOISE_STD_N,
    )

    cutting_force_noise_seed: int = Field(
        default=42,
        ge=0,
    )


class RobustEvaluationRequest(BaseModel):
    model_path: str | None = None
    amplitudes_n: list[float] = Field(default_factory=lambda: list(ROBUST_AMPLITUDES_N), min_length=1, max_length=25)
    frequencies_hz: list[float] = Field(default_factory=lambda: list(ROBUST_FREQUENCIES_HZ), min_length=1, max_length=25)
    phases_rad: list[float] = Field(default_factory=lambda: list(ROBUST_PHASES_RAD), min_length=1, max_length=25)
    duration_s: float = Field(default=ROBUST_DEFAULT_DURATION_S, gt=0.0, le=2.0)
    plant_variations: dict[str, Any] = Field(default_factory=dict)
    process_noise_std_n: float = Field(default=0.0, ge=0.0, le=MAX_PROCESS_NOISE_STD_N)
    process_noise_seed: int = Field(default=42, ge=0)
    harmonic_amplitudes_ratio: list[float] = Field(default_factory=list, max_length=MAX_HARMONICS)
    harmonic_phase_rad: list[float] = Field(default_factory=list, max_length=MAX_HARMONICS)
    cutting_force_noise_std_n: float = Field(default=0.0, ge=0.0, le=MAX_CUTTING_FORCE_NOISE_STD_N)
    cutting_force_noise_seed: int = Field(default=42, ge=0)
    pid_gains: dict[str, float] = Field(default_factory=lambda: {
        "kp": 180000.0, "ki": 0.0, "kd": 0.0, "integral_limit_m_s": 1.0e-3
    })


class SACTrainingRequest(BaseModel):
    model_name: str = Field(min_length=1, max_length=180)
    total_timesteps: int = Field(default=300_000, gt=0, le=20_000_000)
    seed: int = Field(default=42, ge=0)
    episode_duration_s: float = Field(default=0.2, gt=0.0, le=2.0)
    amplitude_range_n: list[float] = Field(default_factory=lambda: [0.5, 1.5], min_length=2, max_length=2)
    frequency_range_hz: list[float] = Field(default_factory=lambda: [150.0, 200.0], min_length=2, max_length=2)
    phase_range_rad: list[float] = Field(default_factory=lambda: [-3.141592653589793, 3.141592653589793], min_length=2, max_length=2)
    harmonic_amplitudes_ratio: list[float] = Field(default_factory=list, max_length=5)
    harmonic_phase_rad: list[float] = Field(default_factory=list, max_length=5)
    cutting_force_noise_std_n: float = Field(default=0.0, ge=0.0, le=MAX_CUTTING_FORCE_NOISE_STD_N)
    cutting_force_noise_seed: int = Field(default=42, ge=0)
    process_noise_std_n: float = Field(default=0.0, ge=0.0, le=MAX_PROCESS_NOISE_STD_N)
    process_noise_seed: int = Field(default=42, ge=0)
    plant_variations: dict[str, Any] = Field(default_factory=dict)
    sac_action_repeat: int = Field(default=5, ge=1, le=100)
    observation_history_length: int = Field(default=8, ge=1, le=64)
    learning_rate: float = Field(default=3e-4, gt=0.0)
    buffer_size: int = Field(default=200_000, gt=0, le=5_000_000)
    learning_starts: int = Field(default=5_000, ge=0)
    batch_size: int = Field(default=256, gt=0, le=8192)
    tau: float = Field(default=0.005, gt=0.0, le=1.0)
    gamma: float = Field(default=0.99, gt=0.0, le=1.0)
    train_freq: int = Field(default=1, ge=1, le=1000)
    gradient_steps: int = Field(default=1, ge=1, le=1000)
    training_mode: Literal["specified", "generalized"] = "specified"
    generalized_randomization: dict[str, Any] = Field(default_factory=dict)


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
    status = get_model_status(
        MODEL_PATH
    )

    return {
        "status": "ok",
        "simulation_only": True,
        "sac_model": status,
        "plant_parameter_variation_limit_percent": (
            MAX_PLANT_VARIATION_PERCENT
        ),
        "process_noise_max_std_n": (
            MAX_PROCESS_NOISE_STD_N
        ),
        "cutting_force_noise_max_std_n": MAX_CUTTING_FORCE_NOISE_STD_N,
        "max_harmonics": MAX_HARMONICS,
    }


@app.get("/model")
def model_status():
    return get_model_status(
        MODEL_PATH
    )


@app.post("/simulate")
def simulate(request: SimulationRequest):
    try:
        return run_simulation(
            controller_name=request.controller,
            amplitude_n=request.amplitude_n,
            frequency_hz=request.frequency_hz,
            phase_rad=request.phase_rad,
            duration_s=request.duration_s,
            model_path=request.model_path or MODEL_PATH,
            plant_variations=request.plant_variations,
            process_noise_std_n=request.process_noise_std_n,
            process_noise_seed=request.process_noise_seed,
            harmonic_amplitudes_ratio=request.harmonic_amplitudes_ratio,
            harmonic_phase_rad=request.harmonic_phase_rad,
            cutting_force_noise_std_n=request.cutting_force_noise_std_n,
            cutting_force_noise_seed=request.cutting_force_noise_seed,
            pid_gains=request.pid_gains,
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Simulation failed. Check the server terminal logs."
            ),
        ) from exc


@app.post("/robust-evaluation")
def robust_evaluation(request: RobustEvaluationRequest):
    try:
        return run_robust_evaluation(
            model_path=request.model_path or MODEL_PATH,
            amplitudes_n=request.amplitudes_n,
            frequencies_hz=request.frequencies_hz,
            phases_rad=request.phases_rad,
            duration_s=request.duration_s,
            plant_variations=request.plant_variations,
            process_noise_std_n=request.process_noise_std_n,
            process_noise_seed=request.process_noise_seed,
            harmonic_amplitudes_ratio=request.harmonic_amplitudes_ratio,
            harmonic_phase_rad=request.harmonic_phase_rad,
            cutting_force_noise_std_n=request.cutting_force_noise_std_n,
            cutting_force_noise_seed=request.cutting_force_noise_seed,
            pid_gains=request.pid_gains,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Robust evaluation failed. Check the server terminal logs.") from exc


@app.get("/models")
def list_models():
    root = os.path.dirname(MODEL_PATH) or "."
    directory = os.path.abspath(root)
    if not os.path.isdir(directory):
        return {"models": [], "directory": directory}

    models = []
    for path in sorted(
        (p for p in __import__("pathlib").Path(directory).glob("*.zip") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        metadata_path = path.with_suffix(".json")
        metadata = None
        if metadata_path.is_file():
            try:
                import json
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                metadata = None
        models.append({
            "name": path.name,
            "path": str(path),
            "modified_time": path.stat().st_mtime,
            "metadata": metadata,
        })

    return {"models": models, "directory": directory}


@app.post("/models/import")
async def import_model(file: UploadFile = File(...)):
    """Import a SAC .zip model supplied by the user into results/external_models."""
    filename = Path(file.filename or "").name
    if not filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Please select a SAC model .zip file.")

    target_dir = Path(os.path.dirname(MODEL_PATH) or ".") / "external_models"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / filename

    with target_path.open("wb") as output:
        shutil.copyfileobj(file.file, output)

    _load_sac_model.cache_clear()
    return {
        "status": "imported",
        "model_path": str(target_path.resolve()),
        "model_name": target_path.name,
    }


@app.post("/train")
def train(request: SACTrainingRequest):
    try:
        name = request.model_name.strip()
        if not name.lower().endswith(".zip"):
            name = name + ".zip"
        model_path = os.path.join(os.path.dirname(MODEL_PATH) or ".", name)

        model, metadata = train_sac(
            total_timesteps=request.total_timesteps,
            seed=request.seed,
            model_path=model_path,
            plant_variations=request.plant_variations,
            amplitude_range_n=request.amplitude_range_n,
            frequency_range_hz=request.frequency_range_hz,
            phase_range_rad=request.phase_range_rad,
            harmonic_amplitudes_ratio=request.harmonic_amplitudes_ratio,
            harmonic_phase_rad=request.harmonic_phase_rad,
            cutting_force_noise_std_n=request.cutting_force_noise_std_n,
            cutting_force_noise_seed=request.cutting_force_noise_seed,
            process_noise_std_n=request.process_noise_std_n,
            process_noise_seed=request.process_noise_seed,
            episode_duration_s=request.episode_duration_s,
            sac_action_repeat=request.sac_action_repeat,
            observation_history_length=request.observation_history_length,
            learning_rate=request.learning_rate,
            buffer_size=request.buffer_size,
            learning_starts=request.learning_starts,
            batch_size=request.batch_size,
            tau=request.tau,
            gamma=request.gamma,
            train_freq=request.train_freq,
            gradient_steps=request.gradient_steps,
            training_mode=request.training_mode,
            generalized_randomization=request.generalized_randomization,
        )
        _load_sac_model.cache_clear()
        return {"status": "completed", "model_path": str(model_path), "metadata": metadata}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"SAC training failed. Check server logs: {exc}") from exc


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
                model_path=request.model_path or MODEL_PATH,
                plant_variations=request.plant_variations,
                process_noise_std_n=request.process_noise_std_n,
                process_noise_seed=request.process_noise_seed,
                harmonic_amplitudes_ratio=request.harmonic_amplitudes_ratio,
                harmonic_phase_rad=request.harmonic_phase_rad,
                cutting_force_noise_std_n=request.cutting_force_noise_std_n,
                cutting_force_noise_seed=request.cutting_force_noise_seed,
                pid_gains=request.pid_gains,
            ),
        }

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Controller comparison failed. Check server logs."
            ),
        ) from exc

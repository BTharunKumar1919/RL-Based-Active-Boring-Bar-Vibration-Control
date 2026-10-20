"""FastAPI endpoints for the simulated boring-bar vibration-control demo."""

import os
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from deployment_service import (
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
            model_path=MODEL_PATH,
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
            model_path=MODEL_PATH,
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

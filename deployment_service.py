"""Simulation service for the current-driven seven-mode vibration model.

This service does not connect to physical sensors, amplifiers, or machinery.
It accepts optional plant-parameter perturbations so the same saved controller
can be evaluated on nearby plant models without retraining it.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

from config import (
    PLANT,
    TRAINING,
    EVALUATION,
    DisturbanceConfig,
)
from models import build_plant_config
from experiments import simulate_controller, calculate_metrics


DEFAULT_MODEL_PATH = "results/sac_vibration_model_7mode.zip"
ALLOWED_CONTROLLERS = (
    "No control",
    "PID",
    "SAC",
)

MAX_PLANT_VARIATION_PERCENT = 20.0
MAX_PROCESS_NOISE_STD_N = 1.0


@lru_cache(maxsize=2)
def _load_sac_model(model_path):
    return SAC.load(model_path)


def get_model_status(model_path=DEFAULT_MODEL_PATH):
    path = Path(model_path)

    return {
        "model_path": str(path),
        "model_exists": path.is_file(),
        "model_type": "seven-mode current-driven modal model",
        "message": (
            "Trained SAC model is available."
            if path.is_file()
            else "Trained SAC model not found. Retrain SAC for the updated plant."
        ),
    }


def run_simulation(
    controller_name="SAC",
    amplitude_n=EVALUATION.nominal_amplitude_n,
    frequency_hz=EVALUATION.nominal_frequency_hz,
    phase_rad=0.0,
    duration_s=EVALUATION.duration_s,
    model_path=DEFAULT_MODEL_PATH,
    plant_variations=None,
    process_noise_std_n=0.0,
    process_noise_seed=42,
):
    """Run one controller on one disturbance and one plant model."""

    if controller_name not in ALLOWED_CONTROLLERS:
        raise ValueError(
            f"controller_name must be one of {ALLOWED_CONTROLLERS}."
        )

    values = {
        "amplitude_n": amplitude_n,
        "frequency_hz": frequency_hz,
        "phase_rad": phase_rad,
        "duration_s": duration_s,
        "process_noise_std_n": process_noise_std_n,
    }

    for name, value in values.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
        ):
            raise ValueError(
                f"{name} must be a finite number."
            )

    if amplitude_n < 0 or frequency_hz < 0:
        raise ValueError(
            "Amplitude and frequency must be non-negative."
        )

    if duration_s <= 0 or duration_s > 2.0:
        raise ValueError(
            "Duration must be in (0, 2] seconds."
        )

    if process_noise_std_n < 0 or process_noise_std_n > MAX_PROCESS_NOISE_STD_N:
        raise ValueError(
            f"process_noise_std_n must be between 0 and {MAX_PROCESS_NOISE_STD_N:.2f} N."
        )

    if isinstance(process_noise_seed, bool) or not isinstance(process_noise_seed, int):
        raise ValueError("process_noise_seed must be an integer.")

    plant_config = build_plant_config(
        plant_variations,
        max_percent=MAX_PLANT_VARIATION_PERCENT,
    )

    step_count = int(
        round(duration_s / plant_config.dt_s)
    )

    if step_count < 1 or step_count > 20_000:
        raise ValueError(
            "Requested duration produces an invalid step count."
        )

    disturbance = DisturbanceConfig(
        float(amplitude_n),
        float(frequency_hz),
        float(phase_rad),
    )

    model = None

    if controller_name == "SAC":
        status = get_model_status(model_path)

        if not status["model_exists"]:
            raise FileNotFoundError(
                status["message"]
            )

        model = _load_sac_model(
            str(model_path)
        )

    data = simulate_controller(
        controller_name,
        disturbance,
        float(duration_s),
        model=model,
        plant_config=plant_config,
        process_noise_std_n=float(process_noise_std_n),
        process_noise_seed=int(process_noise_seed),
    )

    metrics = calculate_metrics(
        data
    )

    records = data.replace(
        [np.inf, -np.inf],
        np.nan,
    ).to_dict(orient="records")

    if any(
        pd.isna(value)
        for record in records
        for value in record.values()
    ):
        raise RuntimeError(
            "Simulation produced invalid numeric output."
        )

    return {
        "simulation_only": True,
        "model_type": "seven-mode current-driven modal model",
        "controller": controller_name,
        "inputs": {
            "amplitude_n": float(amplitude_n),
            "frequency_hz": float(frequency_hz),
            "phase_rad": float(phase_rad),
            "duration_s": float(duration_s),
        },
        "plant_variations": plant_variations or {},
        "plant_variation_limit_percent": MAX_PLANT_VARIATION_PERCENT,
        "process_noise": {
            "enabled": bool(process_noise_std_n > 0.0),
            "std_n": float(process_noise_std_n),
            "seed": int(process_noise_seed),
            "description": "Additive zero-mean force noise applied to the force-driven plant mode.",
        },
        "plant_parameters": {
            "current_modes": [
                {
                    "natural_frequency_hz": mode.natural_frequency_hz,
                    "damping_ratio": mode.damping_ratio,
                    "k_yI": mode.k_yI,
                }
                for mode in plant_config.current_modes
            ],
            "force_mode": {
                "natural_frequency_hz": plant_config.force_mode.natural_frequency_hz,
                "damping_ratio": plant_config.force_mode.damping_ratio,
                "stiffness_n_m": plant_config.force_mode.stiffness_n_m,
            },
            "actuator_bandwidth_hz": plant_config.actuator_bandwidth_hz,
            "current_slew_rate_a_s": plant_config.current_slew_rate_a_s,
            "continuous_current_limit_a": plant_config.continuous_current_limit_a,
            "dt_s": plant_config.dt_s,
        },
        "current_limits_a": {
            "continuous": plant_config.continuous_current_limit_a,
            "peak": plant_config.peak_current_limit_a,
        },
        "training_frequency_range_hz": [
            TRAINING.force_frequency_min_hz,
            TRAINING.force_frequency_max_hz,
        ],
        "frequency_outside_training_range": not (
            TRAINING.force_frequency_min_hz
            <= frequency_hz
            <= TRAINING.force_frequency_max_hz
        ),
        "sac_action_repeat": int(
            TRAINING.sac_action_repeat
        ),
        "sac_action_interval_s": float(
            plant_config.dt_s
            * TRAINING.sac_action_repeat
        ),
        "observation_history_length": int(
            TRAINING.observation_history_length
        ),
        "metrics": metrics,
        "timeseries": records,
    }


def compare_controllers(
    amplitude_n=EVALUATION.nominal_amplitude_n,
    frequency_hz=EVALUATION.nominal_frequency_hz,
    phase_rad=0.0,
    duration_s=EVALUATION.duration_s,
    model_path=DEFAULT_MODEL_PATH,
    plant_variations=None,
    process_noise_std_n=0.0,
    process_noise_seed=42,
):
    """Compare all three controllers on exactly the same plant/noise."""

    results = []

    for name in ALLOWED_CONTROLLERS:
        result = run_simulation(
            name,
            amplitude_n,
            frequency_hz,
            phase_rad,
            duration_s,
            model_path,
            plant_variations,
            process_noise_std_n,
            process_noise_seed,
        )

        results.append(
            {
                "controller": name,
                **result["metrics"],
            }
        )

    return results

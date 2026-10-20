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
MAX_CUTTING_FORCE_NOISE_STD_N = 1.0
MAX_HARMONICS = 5

# Default robust-evaluation grid. This matches the standalone V3 evaluator
# while using the same simulation path as the App.
ROBUST_AMPLITUDES_N = (0.5, 1.0, 1.5)
ROBUST_FREQUENCIES_HZ = (150.0, 160.0, 174.0, 185.0, 200.0, 205.0, 215.0)
ROBUST_PHASES_RAD = (0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0)
ROBUST_DEFAULT_DURATION_S = 0.2
ROBUST_TRANSIENT_FRACTION = 0.2


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
    harmonic_amplitudes_ratio=None,
    harmonic_phase_rad=None,
    cutting_force_noise_std_n=0.0,
    cutting_force_noise_seed=42,
    pid_gains=None,
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
        "cutting_force_noise_std_n": cutting_force_noise_std_n,
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

    if cutting_force_noise_std_n < 0 or cutting_force_noise_std_n > MAX_CUTTING_FORCE_NOISE_STD_N:
        raise ValueError(f"cutting_force_noise_std_n must be between 0 and {MAX_CUTTING_FORCE_NOISE_STD_N:.2f} N.")

    if isinstance(cutting_force_noise_seed, bool) or not isinstance(cutting_force_noise_seed, int):
        raise ValueError("cutting_force_noise_seed must be an integer.")

    harmonic_amplitudes_ratio = [] if harmonic_amplitudes_ratio is None else list(harmonic_amplitudes_ratio)
    harmonic_phase_rad = [] if harmonic_phase_rad is None else list(harmonic_phase_rad)
    if len(harmonic_amplitudes_ratio) != len(harmonic_phase_rad):
        raise ValueError("harmonic_amplitudes_ratio and harmonic_phase_rad must have the same length.")
    if len(harmonic_amplitudes_ratio) > MAX_HARMONICS:
        raise ValueError(f"At most {MAX_HARMONICS} harmonics are allowed.")
    for index, value in enumerate(harmonic_amplitudes_ratio, start=2):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0 or value > 1.0:
            raise ValueError(f"Harmonic {index} amplitude ratio must be between 0 and 1.")
    for index, value in enumerate(harmonic_phase_rad, start=2):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"Harmonic {index} phase must be finite.")

    pid_gains = {} if pid_gains is None else dict(pid_gains)
    for key in ("kp", "ki", "kd", "integral_limit_m_s"):
        value = pid_gains.get(key, None)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"PID {key} must be a finite non-negative number.")

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
        amplitude_n=float(amplitude_n),
        frequency_hz=float(frequency_hz),
        phase_rad=float(phase_rad),
        harmonic_amplitudes_ratio=tuple(float(value) for value in harmonic_amplitudes_ratio),
        harmonic_phase_rad=tuple(float(value) for value in harmonic_phase_rad),
        noise_std_n=float(cutting_force_noise_std_n),
        noise_seed=int(cutting_force_noise_seed),
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
        pid_gains=pid_gains,
        process_noise_std_n=float(process_noise_std_n),
        process_noise_seed=int(process_noise_seed),
    )

    metrics = calculate_metrics(
        data,
        plant_config=plant_config,
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
            "harmonic_amplitudes_ratio": [float(value) for value in harmonic_amplitudes_ratio],
            "harmonic_phase_rad": [float(value) for value in harmonic_phase_rad],
            "cutting_force_noise_std_n": float(cutting_force_noise_std_n),
            "cutting_force_noise_seed": int(cutting_force_noise_seed),
            "pid_gains": {
                "kp": float(pid_gains.get("kp", 180000.0)),
                "ki": float(pid_gains.get("ki", 0.0)),
                "kd": float(pid_gains.get("kd", 0.0)),
                "integral_limit_m_s": float(pid_gains.get("integral_limit_m_s", 1.0e-3)),
            },
        },
        "plant_variations": plant_variations or {},
        "plant_variation_limit_percent": MAX_PLANT_VARIATION_PERCENT,
        "cutting_force": {
            "harmonics": [
                {"order": int(order), "amplitude_ratio": float(ratio), "phase_rad": float(phase), "amplitude_n": float(amplitude_n * ratio)}
                for order, (ratio, phase) in enumerate(zip(harmonic_amplitudes_ratio, harmonic_phase_rad), start=2)
            ],
            "noise": {"enabled": bool(cutting_force_noise_std_n > 0.0), "std_n": float(cutting_force_noise_std_n), "seed": int(cutting_force_noise_seed)},
        },
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
    harmonic_amplitudes_ratio=None,
    harmonic_phase_rad=None,
    cutting_force_noise_std_n=0.0,
    cutting_force_noise_seed=42,
    pid_gains=None,
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
            harmonic_amplitudes_ratio,
            harmonic_phase_rad,
            cutting_force_noise_std_n,
            cutting_force_noise_seed,
            pid_gains,
        )

        results.append(
            {
                "controller": name,
                **result["metrics"],
            }
        )

    return results



def _robust_score(vibration_reduction_pct, rms_current_a, saturation_pct, current_limit_a):
    """Return a transparent composite score; higher is better.

    70% is vibration reduction relative to No Control and 30% rewards low
    actuator effort. Saturation is reported separately rather than hidden in
    the score so the user can inspect the trade-off directly.
    """
    vibration_term = float(np.clip(vibration_reduction_pct / 100.0, -1.0, 1.0))
    effort_term = float(np.clip(1.0 - rms_current_a / current_limit_a, -1.0, 1.0))
    return 100.0 * (0.70 * vibration_term + 0.30 * effort_term)


def run_robust_evaluation(
    model_path=DEFAULT_MODEL_PATH,
    amplitudes_n=ROBUST_AMPLITUDES_N,
    frequencies_hz=ROBUST_FREQUENCIES_HZ,
    phases_rad=ROBUST_PHASES_RAD,
    duration_s=ROBUST_DEFAULT_DURATION_S,
    plant_variations=None,
    process_noise_std_n=0.0,
    process_noise_seed=42,
    harmonic_amplitudes_ratio=None,
    harmonic_phase_rad=None,
    cutting_force_noise_std_n=0.0,
    cutting_force_noise_seed=42,
    pid_gains=None,
):
    """Evaluate all controllers over a grid of operating conditions.

    Every controller sees the same plant, cutting-force waveform, cutting-force
    noise seed, and process-noise seed for each condition.
    """
    amplitudes_n = tuple(float(x) for x in amplitudes_n)
    frequencies_hz = tuple(float(x) for x in frequencies_hz)
    phases_rad = tuple(float(x) for x in phases_rad)

    if not amplitudes_n or not frequencies_hz or not phases_rad:
        raise ValueError("Robust evaluation requires non-empty amplitude, frequency, and phase grids.")
    if duration_s <= 0 or duration_s > 2.0:
        raise ValueError("Robust evaluation duration must be in (0, 2] seconds.")

    total_conditions = len(amplitudes_n) * len(frequencies_hz) * len(phases_rad)
    if total_conditions > 500:
        raise ValueError("Robust evaluation is limited to 500 operating conditions per run.")

    # Load SAC once rather than once per condition.
    model = None
    if not get_model_status(model_path)["model_exists"]:
        raise FileNotFoundError(get_model_status(model_path)["message"])
    model = _load_sac_model(str(model_path))

    rows = []
    case_index = 0

    for amplitude_n in amplitudes_n:
        for frequency_hz in frequencies_hz:
            for phase_rad in phases_rad:
                case_index += 1
                case_results = {}

                for controller_name in ALLOWED_CONTROLLERS:
                    result = run_simulation(
                        controller_name=controller_name,
                        amplitude_n=amplitude_n,
                        frequency_hz=frequency_hz,
                        phase_rad=phase_rad,
                        duration_s=duration_s,
                        model_path=model_path,
                        plant_variations=plant_variations,
                        process_noise_std_n=process_noise_std_n,
                        process_noise_seed=process_noise_seed,
                        harmonic_amplitudes_ratio=harmonic_amplitudes_ratio,
                        harmonic_phase_rad=harmonic_phase_rad,
                        cutting_force_noise_std_n=cutting_force_noise_std_n,
                        cutting_force_noise_seed=cutting_force_noise_seed,
                        pid_gains=pid_gains,
                    )
                    case_results[controller_name] = result["metrics"]

                baseline = case_results["No control"]
                for controller_name, metrics in case_results.items():
                    baseline_rms = baseline["steady_rms_vibration_um"]
                    current_rms = metrics["rms_control_current_a"]
                    reduction = (
                        100.0 * (1.0 - metrics["steady_rms_vibration_um"] / baseline_rms)
                        if baseline_rms > 0
                        else 0.0
                    )

                    rows.append({
                        "case": case_index,
                        "controller": controller_name,
                        "amplitude_n": amplitude_n,
                        "frequency_hz": frequency_hz,
                        "phase_rad": phase_rad,
                        "steady_rms_vibration_um": metrics["steady_rms_vibration_um"],
                        "rms_vibration_um": metrics["rms_vibration_um"],
                        "peak_vibration_um": metrics["peak_vibration_um"],
                        "rms_control_current_a": current_rms,
                        "peak_control_current_a": metrics["peak_control_current_a"],
                        "near_saturation_percent": 100.0 * metrics["actuator_near_saturation_fraction"],
                        "hard_saturation_percent": 100.0 * metrics["actuator_saturation_fraction"],
                        "steady_rms_reduction_vs_no_control_percent": reduction,
                    })

    frame = pd.DataFrame(rows)

    summary_rows = []
    for controller_name in ALLOWED_CONTROLLERS:
        subset = frame[frame["controller"] == controller_name].copy()
        reductions = subset["steady_rms_reduction_vs_no_control_percent"]
        current_limit = build_plant_config(plant_variations).continuous_current_limit_a
        summary_rows.append({
            "controller": controller_name,
            "cases": int(len(subset)),
            "mean_steady_rms_vibration_um": float(subset["steady_rms_vibration_um"].mean()),
            "median_steady_rms_vibration_um": float(subset["steady_rms_vibration_um"].median()),
            "best_case_steady_rms_vibration_um": float(subset["steady_rms_vibration_um"].min()),
            "worst_case_steady_rms_vibration_um": float(subset["steady_rms_vibration_um"].max()),
            "mean_peak_vibration_um": float(subset["peak_vibration_um"].mean()),
            "mean_rms_control_current_a": float(subset["rms_control_current_a"].mean()),
            "mean_hard_saturation_percent": float(subset["hard_saturation_percent"].mean()),
            "mean_vibration_reduction_percent": float(reductions.mean()),
            "best_vibration_reduction_percent": float(reductions.max()),
            "worst_vibration_reduction_percent": float(reductions.min()),
            "success_rate_percent": float(np.mean(reductions > 0.0) * 100.0),
            "composite_robust_score": _robust_score(
                float(reductions.mean()),
                float(subset["rms_control_current_a"].mean()),
                float(subset["hard_saturation_percent"].mean()),
                current_limit,
            ),
        })

    summary = pd.DataFrame(summary_rows)

    return {
        "simulation_only": True,
        "conditions": {
            "amplitudes_n": list(amplitudes_n),
            "frequencies_hz": list(frequencies_hz),
            "phases_rad": list(phases_rad),
            "duration_s": float(duration_s),
            "total_conditions": int(total_conditions),
        },
        "pid_gains": pid_gains or {
            "kp": 180000.0, "ki": 0.0, "kd": 0.0, "integral_limit_m_s": 1.0e-3
        },
        "summary": summary.to_dict(orient="records"),
        "cases": frame.to_dict(orient="records"),
        "score_definition": {
            "vibration_weight": 0.70,
            "effort_weight": 0.30,
            "description": "70% mean vibration reduction vs No Control + 30% low RMS-current term; saturation is reported separately.",
        },
    }

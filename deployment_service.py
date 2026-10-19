"""Simulation service for the current-driven seven-mode vibration model.
This service does not connect to physical sensors, amplifiers, or machinery.
"""
from functools import lru_cache
import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from config import PLANT, TRAINING, EVALUATION, DisturbanceConfig
from controllers import NoControl, PIDController
from experiments import simulate_controller, calculate_metrics

DEFAULT_MODEL_PATH = "results/sac_vibration_model_7mode.zip"
ALLOWED_CONTROLLERS = ("No control", "PID", "SAC")

@lru_cache(maxsize=2)
def _load_sac_model(model_path):
    return SAC.load(model_path)

def get_model_status(model_path=DEFAULT_MODEL_PATH):
    from pathlib import Path
    path = Path(model_path)
    return {"model_path": str(path), "model_exists": path.is_file(),
            "model_type": "seven-mode current-driven modal model",
            "message": "Trained SAC model is available." if path.is_file() else "Trained SAC model not found. Retrain SAC for the updated plant."}

def run_simulation(controller_name="SAC", amplitude_n=EVALUATION.nominal_amplitude_n,
                   frequency_hz=EVALUATION.nominal_frequency_hz, phase_rad=0.0,
                   duration_s=EVALUATION.duration_s, model_path=DEFAULT_MODEL_PATH):
    if controller_name not in ALLOWED_CONTROLLERS: raise ValueError(f"controller_name must be one of {ALLOWED_CONTROLLERS}.")
    values = {"amplitude_n": amplitude_n, "frequency_hz": frequency_hz, "phase_rad": phase_rad, "duration_s": duration_s}
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value): raise ValueError(f"{name} must be a finite number.")
    if amplitude_n < 0 or frequency_hz < 0 or duration_s <= 0 or duration_s > 2.0: raise ValueError("Amplitude/frequency must be non-negative and duration must be in (0, 2] seconds.")
    step_count = int(round(duration_s / PLANT.dt_s))
    if step_count < 1 or step_count > 20_000: raise ValueError("Requested duration produces an invalid step count.")
    disturbance = DisturbanceConfig(float(amplitude_n), float(frequency_hz), float(phase_rad))
    model = None
    if controller_name == "SAC":
        status = get_model_status(model_path)
        if not status["model_exists"]: raise FileNotFoundError(status["message"])
        model = _load_sac_model(str(model_path))
    data = simulate_controller(controller_name, disturbance, float(duration_s), model=model)
    metrics = calculate_metrics(data)
    records = data.replace([np.inf, -np.inf], np.nan).to_dict(orient="records")
    if any(pd.isna(value) for record in records for value in record.values()): raise RuntimeError("Simulation produced invalid numeric output.")
    return {"simulation_only": True, "model_type": "seven-mode current-driven modal model",
            "controller": controller_name,
            "inputs": {k: float(v) for k, v in values.items()},
            "current_limits_a": {"continuous": PLANT.continuous_current_limit_a, "peak": PLANT.peak_current_limit_a},
            "training_frequency_range_hz": [TRAINING.force_frequency_min_hz, TRAINING.force_frequency_max_hz],
            "frequency_outside_training_range": not (TRAINING.force_frequency_min_hz <= frequency_hz <= TRAINING.force_frequency_max_hz),
            "metrics": metrics, "timeseries": records}

def compare_controllers(amplitude_n=EVALUATION.nominal_amplitude_n, frequency_hz=EVALUATION.nominal_frequency_hz,
                        phase_rad=0.0, duration_s=EVALUATION.duration_s, model_path=DEFAULT_MODEL_PATH):
    results = []
    for name in ALLOWED_CONTROLLERS:
        result = run_simulation(name, amplitude_n, frequency_hz, phase_rad, duration_s, model_path)
        results.append({"controller": name, **result["metrics"]})
    return results

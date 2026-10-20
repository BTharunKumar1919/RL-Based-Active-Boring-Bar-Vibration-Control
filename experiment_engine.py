"""Configurable experiment runner for the seven-mode boring-bar model.

This is a simulation-only extension. Noise terms are explicit additive model
assumptions and are recorded with each experiment for reproducibility.
"""
from __future__ import annotations
from dataclasses import asdict, replace
import math
import numpy as np
import pandas as pd
from collections import deque
from config import PLANT, TRAINING, CurrentDrivenMode, ForceDrivenMode, PlantConfig
from models import SevenModePlant
from controllers import NoControl, PIDController


def plant_to_dict(config=PLANT):
    return {
        "current_modes": [asdict(x) for x in config.current_modes],
        "force_mode": asdict(config.force_mode),
        "dt_s": config.dt_s,
        "max_continuous_force_n": config.max_continuous_force_n,
        "max_peak_force_n": config.max_peak_force_n,
        "max_dynamic_force_at_2000_hz_n": config.max_dynamic_force_at_2000_hz_n,
        "continuous_current_limit_a": config.continuous_current_limit_a,
        "peak_current_limit_a": config.peak_current_limit_a,
        "current_loop_bandwidth_hz": config.current_loop_bandwidth_hz,
        "actuator_bandwidth_hz": config.actuator_bandwidth_hz,
        "current_slew_rate_a_s": config.current_slew_rate_a_s,
    }


def plant_from_dict(data=None):
    data = data or {}
    base = plant_to_dict(PLANT)
    base.update({k: v for k, v in data.items() if k in base})
    modes = base["current_modes"]
    if len(modes) != len(PLANT.current_modes):
        raise ValueError(f"This plant definition requires {len(PLANT.current_modes)} current-driven modes.")
    return PlantConfig(
        current_modes=tuple(CurrentDrivenMode(**m) for m in modes),
        force_mode=ForceDrivenMode(**base["force_mode"]),
        **{k: v for k, v in base.items() if k not in ("current_modes", "force_mode")},
    )


def apply_plant_variations(base_config, variations=None):
    """Apply per-mode percentage changes. Each item supports frequency_pct,
    damping_pct, and gain_pct (or stiffness_pct for force mode).
    """
    variations = variations or {}
    modes = []
    for i, mode in enumerate(base_config.current_modes):
        v = variations.get(f"mode_{i+1}", {})
        modes.append(replace(
            mode,
            natural_frequency_hz=mode.natural_frequency_hz * (1 + float(v.get("frequency_pct", 0))/100),
            damping_ratio=mode.damping_ratio * (1 + float(v.get("damping_pct", 0))/100),
            k_yI=mode.k_yI * (1 + float(v.get("gain_pct", 0))/100),
        ))
    fv = variations.get("force_mode", {})
    force_mode = replace(
        base_config.force_mode,
        natural_frequency_hz=base_config.force_mode.natural_frequency_hz * (1 + float(fv.get("frequency_pct", 0))/100),
        damping_ratio=base_config.force_mode.damping_ratio * (1 + float(fv.get("damping_pct", 0))/100),
        stiffness_n_m=base_config.force_mode.stiffness_n_m * (1 + float(fv.get("stiffness_pct", 0))/100),
    )
    cfg = replace(base_config, current_modes=tuple(modes), force_mode=force_mode)
    # Run same invariants as the plant constructor before the experiment begins.
    for mode in (*cfg.current_modes, cfg.force_mode):
        if mode.natural_frequency_hz <= 0 or not 0 < mode.damping_ratio < 1:
            raise ValueError("Parameter variation produced invalid frequency or damping; keep frequency positive and damping in (0, 1).")
    if any(m.k_yI == 0 for m in cfg.current_modes) or cfg.force_mode.stiffness_n_m <= 0:
        raise ValueError("Parameter variation produced a zero/negative modal gain or stiffness.")
    return cfg


class CuttingForce:
    """Sinusoidal, multi-harmonic, amplitude-modulated, or mixed force."""
    def __init__(self, cfg, rng):
        self.cfg = cfg or {"kind": "sinusoidal", "amplitude_n": 1.0, "frequency_hz": 174.0, "phase_rad": 0.0}
        self.rng = rng
        self.kind = self.cfg.get("kind", "sinusoidal")
        self._last_noise = 0.0
        if self.kind not in {"sinusoidal", "multi_harmonic", "modulated", "mixed"}:
            raise ValueError("Unknown cutting-force model.")
        for key in ("amplitude_n", "frequency_hz", "noise_std_n", "modulation_frequency_hz"):
            value = float(self.cfg.get(key, 0.0))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and non-negative.")
        depth = float(self.cfg.get("modulation_depth", 0.0))
        if not np.isfinite(depth) or not 0 <= depth <= 1:
            raise ValueError("modulation_depth must be between 0 and 1.")
        for harmonic in self.cfg.get("harmonics", []):
            if int(harmonic.get("order", 1)) < 1 or float(harmonic.get("amplitude_n", 0)) < 0:
                raise ValueError("Harmonic orders must be positive and amplitudes non-negative.")

    def __call__(self, t):
        c = self.cfg
        amp = float(c.get("amplitude_n", 1.0)); freq = float(c.get("frequency_hz", 174.0)); phase = float(c.get("phase_rad", 0.0))
        value = amp * math.sin(2 * math.pi * freq * t + phase)
        if self.kind in ("multi_harmonic", "mixed"):
            for h in c.get("harmonics", [{"order": 2, "amplitude_n": 0.25, "phase_rad": 0.0}]):
                value += float(h.get("amplitude_n", 0.0)) * math.sin(2*math.pi*freq*float(h.get("order", 2))*t + float(h.get("phase_rad", 0.0)))
        if self.kind in ("modulated", "mixed"):
            depth = float(c.get("modulation_depth", 0.25)); mf = float(c.get("modulation_frequency_hz", 5.0))
            value *= max(0.0, 1.0 + depth * math.sin(2*math.pi*mf*t))
        noise_std = float(c.get("noise_std_n", 0.0))
        self._last_noise = float(self.rng.normal(0, noise_std)) if noise_std > 0 else 0.0
        return value + self._last_noise


def _calculate_metrics(data, current_limit_a):
    if len(data) == 0: raise ValueError("Cannot calculate metrics for empty data.")
    displacement=data["displacement_um"].to_numpy(); current=data["control_current_a"].to_numpy()
    start=min(int(0.2*len(data)),len(data)-1); steady_displacement=displacement[start:]; steady_current=current[start:]
    return {
        "rms_vibration_um":float(np.sqrt(np.mean(displacement**2))),
        "peak_vibration_um":float(np.max(np.abs(displacement))),
        "steady_rms_vibration_um":float(np.sqrt(np.mean(steady_displacement**2))),
        "rms_control_current_a":float(np.sqrt(np.mean(steady_current**2))),
        "peak_control_current_a":float(np.max(np.abs(current))),
        "actuator_near_saturation_fraction":float(np.mean(np.abs(current)>=0.95*current_limit_a)),
        "actuator_saturation_fraction":float(np.mean(np.abs(current)>=0.999*current_limit_a)),
    }


def simulate_experiment(payload, model=None):
    seed = int(payload.get("seed", 42)); rng = np.random.default_rng(seed)
    base = plant_from_dict(payload.get("plant"))
    plant = apply_plant_variations(base, payload.get("variations"))
    force_cfg = payload.get("cutting_force", {})
    force = CuttingForce(force_cfg, rng)
    controller_name = payload.get("controller", "SAC")
    if controller_name == "SAC" and model is None:
        raise ValueError("Select an existing trained SAC policy or train one first.")
    if controller_name == "PID": controller = PIDController(plant_config=plant)
    elif controller_name == "No control": controller = NoControl()
    elif controller_name == "SAC": controller = None
    else: raise ValueError("controller must be No control, PID, or SAC")
    if controller is not None: controller.reset()
    duration = float(payload.get("duration_s", 0.2))
    if not 0 < duration <= 2.0: raise ValueError("duration_s must be in (0, 2].")
    dt = plant.dt_s; n_steps = int(round(duration/dt))
    if not 1 <= n_steps <= 20000: raise ValueError("Invalid simulation step count.")
    sensor_std = float(payload.get("noise", {}).get("sensor_std_m", 0.0))
    process_std = float(payload.get("noise", {}).get("process_std_n", 0.0))
    if not np.isfinite(sensor_std) or not np.isfinite(process_std) or sensor_std < 0 or process_std < 0:
        raise ValueError("Noise standard deviations must be finite and non-negative.")
    plant_obj = SevenModePlant(plant); plant_obj.reset()
    history = deque(maxlen=TRAINING.observation_history_length)
    def obs():
        s = plant_obj.state
        return np.asarray([ (s.displacement_m + (rng.normal(0,sensor_std) if sensor_std else 0))/TRAINING.displacement_scale_m,
                            s.velocity_m_s/TRAINING.velocity_scale_m_s,
                            plant_obj.actual_current_a/plant.continuous_current_limit_a], dtype=np.float32)
    for _ in range(TRAINING.observation_history_length): history.append(obs())
    action_repeat = TRAINING.sac_action_repeat; command = 0.0; rows=[]
    for i in range(n_steps):
        t=i*dt
        if controller_name == "SAC":
            if i % action_repeat == 0:
                measured=obs()
                history.append(measured)
                action, _ = model.predict(np.concatenate(list(history)).astype(np.float32), deterministic=True)
                command=float(np.clip(np.asarray(action).reshape(-1)[0],-1,1))*plant.continuous_current_limit_a
        else:
            s=plant_obj.state
            y=s.displacement_m + (rng.normal(0,sensor_std) if sensor_std else 0)
            command=float(np.clip(controller.act(y,s.velocity_m_s,dt),-plant.continuous_current_limit_a,plant.continuous_current_limit_a))
        f=force(t); process_noise=float(rng.normal(0,process_std)) if process_std else 0.0
        state=plant_obj.step(disturbance_n=f+process_noise,current_a=command)
        rows.append({"time_s":t+dt,"displacement_um":state.displacement_m*1e6,"velocity_mm_s":state.velocity_m_s*1e3,
                     "disturbance_force_n":f,"process_noise_n":process_noise,"commanded_current_a":command,
                     "control_current_a":plant_obj.actual_current_a,"normalized_action":command/plant.continuous_current_limit_a})
    df=pd.DataFrame(rows); metrics=_calculate_metrics(df, plant.continuous_current_limit_a)
    return {"simulation_only":True,"controller":controller_name,"seed":seed,"plant":plant_to_dict(plant),
            "cutting_force":force_cfg,"noise":payload.get("noise",{}),"variations":payload.get("variations",{}),
            "metrics":metrics,"timeseries":df.to_dict(orient="records")}

PLANT_PRESETS = {
    "paper_nominal": {"label": "Paper nominal (baseline)", "variations": {}},
    "frequencies_minus_5pct": {"label": "All modal frequencies −5%", "variations": {**{f"mode_{i}": {"frequency_pct": -5} for i in range(1,7)}, "force_mode":{"frequency_pct":-5}}},
    "frequencies_plus_5pct": {"label": "All modal frequencies +5%", "variations": {**{f"mode_{i}": {"frequency_pct": 5} for i in range(1,7)}, "force_mode":{"frequency_pct":5}}},
    "reduced_damping_10pct": {"label": "All modal damping −10%", "variations": {**{f"mode_{i}": {"damping_pct": -10} for i in range(1,7)}, "force_mode":{"damping_pct":-10}}},
}

def resolve_plant_preset(name="paper_nominal", custom_variations=None):
    if name not in PLANT_PRESETS: raise ValueError(f"Unknown plant preset: {name}")
    merged={k:dict(v) for k,v in PLANT_PRESETS[name]["variations"].items()}
    for k, values in (custom_variations or {}).items():
        merged.setdefault(k,{})
        for parameter, value in values.items():
            # Preset offset and user offset are additive percentage points, so
            # zero-valued GUI controls preserve the selected preset.
            merged[k][parameter] = float(merged[k].get(parameter, 0.0)) + float(value)
    return apply_plant_variations(PLANT, merged), merged

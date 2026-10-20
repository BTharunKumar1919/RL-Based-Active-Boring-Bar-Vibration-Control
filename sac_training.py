"""Configurable Soft Actor-Critic training for the simulated boring-bar plant."""

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from config import PLANT, TRAINING, DisturbanceConfig
from env import VibrationEnv
from models import build_plant_config


def _clean_model_name(name: str) -> str:
    name = str(name).strip()
    if not name:
        raise ValueError("Model name cannot be empty.")
    if name.lower().endswith(".zip"):
        name = name[:-4]
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
    safe = safe.strip("._")
    if not safe:
        raise ValueError("Model name contains no valid filename characters.")
    return safe


def _build_metadata(model_path, training_config, plant_variations, disturbance_config, requested):
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "algorithm": "SAC",
        "environment": "VibrationEnv",
        "plant_variations": plant_variations or {},
        "disturbance": {
            "amplitude_range_n": [training_config.force_amplitude_min_n, training_config.force_amplitude_max_n],
            "frequency_range_hz": [training_config.force_frequency_min_hz, training_config.force_frequency_max_hz],
            "phase_range_rad": list(requested.get("phase_range_rad", [-3.141592653589793, 3.141592653589793])),
            "harmonic_amplitudes_ratio": list(disturbance_config.harmonic_amplitudes_ratio),
            "harmonic_phase_rad": list(disturbance_config.harmonic_phase_rad),
            "harmonic_randomization": requested.get("generalized_randomization", {}).get("harmonics", []),
            "harmonic_count_randomization": requested.get("generalized_randomization", {}).get("harmonic_count", {}),
            "cutting_force_noise_std_n": disturbance_config.noise_std_n,
            "cutting_force_noise_seed": disturbance_config.noise_seed,
        },
        "process_noise_std_n": requested.get("process_noise_std_n", 0.0),
        "process_noise_seed": requested.get("process_noise_seed", 42),
        "sac": {
            "total_timesteps": requested.get("total_timesteps", training_config.total_timesteps),
            "seed": training_config.seed,
            "episode_duration_s": training_config.episode_duration_s,
            "action_repeat": training_config.sac_action_repeat,
            "observation_history_length": training_config.observation_history_length,
            "learning_rate": training_config.learning_rate,
            "buffer_size": training_config.buffer_size,
            "learning_starts": training_config.learning_starts,
            "batch_size": training_config.batch_size,
            "tau": training_config.tau,
            "gamma": training_config.gamma,
            "train_freq": training_config.train_freq,
            "gradient_steps": training_config.gradient_steps,
        },
        "training_mode": requested.get("training_mode", "specified"),
        "generalized_randomization": requested.get("generalized_randomization", {}),
        "requested": requested,
    }


def train_sac(
    total_timesteps=None,
    seed=None,
    model_path="results/sac_vibration_model_7mode",
    plant_variations=None,
    amplitude_range_n=None,
    frequency_range_hz=None,
    phase_range_rad=None,
    harmonic_amplitudes_ratio=None,
    harmonic_phase_rad=None,
    cutting_force_noise_std_n=0.0,
    cutting_force_noise_seed=42,
    process_noise_std_n=0.0,
    process_noise_seed=42,
    episode_duration_s=None,
    sac_action_repeat=None,
    observation_history_length=None,
    learning_rate=None,
    buffer_size=None,
    learning_starts=None,
    batch_size=None,
    tau=None,
    gamma=None,
    train_freq=None,
    gradient_steps=None,
    training_mode="specified",
    generalized_randomization=None,
):
    """Train SAC with an explicit, reproducible configuration."""

    total_timesteps = TRAINING.total_timesteps if total_timesteps is None else int(total_timesteps)
    seed = TRAINING.seed if seed is None else int(seed)

    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive.")

    amplitude_range_n = amplitude_range_n or [TRAINING.force_amplitude_min_n, TRAINING.force_amplitude_max_n]
    frequency_range_hz = frequency_range_hz or [TRAINING.force_frequency_min_hz, TRAINING.force_frequency_max_hz]
    phase_range_rad = phase_range_rad or [-3.141592653589793, 3.141592653589793]

    if len(amplitude_range_n) != 2 or amplitude_range_n[0] < 0 or amplitude_range_n[1] < amplitude_range_n[0]:
        raise ValueError("Amplitude range must be [min, max] with 0 <= min <= max.")
    if len(frequency_range_hz) != 2 or frequency_range_hz[0] < 0 or frequency_range_hz[1] < frequency_range_hz[0]:
        raise ValueError("Frequency range must be [min, max] with 0 <= min <= max.")
    if len(phase_range_rad) != 2 or phase_range_rad[1] < phase_range_rad[0]:
        raise ValueError("Phase range must be [min, max].")

    training_mode = str(training_mode).strip().lower()
    if training_mode not in {"specified", "generalized"}:
        raise ValueError("training_mode must be 'specified' or 'generalized'.")

    generalized_randomization = generalized_randomization or {}
    if training_mode == "specified" and generalized_randomization:
        raise ValueError("Generalized randomization settings are only valid in generalized training mode.")

    harmonic_amplitudes_ratio = [] if harmonic_amplitudes_ratio is None else list(harmonic_amplitudes_ratio)
    harmonic_phase_rad = [] if harmonic_phase_rad is None else list(harmonic_phase_rad)
    if len(harmonic_amplitudes_ratio) != len(harmonic_phase_rad):
        raise ValueError("Harmonic amplitude and phase lists must have the same length.")

    training_config = replace(
        TRAINING,
        seed=seed,
        total_timesteps=total_timesteps,
        episode_duration_s=TRAINING.episode_duration_s if episode_duration_s is None else float(episode_duration_s),
        sac_action_repeat=TRAINING.sac_action_repeat if sac_action_repeat is None else int(sac_action_repeat),
        observation_history_length=TRAINING.observation_history_length if observation_history_length is None else int(observation_history_length),
        learning_rate=TRAINING.learning_rate if learning_rate is None else float(learning_rate),
        buffer_size=TRAINING.buffer_size if buffer_size is None else int(buffer_size),
        learning_starts=TRAINING.learning_starts if learning_starts is None else int(learning_starts),
        batch_size=TRAINING.batch_size if batch_size is None else int(batch_size),
        tau=TRAINING.tau if tau is None else float(tau),
        gamma=TRAINING.gamma if gamma is None else float(gamma),
        train_freq=TRAINING.train_freq if train_freq is None else int(train_freq),
        gradient_steps=TRAINING.gradient_steps if gradient_steps is None else int(gradient_steps),
        force_amplitude_min_n=float(amplitude_range_n[0]),
        force_amplitude_max_n=float(amplitude_range_n[1]),
        force_frequency_min_hz=float(frequency_range_hz[0]),
        force_frequency_max_hz=float(frequency_range_hz[1]),
    )

    if training_config.episode_duration_s <= 0:
        raise ValueError("episode_duration_s must be positive.")
    if training_config.sac_action_repeat < 1:
        raise ValueError("sac_action_repeat must be at least 1.")
    if training_config.observation_history_length < 1:
        raise ValueError("observation_history_length must be at least 1.")
    if training_config.buffer_size <= 0 or training_config.learning_starts < 0 or training_config.batch_size <= 0:
        raise ValueError("Invalid SAC buffer/learning parameters.")
    if not 0 < training_config.tau <= 1 or not 0 < training_config.gamma <= 1:
        raise ValueError("tau and gamma must be in (0, 1].")
    if training_config.train_freq < 1 or training_config.gradient_steps < 1:
        raise ValueError("train_freq and gradient_steps must be at least 1.")
    if training_config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    plant_config = build_plant_config(plant_variations, max_percent=20.0)
    if training_mode == "specified":
        amplitude_value = float(amplitude_range_n[0])
        frequency_value = float(frequency_range_hz[0])
        phase_value = float(phase_range_rad[0])
    else:
        amplitude_value = float(sum(amplitude_range_n) / 2.0)
        frequency_value = float(sum(frequency_range_hz) / 2.0)
        phase_value = float(sum(phase_range_rad) / 2.0)

    disturbance = DisturbanceConfig(
        amplitude_n=amplitude_value,
        frequency_hz=frequency_value,
        phase_rad=phase_value,
        harmonic_amplitudes_ratio=tuple(float(x) for x in harmonic_amplitudes_ratio),
        harmonic_phase_rad=tuple(float(x) for x in harmonic_phase_rad),
        noise_std_n=float(cutting_force_noise_std_n),
        noise_seed=int(cutting_force_noise_seed),
    )

    set_random_seed(seed)

    env = Monitor(
        VibrationEnv(
            plant_config=plant_config,
            disturbance_config=disturbance,
            training_config=training_config,
            episode_duration_s=training_config.episode_duration_s,
            randomize_episode=(training_mode == "generalized"),
            phase_range_rad=tuple(float(x) for x in phase_range_rad),
            process_noise_std_n=float(process_noise_std_n),
            process_noise_seed=int(process_noise_seed),
            generalized_randomization=generalized_randomization,
        )
    )

    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=training_config.learning_rate,
        buffer_size=training_config.buffer_size,
        learning_starts=training_config.learning_starts,
        batch_size=training_config.batch_size,
        tau=training_config.tau,
        gamma=training_config.gamma,
        train_freq=training_config.train_freq,
        gradient_steps=training_config.gradient_steps,
        seed=seed,
        verbose=1,
        device="auto",
        tensorboard_log="results/tensorboard",
    )

    print(f"Starting SAC training for {total_timesteps:,} timesteps.")
    model.learn(total_timesteps=total_timesteps, progress_bar=False)

    output_path = Path(model_path)
    if output_path.suffix.lower() == ".zip":
        output_path = output_path.with_suffix("")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.save(str(output_path))
    env.close()

    metadata = _build_metadata(
        output_path.with_suffix(".zip"),
        training_config,
        plant_variations,
        disturbance,
        {
            "total_timesteps": total_timesteps,
            "phase_range_rad": list(phase_range_rad),
            "process_noise_std_n": process_noise_std_n,
            "process_noise_seed": process_noise_seed,
            "training_mode": training_mode,
            "generalized_randomization": generalized_randomization,
        },
    )
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved SAC model to: {output_path}.zip")
    return model, metadata

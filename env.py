
"""Randomized Gymnasium environment for SAC vibration control."""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import (
    PLANT,
    DISTURBANCE,
    TRAINING,
    PlantConfig,
    DisturbanceConfig,
    TrainingConfig,
)
from models import SingleModePlant, SinusoidalCuttingForce


class VibrationEnv(gym.Env):
    """Single-mode vibration-control environment.

    Observation:
        [normalized displacement, normalized velocity, previous action]

    Action:
        One normalized force command in [-1, 1], scaled to the plant's
        configured maximum generalized control force.

    Reward:
        Penalizes squared normalized displacement, control effort, and
        changes in normalized action.

    The disturbance is harmonic and does not model regenerative chatter.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        plant_config: PlantConfig = PLANT,
        disturbance_config: DisturbanceConfig = DISTURBANCE,
        training_config: TrainingConfig = TRAINING,
        episode_duration_s=None,
        randomize_episode=True,
    ):
        super().__init__()

        self.plant_config = plant_config
        self.base_disturbance = disturbance_config
        self.training_config = training_config

        if episode_duration_s is None:
            episode_duration_s = training_config.episode_duration_s

        self.episode_duration_s = float(episode_duration_s)
        if self.episode_duration_s <= 0:
            raise ValueError("Episode duration must be positive.")

        self.max_steps = int(
            round(self.episode_duration_s / plant_config.dt_s)
        )
        if self.max_steps < 1:
            raise ValueError("Episode duration must include at least one step.")

        if training_config.displacement_scale_m <= 0:
            raise ValueError("Displacement normalization scale must be positive.")
        if training_config.velocity_scale_m_s <= 0:
            raise ValueError("Velocity normalization scale must be positive.")

        self.randomize_episode = bool(randomize_episode)
        self.plant = SingleModePlant(plant_config)
        self.force_model = SinusoidalCuttingForce(disturbance_config)

        self.step_count = 0
        self.previous_action = 0.0

        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=np.full(3, -np.inf, dtype=np.float32),
            high=np.full(3, np.inf, dtype=np.float32),
            dtype=np.float32,
        )

    def _observation(self):
        state = self.plant.state
        cfg = self.training_config

        observation = np.asarray(
            [
                state.displacement_m / cfg.displacement_scale_m,
                state.velocity_m_s / cfg.velocity_scale_m_s,
                self.previous_action,
            ],
            dtype=np.float32,
        )

        return observation

    def reset(self, *, seed=None, options=None):
        """Start a new episode, optionally randomizing its conditions."""
        super().reset(seed=seed)

        cfg = self.training_config

        if self.randomize_episode:
            amplitude_n = self.np_random.uniform(
                cfg.force_amplitude_min_n,
                cfg.force_amplitude_max_n,
            )
            frequency_hz = self.np_random.uniform(
                cfg.force_frequency_min_hz,
                cfg.force_frequency_max_hz,
            )
            phase_rad = self.np_random.uniform(-np.pi, np.pi)

            initial_displacement_m = self.np_random.uniform(
                -cfg.initial_displacement_max_m,
                cfg.initial_displacement_max_m,
            )
            initial_velocity_m_s = self.np_random.uniform(
                -cfg.initial_velocity_max_m_s,
                cfg.initial_velocity_max_m_s,
            )

            episode_disturbance = DisturbanceConfig(
                amplitude_n=float(amplitude_n),
                frequency_hz=float(frequency_hz),
                phase_rad=float(phase_rad),
            )
        else:
            initial_displacement_m = 0.0
            initial_velocity_m_s = 0.0
            episode_disturbance = self.base_disturbance

        self.force_model = SinusoidalCuttingForce(episode_disturbance)
        self.plant.reset(
            displacement_m=initial_displacement_m,
            velocity_m_s=initial_velocity_m_s,
        )

        self.step_count = 0
        self.previous_action = 0.0

        info = {
            "disturbance_amplitude_n": episode_disturbance.amplitude_n,
            "disturbance_frequency_hz": episode_disturbance.frequency_hz,
            "disturbance_phase_rad": episode_disturbance.phase_rad,
        }

        return self._observation(), info

    def step(self, action):
        """Apply one bounded action and advance the shared plant model."""
        action_array = np.asarray(action, dtype=np.float64).reshape(-1)

        if action_array.size != 1:
            raise ValueError(
                f"Expected one continuous action, got shape {action_array.shape}."
            )

        normalized_action = float(
            np.clip(action_array[0], -1.0, 1.0)
        )

        control_force_n = (
            normalized_action * self.plant_config.max_control_force_n
        )

        time_s = self.step_count * self.plant_config.dt_s
        disturbance_force_n = self.force_model(time_s)

        state = self.plant.step(
            disturbance_n=disturbance_force_n,
            control_n=control_force_n,
        )

        self.step_count += 1

        cfg = self.training_config
        normalized_displacement = (
            state.displacement_m / cfg.displacement_scale_m
        )
        normalized_control = (
            control_force_n / self.plant_config.max_control_force_n
        )
        action_change = normalized_action - self.previous_action

        reward = -(
            cfg.displacement_reward_weight * normalized_displacement**2
            + cfg.control_effort_reward_weight * normalized_control**2
            + cfg.action_change_reward_weight * action_change**2
        )

        self.previous_action = normalized_action

        # A fixed-duration control episode ends by truncation, not termination.
        terminated = False
        truncated = self.step_count >= self.max_steps

        info = {
            "time_s": self.step_count * self.plant_config.dt_s,
            "displacement_m": state.displacement_m,
            "velocity_m_s": state.velocity_m_s,
            "disturbance_force_n": disturbance_force_n,
            "control_force_n": control_force_n,
            "normalized_action": normalized_action,
        }

        observation = self._observation()
        return observation, float(reward), terminated, truncated, info


"""Gymnasium environment for SAC control of the seven-mode plant."""

from collections import deque
from dataclasses import replace

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

from models import (
    SevenModePlant,
    SinusoidalCuttingForce,
)


class VibrationEnv(gym.Env):
    """SAC environment for active vibration suppression.

    Plant integration:
        0.1 ms

    SAC control interval:
        5 plant steps = 0.5 ms

    Observation:
        A short history of:

            1. normalized displacement
            2. normalized velocity
            3. normalized actual actuator current

        The history contains 8 observations, corresponding to
        approximately 4 ms of recent system history.

    Action:
        Normalized commanded actuator current in [-1, 1].

    The commanded current passes through the actuator dynamics
    implemented in SevenModePlant before entering the modal plant.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        plant_config: PlantConfig = PLANT,
        disturbance_config: DisturbanceConfig = DISTURBANCE,
        training_config: TrainingConfig = TRAINING,
        episode_duration_s=None,
        randomize_episode=True,
        generalized_plant=False,
        plant_variation_pct=10.0,
        force_model_config=None,
        noise_config=None,
    ):
        super().__init__()

        self.plant_config = plant_config
        self.base_disturbance = disturbance_config
        self.training_config = training_config
        self.generalized_plant = bool(generalized_plant)
        self.plant_variation_pct = float(plant_variation_pct)
        self.force_model_config = force_model_config or {"kind": "sinusoidal"}
        self.noise_config = noise_config or {}
        self._sensor_rng = np.random.default_rng(training_config.seed)

        if episode_duration_s is None:
            episode_duration_s = training_config.episode_duration_s

        self.episode_duration_s = float(episode_duration_s)

        if self.episode_duration_s <= 0:
            raise ValueError(
                "Episode duration must be positive."
            )

        # ---------------------------------------------------------
        # SAC timing
        # ---------------------------------------------------------

        self.action_repeat = int(
            training_config.sac_action_repeat
        )

        if self.action_repeat < 1:
            raise ValueError(
                "sac_action_repeat must be at least 1."
            )

        self.rl_dt_s = (
            plant_config.dt_s * self.action_repeat
        )

        # Number of SAC decisions per episode.
        self.max_steps = int(
            round(
                self.episode_duration_s
                / self.rl_dt_s
            )
        )

        if self.max_steps < 1:
            raise ValueError(
                "Episode duration must include at least one SAC step."
            )

        # ---------------------------------------------------------
        # Observation history
        # ---------------------------------------------------------

        self.history_length = int(
            training_config.observation_history_length
        )

        if self.history_length < 1:
            raise ValueError(
                "observation_history_length must be at least 1."
            )

        self.observation_dimension = (
            3 * self.history_length
        )

        self._observation_history = deque(
            maxlen=self.history_length
        )

        # ---------------------------------------------------------
        # Validation
        # ---------------------------------------------------------

        if (
            training_config.displacement_scale_m <= 0
            or training_config.velocity_scale_m_s <= 0
        ):
            raise ValueError(
                "Observation scales must be positive."
            )

        self.randomize_episode = bool(
            randomize_episode
        )

        # ---------------------------------------------------------
        # Plant
        # ---------------------------------------------------------

        self.plant = SevenModePlant(
            plant_config
        )

        self.force_model = SinusoidalCuttingForce(
            disturbance_config
        )

        self.step_count = 0

        self._previous_actual_current_normalized = 0.0

        # ---------------------------------------------------------
        # Action space
        # ---------------------------------------------------------

        self.action_space = spaces.Box(
            low=np.array(
                [-1.0],
                dtype=np.float32,
            ),
            high=np.array(
                [1.0],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        # ---------------------------------------------------------
        # Observation space
        # ---------------------------------------------------------

        self.observation_space = spaces.Box(
            low=np.full(
                self.observation_dimension,
                -np.inf,
                dtype=np.float32,
            ),
            high=np.full(
                self.observation_dimension,
                np.inf,
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

    # =============================================================
    # Observation handling
    # =============================================================

    def _single_observation(self):
        """Return the current normalized 3-element observation."""

        state = self.plant.state

        cfg = self.training_config

        current_limit = (
            self.plant_config.continuous_current_limit_a
        )

        sensor_std = float(self.noise_config.get("sensor_std_m", 0.0))
        measured_displacement = state.displacement_m + (self._sensor_rng.normal(0, sensor_std) if sensor_std > 0 else 0.0)
        return np.asarray(
            [
                measured_displacement
                / cfg.displacement_scale_m,

                state.velocity_m_s
                / cfg.velocity_scale_m_s,

                self.plant.actual_current_a
                / current_limit,
            ],
            dtype=np.float32,
        )

    def _update_history(self):
        """Append the latest observation to the history."""

        current_observation = (
            self._single_observation()
        )

        self._observation_history.append(
            current_observation
        )

    def _observation(self):
        """Return flattened observation history.

        During the beginning of an episode, the history is padded
        with the earliest available observation.
        """

        if len(self._observation_history) == 0:
            current = self._single_observation()

            for _ in range(
                self.history_length
            ):
                self._observation_history.append(
                    current.copy()
                )

        elif len(self._observation_history) < self.history_length:
            first = self._observation_history[0]

            while len(
                self._observation_history
            ) < self.history_length:
                self._observation_history.appendleft(
                    first.copy()
                )

        return np.concatenate(
            list(self._observation_history)
        ).astype(np.float32)

    # =============================================================
    # Reset
    # =============================================================

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        """Reset the environment."""

        super().reset(seed=seed)

        cfg = self.training_config

        if self.generalized_plant:
            pct = self.plant_variation_pct / 100.0
            varied_modes = tuple(replace(m,
                natural_frequency_hz=m.natural_frequency_hz * self.np_random.uniform(1-pct, 1+pct),
                damping_ratio=float(np.clip(m.damping_ratio * self.np_random.uniform(1-pct, 1+pct), 0.001, 0.95)),
                k_yI=m.k_yI * self.np_random.uniform(1-pct, 1+pct)) for m in self.plant_config.current_modes)
            fm = self.plant_config.force_mode
            varied_force = replace(fm, natural_frequency_hz=fm.natural_frequency_hz * self.np_random.uniform(1-pct,1+pct),
                damping_ratio=float(np.clip(fm.damping_ratio * self.np_random.uniform(1-pct,1+pct),0.001,0.95)),
                stiffness_n_m=fm.stiffness_n_m * self.np_random.uniform(1-pct,1+pct))
            self.plant = SevenModePlant(replace(self.plant_config, current_modes=varied_modes, force_mode=varied_force))
        else:
            self.plant = SevenModePlant(self.plant_config)

        if self.randomize_episode:

            amplitude_n = self.np_random.uniform(
                cfg.force_amplitude_min_n,
                cfg.force_amplitude_max_n,
            )

            frequency_hz = self.np_random.uniform(
                cfg.force_frequency_min_hz,
                cfg.force_frequency_max_hz,
            )

            phase_rad = self.np_random.uniform(
                -np.pi,
                np.pi,
            )

            initial_displacement_m = (
                self.np_random.uniform(
                    -cfg.initial_displacement_max_m,
                    cfg.initial_displacement_max_m,
                )
            )

            initial_velocity_m_s = (
                self.np_random.uniform(
                    -cfg.initial_velocity_max_m_s,
                    cfg.initial_velocity_max_m_s,
                )
            )

            episode_disturbance = (
                DisturbanceConfig(
                    float(amplitude_n),
                    float(frequency_hz),
                    float(phase_rad),
                )
            )

        else:

            initial_displacement_m = 0.0
            initial_velocity_m_s = 0.0

            episode_disturbance = (
                self.base_disturbance
            )

        # The generalized training path currently randomizes plant parameters
        # and sinusoidal force conditions. Force model classes are selected below.
        kind = self.force_model_config.get("kind", "sinusoidal")
        force_cfg = dict(self.force_model_config)
        if self.randomize_episode and self.generalized_plant:
            force_cfg["amplitude_n"] = float(amplitude_n)
            force_cfg["frequency_hz"] = float(frequency_hz)
            force_cfg["phase_rad"] = float(phase_rad)
        elif self.randomize_episode and kind == "sinusoidal":
            force_cfg["amplitude_n"] = float(amplitude_n)
            force_cfg["frequency_hz"] = float(frequency_hz)
            force_cfg["phase_rad"] = float(phase_rad)
        if kind == "sinusoidal" and float(force_cfg.get("noise_std_n", 0.0)) == 0.0:
            self.force_model = SinusoidalCuttingForce(DisturbanceConfig(
                float(force_cfg.get("amplitude_n", episode_disturbance.amplitude_n)),
                float(force_cfg.get("frequency_hz", episode_disturbance.frequency_hz)),
                float(force_cfg.get("phase_rad", episode_disturbance.phase_rad))))
        else:
            # Same force-model implementation as evaluation, including additive force noise.
            from experiment_engine import CuttingForce
            self.force_model = CuttingForce(force_cfg, self._sensor_rng)

        self.plant.reset(
            initial_displacement_m,
            initial_velocity_m_s,
        )

        self.step_count = 0

        self._previous_actual_current_normalized = (
            0.0
        )

        self._observation_history.clear()

        initial_observation = (
            self._single_observation()
        )

        # Pad the history with the initial state.
        for _ in range(
            self.history_length
        ):
            self._observation_history.append(
                initial_observation.copy()
            )

        return (
            self._observation(),
            {
                "disturbance_amplitude_n":
                    episode_disturbance.amplitude_n,

                "disturbance_frequency_hz":
                    episode_disturbance.frequency_hz,

                "disturbance_phase_rad":
                    episode_disturbance.phase_rad,
            },
        )

    # =============================================================
    # Step
    # =============================================================

    def step(self, action):
        """Advance the environment by one SAC control interval."""

        action_array = np.asarray(
            action,
            dtype=np.float64,
        ).reshape(-1)

        if action_array.size != 1:
            raise ValueError(
                f"Expected one continuous action, "
                f"got shape {action_array.shape}."
            )

        # ---------------------------------------------------------
        # SAC action
        # ---------------------------------------------------------

        normalized_action = float(
            np.clip(
                action_array[0],
                -1.0,
                1.0,
            )
        )

        commanded_current_a = (
            normalized_action
            * self.plant_config.continuous_current_limit_a
        )

        # ---------------------------------------------------------
        # Hold this command for several physical plant steps.
        # ---------------------------------------------------------

        last_disturbance_n = 0.0
        state = self.plant.state

        for substep in range(
            self.action_repeat
        ):

            physical_step_index = (
                self.step_count
                * self.action_repeat
                + substep
            )

            time_s = (
                physical_step_index
                * self.plant_config.dt_s
            )

            disturbance_n = self.force_model(time_s)
            process_std = float(self.noise_config.get("process_std_n", 0.0))
            if process_std > 0:
                disturbance_n += float(self._sensor_rng.normal(0, process_std))

            state = self.plant.step(
                disturbance_n=disturbance_n,
                current_a=commanded_current_a,
            )

            last_disturbance_n = (
                disturbance_n
            )

        # ---------------------------------------------------------
        # Update SAC observation history.
        # ---------------------------------------------------------

        self._update_history()

        self.step_count += 1

        cfg = self.training_config

        current_limit = (
            self.plant_config.continuous_current_limit_a
        )

        normalized_displacement = (
            state.displacement_m
            / cfg.displacement_scale_m
        )

        actual_current_normalized = (
            self.plant.actual_current_a
            / current_limit
        )

        # ---------------------------------------------------------
        # Actual-current change penalty.
        # ---------------------------------------------------------

        action_change = (
            actual_current_normalized
            - self._previous_actual_current_normalized
        )

        # ---------------------------------------------------------
        # Reward
        # ---------------------------------------------------------

        reward = -(
            cfg.displacement_reward_weight
            * normalized_displacement**2

            + cfg.control_effort_reward_weight
            * actual_current_normalized**2

            + cfg.action_change_reward_weight
            * action_change**2
        )

        self._previous_actual_current_normalized = (
            actual_current_normalized
        )

        # ---------------------------------------------------------
        # Episode termination
        # ---------------------------------------------------------

        terminated = False

        truncated = (
            self.step_count
            >= self.max_steps
        )

        # ---------------------------------------------------------
        # Information
        # ---------------------------------------------------------

        info = {
            "time_s":
                self.step_count
                * self.rl_dt_s,

            "displacement_m":
                state.displacement_m,

            "velocity_m_s":
                state.velocity_m_s,

            "disturbance_force_n":
                last_disturbance_n,

            "commanded_current_a":
                commanded_current_a,

            "control_current_a":
                self.plant.actual_current_a,

            "normalized_action":
                normalized_action,

            "normalized_actual_current":
                actual_current_normalized,

            "sac_action_interval_s":
                self.rl_dt_s,

            "plant_steps_per_action":
                self.action_repeat,
        }

        return (
            self._observation(),
            float(reward),
            terminated,
            truncated,
            info,
        )
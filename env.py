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
        phase_range_rad=(-np.pi, np.pi),
        process_noise_std_n=0.0,
        process_noise_seed=42,
        generalized_randomization=None,
    ):
        super().__init__()

        self.base_plant_config = plant_config
        self.plant_config = plant_config
        self.base_disturbance = disturbance_config
        self.training_config = training_config
        self.generalized_randomization = generalized_randomization or {}

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

        self.phase_range_rad = (float(phase_range_rad[0]), float(phase_range_rad[1]))
        if self.phase_range_rad[1] < self.phase_range_rad[0]:
            raise ValueError("phase_range_rad must be [min, max].")

        self.process_noise_std_n = float(process_noise_std_n)
        self.process_noise_seed = int(process_noise_seed)

        # Generalized-training randomization settings. These are deliberately
        # limited to continuous parameters in Step 6A.
        self._validate_generalized_randomization()

        # ---------------------------------------------------------
        # Plant
        # ---------------------------------------------------------

        self.plant = SevenModePlant(
            plant_config,
            process_noise_std_n=self.process_noise_std_n,
            process_noise_seed=self.process_noise_seed,
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

    def _validate_generalized_randomization(self):
        """Validate Step 6A continuous-parameter randomization settings."""
        allowed = {
            "amplitude_n", "frequency_hz", "phase_rad",
            "cutting_force_noise_std_n", "process_noise_std_n",
            "plant_variation_percent", "harmonics", "harmonic_count",
        }
        for key, spec in self.generalized_randomization.items():
            if key not in allowed:
                raise ValueError(f"Unsupported generalized randomization parameter: {key}")
            if not isinstance(spec, dict):
                raise ValueError(f"Randomization setting for {key} must be a dictionary.")
            if not bool(spec.get("enabled", False)):
                continue
            lo = float(spec.get("min"))
            hi = float(spec.get("max"))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi < lo:
                raise ValueError(f"Invalid randomization range for {key}.")
            if key in {"amplitude_n", "frequency_hz", "cutting_force_noise_std_n", "process_noise_std_n"} and lo < 0:
                raise ValueError(f"{key} cannot have a negative minimum.")
            if key == "plant_variation_percent" and (lo < -20.0 or hi > 20.0):
                raise ValueError("Plant randomization must stay within +/-20%.")
            if key == "harmonic_count" and (lo < 0 or hi > 5 or not float(lo).is_integer() or not float(hi).is_integer()):
                raise ValueError("harmonic_count must use integer values from 0 to 5.")
        harmonic_specs = self.generalized_randomization.get("harmonics", [])
        if not isinstance(harmonic_specs, list) or len(harmonic_specs) > 5:
            raise ValueError("harmonics must contain at most five harmonic specifications.")
        for index, spec in enumerate(harmonic_specs, start=2):
            if not isinstance(spec, dict):
                raise ValueError(f"Harmonic {index} specification must be a dictionary.")
            for field in ("amplitude_ratio", "phase_rad"):
                field_spec = spec.get(field, {})
                if not isinstance(field_spec, dict):
                    raise ValueError(f"Harmonic {index} {field} setting must be a dictionary.")
                if not bool(field_spec.get("enabled", False)):
                    continue
                lo = float(field_spec.get("min")); hi = float(field_spec.get("max"))
                if not np.isfinite(lo) or not np.isfinite(hi) or hi < lo:
                    raise ValueError(f"Invalid harmonic {index} {field} range.")
                if field == "amplitude_ratio" and (lo < 0 or hi > 1.0):
                    raise ValueError(f"Harmonic {index} amplitude ratio must be 0 to 100%.")

    def _sample_value(self, key, fixed_value):
        spec = self.generalized_randomization.get(key, {})
        if bool(spec.get("enabled", False)):
            return float(self.np_random.uniform(float(spec["min"]), float(spec["max"])))
        return float(fixed_value)

    def _sample_plant_config(self):
        """Sample physical plant parameters independently within one common percent range."""
        spec = self.generalized_randomization.get("plant_variation_percent", {})
        if not bool(spec.get("enabled", False)):
            return self.base_plant_config

        lo = float(spec["min"])
        hi = float(spec["max"])

        current_modes = []
        for mode in self.base_plant_config.current_modes:
            current_modes.append(
                replace(
                    mode,
                    natural_frequency_hz=mode.natural_frequency_hz * (1.0 + self.np_random.uniform(lo, hi) / 100.0),
                    damping_ratio=mode.damping_ratio * (1.0 + self.np_random.uniform(lo, hi) / 100.0),
                    k_yI=mode.k_yI * (1.0 + self.np_random.uniform(lo, hi) / 100.0),
                )
            )

        force_mode = self.base_plant_config.force_mode
        force_mode = replace(
            force_mode,
            natural_frequency_hz=force_mode.natural_frequency_hz * (1.0 + self.np_random.uniform(lo, hi) / 100.0),
            damping_ratio=force_mode.damping_ratio * (1.0 + self.np_random.uniform(lo, hi) / 100.0),
            stiffness_n_m=force_mode.stiffness_n_m * (1.0 + self.np_random.uniform(lo, hi) / 100.0),
        )

        return replace(
            self.base_plant_config,
            current_modes=tuple(current_modes),
            force_mode=force_mode,
            actuator_bandwidth_hz=self.base_plant_config.actuator_bandwidth_hz * (1.0 + self.np_random.uniform(lo, hi) / 100.0),
            current_slew_rate_a_s=self.base_plant_config.current_slew_rate_a_s * (1.0 + self.np_random.uniform(lo, hi) / 100.0),
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

        return np.asarray(
            [
                state.displacement_m
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

        if self.randomize_episode or self.generalized_randomization:
            amplitude_n = self._sample_value(
                "amplitude_n", self.base_disturbance.amplitude_n
            )
            frequency_hz = self._sample_value(
                "frequency_hz", self.base_disturbance.frequency_hz
            )
            phase_rad = self._sample_value(
                "phase_rad", self.base_disturbance.phase_rad
            )

            cutting_noise_std_n = self._sample_value(
                "cutting_force_noise_std_n",
                self.base_disturbance.noise_std_n,
            )
            process_noise_std_n = self._sample_value(
                "process_noise_std_n",
                self.process_noise_std_n,
            )

            if self.randomize_episode and not self.generalized_randomization:
                # Backward-compatible Step 5 behavior: amplitude/frequency/phase
                # are randomized from the training ranges.
                cfg = self.training_config
                amplitude_n = float(self.np_random.uniform(
                    cfg.force_amplitude_min_n,
                    cfg.force_amplitude_max_n,
                ))
                frequency_hz = float(self.np_random.uniform(
                    cfg.force_frequency_min_hz,
                    cfg.force_frequency_max_hz,
                ))
                phase_rad = float(self.np_random.uniform(
                    self.phase_range_rad[0],
                    self.phase_range_rad[1],
                ))

            initial_displacement_m = float(self.np_random.uniform(
                -cfg.initial_displacement_max_m,
                cfg.initial_displacement_max_m,
            ))
            initial_velocity_m_s = float(self.np_random.uniform(
                -cfg.initial_velocity_max_m_s,
                cfg.initial_velocity_max_m_s,
            ))

            harmonic_ratios = list(self.base_disturbance.harmonic_amplitudes_ratio)
            harmonic_phases = list(self.base_disturbance.harmonic_phase_rad)
            harmonic_specs = self.generalized_randomization.get("harmonics", [])
            if harmonic_specs:
                while len(harmonic_ratios) < 5:
                    harmonic_ratios.append(0.0)
                    harmonic_phases.append(0.0)
                for idx, spec in enumerate(harmonic_specs):
                    if idx >= 5:
                        break
                    amp_spec = spec.get("amplitude_ratio", {})
                    phase_spec = spec.get("phase_rad", {})
                    if bool(amp_spec.get("enabled", False)):
                        harmonic_ratios[idx] = float(self.np_random.uniform(float(amp_spec["min"]), float(amp_spec["max"])))
                    if bool(phase_spec.get("enabled", False)):
                        harmonic_phases[idx] = float(self.np_random.uniform(float(phase_spec["min"]), float(phase_spec["max"])))
                count_spec = self.generalized_randomization.get("harmonic_count", {})
                if bool(count_spec.get("enabled", False)):
                    count = int(self.np_random.integers(int(count_spec["min"]), int(count_spec["max"]) + 1))
                    harmonic_ratios = harmonic_ratios[:count]
                    harmonic_phases = harmonic_phases[:count]
                else:
                    active_count = 0
                    for i, ratio in enumerate(harmonic_ratios):
                        spec = harmonic_specs[i] if i < len(harmonic_specs) else {}
                        if abs(float(ratio)) > 0.0 or bool(spec.get("enabled", False)):
                            active_count = i + 1
                    harmonic_ratios = harmonic_ratios[:active_count]
                    harmonic_phases = harmonic_phases[:active_count]

            episode_disturbance = DisturbanceConfig(
                amplitude_n=amplitude_n,
                frequency_hz=frequency_hz,
                phase_rad=phase_rad,
                harmonic_amplitudes_ratio=tuple(harmonic_ratios),
                harmonic_phase_rad=tuple(harmonic_phases),
                noise_std_n=cutting_noise_std_n,
                noise_seed=int(self.np_random.integers(0, 2**31 - 1)),
            )
        else:
            initial_displacement_m = 0.0
            initial_velocity_m_s = 0.0
            episode_disturbance = self.base_disturbance
            process_noise_std_n = self.process_noise_std_n

        self.plant_config = self._sample_plant_config()
        self.process_noise_std_n = float(process_noise_std_n)
        self.plant = SevenModePlant(
            self.plant_config,
            process_noise_std_n=self.process_noise_std_n,
            process_noise_seed=int(self.np_random.integers(0, 2**31 - 1)),
        )

        self.force_model = (
            SinusoidalCuttingForce(
                episode_disturbance
            )
        )

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

            disturbance_n = (
                self.force_model(time_s)
            )

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
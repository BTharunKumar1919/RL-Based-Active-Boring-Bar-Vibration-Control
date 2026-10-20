"""Seven-mode current-driven plant with actuator dynamics.

The plant implements:

    y(s) = G_yI(s) I(s) + G_yF(s) F_c(s)

The current-driven modal dynamics are preceded by a first-order actuator
model with a finite bandwidth and current slew-rate limit.

This module also provides a safe way to create perturbed plant models from
the nominal PLANT configuration. The perturbations are expressed as
percentage changes from the nominal physical parameters.

Process noise is represented as zero-mean additive force noise on the
force-driven mode. It is intended as a simple model of unmodelled/process
excitation, not sensor noise.
"""

from dataclasses import dataclass, replace
import math
import numpy as np

from config import PLANT, DISTURBANCE, PlantConfig, DisturbanceConfig


@dataclass(frozen=True)
class PlantState:
    displacement_m: float
    velocity_m_s: float


def _scaled(value, percent):
    """Apply a percentage change while preserving the sign of value."""
    return float(value) * (1.0 + float(percent) / 100.0)


def build_plant_config(variations=None, max_percent=20.0):
    """Create a perturbed PlantConfig from the nominal PLANT.

    Expected structure::

        {
            "current_modes": [
                {"natural_frequency_percent": 0.0,
                 "damping_ratio_percent": 0.0,
                 "k_yI_percent": 0.0},
                ... six entries ...
            ],
            "force_mode": {
                "natural_frequency_percent": 0.0,
                "damping_ratio_percent": 0.0,
                "stiffness_percent": 0.0,
            },
            "actuator": {
                "bandwidth_percent": 0.0,
                "slew_rate_percent": 0.0,
            },
        }

    Only physical parameters that do not change the controller's action
    interface or simulation timestep are exposed here. In particular,
    dt and the +/-5 A current interface remain fixed so that a saved SAC
    policy remains compatible with the observation/action space.
    """

    if variations is None:
        return PLANT

    if not isinstance(variations, dict):
        raise ValueError("plant_variations must be a dictionary.")

    max_percent = float(max_percent)

    def get_percent(mapping, key):
        value = mapping.get(key, 0.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Plant variation '{key}' must be numeric.")
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"Plant variation '{key}' must be finite.")
        if abs(value) > max_percent:
            raise ValueError(
                f"Plant variation '{key}' must be within +/-{max_percent:.1f}%."
            )
        return value

    mode_variations = variations.get("current_modes", [])
    if not isinstance(mode_variations, list):
        raise ValueError("plant_variations.current_modes must be a list.")
    if len(mode_variations) > len(PLANT.current_modes):
        raise ValueError("Too many current-mode variation entries.")

    new_modes = []

    for index, mode in enumerate(PLANT.current_modes):
        variation = mode_variations[index] if index < len(mode_variations) else {}
        if not isinstance(variation, dict):
            raise ValueError(f"Variation for current mode {index + 1} must be a dictionary.")

        new_modes.append(
            type(mode)(
                natural_frequency_hz=_scaled(
                    mode.natural_frequency_hz,
                    get_percent(variation, "natural_frequency_percent"),
                ),
                damping_ratio=_scaled(
                    mode.damping_ratio,
                    get_percent(variation, "damping_ratio_percent"),
                ),
                k_yI=_scaled(
                    mode.k_yI,
                    get_percent(variation, "k_yI_percent"),
                ),
            )
        )

    force_variation = variations.get("force_mode", {})
    if not isinstance(force_variation, dict):
        raise ValueError("plant_variations.force_mode must be a dictionary.")

    force_mode = type(PLANT.force_mode)(
        natural_frequency_hz=_scaled(
            PLANT.force_mode.natural_frequency_hz,
            get_percent(force_variation, "natural_frequency_percent"),
        ),
        damping_ratio=_scaled(
            PLANT.force_mode.damping_ratio,
            get_percent(force_variation, "damping_ratio_percent"),
        ),
        stiffness_n_m=_scaled(
            PLANT.force_mode.stiffness_n_m,
            get_percent(force_variation, "stiffness_percent"),
        ),
    )

    actuator_variation = variations.get("actuator", {})
    if not isinstance(actuator_variation, dict):
        raise ValueError("plant_variations.actuator must be a dictionary.")

    return replace(
        PLANT,
        current_modes=tuple(new_modes),
        force_mode=force_mode,
        actuator_bandwidth_hz=_scaled(
            PLANT.actuator_bandwidth_hz,
            get_percent(actuator_variation, "bandwidth_percent"),
        ),
        current_slew_rate_a_s=_scaled(
            PLANT.current_slew_rate_a_s,
            get_percent(actuator_variation, "slew_rate_percent"),
        ),
    )


class SinusoidalCuttingForce:
    """Prescribed harmonic disturbance; not a regenerative chatter model."""

    def __init__(self, config: DisturbanceConfig = DISTURBANCE):
        if config.amplitude_n < 0:
            raise ValueError(
                "Disturbance amplitude must be non-negative."
            )

        if config.frequency_hz < 0:
            raise ValueError(
                "Disturbance frequency must be non-negative."
            )

        self.config = config

    def __call__(self, time_s: float) -> float:
        return self.config.amplitude_n * math.sin(
            2.0
            * math.pi
            * self.config.frequency_hz
            * time_s
            + self.config.phase_rad
        )


class SevenModePlant:
    """Six current-driven modes plus one cutting-force-driven mode.

    The controller supplies commanded current.

    The actual actuator current is obtained through:

        commanded current
                 |
                 v
        first-order bandwidth
                 |
                 v
        slew-rate limitation
                 |
                 v
        hard current saturation
                 |
                 v
        current-driven modal dynamics

    ``process_noise_std_n`` adds zero-mean Gaussian force noise to the
    force-driven mode at each physical integration step.
    """

    def __init__(
        self,
        config: PlantConfig = PLANT,
        process_noise_std_n: float = 0.0,
        process_noise_seed=None,
    ):
        if config.dt_s <= 0:
            raise ValueError(
                "Simulation time step must be positive."
            )

        if not config.current_modes:
            raise ValueError(
                "At least one current-driven mode is required."
            )

        if config.continuous_current_limit_a <= 0:
            raise ValueError(
                "Continuous current limit must be positive."
            )

        if config.actuator_bandwidth_hz <= 0:
            raise ValueError(
                "Actuator bandwidth must be positive."
            )

        if config.current_slew_rate_a_s <= 0:
            raise ValueError(
                "Current slew rate must be positive."
            )

        process_noise_std_n = float(process_noise_std_n)
        if not np.isfinite(process_noise_std_n) or process_noise_std_n < 0:
            raise ValueError(
                "process_noise_std_n must be a finite non-negative value."
            )

        for mode in (*config.current_modes, config.force_mode):
            if mode.natural_frequency_hz <= 0:
                raise ValueError(
                    "Modal natural frequencies must be positive."
                )

            if not 0 < mode.damping_ratio < 1:
                raise ValueError(
                    "Modal damping ratios must be between 0 and 1."
                )

        if any(mode.k_yI == 0 for mode in config.current_modes):
            raise ValueError(
                "Current modal coefficients must be non-zero."
            )

        if config.force_mode.stiffness_n_m <= 0:
            raise ValueError(
                "Force-mode stiffness must be positive."
            )

        self.config = config
        self.process_noise_std_n = process_noise_std_n
        self.rng = np.random.default_rng(process_noise_seed)
        self.last_process_noise_n = 0.0

        self.current_q = np.zeros(
            len(config.current_modes),
            dtype=float,
        )

        self.current_v = np.zeros(
            len(config.current_modes),
            dtype=float,
        )

        self.force_q = 0.0
        self.force_v = 0.0

        # Actual actuator current.
        self.actuator_current_a = 0.0

    @property
    def state(self) -> PlantState:
        """Return the total modal displacement and velocity."""

        return PlantState(
            displacement_m=float(
                np.sum(self.current_q) + self.force_q
            ),
            velocity_m_s=float(
                np.sum(self.current_v) + self.force_v
            ),
        )

    @property
    def actual_current_a(self) -> float:
        """Actual physical actuator current applied to the plant."""

        return float(self.actuator_current_a)

    def reset(
        self,
        displacement_m: float = 0.0,
        velocity_m_s: float = 0.0,
    ) -> PlantState:
        """Reset all modal and actuator states."""

        self.current_q.fill(0.0)
        self.current_v.fill(0.0)

        self.force_q = float(displacement_m)
        self.force_v = float(velocity_m_s)

        self.actuator_current_a = 0.0
        self.last_process_noise_n = 0.0

        return self.state

    @staticmethod
    def _derivative(
        q,
        v,
        omega,
        zeta,
        input_gain,
        input_value,
    ):
        acceleration = (
            -2.0 * zeta * omega * v
            - omega**2 * q
            + input_gain * input_value
        )

        return v, acceleration

    def _rk4_mode(
        self,
        q,
        v,
        omega,
        zeta,
        input_gain,
        input_value,
    ):
        """Advance one second-order modal state using RK4."""

        dt = self.config.dt_s

        f = lambda qq, vv: self._derivative(
            qq,
            vv,
            omega,
            zeta,
            input_gain,
            input_value,
        )

        k1q, k1v = f(q, v)

        k2q, k2v = f(
            q + 0.5 * dt * k1q,
            v + 0.5 * dt * k1v,
        )

        k3q, k3v = f(
            q + 0.5 * dt * k2q,
            v + 0.5 * dt * k2v,
        )

        k4q, k4v = f(
            q + dt * k3q,
            v + dt * k3v,
        )

        q_next = (
            q
            + dt
            * (
                k1q
                + 2.0 * k2q
                + 2.0 * k3q
                + k4q
            )
            / 6.0
        )

        v_next = (
            v
            + dt
            * (
                k1v
                + 2.0 * k2v
                + 2.0 * k3v
                + k4v
            )
            / 6.0
        )

        return q_next, v_next

    def _update_actuator_current(
        self,
        commanded_current_a,
    ):
        """Update the actual actuator current."""

        dt = self.config.dt_s
        limit = self.config.continuous_current_limit_a

        commanded_current_a = float(
            np.clip(
                commanded_current_a,
                -limit,
                limit,
            )
        )

        bandwidth_hz = self.config.actuator_bandwidth_hz

        tau = 1.0 / (
            2.0 * math.pi * bandwidth_hz
        )

        alpha = dt / (tau + dt)

        filtered_current = (
            self.actuator_current_a
            + alpha
            * (
                commanded_current_a
                - self.actuator_current_a
            )
        )

        max_delta = (
            self.config.current_slew_rate_a_s
            * dt
        )

        delta = np.clip(
            filtered_current - self.actuator_current_a,
            -max_delta,
            max_delta,
        )

        self.actuator_current_a += delta

        self.actuator_current_a = float(
            np.clip(
                self.actuator_current_a,
                -limit,
                limit,
            )
        )

        return self.actuator_current_a

    def step(
        self,
        disturbance_n: float,
        current_a: float,
    ) -> PlantState:
        """Advance the seven modal states by one RK4 step."""

        actual_current_a = self._update_actuator_current(
            current_a
        )

        disturbance_n = float(disturbance_n)

        process_noise_n = 0.0
        if self.process_noise_std_n > 0.0:
            process_noise_n = float(
                self.rng.normal(
                    0.0,
                    self.process_noise_std_n,
                )
            )

        self.last_process_noise_n = process_noise_n

        effective_disturbance_n = (
            disturbance_n + process_noise_n
        )

        for idx, mode in enumerate(
            self.config.current_modes
        ):
            omega = (
                2.0
                * math.pi
                * mode.natural_frequency_hz
            )

            gain = omega**2 / mode.k_yI

            self.current_q[idx], self.current_v[idx] = (
                self._rk4_mode(
                    self.current_q[idx],
                    self.current_v[idx],
                    omega,
                    mode.damping_ratio,
                    gain,
                    actual_current_a,
                )
            )

        mode = self.config.force_mode

        omega = (
            2.0
            * math.pi
            * mode.natural_frequency_hz
        )

        gain = omega**2 / mode.stiffness_n_m

        self.force_q, self.force_v = self._rk4_mode(
            self.force_q,
            self.force_v,
            omega,
            mode.damping_ratio,
            gain,
            effective_disturbance_n,
        )

        return self.state


# Keep the old import name as an alias.
SingleModePlant = SevenModePlant

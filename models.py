"""Seven-mode current-driven plant with actuator dynamics.

The plant implements:

    y(s) = G_yI(s) I(s) + G_yF(s) F_c(s)

The current-driven modal dynamics are preceded by a first-order actuator
model with a finite bandwidth and current slew-rate limit.

Integration uses RK4 with the actual actuator current and cutting force
held constant over one simulation step.
"""

from dataclasses import dataclass
import math
import numpy as np

from config import PLANT, DISTURBANCE, PlantConfig, DisturbanceConfig


@dataclass(frozen=True)
class PlantState:
    displacement_m: float
    velocity_m_s: float


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
    """

    def __init__(self, config: PlantConfig = PLANT):
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

        # The actuator starts at zero current.
        self.actuator_current_a = 0.0

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
        """Update the actual actuator current.

        The actuator is modeled as a first-order system:

            tau dI/dt + I = I_command

        followed by a physical current slew-rate limit and hard saturation.
        """

        dt = self.config.dt_s

        limit = self.config.continuous_current_limit_a

        commanded_current_a = float(
            np.clip(
                commanded_current_a,
                -limit,
                limit,
            )
        )

        # First-order actuator time constant corresponding to the
        # specified -3 dB bandwidth.
        bandwidth_hz = self.config.actuator_bandwidth_hz

        tau = 1.0 / (
            2.0 * math.pi * bandwidth_hz
        )

        # Exact-discrete-equivalent style update.
        alpha = dt / (tau + dt)

        filtered_current = (
            self.actuator_current_a
            + alpha
            * (
                commanded_current_a
                - self.actuator_current_a
            )
        )

        # Apply physical slew-rate limitation.
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

        # Final hard continuous-current limit.
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
        """Advance the seven modal states by one RK4 step.

        Parameters
        ----------
        disturbance_n:
            Cutting-force disturbance in N.

        current_a:
            Commanded actuator current in A.

        Returns
        -------
        PlantState
            Updated tool-tip displacement and velocity.
        """

        # Convert commanded current to actual physical actuator current.
        actual_current_a = self._update_actuator_current(
            current_a
        )

        disturbance_n = float(disturbance_n)

        # Current-driven modes.
        for idx, mode in enumerate(
            self.config.current_modes
        ):
            omega = (
                2.0
                * math.pi
                * mode.natural_frequency_hz
            )

            # Numerator from G_qi(s)/k_yI,i:
            #
            #     omega_i^2 / k_yI,i
            #
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

        # Cutting-force-driven mode.
        mode = self.config.force_mode

        omega = (
            2.0
            * math.pi
            * mode.natural_frequency_hz
        )

        # Numerator from G_yF(s):
        #
        #     omega_o^2 / k_yF,o
        #
        gain = omega**2 / mode.stiffness_n_m

        self.force_q, self.force_v = self._rk4_mode(
            self.force_q,
            self.force_v,
            omega,
            mode.damping_ratio,
            gain,
            disturbance_n,
        )

        return self.state


# Keep the old import name as an alias.
SingleModePlant = SevenModePlant
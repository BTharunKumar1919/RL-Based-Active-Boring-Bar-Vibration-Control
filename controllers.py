
"""Baseline controllers for the single-mode vibration-control model."""
from dataclasses import dataclass

import numpy as np

from config import PLANT, PlantConfig


class NoControl:
    """Baseline with no active control force."""

    def reset(self):
        """Reset controller state; this controller has no internal state."""
        pass

    def act(self, displacement_m, velocity_m_s, dt_s):
        """Return zero control force."""
        return 0.0


@dataclass
class PIDController:
    """PID baseline with force saturation and integral anti-windup.

    The default gains are initial simulation values, not tuned industrial
    gains. The controller uses displacement and velocity in SI units.
    """

    kp: float = 50000.0
    ki: float = 0.0
    kd: float = 20.0
    plant_config: PlantConfig = PLANT
    integral_limit_m_s: float = 1.0e-3

    integral_error: float = 0.0

    def __post_init__(self):
        if self.kp < 0 or self.ki < 0 or self.kd < 0:
            raise ValueError("PID gains must be non-negative.")
        if self.integral_limit_m_s <= 0:
            raise ValueError("Integral limit must be positive.")

    def reset(self):
        """Clear the integral state before a new simulation."""
        self.integral_error = 0.0

    def act(self, displacement_m, velocity_m_s, dt_s):
        """Calculate a bounded force command opposing the measured motion."""
        if dt_s <= 0:
            raise ValueError("dt_s must be positive.")

        displacement_m = float(displacement_m)
        velocity_m_s = float(velocity_m_s)

        old_integral = self.integral_error
        candidate_integral = float(
            np.clip(
                old_integral + displacement_m * dt_s,
                -self.integral_limit_m_s,
                self.integral_limit_m_s,
            )
        )

        unsaturated_force_n = (
            -self.kp * displacement_m
            -self.kd * velocity_m_s
            -self.ki * candidate_integral
        )

        force_limit_n = self.plant_config.max_control_force_n
        saturated_force_n = float(
            np.clip(
                unsaturated_force_n,
                -force_limit_n,
                force_limit_n,
            )
        )

        # Conditional integration: avoid integrating farther into saturation.
        saturated_high = unsaturated_force_n > force_limit_n
        saturated_low = unsaturated_force_n < -force_limit_n

        if not (
            (saturated_high and displacement_m < 0)
            or (saturated_low and displacement_m > 0)
        ):
            self.integral_error = candidate_integral

        return saturated_force_n

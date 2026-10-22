"""Baseline controllers for the current-driven seven-mode plant."""
from dataclasses import dataclass
import numpy as np

from config import PLANT, PlantConfig


class NoControl:
    """Baseline with zero actuator current."""
    def reset(self):
        pass

    def act(self, displacement_m, velocity_m_s, dt_s):
        return 0.0


@dataclass
class PIDController:
    """PID controller whose output is current in amperes.

    Units:
        kp [A/m], ki [A/(m s)], kd [A/(m/s)].

    The gains are deliberately configurable from the dashboard so the PID
    can be retuned for different plant models instead of being hard-coded.
    """
    kp: float = 180000.0
    ki: float = 0.0
    kd: float = 0.0
    plant_config: PlantConfig = PLANT
    integral_limit_m_s: float = 1.0e-3
    integral_error: float = 0.0

    def __post_init__(self):
        # Signed gains are permitted for controller experiments. A negative
        # gain reverses that term's contribution to the control current.
        for name in ("kp", "ki", "kd"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f"PID {name} must be a finite number.")

        if (
            isinstance(self.integral_limit_m_s, bool)
            or not isinstance(self.integral_limit_m_s, (int, float))
            or not np.isfinite(self.integral_limit_m_s)
            or self.integral_limit_m_s <= 0
        ):
            raise ValueError("Integral limit must be a finite positive number.")

    def reset(self):
        self.integral_error = 0.0

    def act(self, displacement_m, velocity_m_s, dt_s):
        if dt_s <= 0:
            raise ValueError("dt_s must be positive.")

        x = float(displacement_m)
        v = float(velocity_m_s)

        candidate_integral = float(
            np.clip(
                self.integral_error + x * dt_s,
                -self.integral_limit_m_s,
                self.integral_limit_m_s,
            )
        )

        unsaturated_current = (
            -self.kp * x
            -self.kd * v
            -self.ki * candidate_integral
        )

        limit = self.plant_config.continuous_current_limit_a
        current = float(
            np.clip(
                unsaturated_current,
                -limit,
                limit,
            )
        )

        sat_high = unsaturated_current > limit
        sat_low = unsaturated_current < -limit

        # Simple conditional-integration anti-windup.
        if not ((sat_high and x < 0) or (sat_low and x > 0)):
            self.integral_error = candidate_integral

        return current

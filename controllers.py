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

    Units: kp [A/m], ki [A/(m s)], kd [A/(m/s)].
    The gains must be retuned for the new plant and current-limited output.
    """
    kp: float = 180000
    ki: float = 0.0
    kd: float = 0.0
    plant_config: PlantConfig = PLANT
    integral_limit_m_s: float = 1.0e-3
    integral_error: float = 0.0

    def __post_init__(self):
        if self.kp < 0 or self.ki < 0 or self.kd < 0:
            raise ValueError("PID gains must be non-negative.")
        if self.integral_limit_m_s <= 0:
            raise ValueError("Integral limit must be positive.")

    def reset(self):
        self.integral_error = 0.0

    def act(self, displacement_m, velocity_m_s, dt_s):
        if dt_s <= 0:
            raise ValueError("dt_s must be positive.")
        x = float(displacement_m)
        v = float(velocity_m_s)
        candidate_integral = float(np.clip(
            self.integral_error + x * dt_s,
            -self.integral_limit_m_s,
            self.integral_limit_m_s,
        ))
        unsaturated_current = -self.kp*x - self.kd*v - self.ki*candidate_integral
        limit = self.plant_config.continuous_current_limit_a
        current = float(np.clip(unsaturated_current, -limit, limit))
        sat_high = unsaturated_current > limit
        sat_low = unsaturated_current < -limit
        if not ((sat_high and x < 0) or (sat_low and x > 0)):
            self.integral_error = candidate_integral
        return current

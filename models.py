
"""Single-mode modal plant and harmonic test disturbance."""
from dataclasses import dataclass
import math

from config import PLANT, DISTURBANCE, PlantConfig, DisturbanceConfig


@dataclass
class PlantState:
    displacement_m: float = 0.0
    velocity_m_s: float = 0.0


class SinusoidalCuttingForce:
    """Harmonic generalized-force disturbance.

    This is a controlled test input, NOT a regenerative chatter model.
    """
    def __init__(self, config: DisturbanceConfig = DISTURBANCE):
        if config.amplitude_n < 0:
            raise ValueError("Disturbance amplitude must be non-negative.")
        if config.frequency_hz < 0:
            raise ValueError("Disturbance frequency must be non-negative.")

        self.config = config

    def __call__(self, time_s: float) -> float:
        return self.config.amplitude_n * math.sin(
            2.0 * math.pi * self.config.frequency_hz * time_s
            + self.config.phase_rad
        )


class SingleModePlant:
    """Single-degree-of-freedom modal-coordinate model.

    Governing equation:
        m*q_ddot + c*q_dot + k*q = F_disturbance + F_control

    A semi-implicit Euler integration step is used consistently for all
    controllers. The coordinate q is expressed in metres. Converting it
    to physical tool-tip displacement requires a mode-shape scaling factor,
    which has not been identified in this starter model.
    """
    def __init__(self, config: PlantConfig = PLANT):
        if config.modal_mass_kg <= 0:
            raise ValueError("Modal mass must be positive.")
        if config.natural_frequency_hz <= 0:
            raise ValueError("Natural frequency must be positive.")
        if not 0 < config.damping_ratio < 1:
            raise ValueError("Damping ratio must be between 0 and 1.")
        if config.dt_s <= 0:
            raise ValueError("Integration time step must be positive.")
        if config.max_control_force_n <= 0:
            raise ValueError("Maximum control force must be positive.")

        self.config = config
        self.state = PlantState()

    def reset(
        self,
        displacement_m: float = 0.0,
        velocity_m_s: float = 0.0,
    ) -> PlantState:
        """Reset the plant to the requested initial state."""
        self.state = PlantState(
            displacement_m=float(displacement_m),
            velocity_m_s=float(velocity_m_s),
        )
        return self.state

    def step(
        self,
        disturbance_n: float,
        control_n: float,
    ) -> PlantState:
        """Advance the plant by one semi-implicit Euler step."""
        c = self.config
        s = self.state

        acceleration_m_s2 = (
            float(disturbance_n)
            + float(control_n)
            - c.damping_ns_m * s.velocity_m_s
            - c.stiffness_n_m * s.displacement_m
        ) / c.modal_mass_kg

        # Semi-implicit Euler: update velocity before displacement.
        velocity_next_m_s = (
            s.velocity_m_s + acceleration_m_s2 * c.dt_s
        )
        displacement_next_m = (
            s.displacement_m + velocity_next_m_s * c.dt_s
        )

        self.state = PlantState(
            displacement_m=displacement_next_m,
            velocity_m_s=velocity_next_m_s,
        )
        return self.state

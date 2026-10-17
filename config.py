
"""Central parameters for the single-mode boring-bar control surrogate.

Values are separated into model parameters and experiment assumptions.
This is an educational reduced-order model, not an identified digital twin.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PlantConfig:
    # Working modal-frequency hypothesis. Verify the source before citing it.
    natural_frequency_hz: float = 174.0
    damping_ratio: float = 0.012

    # Normalization assumption, NOT an experimentally identified modal mass.
    modal_mass_kg: float = 1.0

    # Integration step: 10 kHz simulation update rate.
    dt_s: float = 1.0e-4

    # Assumed generalized actuator force limit; not a measured actuator rating.
    max_control_force_n: float = 2.0

    @property
    def natural_frequency_rad_s(self) -> float:
        return 2.0 * math.pi * self.natural_frequency_hz

    @property
    def stiffness_n_m(self) -> float:
        return self.modal_mass_kg * self.natural_frequency_rad_s**2

    @property
    def damping_ns_m(self) -> float:
        return (
            2.0
            * self.damping_ratio
            * self.modal_mass_kg
            * self.natural_frequency_rad_s
        )


@dataclass(frozen=True)
class DisturbanceConfig:
    amplitude_n: float = 1.0
    frequency_hz: float = 174.0
    phase_rad: float = 0.0


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    total_timesteps: int = 300_000

    # A 0.2 s episode contains 2,000 simulation steps at dt = 1e-4 s.
    episode_duration_s: float = 0.2

    learning_rate: float = 3e-4
    buffer_size: int = 200_000
    learning_starts: int = 5_000
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1

    # Training randomization ranges; these are experiment assumptions.
    force_amplitude_min_n: float = 0.5
    force_amplitude_max_n: float = 1.5
    force_frequency_min_hz: float = 150.0
    force_frequency_max_hz: float = 200.0

    initial_displacement_max_m: float = 1.0e-5
    initial_velocity_max_m_s: float = 0.005

    # Observation normalization scales.
    displacement_scale_m: float = 1.0e-4
    velocity_scale_m_s: float = 0.05

    # Reward weights.
    displacement_reward_weight: float = 1.0
    control_effort_reward_weight: float = 0.05
    action_change_reward_weight: float = 0.01


@dataclass(frozen=True)
class EvaluationConfig:
    duration_s: float = 0.2

    nominal_amplitude_n: float = 1.0
    nominal_frequency_hz: float = 174.0

    # Shifted test condition: outside the training frequency range.
    unseen_amplitude_n: float = 1.4
    unseen_frequency_hz: float = 205.0

    test_phase_rad: float = 0.73
    plot_time_window_s: float = 0.2


PLANT = PlantConfig()
DISTURBANCE = DisturbanceConfig()
TRAINING = TrainingConfig()
EVALUATION = EvaluationConfig()
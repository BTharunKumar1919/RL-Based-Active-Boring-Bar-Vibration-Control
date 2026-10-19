"""Configuration for the seven-mode current-driven boring-bar model."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CurrentDrivenMode:
    natural_frequency_hz: float
    damping_ratio: float
    k_yI: float


@dataclass(frozen=True)
class ForceDrivenMode:
    natural_frequency_hz: float
    damping_ratio: float
    stiffness_n_m: float


@dataclass(frozen=True)
class PlantConfig:
    current_modes: tuple = (
        CurrentDrivenMode(51.0, 0.03, -2.0e6),
        CurrentDrivenMode(67.0, 0.08, -4.0e5),
        CurrentDrivenMode(98.0, 0.03, -9.0e6),
        CurrentDrivenMode(138.0, 0.08, -8.0e5),
        CurrentDrivenMode(273.0, 0.05, 4.0e6),
        CurrentDrivenMode(450.0, 0.05, -9.5e5),
    )

    force_mode: ForceDrivenMode = ForceDrivenMode(
        natural_frequency_hz=188.0,
        damping_ratio=0.034,
        stiffness_n_m=2.6e6,
    )

    # Physical plant integration timestep.
    # 0.1 ms = 10 kHz.
    dt_s: float = 1.0e-4

    max_continuous_force_n: float = 248.0
    max_peak_force_n: float = 670.0
    max_dynamic_force_at_2000_hz_n: float = 107.0

    continuous_current_limit_a: float = 5.0
    peak_current_limit_a: float = 15.0

    current_loop_bandwidth_hz: float = 8000.0

    # Effective actuator bandwidth used by the RL plant.
    actuator_bandwidth_hz: float = 800.0

    # Physical current slew-rate limit.
    current_slew_rate_a_s: float = 10000.0


@dataclass(frozen=True)
class DisturbanceConfig:
    amplitude_n: float = 1.0
    frequency_hz: float = 174.0
    phase_rad: float = 0.0


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42

    total_timesteps: int = 300_000

    episode_duration_s: float = 0.2

    # ---------------------------------------------------------
    # RL control timing
    # ---------------------------------------------------------

    # Plant still integrates every 0.1 ms.
    # SAC updates its command every 5 plant steps = 0.5 ms.
    sac_action_repeat: int = 5

    # Number of recent SAC observations included in the policy input.
    #
    # 8 observations × 0.5 ms = 4 ms of history.
    observation_history_length: int = 8

    # ---------------------------------------------------------
    # SAC hyperparameters
    # ---------------------------------------------------------

    learning_rate: float = 3e-4
    buffer_size: int = 200_000
    learning_starts: int = 5_000
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1

    # ---------------------------------------------------------
    # Disturbance randomization
    # ---------------------------------------------------------

    force_amplitude_min_n: float = 0.5
    force_amplitude_max_n: float = 1.5

    force_frequency_min_hz: float = 150.0
    force_frequency_max_hz: float = 200.0

    initial_displacement_max_m: float = 1.0e-5
    initial_velocity_max_m_s: float = 0.005

    # ---------------------------------------------------------
    # Observation scaling
    # ---------------------------------------------------------

    displacement_scale_m: float = 5.0e-6
    velocity_scale_m_s: float = 0.005

    # ---------------------------------------------------------
    # Reward
    # ---------------------------------------------------------

    displacement_reward_weight: float = 1.0

    control_effort_reward_weight: float = 0.10

    action_change_reward_weight: float = 0.05


@dataclass(frozen=True)
class EvaluationConfig:
    duration_s: float = 0.2

    nominal_amplitude_n: float = 1.0
    nominal_frequency_hz: float = 174.0

    unseen_amplitude_n: float = 1.4
    unseen_frequency_hz: float = 205.0

    test_phase_rad: float = 0.73

    plot_time_window_s: float = 0.2


PLANT = PlantConfig()
DISTURBANCE = DisturbanceConfig()
TRAINING = TrainingConfig()
EVALUATION = EvaluationConfig()
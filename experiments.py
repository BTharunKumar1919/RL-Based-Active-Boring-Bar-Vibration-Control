"""Run fair controller comparisons for the seven-mode plant."""

from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stable_baselines3 import SAC

from config import (
    PLANT,
    TRAINING,
    EVALUATION,
    DisturbanceConfig,
)

from models import (
    SevenModePlant,
    SinusoidalCuttingForce,
)

from controllers import (
    NoControl,
    PIDController,
)


def _make_sac_observation(
    plant,
    history,
):
    """Construct the SAC observation from recent history."""

    state = plant.state

    current_limit = (
        plant.config.continuous_current_limit_a
    )

    current_observation = np.asarray(
        [
            state.displacement_m
            / TRAINING.displacement_scale_m,

            state.velocity_m_s
            / TRAINING.velocity_scale_m_s,

            plant.actual_current_a
            / current_limit,
        ],
        dtype=np.float32,
    )

    history.append(
        current_observation
    )

    if len(history) < TRAINING.observation_history_length:

        first = history[0]

        while len(history) < (
            TRAINING.observation_history_length
        ):
            history.appendleft(
                first.copy()
            )

    return np.concatenate(
        list(history)
    ).astype(np.float32)


def simulate_controller(
    controller_name,
    disturbance_config,
    duration_s,
    model=None,
    plant_config=PLANT,
    process_noise_std_n=0.0,
    process_noise_seed=None,
    pid_gains=None,
):
    """Simulate one controller under one disturbance condition."""

    if duration_s <= 0:
        raise ValueError(
            "duration_s must be positive."
        )

    plant = SevenModePlant(
        plant_config,
        process_noise_std_n=process_noise_std_n,
        process_noise_seed=process_noise_seed,
    )
    plant.reset()

    force_model = SinusoidalCuttingForce(
        disturbance_config
    )

    if controller_name == "No control":

        controller = NoControl()

    elif controller_name == "PID":

        pid_gains = {} if pid_gains is None else dict(pid_gains)
        controller = PIDController(
            kp=float(pid_gains.get("kp", 180000.0)),
            ki=float(pid_gains.get("ki", 0.0)),
            kd=float(pid_gains.get("kd", 0.0)),
            integral_limit_m_s=float(pid_gains.get("integral_limit_m_s", 1.0e-3)),
            plant_config=plant_config,
        )

    elif controller_name == "SAC":

        if model is None:
            raise ValueError(
                "A trained SAC model is required."
            )

        controller = None

    else:

        raise ValueError(
            f"Unknown controller: {controller_name}"
        )

    if controller is not None:
        controller.reset()

    dt = plant_config.dt_s

    action_repeat = (
        TRAINING.sac_action_repeat
    )

    rl_dt = dt * action_repeat

    n_steps = int(
        round(duration_s / dt)
    )

    records = []

    # ---------------------------------------------------------
    # SAC history.
    # ---------------------------------------------------------

    history = deque(
        maxlen=TRAINING.observation_history_length
    )

    if controller_name == "SAC":

        initial_observation = np.asarray(
            [
                plant.state.displacement_m
                / TRAINING.displacement_scale_m,

                plant.state.velocity_m_s
                / TRAINING.velocity_scale_m_s,

                plant.actual_current_a
                / plant_config.continuous_current_limit_a,
            ],
            dtype=np.float32,
        )

        for _ in range(
            TRAINING.observation_history_length
        ):
            history.append(
                initial_observation.copy()
            )

    # Current SAC command.
    sac_commanded_current_a = 0.0

    for step_index in range(
        n_steps
    ):

        time_s = (
            step_index * dt
        )

        state = plant.state

        disturbance_n = force_model(
            time_s
        )

        # =====================================================
        # SAC
        # =====================================================

        if controller_name == "SAC":

            # SAC selects a new command every
            # action_repeat physical plant steps.
            if step_index % action_repeat == 0:

                observation = (
                    _make_sac_observation(
                        plant,
                        history,
                    )
                )

                action, _ = model.predict(
                    observation,
                    deterministic=True,
                )

                normalized_action = float(
                    np.clip(
                        np.asarray(
                            action
                        ).reshape(-1)[0],
                        -1.0,
                        1.0,
                    )
                )

                sac_commanded_current_a = (
                    normalized_action
                    * plant_config.continuous_current_limit_a
                )

            commanded_current_a = (
                sac_commanded_current_a
            )

            normalized_action = (
                commanded_current_a
                / plant_config.continuous_current_limit_a
            )

        # =====================================================
        # PID / No Control
        # =====================================================

        else:

            commanded_current_a = float(
                controller.act(
                    state.displacement_m,
                    state.velocity_m_s,
                    dt,
                )
            )

            commanded_current_a = float(
                np.clip(
                    commanded_current_a,
                    -plant_config.continuous_current_limit_a,
                    plant_config.continuous_current_limit_a,
                )
            )

            normalized_action = (
                commanded_current_a
                / plant_config.continuous_current_limit_a
            )

        # =====================================================
        # Advance physical plant.
        # =====================================================

        next_state = plant.step(
            disturbance_n=disturbance_n,
            current_a=commanded_current_a,
        )

        records.append(
            {
                "time_s":
                    time_s + dt,

                "displacement_um":
                    next_state.displacement_m
                    * 1e6,

                "velocity_mm_s":
                    next_state.velocity_m_s
                    * 1e3,

                "fundamental_force_n":
                    force_model.last_fundamental_force_n,

                "harmonic_force_n":
                    force_model.last_harmonic_force_n,

                "cutting_force_noise_n":
                    force_model.last_noise_n,

                "deterministic_cutting_force_n":
                    force_model.last_deterministic_force_n,

                "disturbance_force_n":
                    disturbance_n,

                "process_noise_n":
                    plant.last_process_noise_n,

                "effective_disturbance_force_n":
                    disturbance_n + plant.last_process_noise_n,

                "commanded_current_a":
                    commanded_current_a,

                "control_current_a":
                    plant.actual_current_a,

                "normalized_action":
                    normalized_action,
            }
        )

    return pd.DataFrame(
        records
    )


def calculate_metrics(data, plant_config=PLANT):
    """Calculate vibration and actuator metrics."""

    if len(data) == 0:
        raise ValueError(
            "Cannot calculate metrics for empty data."
        )

    displacement = data[
        "displacement_um"
    ].to_numpy()

    current = data[
        "control_current_a"
    ].to_numpy()

    start = min(
        int(0.2 * len(data)),
        len(data) - 1,
    )

    steady_displacement = (
        displacement[start:]
    )

    steady_current = (
        current[start:]
    )

    limit = plant_config.continuous_current_limit_a

    near_saturation = (
        np.abs(current)
        >= 0.95 * limit
    )

    hard_saturation = (
        np.abs(current)
        >= 0.999 * limit
    )

    return {
        "rms_vibration_um":
            float(
                np.sqrt(
                    np.mean(
                        displacement**2
                    )
                )
            ),

        "peak_vibration_um":
            float(
                np.max(
                    np.abs(displacement)
                )
            ),

        "steady_rms_vibration_um":
            float(
                np.sqrt(
                    np.mean(
                        steady_displacement**2
                    )
                )
            ),

        "rms_control_current_a":
            float(
                np.sqrt(
                    np.mean(
                        steady_current**2
                    )
                )
            ),

        "peak_control_current_a":
            float(
                np.max(
                    np.abs(current)
                )
            ),

        "actuator_near_saturation_fraction":
            float(
                np.mean(
                    near_saturation
                )
            ),

        "actuator_saturation_fraction":
            float(
                np.mean(
                    hard_saturation
                )
            ),
    }


def _save_plots(
    condition_name,
    condition_data,
    output_dir,
):
    """Save displacement and current comparison plots."""

    output_dir = Path(
        output_dir
    )

    time_ms = (
        condition_data["No control"][
            "time_s"
        ].to_numpy()
        * 1000.0
    )

    # ---------------------------------------------------------
    # Displacement comparison.
    # ---------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    for name, data in (
        condition_data.items()
    ):

        ax.plot(
            time_ms,
            data["displacement_um"],
            label=name,
            linewidth=1.1,
        )

    ax.set_title(
        f"Displacement comparison — "
        f"{condition_name}"
    )

    ax.set_xlabel(
        "Time (ms)"
    )

    ax.set_ylabel(
        "Tool-tip displacement "
        "(µm; modal sum)"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir
        / f"displacement_{condition_name}_overlay.png",
        dpi=160,
    )

    plt.close(fig)

    # ---------------------------------------------------------
    # SAC displacement detail.
    # ---------------------------------------------------------

    sac_data = (
        condition_data["SAC"]
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        sac_data["time_s"] * 1000.0,
        sac_data["displacement_um"],
        linewidth=1.2,
        label="SAC",
    )

    ax.set_title(
        f"SAC displacement detail — "
        f"{condition_name}"
    )

    ax.set_xlabel(
        "Time (ms)"
    )

    ax.set_ylabel(
        "Tool-tip displacement "
        "(µm; modal sum)"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir
        / f"displacement_{condition_name}_sac_zoom.png",
        dpi=160,
    )

    plt.close(fig)

    # ---------------------------------------------------------
    # Actuator current.
    # ---------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    for name, data in (
        condition_data.items()
    ):

        ax.plot(
            data["time_s"] * 1000.0,
            data["control_current_a"],
            label=name,
            linewidth=1.0,
        )

    ax.axhline(
        PLANT.continuous_current_limit_a,
        linestyle="--",
        linewidth=0.8,
        label="Continuous current limit",
    )

    ax.axhline(
        -PLANT.continuous_current_limit_a,
        linestyle="--",
        linewidth=0.8,
    )

    ax.set_title(
        f"Actuator current — "
        f"{condition_name}"
    )

    ax.set_xlabel(
        "Time (ms)"
    )

    ax.set_ylabel(
        "Control current (A)"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir
        / f"control_{condition_name}.png",
        dpi=160,
    )

    plt.close(fig)


def run_experiments(
    model_path="results/sac_vibration_model_7mode.zip",
    output_dir="results",
):
    """Run nominal and unseen controller comparisons."""

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = SAC.load(
        str(model_path)
    )

    conditions = {
        "nominal": DisturbanceConfig(
            EVALUATION.nominal_amplitude_n,
            EVALUATION.nominal_frequency_hz,
            0.0,
        ),

        "unseen": DisturbanceConfig(
            EVALUATION.unseen_amplitude_n,
            EVALUATION.unseen_frequency_hz,
            EVALUATION.test_phase_rad,
        ),
    }

    all_metrics = []
    all_timeseries = []

    for (
        condition_name,
        disturbance,
    ) in conditions.items():

        condition_data = {}

        for controller_name in (
            "No control",
            "PID",
            "SAC",
        ):

            data = simulate_controller(
                controller_name,
                disturbance,
                EVALUATION.duration_s,
                model=(
                    model
                    if controller_name == "SAC"
                    else None
                ),
            )

            data.insert(
                0,
                "controller",
                controller_name,
            )

            data.insert(
                0,
                "condition",
                condition_name,
            )

            condition_data[
                controller_name
            ] = data

            metrics = calculate_metrics(
                data
            )

            metrics.update(
                condition=condition_name,
                controller=controller_name,
            )

            all_metrics.append(
                metrics
            )

            all_timeseries.append(
                data
            )

        _save_plots(
            condition_name,
            condition_data,
            output_dir,
        )

    metrics_frame = pd.DataFrame(
        all_metrics
    )

    timeseries_frame = pd.concat(
        all_timeseries,
        ignore_index=True,
    )

    metrics_frame.to_csv(
        output_dir
        / "controller_metrics.csv",
        index=False,
    )

    timeseries_frame.to_csv(
        output_dir
        / "controller_timeseries.csv",
        index=False,
    )

    print(
        "\nController metrics:"
    )

    print(
        metrics_frame.to_string(
            index=False,
            float_format=lambda x: f"{x:.6g}",
        )
    )

    print(
        f"\nResults saved under: "
        f"{output_dir.resolve()}"
    )

    return metrics_frame


if __name__ == "__main__":
    run_experiments()
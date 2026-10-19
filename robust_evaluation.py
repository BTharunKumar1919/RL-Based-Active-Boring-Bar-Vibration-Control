"""Robust evaluation of No Control, PID, and SAC."""

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
)

from models import (
    SevenModePlant,
    SinusoidalCuttingForce,
)

from controllers import (
    NoControl,
    PIDController,
)


AMPLITUDES_N = (
    0.5,
    1.0,
    1.5,
)

FREQUENCIES_HZ = (
    150.0,
    160.0,
    174.0,
    185.0,
    200.0,
    205.0,
    215.0,
)

PHASES_RAD = (
    0.0,
    np.pi / 2.0,
    np.pi,
    3.0 * np.pi / 2.0,
)

DURATION_S = 0.2


def make_sac_observation(
    plant,
    history,
):
    """Build the same observation used during SAC training."""

    state = plant.state

    current_observation = np.asarray(
        [
            state.displacement_m
            / TRAINING.displacement_scale_m,

            state.velocity_m_s
            / TRAINING.velocity_scale_m_s,

            plant.actual_current_a
            / PLANT.continuous_current_limit_a,
        ],
        dtype=np.float32,
    )

    history.append(
        current_observation
    )

    if len(history) < (
        TRAINING.observation_history_length
    ):

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


def simulate(
    controller_name,
    disturbance,
    model=None,
):
    """Simulate one controller."""

    plant = SevenModePlant(
        PLANT
    )

    plant.reset()

    force_model = (
        SinusoidalCuttingForce(
            disturbance
        )
    )

    if controller_name == "No control":

        controller = NoControl()

    elif controller_name == "PID":

        controller = PIDController(
            plant_config=PLANT
        )

    elif controller_name == "SAC":

        if model is None:
            raise ValueError(
                "SAC model is required."
            )

        controller = None

    else:

        raise ValueError(
            f"Unknown controller: "
            f"{controller_name}"
        )

    if controller is not None:
        controller.reset()

    dt = PLANT.dt_s

    action_repeat = (
        TRAINING.sac_action_repeat
    )

    n_steps = int(
        round(
            DURATION_S / dt
        )
    )

    records = []

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
                / PLANT.continuous_current_limit_a,
            ],
            dtype=np.float32,
        )

        for _ in range(
            TRAINING.observation_history_length
        ):

            history.append(
                initial_observation.copy()
            )

    sac_commanded_current_a = 0.0

    for step_index in range(
        n_steps
    ):

        time_s = (
            step_index * dt
        )

        state = plant.state

        disturbance_n = (
            force_model(time_s)
        )

        # -----------------------------------------------------
        # SAC
        # -----------------------------------------------------

        if controller_name == "SAC":

            if step_index % action_repeat == 0:

                observation = (
                    make_sac_observation(
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
                    * PLANT.continuous_current_limit_a
                )

            commanded_current_a = (
                sac_commanded_current_a
            )

            normalized_action = (
                commanded_current_a
                / PLANT.continuous_current_limit_a
            )

        # -----------------------------------------------------
        # PID / No Control
        # -----------------------------------------------------

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
                    -PLANT.continuous_current_limit_a,
                    PLANT.continuous_current_limit_a,
                )
            )

            normalized_action = (
                commanded_current_a
                / PLANT.continuous_current_limit_a
            )

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

                "disturbance_force_n":
                    disturbance_n,

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


def calculate_metrics(data):
    """Calculate robust-evaluation metrics."""

    displacement = (
        data["displacement_um"]
        .to_numpy()
    )

    current = (
        data["control_current_a"]
        .to_numpy()
    )

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

    limit = (
        PLANT.continuous_current_limit_a
    )

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

        "near_saturation_percent":
            float(
                100.0
                * np.mean(
                    near_saturation
                )
            ),

        "hard_saturation_percent":
            float(
                100.0
                * np.mean(
                    hard_saturation
                )
            ),
    }


def main():
    """Run the complete robust evaluation."""

    model_path = (
        "results/"
        "sac_vibration_model_7mode.zip"
    )

    output_dir = Path(
        "results"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = SAC.load(
        model_path
    )

    results = []

    total_cases = (
        len(AMPLITUDES_N)
        * len(FREQUENCIES_HZ)
        * len(PHASES_RAD)
    )

    case_number = 0

    for amplitude_n in AMPLITUDES_N:

        for frequency_hz in FREQUENCIES_HZ:

            for phase_rad in PHASES_RAD:

                disturbance = (
                    type(
                        "Disturbance",
                        (),
                        {
                            "amplitude_n":
                                amplitude_n,

                            "frequency_hz":
                                frequency_hz,

                            "phase_rad":
                                phase_rad,
                        },
                    )()
                )

                for controller_name in (
                    "No control",
                    "PID",
                    "SAC",
                ):

                    data = simulate(
                        controller_name,
                        disturbance,
                        model=(
                            model
                            if controller_name == "SAC"
                            else None
                        ),
                    )

                    metrics = (
                        calculate_metrics(
                            data
                        )
                    )

                    metrics.update(
                        controller=controller_name,
                        amplitude_n=amplitude_n,
                        frequency_hz=frequency_hz,
                        phase_rad=phase_rad,
                    )

                    results.append(
                        metrics
                    )

                case_number += 1

                print(
                    f"Completed "
                    f"{case_number}/{total_cases}"
                )

    results_frame = pd.DataFrame(
        results
    )

    output_csv = (
        output_dir
        / "robust_evaluation.csv"
    )

    results_frame.to_csv(
        output_csv,
        index=False,
    )

    print(
        "\nRobust evaluation saved to:"
    )

    print(
        output_csv.resolve()
    )

    print(
        "\nSummary:"
    )

    summary = (
        results_frame
        .groupby("controller")
        [
            [
                "rms_vibration_um",
                "steady_rms_vibration_um",
                "peak_vibration_um",
                "rms_control_current_a",
                "peak_control_current_a",
                "near_saturation_percent",
                "hard_saturation_percent",
            ]
        ]
        .mean()
    )

    print(
        summary.to_string(
            float_format=lambda x: f"{x:.6g}"
        )
    )


if __name__ == "__main__":
    main()
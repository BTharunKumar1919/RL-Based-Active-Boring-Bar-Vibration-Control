"""Robust evaluation of No Control, PID, and SAC.

This script evaluates the current V3 seven-mode boring-bar model over a
predefined grid of disturbance amplitudes, frequencies, and phases.

It uses the same SAC deployment logic as the current training setup:
    - 8-state observation history
    - actual actuator current in the observation
    - SAC action held for sac_action_repeat plant steps
    - commanded current limited to the continuous current limit
    - actuator bandwidth, slew-rate limiting, and saturation handled by
      SevenModePlant

Every run regenerates the robust-evaluation CSV files and PNG plots in
the results/ directory so that stale results are not accidentally reused.
"""

from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stable_baselines3 import SAC

from config import PLANT, TRAINING
from models import SevenModePlant, SinusoidalCuttingForce
from controllers import NoControl, PIDController


# ---------------------------------------------------------------------------
# Evaluation grid
# ---------------------------------------------------------------------------

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

CONTROLLERS = (
    "No control",
    "PID",
    "SAC",
)


# ---------------------------------------------------------------------------
# SAC observation
# ---------------------------------------------------------------------------

def make_sac_observation(plant, history):
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

    history.append(current_observation)

    if len(history) < TRAINING.observation_history_length:

        first = history[0]

        while len(history) < TRAINING.observation_history_length:
            history.appendleft(first.copy())

    return np.concatenate(
        list(history)
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate(
    controller_name,
    disturbance,
    model=None,
):
    """Simulate one controller for one disturbance condition."""

    plant = SevenModePlant(PLANT)
    plant.reset()

    force_model = SinusoidalCuttingForce(disturbance)

    if controller_name == "No control":

        controller = NoControl()

    elif controller_name == "PID":

        controller = PIDController(
            plant_config=PLANT
        )

    elif controller_name == "SAC":

        if model is None:
            raise ValueError(
                "SAC model is required for SAC evaluation."
            )

        controller = None

    else:

        raise ValueError(
            f"Unknown controller: {controller_name}"
        )

    if controller is not None:
        controller.reset()

    dt = PLANT.dt_s

    action_repeat = int(
        TRAINING.sac_action_repeat
    )

    n_steps = int(
        round(DURATION_S / dt)
    )

    records = []

    history = deque(
        maxlen=TRAINING.observation_history_length
    )

    # ---------------------------------------------------------------
    # Initialize SAC observation history exactly as during training.
    # ---------------------------------------------------------------

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

    # SAC command is held between policy updates.
    sac_commanded_current_a = 0.0

    for step_index in range(n_steps):

        time_s = step_index * dt

        state = plant.state

        disturbance_n = force_model(time_s)

        # -----------------------------------------------------------
        # SAC
        # -----------------------------------------------------------

        if controller_name == "SAC":

            # Update the SAC action only every action_repeat plant steps.
            if step_index % action_repeat == 0:

                observation = make_sac_observation(
                    plant,
                    history,
                )

                action, _ = model.predict(
                    observation,
                    deterministic=True,
                )

                normalized_action = float(
                    np.clip(
                        np.asarray(action).reshape(-1)[0],
                        -1.0,
                        1.0,
                    )
                )

                sac_commanded_current_a = (
                    normalized_action
                    * PLANT.continuous_current_limit_a
                )

            commanded_current_a = sac_commanded_current_a

            normalized_action = (
                commanded_current_a
                / PLANT.continuous_current_limit_a
            )

        # -----------------------------------------------------------
        # PID / No Control
        # -----------------------------------------------------------

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

        # -----------------------------------------------------------
        # Plant update
        #
        # The plant applies:
        #   commanded current
        #       -> actuator bandwidth
        #       -> slew-rate limit
        #       -> current saturation
        #       -> modal dynamics
        # -----------------------------------------------------------

        next_state = plant.step(
            disturbance_n=disturbance_n,
            current_a=commanded_current_a,
        )

        records.append(
            {
                "time_s": time_s + dt,

                "displacement_um":
                    next_state.displacement_m * 1e6,

                "velocity_mm_s":
                    next_state.velocity_m_s * 1e3,

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

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def calculate_metrics(data):
    """Calculate the robust-evaluation performance metrics."""

    displacement = data[
        "displacement_um"
    ].to_numpy()

    current = data[
        "control_current_a"
    ].to_numpy()

    # First 20% treated as transient.
    start = min(
        int(0.2 * len(data)),
        len(data) - 1,
    )

    steady_displacement = displacement[start:]
    steady_current = current[start:]

    limit = PLANT.continuous_current_limit_a

    near_saturation = (
        np.abs(current)
        >= 0.95 * limit
    )

    hard_saturation = (
        np.abs(current)
        >= 0.999 * limit
    )

    return {
        "rms_vibration_um": float(
            np.sqrt(
                np.mean(
                    displacement ** 2
                )
            )
        ),

        "peak_vibration_um": float(
            np.max(
                np.abs(displacement)
            )
        ),

        "steady_rms_vibration_um": float(
            np.sqrt(
                np.mean(
                    steady_displacement ** 2
                )
            )
        ),

        "rms_control_current_a": float(
            np.sqrt(
                np.mean(
                    steady_current ** 2
                )
            )
        ),

        "peak_control_current_a": float(
            np.max(
                np.abs(current)
            )
        ),

        "near_saturation_percent": float(
            100.0
            * np.mean(
                near_saturation
            )
        ),

        "hard_saturation_percent": float(
            100.0
            * np.mean(
                hard_saturation
            )
        ),
    }


# ---------------------------------------------------------------------------
# Result processing
# ---------------------------------------------------------------------------

def add_reduction_metrics(results_frame):
    """Add vibration-reduction percentages relative to No control."""

    key_columns = [
        "amplitude_n",
        "frequency_hz",
        "phase_rad",
    ]

    baseline = (
        results_frame[
            results_frame["controller"] == "No control"
        ][
            key_columns + [
                "rms_vibration_um",
                "steady_rms_vibration_um",
                "peak_vibration_um",
            ]
        ]
        .rename(
            columns={
                "rms_vibration_um":
                    "baseline_rms_vibration_um",

                "steady_rms_vibration_um":
                    "baseline_steady_rms_vibration_um",

                "peak_vibration_um":
                    "baseline_peak_vibration_um",
            }
        )
    )

    merged = results_frame.merge(
        baseline,
        on=key_columns,
        how="left",
    )

    merged["rms_reduction_percent"] = (
        100.0
        * (
            1.0
            - merged["rms_vibration_um"]
            / merged["baseline_rms_vibration_um"]
        )
    )

    merged["steady_rms_reduction_percent"] = (
        100.0
        * (
            1.0
            - merged["steady_rms_vibration_um"]
            / merged["baseline_steady_rms_vibration_um"]
        )
    )

    merged["peak_reduction_percent"] = (
        100.0
        * (
            1.0
            - merged["peak_vibration_um"]
            / merged["baseline_peak_vibration_um"]
        )
    )

    return merged


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _controller_order(frame):
    """Return controllers in the standard evaluation order."""

    present = set(
        frame["controller"].unique()
    )

    return [
        controller
        for controller in CONTROLLERS
        if controller in present
    ]


def save_rms_plot(results_frame, output_dir):
    """Save mean steady RMS vibration comparison."""

    summary = (
        results_frame
        .groupby("controller")[
            "steady_rms_vibration_um"
        ]
        .mean()
        .reindex(
            _controller_order(results_frame)
        )
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        summary.index,
        summary.values,
    )

    ax.set_xlabel("Controller")
    ax.set_ylabel("Mean steady RMS vibration (µm)")
    ax.set_title(
        "Robust evaluation: mean steady RMS vibration"
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        output_dir / "robust_evaluation_rms.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_reduction_plot(results_frame, output_dir):
    """Save mean vibration-reduction comparison."""

    controller_results = results_frame[
        results_frame["controller"].isin(
            ["PID", "SAC"]
        )
    ]

    summary = (
        controller_results
        .groupby("controller")[
            "steady_rms_reduction_percent"
        ]
        .mean()
        .reindex(
            ["PID", "SAC"]
        )
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        summary.index,
        summary.values,
    )

    ax.axhline(
        0.0,
        linewidth=1.0,
    )

    ax.set_xlabel("Controller")
    ax.set_ylabel("Mean steady RMS reduction (%)")
    ax.set_title(
        "Robust evaluation: vibration reduction vs No control"
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        output_dir / "robust_evaluation_reduction.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_current_plot(results_frame, output_dir):
    """Save mean steady RMS actuator-current comparison."""

    summary = (
        results_frame
        .groupby("controller")[
            "rms_control_current_a"
        ]
        .mean()
        .reindex(
            _controller_order(results_frame)
        )
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        summary.index,
        summary.values,
    )

    ax.set_xlabel("Controller")
    ax.set_ylabel("Mean steady RMS current (A)")
    ax.set_title(
        "Robust evaluation: actuator current"
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        output_dir / "robust_evaluation_current.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_peak_plot(results_frame, output_dir):
    """Save mean peak vibration comparison."""

    summary = (
        results_frame
        .groupby("controller")[
            "peak_vibration_um"
        ]
        .mean()
        .reindex(
            _controller_order(results_frame)
        )
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        summary.index,
        summary.values,
    )

    ax.set_xlabel("Controller")
    ax.set_ylabel("Mean peak vibration (µm)")
    ax.set_title(
        "Robust evaluation: mean peak vibration"
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        output_dir / "robust_evaluation_peak.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_frequency_plot(results_frame, output_dir):
    """Save steady RMS vibration versus frequency."""

    summary = (
        results_frame
        .groupby(
            [
                "frequency_hz",
                "controller",
            ]
        )[
            "steady_rms_vibration_um"
        ]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    for controller in CONTROLLERS:

        subset = summary[
            summary["controller"] == controller
        ]

        if subset.empty:
            continue

        ax.plot(
            subset["frequency_hz"],
            subset["steady_rms_vibration_um"],
            marker="o",
            label=controller,
        )

    ax.set_xlabel("Disturbance frequency (Hz)")
    ax.set_ylabel("Mean steady RMS vibration (µm)")
    ax.set_title(
        "Robust evaluation: vibration across disturbance frequency"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir / "robust_evaluation_frequency.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main():
    """Run the complete robust evaluation and regenerate all results."""

    model_path = Path(
        "results"
    ) / "sac_vibration_model_7mode.zip"

    output_dir = Path(
        "results"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not model_path.is_file():
        raise FileNotFoundError(
            f"SAC model not found: {model_path}"
        )

    print("=" * 70)
    print("ROBUST EVALUATION")
    print("=" * 70)

    print(
        f"Amplitude range: {AMPLITUDES_N}"
    )

    print(
        f"Frequency range: {FREQUENCIES_HZ}"
    )

    print(
        f"Phase count: {len(PHASES_RAD)}"
    )

    print(
        f"Duration: {DURATION_S:.3f} s"
    )

    print(
        f"SAC observation history: "
        f"{TRAINING.observation_history_length}"
    )

    print(
        f"SAC action repeat: "
        f"{TRAINING.sac_action_repeat} plant steps"
    )

    print(
        f"SAC action interval: "
        f"{TRAINING.sac_action_repeat * PLANT.dt_s * 1e3:.3f} ms"
    )

    print(
        f"Plant timestep: "
        f"{PLANT.dt_s * 1e6:.1f} µs"
    )

    total_cases = (
        len(AMPLITUDES_N)
        * len(FREQUENCIES_HZ)
        * len(PHASES_RAD)
    )

    total_simulations = (
        total_cases
        * len(CONTROLLERS)
    )

    print(
        f"Operating conditions: {total_cases}"
    )

    print(
        f"Total simulations: {total_simulations}"
    )

    print("=" * 70)

    # ---------------------------------------------------------------
    # Load SAC model once.
    # ---------------------------------------------------------------

    model = SAC.load(
        str(model_path)
    )

    results = []

    case_number = 0

    # ---------------------------------------------------------------
    # Evaluate all conditions.
    # ---------------------------------------------------------------

    for amplitude_n in AMPLITUDES_N:

        for frequency_hz in FREQUENCIES_HZ:

            for phase_rad in PHASES_RAD:

                disturbance = type(
                    "Disturbance",
                    (),
                    {
                        "amplitude_n": amplitude_n,
                        "frequency_hz": frequency_hz,
                        "phase_rad": phase_rad,
                    },
                )()

                for controller_name in CONTROLLERS:

                    data = simulate(
                        controller_name,
                        disturbance,
                        model=(
                            model
                            if controller_name == "SAC"
                            else None
                        ),
                    )

                    metrics = calculate_metrics(
                        data
                    )

                    metrics.update(
                        {
                            "controller":
                                controller_name,

                            "amplitude_n":
                                amplitude_n,

                            "frequency_hz":
                                frequency_hz,

                            "phase_rad":
                                phase_rad,
                        }
                    )

                    results.append(
                        metrics
                    )

                case_number += 1

                print(
                    f"Completed "
                    f"{case_number}/{total_cases} "
                    f"conditions"
                )

    # ---------------------------------------------------------------
    # Assemble results.
    # ---------------------------------------------------------------

    results_frame = pd.DataFrame(
        results
    )

    results_frame = add_reduction_metrics(
        results_frame
    )

    # ---------------------------------------------------------------
    # Remove old generated robust-evaluation outputs.
    #
    # This is deliberate. It prevents an old PNG/CSV from remaining
    # in results/ and being mistaken for the current evaluation.
    # ---------------------------------------------------------------

    old_output_files = [
        "robust_evaluation.csv",
        "robust_evaluation_by_condition.csv",
        "robust_evaluation_summary.csv",
        "robust_evaluation_rms.png",
        "robust_evaluation_reduction.png",
        "robust_evaluation_current.png",
        "robust_evaluation_peak.png",
        "robust_evaluation_frequency.png",
    ]

    for filename in old_output_files:

        old_file = output_dir / filename

        if old_file.exists():
            old_file.unlink()

    # ---------------------------------------------------------------
    # Save complete raw result table.
    # ---------------------------------------------------------------

    results_csv = (
        output_dir
        / "robust_evaluation.csv"
    )

    results_frame.to_csv(
        results_csv,
        index=False,
    )

    # ---------------------------------------------------------------
    # Save controller-by-condition table.
    #
    # Each operating condition is one row with controller metrics
    # arranged as separate columns.
    # ---------------------------------------------------------------

    key_columns = [
        "amplitude_n",
        "frequency_hz",
        "phase_rad",
    ]

    value_columns = [
        "rms_vibration_um",
        "steady_rms_vibration_um",
        "peak_vibration_um",
        "rms_control_current_a",
        "peak_control_current_a",
        "near_saturation_percent",
        "hard_saturation_percent",
        "rms_reduction_percent",
        "steady_rms_reduction_percent",
        "peak_reduction_percent",
    ]

    by_condition = (
        results_frame
        .pivot(
            index=key_columns,
            columns="controller",
            values=value_columns,
        )
        .reset_index()
    )

    # Flatten MultiIndex column names.
    flattened_columns = []

    for column in by_condition.columns:

        if isinstance(column, tuple):

            metric, controller = column

            if controller == "":
                flattened_columns.append(
                    metric
                )
            else:
                flattened_columns.append(
                    f"{controller}_{metric}"
                )

        else:

            flattened_columns.append(
                column
            )

    by_condition.columns = flattened_columns

    by_condition_csv = (
        output_dir
        / "robust_evaluation_by_condition.csv"
    )

    by_condition.to_csv(
        by_condition_csv,
        index=False,
    )

    # ---------------------------------------------------------------
    # Save controller-level summary.
    # ---------------------------------------------------------------

    summary_metrics = [
        "rms_vibration_um",
        "steady_rms_vibration_um",
        "peak_vibration_um",
        "rms_control_current_a",
        "peak_control_current_a",
        "near_saturation_percent",
        "hard_saturation_percent",
        "rms_reduction_percent",
        "steady_rms_reduction_percent",
        "peak_reduction_percent",
    ]

    summary = (
        results_frame
        .groupby("controller")[
            summary_metrics
        ]
        .mean()
        .reindex(CONTROLLERS)
    )

    summary_csv = (
        output_dir
        / "robust_evaluation_summary.csv"
    )

    summary.to_csv(
        summary_csv
    )

    # ---------------------------------------------------------------
    # Generate CURRENT plots.
    # ---------------------------------------------------------------

    save_rms_plot(
        results_frame,
        output_dir,
    )

    save_reduction_plot(
        results_frame,
        output_dir,
    )

    save_current_plot(
        results_frame,
        output_dir,
    )

    save_peak_plot(
        results_frame,
        output_dir,
    )

    save_frequency_plot(
        results_frame,
        output_dir,
    )

    # ---------------------------------------------------------------
    # Console summary.
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print("ROBUST EVALUATION COMPLETE")
    print("=" * 70)

    print()
    print("Summary:")
    print()

    print(
        summary.to_string(
            float_format=lambda x: f"{x:.6g}"
        )
    )

    print()
    print("Updated result files:")

    print(
        f"  {results_csv.resolve()}"
    )

    print(
        f"  {by_condition_csv.resolve()}"
    )

    print(
        f"  {summary_csv.resolve()}"
    )

    print()
    print("Updated plots:")

    for filename in (
        "robust_evaluation_rms.png",
        "robust_evaluation_reduction.png",
        "robust_evaluation_current.png",
        "robust_evaluation_peak.png",
        "robust_evaluation_frequency.png",
    ):

        print(
            f"  {(output_dir / filename).resolve()}"
        )

    print()
    print(
        f"Rows in robust_evaluation.csv: "
        f"{len(results_frame)}"
    )

    print(
        f"Expected rows: "
        f"{total_simulations}"
    )

    if len(results_frame) != total_simulations:
        raise RuntimeError(
            "Unexpected number of robust-evaluation rows."
        )

    print()
    print("All robust-evaluation outputs were regenerated.")
    print("=" * 70)


if __name__ == "__main__":
    main()
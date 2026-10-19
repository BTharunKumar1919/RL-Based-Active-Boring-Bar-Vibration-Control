"""Fast two-stage PD/PID tuning for the seven-mode current-driven plant.

The PID/PD controller is evaluated through the same actuator dynamics used
by SAC:

    commanded current
            ↓
    actuator bandwidth
            ↓
    current slew-rate limit
            ↓
    current saturation
            ↓
    seven-mode mechanical plant

Stage 1:
    Coarse search over the training operating conditions.

Stage 2:
    Fine search around the best Stage-1 gains.

Final:
    Evaluate the selected gains on separate validation conditions.
"""

from pathlib import Path
import itertools
import json
import math

import numpy as np
import pandas as pd

from config import PLANT, DisturbanceConfig, TRAINING
from controllers import PIDController
from models import SevenModePlant, SinusoidalCuttingForce


# ============================================================
# GENERAL SETTINGS
# ============================================================

DURATION_S = 0.2

# Ignore the first 20% of the simulation when calculating
# steady-state performance.
STEADY_START_FRACTION = 0.20


# ============================================================
# STAGE 1 — COARSE SEARCH
# ============================================================

# Deliberately smaller than the previous 15 x 15 grid.
#
# The purpose of Stage 1 is to identify the useful region,
# not to find the exact optimum.

KP_STAGE1 = [
    0,
    25_000,
    50_000,
    75_000,
    100_000,
    150_000,
    200_000,
    300_000,
    400_000,
    500_000,
]

KD_STAGE1 = [
    0,
    5,
    10,
    20,
    40,
    75,
    100,
    150,
    250,
]


# ============================================================
# STAGE 2 — FINE SEARCH
# ============================================================

# These are generated automatically around the Stage-1 winner.
#
# Example:
#   best Kp = 150000
#
# searches approximately:
#   105000 ... 195000
#
# around that point.

KP_FINE_MULTIPLIERS = [
    0.70,
    0.80,
    0.90,
    0.95,
    1.00,
    1.05,
    1.10,
    1.20,
    1.30,
]

KD_FINE_MULTIPLIERS = [
    0.50,
    0.70,
    0.85,
    1.00,
    1.15,
    1.30,
    1.50,
]


# ============================================================
# TRAINING / TUNING CONDITIONS
# ============================================================

# These are deliberately close to the operating range used
# by SAC training.

TRAINING_AMPLITUDES_N = (
    0.5,
    1.0,
    1.5,
)

TRAINING_FREQUENCIES_HZ = (
    150.0,
    174.0,
    200.0,
)

TRAINING_PHASES_RAD = (
    0.0,
    math.pi / 2,
)


TRAINING_CONDITIONS = [
    (amplitude, frequency, phase)
    for amplitude, frequency, phase in itertools.product(
        TRAINING_AMPLITUDES_N,
        TRAINING_FREQUENCIES_HZ,
        TRAINING_PHASES_RAD,
    )
]


# ============================================================
# VALIDATION CONDITIONS
# ============================================================

# These are NOT used to select the gains.
#
# They are deliberately different from the tuning conditions.

VALIDATION_AMPLITUDES_N = (
    0.75,
    1.25,
)

VALIDATION_FREQUENCIES_HZ = (
    140.0,
    160.0,
    185.0,
    205.0,
    215.0,
)

VALIDATION_PHASES_RAD = (
    0.0,
    math.pi / 2,
    math.pi,
    3.0 * math.pi / 2,
)


VALIDATION_CONDITIONS = [
    (amplitude, frequency, phase)
    for amplitude, frequency, phase in itertools.product(
        VALIDATION_AMPLITUDES_N,
        VALIDATION_FREQUENCIES_HZ,
        VALIDATION_PHASES_RAD,
    )
]


# ============================================================
# OBJECTIVE WEIGHTS
# ============================================================

# Vibration remains the dominant objective.
#
# A modest current penalty discourages unnecessarily aggressive
# control without forcing the optimizer toward zero control.
#
# Saturation is penalized more strongly because persistent
# saturation is undesirable for the physical actuator.

DISPLACEMENT_WEIGHT = 1.0
CURRENT_WEIGHT = 0.05
SATURATION_WEIGHT = 2.0


# ============================================================
# SIMULATION
# ============================================================

def evaluate_gains(kp, kd, conditions):
    """Evaluate one PD controller across a set of conditions."""

    rms_displacement_um = []
    peak_displacement_um = []
    rms_current_a = []
    current_change_rms_a = []
    saturation_fraction = []

    for amplitude, frequency, phase in conditions:

        # ----------------------------------------------------
        # Fresh plant for every operating condition.
        # ----------------------------------------------------

        plant = SevenModePlant(PLANT)

        plant.reset()

        disturbance = SinusoidalCuttingForce(
            DisturbanceConfig(
                amplitude_n=amplitude,
                frequency_hz=frequency,
                phase_rad=phase,
            )
        )

        controller = PIDController(
            kp=float(kp),
            ki=0.0,
            kd=float(kd),
            plant_config=PLANT,
        )

        controller.reset()

        n_steps = int(
            round(DURATION_S / PLANT.dt_s)
        )

        steady_start = int(
            STEADY_START_FRACTION * n_steps
        )

        displacement = np.empty(n_steps)
        current = np.empty(n_steps)

        # ----------------------------------------------------
        # Run simulation.
        # ----------------------------------------------------

        for i in range(n_steps):

            state = plant.state

            command_current_a = controller.act(
                state.displacement_m,
                state.velocity_m_s,
                PLANT.dt_s,
            )

            disturbance_force_n = disturbance(
                i * PLANT.dt_s
            )

            new_state = plant.step(
                disturbance_n=disturbance_force_n,
                current_a=command_current_a,
            )

            displacement[i] = (
                new_state.displacement_m
            )

            current[i] = (
                plant.actual_current_a
            )

        # ----------------------------------------------------
        # Steady-state data.
        # ----------------------------------------------------

        x = displacement[steady_start:]
        u = current[steady_start:]

        x_um = x * 1e6

        # ----------------------------------------------------
        # Metrics.
        # ----------------------------------------------------

        rms_x = float(
            np.sqrt(
                np.mean(x_um ** 2)
            )
        )

        peak_x = float(
            np.max(
                np.abs(x_um)
            )
        )

        rms_u = float(
            np.sqrt(
                np.mean(u ** 2)
            )
        )

        du = np.diff(u)

        if du.size > 0:
            rms_du = float(
                np.sqrt(
                    np.mean(du ** 2)
                )
            )
        else:
            rms_du = 0.0

        sat = float(
            np.mean(
                np.abs(u)
                >=
                0.99
                * PLANT.continuous_current_limit_a
            )
        )

        rms_displacement_um.append(rms_x)
        peak_displacement_um.append(peak_x)
        rms_current_a.append(rms_u)
        current_change_rms_a.append(rms_du)
        saturation_fraction.append(sat)

    # ========================================================
    # Average over all conditions.
    # ========================================================

    mean_rms_um = float(
        np.mean(rms_displacement_um)
    )

    mean_peak_um = float(
        np.mean(peak_displacement_um)
    )

    mean_current_a = float(
        np.mean(rms_current_a)
    )

    mean_current_change_a = float(
        np.mean(current_change_rms_a)
    )

    mean_saturation = float(
        np.mean(saturation_fraction)
    )

    # ========================================================
    # Objective
    # ========================================================

    # Use the same displacement normalization scale as SAC.

    normalized_displacement = (
        mean_rms_um * 1e-6
        / TRAINING.displacement_scale_m
    )

    normalized_current = (
        mean_current_a
        / PLANT.continuous_current_limit_a
    )

    score = (
        DISPLACEMENT_WEIGHT
        * normalized_displacement ** 2

        +

        CURRENT_WEIGHT
        * normalized_current ** 2

        +

        SATURATION_WEIGHT
        * mean_saturation
    )

    return {
        "kp": float(kp),
        "ki": 0.0,
        "kd": float(kd),

        "mean_steady_rms_um": mean_rms_um,

        "mean_peak_um": mean_peak_um,

        "mean_rms_current_a": mean_current_a,

        "mean_current_change_rms_a":
            mean_current_change_a,

        "mean_saturation_fraction":
            mean_saturation,

        "score": float(score),
    }


# ============================================================
# GRID SEARCH
# ============================================================

def evaluate_grid(
    kp_values,
    kd_values,
    conditions,
    stage_name,
):
    """Evaluate every Kp/Kd combination."""

    gain_pairs = list(
        itertools.product(
            kp_values,
            kd_values,
        )
    )

    total = len(gain_pairs)

    results = []

    print()
    print("=" * 70)
    print(stage_name)
    print("=" * 70)

    print(
        f"Kp candidates: "
        f"{len(kp_values)}"
    )

    print(
        f"Kd candidates: "
        f"{len(kd_values)}"
    )

    print(
        f"Operating conditions: "
        f"{len(conditions)}"
    )

    print(
        f"Total gain combinations: "
        f"{total}"
    )

    print()

    for index, (kp, kd) in enumerate(
        gain_pairs,
        start=1,
    ):

        result = evaluate_gains(
            kp,
            kd,
            conditions,
        )

        results.append(result)

        print(
            f"Tested "
            f"{index}/{total} "
            f"gain combinations.",
            flush=True,
        )

    return (
        pd.DataFrame(results)
        .sort_values("score")
        .reset_index(drop=True)
    )


# ============================================================
# FINE GRID
# ============================================================

def make_fine_grid(
    best_kp,
    best_kd,
):
    """Generate a fine grid around the Stage-1 optimum."""

    # --------------------------------------------------------
    # Kp
    # --------------------------------------------------------

    if best_kp == 0:

        kp_values = [
            0,
            10_000,
            20_000,
            30_000,
            40_000,
            50_000,
            75_000,
        ]

    else:

        kp_values = [
            best_kp * multiplier
            for multiplier
            in KP_FINE_MULTIPLIERS
        ]

    # --------------------------------------------------------
    # Kd
    # --------------------------------------------------------

    if best_kd == 0:

        kd_values = [
            0,
            1,
            2,
            5,
            10,
            20,
            30,
            40,
            60,
            80,
        ]

    else:

        kd_values = [
            best_kd * multiplier
            for multiplier
            in KD_FINE_MULTIPLIERS
        ]

    # --------------------------------------------------------
    # Round and remove duplicates.
    # --------------------------------------------------------

    kp_values = sorted(
        set(
            round(value, 2)
            for value in kp_values
            if value >= 0
        )
    )

    kd_values = sorted(
        set(
            round(value, 4)
            for value in kd_values
            if value >= 0
        )
    )

    return kp_values, kd_values


# ============================================================
# VALIDATION
# ============================================================

def validate_gains(
    kp,
    kd,
    conditions,
):
    """Evaluate one selected controller on validation conditions."""

    rows = []

    for amplitude, frequency, phase in conditions:

        result = evaluate_gains(
            kp,
            kd,
            [
                (
                    amplitude,
                    frequency,
                    phase,
                )
            ],
        )

        result["amplitude_n"] = amplitude
        result["frequency_hz"] = frequency
        result["phase_rad"] = phase

        rows.append(result)

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    output_dir = Path("results")

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 70)
    print("FAST PID / PD TUNING")
    print("=" * 70)

    print(
        f"Simulation duration: "
        f"{DURATION_S:.3f} s"
    )

    print(
        f"Steady-state start: "
        f"{STEADY_START_FRACTION * 100:.0f}%"
    )

    print(
        f"Training conditions: "
        f"{len(TRAINING_CONDITIONS)}"
    )

    print(
        f"Validation conditions: "
        f"{len(VALIDATION_CONDITIONS)}"
    )

    print()

    # ========================================================
    # STAGE 1
    # ========================================================

    stage1_df = evaluate_grid(
        KP_STAGE1,
        KD_STAGE1,
        TRAINING_CONDITIONS,
        "STAGE 1 — COARSE SEARCH",
    )

    stage1_file = (
        output_dir
        / "pid_tuning_stage1_results.csv"
    )

    stage1_df.to_csv(
        stage1_file,
        index=False,
    )

    print()
    print("Best Stage-1 candidates:")
    print(
        stage1_df
        .head(10)
        .to_string(index=False)
    )

    best_stage1 = stage1_df.iloc[0]

    best_kp = float(
        best_stage1["kp"]
    )

    best_kd = float(
        best_stage1["kd"]
    )

    print()
    print("Stage-1 best gains:")
    print(
        f"Kp = {best_kp:g}"
    )
    print(
        "Ki = 0"
    )
    print(
        f"Kd = {best_kd:g}"
    )

    # ========================================================
    # STAGE 2
    # ========================================================

    fine_kp_values, fine_kd_values = (
        make_fine_grid(
            best_kp,
            best_kd,
        )
    )

    stage2_df = evaluate_grid(
        fine_kp_values,
        fine_kd_values,
        TRAINING_CONDITIONS,
        "STAGE 2 — FINE SEARCH",
    )

    stage2_file = (
        output_dir
        / "pid_tuning_fine_results.csv"
    )

    stage2_df.to_csv(
        stage2_file,
        index=False,
    )

    # ========================================================
    # FINAL SELECTED GAINS
    # ========================================================

    best = stage2_df.iloc[0]

    final_kp = float(
        best["kp"]
    )

    final_kd = float(
        best["kd"]
    )

    print()
    print("=" * 70)
    print("FINAL TUNING RESULT")
    print("=" * 70)

    print()
    print("Best gains:")
    print(
        f"Kp = {final_kp:g}"
    )
    print(
        "Ki = 0"
    )
    print(
        f"Kd = {final_kd:g}"
    )

    print()
    print(
        f"Training mean steady RMS: "
        f"{best['mean_steady_rms_um']:.4f} um"
    )

    print(
        f"Training mean peak displacement: "
        f"{best['mean_peak_um']:.4f} um"
    )

    print(
        f"Training mean RMS current: "
        f"{best['mean_rms_current_a']:.4f} A"
    )

    print(
        f"Training mean current-change RMS: "
        f"{best['mean_current_change_rms_a']:.4f} A"
    )

    print(
        f"Training mean saturation: "
        f"{100.0 * best['mean_saturation_fraction']:.2f}%"
    )

    print(
        f"Training objective score: "
        f"{best['score']:.6f}"
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    print()
    print("=" * 70)
    print("VALIDATION OF SELECTED GAINS")
    print("=" * 70)

    print(
        f"Kp = {final_kp:g}, "
        f"Kd = {final_kd:g}"
    )

    validation_df = validate_gains(
        final_kp,
        final_kd,
        VALIDATION_CONDITIONS,
    )

    validation_file = (
        output_dir
        / "pid_validation_conditions.csv"
    )

    validation_df.to_csv(
        validation_file,
        index=False,
    )

    print()

    print(
        validation_df[
            [
                "amplitude_n",
                "frequency_hz",
                "phase_rad",
                "mean_steady_rms_um",
                "mean_peak_um",
                "mean_rms_current_a",
                "mean_saturation_fraction",
            ]
        ].to_string(index=False)
    )

    print()

    print("Validation averages:")

    print(
        f"Mean RMS displacement: "
        f"{validation_df['mean_steady_rms_um'].mean():.4f} um"
    )

    print(
        f"Mean peak displacement: "
        f"{validation_df['mean_peak_um'].mean():.4f} um"
    )

    print(
        f"Mean RMS current: "
        f"{validation_df['mean_rms_current_a'].mean():.4f} A"
    )

    print(
        f"Mean saturation: "
        f"{100.0 * validation_df['mean_saturation_fraction'].mean():.2f}%"
    )

    # ========================================================
    # SAVE SELECTED GAINS
    # ========================================================

    gains = {
        "kp": final_kp,
        "ki": 0.0,
        "kd": final_kd,
        "selection_method":
            "two-stage coarse/fine search",
        "training_condition_count":
            len(TRAINING_CONDITIONS),
        "validation_condition_count":
            len(VALIDATION_CONDITIONS),
        "duration_s":
            DURATION_S,
        "steady_start_fraction":
            STEADY_START_FRACTION,
        "displacement_weight":
            DISPLACEMENT_WEIGHT,
        "current_weight":
            CURRENT_WEIGHT,
        "saturation_weight":
            SATURATION_WEIGHT,
    }

    json_file = (
        output_dir
        / "tuned_pid.json"
    )

    with json_file.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            gains,
            f,
            indent=2,
        )

    # ========================================================
    # FILE SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        f"Stage-1 results:\n"
        f"{stage1_file}"
    )

    print(
        f"Fine-search results:\n"
        f"{stage2_file}"
    )

    print(
        f"Validation results:\n"
        f"{validation_file}"
    )

    print(
        f"Selected gains:\n"
        f"{json_file}"
    )

    print()


if __name__ == "__main__":
    main()
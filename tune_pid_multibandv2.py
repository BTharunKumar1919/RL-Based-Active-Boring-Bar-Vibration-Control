"""Two-stage PID tuning for multiple boring-bar frequency ranges.

The tuner searches Kp, Ki and Kd for several frequency bands using the
same seven-mode current-driven plant and actuator model used by the
main project.

Important:
- The actuator interface is constrained to +/- 5 A.
- Candidate gains that require a raw PID command above +/- 5 A are
  treated as constraint-violating and ranked behind feasible gains.
- Every frequency test uses a fixed 0.5-second simulation, as requested.
- Stage 1 is deliberately coarse.
- Stage 2 is a small local refinement around the Stage-1 winner.
- Results are saved to CSV for every range plus one combined summary.

The requested bands are 0–50 Hz, 0–150 Hz, and 150–250 Hz. A 0-Hz
sinusoid is a static/DC load rather than a vibration frequency, so 0.83 Hz
is used as the lowest dynamic test point. Every test simulation is 0.5 s.
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

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STEADY_START_FRACTION = 0.20

# Physical/current constraint requested by the user.
MAX_COMMAND_CURRENT_A = 5.0
CURRENT_TOLERANCE_A = 1e-6

# Use the same nominal disturbance amplitudes as the SAC studies.
# Only the two extremes are used for tuning to keep the multi-band
# search computationally reasonable.
TUNING_AMPLITUDES_N = (
    0.5,
    1.5,
)

# A fixed phase is sufficient for a linear sinusoidal steady-state
# search and avoids multiplying the already large multi-band search.
# Phase effects are still checked later through validation.
TUNING_PHASE_RAD = 0.0


# ============================================================
# REFERENCE GAIN CENTRES
# ============================================================
#
# The supplied Method-B analytical values are used as the centre of
# the coarse grid where available.
#
# Kp centres supplied by the user from Ziegler–Nichols trial and error.
# Ki and Kd centres are retained from the closest existing bands.
# These centres define the coarse search region; they are not necessarily
# the final tuned gains.
#
# Units:
#   Kp : A/m
#   Ki : A/(m*s)
#   Kd : A*s/m
# ============================================================

GAIN_CENTRES = {
    "0_50": {
        # User-provided Ziegler–Nichols trial-and-error centre at 25 Hz.
        "kp": -200_000.0,
        "ki": 15_000.0,
        "kd": 40.0,
    },
    "0_150": {
        # User-provided Kp centre at 75 Hz.
        "kp": 100_000.0,
        "ki": 20_000.0,
        "kd": 45.0,
    },
    "150_250": {
        # User-provided Kp centre for the 150–250 Hz band.
        "kp": 180_000.0,
        "ki": 1_000.0,
        "kd": 73.2,
    },
}


# ============================================================
# FREQUENCY TEST POINTS
# ============================================================
#
# These are intentionally sparse. Modal frequencies of the actual
# seven-mode plant are explicitly included in the broad ranges so
# that a "generalized" PID is not tuned only on arbitrary points.
#
# 0 Hz itself is not simulated; see module docstring.
# ============================================================

FREQUENCY_RANGES = {
    "0_50": (
        0.83,
        10.0,
        25.0,
        40.0,
        50.0,
    ),
    "0_150": (
        0.83,
        25.0,
        50.0,
        75.0,
        100.0,
        125.0,
        150.0,
    ),
    "150_250": (
        150.0,
        175.0,
        200.0,
        225.0,
        250.0,
    ),
}


# ============================================================
# STAGE 1 — COARSE GRID
# ============================================================
#
# Four values for Kp and Kd, three for Ki.
# This is deliberately much smaller than a dense Cartesian grid.
#
# Multipliers are centred around the analytical/estimated gain scale.
# ============================================================

COARSE_KP_MULTIPLIERS = (
    0.50,
    1.00,
    1.50,
)

COARSE_KI_MULTIPLIERS = (
    0.25,
    1.00,
    1.75,
)

COARSE_KD_MULTIPLIERS = (
    0.50,
    1.00,
    2.00,
)


# ============================================================
# STAGE 2 — FINE GRID
# ============================================================
#
# Only a small local refinement around the Stage-1 winner.
# 3 x 3 x 3 = 27 candidates per frequency range.
# ============================================================

FINE_KP_MULTIPLIERS = (
    0.85,
    1.00,
    1.15,
)

FINE_KI_MULTIPLIERS = (
    0.80,
    1.00,
    1.20,
)

FINE_KD_MULTIPLIERS = (
    0.85,
    1.00,
    1.15,
)


# ============================================================
# OBJECTIVE
# ============================================================
#
# Vibration is dominant.
# Current is a secondary objective.
# Saturation/constraint violation is strongly penalized.
#
# Feasible candidates (raw command never above +/-5 A) always rank
# ahead of infeasible candidates.
# ============================================================

DISPLACEMENT_WEIGHT = 1.0
CURRENT_WEIGHT = 0.05
SATURATION_WEIGHT = 5.0
COMMAND_VIOLATION_WEIGHT = 100.0


# ============================================================
# SIMULATION DURATION
# ============================================================

def simulation_duration_s(frequency_hz, range_name=None):
    """Use a fixed 0.5-second simulation for every frequency and band."""
    return 0.5


# ============================================================
# PID INTEGRAL LIMIT
# ============================================================

def integral_limit_for_ki(ki):
    """Limit the integral contribution to approximately +/-5 A.

    The exact total PID command can still be larger because Kp and Kd
    also contribute. Those raw commands are checked separately.
    """

    if abs(ki) < 1e-12:
        return None

    return MAX_COMMAND_CURRENT_A / abs(ki)


# ============================================================
# CANDIDATE GRID GENERATION
# ============================================================

def make_coarse_grid(centre):
    """Create the Stage-1 grid around a reference gain centre."""

    kp_values = [
        centre["kp"] * m
        for m in COARSE_KP_MULTIPLIERS
    ]

    ki_values = [
        centre["ki"] * m
        for m in COARSE_KI_MULTIPLIERS
    ]

    kd_values = [
        centre["kd"] * m
        for m in COARSE_KD_MULTIPLIERS
    ]

    return (
        sorted(set(round(x, 6) for x in kp_values)),
        sorted(set(round(x, 6) for x in ki_values)),
        sorted(set(round(x, 6) for x in kd_values)),
    )


def make_fine_grid(best_kp, best_ki, best_kd):
    """Create the Stage-2 local refinement grid."""

    kp_values = [
        best_kp * m
        for m in FINE_KP_MULTIPLIERS
    ]

    ki_values = [
        best_ki * m
        for m in FINE_KI_MULTIPLIERS
    ]

    kd_values = [
        best_kd * m
        for m in FINE_KD_MULTIPLIERS
    ]

    return (
        sorted(set(round(x, 6) for x in kp_values)),
        sorted(set(round(x, 6) for x in ki_values)),
        sorted(set(round(x, 6) for x in kd_values)),
    )


# ============================================================
# SINGLE GAIN EVALUATION
# ============================================================

def evaluate_gains(kp, ki, kd, frequencies, range_name):
    """Evaluate one PID controller over one frequency range."""

    rms_displacement_um = []
    peak_displacement_um = []
    rms_current_a = []
    max_actual_current_a = []
    max_command_current_a = []
    saturation_fraction = []

    for frequency in frequencies:

        duration_s = simulation_duration_s(frequency, range_name)

        n_steps = int(
            round(duration_s / PLANT.dt_s)
        )

        steady_start = int(
            STEADY_START_FRACTION * n_steps
        )

        for amplitude in TUNING_AMPLITUDES_N:

            plant = SevenModePlant(PLANT)
            plant.reset()

            disturbance = SinusoidalCuttingForce(
                DisturbanceConfig(
                    amplitude_n=amplitude,
                    frequency_hz=frequency,
                    phase_rad=TUNING_PHASE_RAD,
                    noise_std_n=0.0,
                )
            )

            controller = PIDController(
                kp=float(kp),
                ki=float(ki),
                kd=float(kd),
                plant_config=PLANT,
                integral_limit_m_s=integral_limit_for_ki(ki),
            )

            controller.reset()

            displacement = np.empty(n_steps)
            actual_current = np.empty(n_steps)
            raw_command = np.empty(n_steps)

            for i in range(n_steps):

                state = plant.state

                command_current_a = float(
                    controller.act(
                        state.displacement_m,
                        state.velocity_m_s,
                        PLANT.dt_s,
                    )
                )

                raw_command[i] = command_current_a

                # Explicit +/-5 A command constraint.
                applied_command_a = float(
                    np.clip(
                        command_current_a,
                        -MAX_COMMAND_CURRENT_A,
                        MAX_COMMAND_CURRENT_A,
                    )
                )

                disturbance_force_n = disturbance(
                    i * PLANT.dt_s
                )

                new_state = plant.step(
                    disturbance_n=disturbance_force_n,
                    current_a=applied_command_a,
                )

                displacement[i] = new_state.displacement_m
                actual_current[i] = plant.actual_current_a

            x = displacement[steady_start:]
            u = actual_current[steady_start:]
            cmd = raw_command[steady_start:]

            x_um = x * 1e6

            rms_x = float(
                np.sqrt(np.mean(x_um ** 2))
            )

            peak_x = float(
                np.max(np.abs(x_um))
            )

            rms_u = float(
                np.sqrt(np.mean(u ** 2))
            )

            max_actual_u = float(
                np.max(np.abs(u))
            )

            max_command_u = float(
                np.max(np.abs(cmd))
            )

            sat = float(
                np.mean(
                    np.abs(u)
                    >=
                    0.99
                    * MAX_COMMAND_CURRENT_A
                )
            )

            rms_displacement_um.append(rms_x)
            peak_displacement_um.append(peak_x)
            rms_current_a.append(rms_u)
            max_actual_current_a.append(max_actual_u)
            max_command_current_a.append(max_command_u)
            saturation_fraction.append(sat)

    mean_rms_um = float(
        np.mean(rms_displacement_um)
    )

    mean_peak_um = float(
        np.mean(peak_displacement_um)
    )

    mean_current_a = float(
        np.mean(rms_current_a)
    )

    worst_actual_current_a = float(
        np.max(max_actual_current_a)
    )

    worst_command_current_a = float(
        np.max(max_command_current_a)
    )

    mean_saturation = float(
        np.mean(saturation_fraction)
    )

    # Same displacement scale used by SAC.
    normalized_displacement = (
        mean_rms_um * 1e-6
        / TRAINING.displacement_scale_m
    )

    normalized_current = (
        mean_current_a
        / MAX_COMMAND_CURRENT_A
    )

    command_excess_a = max(
        0.0,
        worst_command_current_a
        - MAX_COMMAND_CURRENT_A,
    )

    normalized_command_excess = (
        command_excess_a
        / MAX_COMMAND_CURRENT_A
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

        +

        COMMAND_VIOLATION_WEIGHT
        * normalized_command_excess ** 2
    )

    feasible = (
        worst_command_current_a
        <=
        MAX_COMMAND_CURRENT_A
        + CURRENT_TOLERANCE_A
    )

    return {
        "kp": float(kp),
        "ki": float(ki),
        "kd": float(kd),

        "mean_steady_rms_um": mean_rms_um,
        "mean_peak_um": mean_peak_um,
        "mean_rms_current_a": mean_current_a,

        "worst_actual_current_a":
            worst_actual_current_a,

        "worst_raw_command_current_a":
            worst_command_current_a,

        "mean_saturation_fraction":
            mean_saturation,

        "command_excess_a":
            command_excess_a,

        "constraint_feasible":
            bool(feasible),

        "score":
            float(score),
    }


# ============================================================
# GRID SEARCH
# ============================================================

def sort_results(df):
    """Rank feasible gains before infeasible gains."""

    return (
        df.sort_values(
            by=[
                "constraint_feasible",
                "score",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )


def evaluate_grid(
    kp_values,
    ki_values,
    kd_values,
    frequencies,
    stage_name,
    range_name,
):
    """Evaluate a Cartesian gain grid."""

    gain_pairs = list(
        itertools.product(
            kp_values,
            ki_values,
            kd_values,
        )
    )

    total = len(gain_pairs)

    print()
    print("=" * 78)
    print(f"{range_name} — {stage_name}")
    print("=" * 78)

    print(
        f"Kp candidates: {len(kp_values)}"
    )

    print(
        f"Ki candidates: {len(ki_values)}"
    )

    print(
        f"Kd candidates: {len(kd_values)}"
    )

    print(
        f"Frequency points: {len(frequencies)}"
    )

    print(
        f"Amplitudes: {len(TUNING_AMPLITUDES_N)}"
    )

    print(
        f"Total gain combinations: {total}"
    )

    results = []

    for index, (kp, ki, kd) in enumerate(
        gain_pairs,
        start=1,
    ):

        result = evaluate_gains(
            kp,
            ki,
            kd,
            frequencies,
            range_name,
        )

        result["range"] = range_name
        result["stage"] = stage_name

        results.append(result)

        print(
            f"{range_name}: "
            f"{stage_name}: "
            f"{index}/{total}",
            flush=True,
        )

    return sort_results(
        pd.DataFrame(results)
    )


# ============================================================
# FINAL VALIDATION
# ============================================================

def validation_frequencies(frequencies):
    """Generate a small independent set of validation frequencies."""

    frequencies = sorted(set(float(f) for f in frequencies))

    if len(frequencies) <= 3:
        return frequencies

    validation = []

    # Use endpoints.
    validation.append(frequencies[0])
    validation.append(frequencies[-1])

    # Add interior midpoints.
    for left, right in zip(
        frequencies[:-1],
        frequencies[1:],
    ):
        midpoint = 0.5 * (left + right)
        validation.append(midpoint)

    # Keep the validation set compact.
    validation = sorted(
        set(round(f, 4) for f in validation)
    )

    if len(validation) > 7:
        indices = np.linspace(
            0,
            len(validation) - 1,
            7,
            dtype=int,
        )
        validation = [
            validation[i]
            for i in sorted(set(indices))
        ]

    return validation


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 78)
    print("MULTI-BAND PID TUNING")
    print("=" * 78)

    print()
    print("Current constraint:")
    print(
        f"Raw PID command <= +/-{MAX_COMMAND_CURRENT_A:.1f} A"
    )

    print()
    print("Tuning amplitudes:")
    print(TUNING_AMPLITUDES_N)

    print()
    print(
        "Stage 1: "
        f"{len(COARSE_KP_MULTIPLIERS)} x "
        f"{len(COARSE_KI_MULTIPLIERS)} x "
        f"{len(COARSE_KD_MULTIPLIERS)} = "
        f"{len(COARSE_KP_MULTIPLIERS) * len(COARSE_KI_MULTIPLIERS) * len(COARSE_KD_MULTIPLIERS)} "
        "gain combinations per range."
    )

    print(
        "Stage 2: "
        f"{len(FINE_KP_MULTIPLIERS)} x "
        f"{len(FINE_KI_MULTIPLIERS)} x "
        f"{len(FINE_KD_MULTIPLIERS)} = "
        f"{len(FINE_KP_MULTIPLIERS) * len(FINE_KI_MULTIPLIERS) * len(FINE_KD_MULTIPLIERS)} "
        "gain combinations per range."
    )

    all_stage_results = []
    summary_rows = []
    validation_rows = []

    for range_name, frequencies in FREQUENCY_RANGES.items():

        centre = GAIN_CENTRES[range_name]

        # ----------------------------------------------------
        # Stage 1
        # ----------------------------------------------------

        coarse_kp, coarse_ki, coarse_kd = (
            make_coarse_grid(centre)
        )

        stage1_df = evaluate_grid(
            coarse_kp,
            coarse_ki,
            coarse_kd,
            frequencies,
            "STAGE 1 — COARSE",
            range_name,
        )

        stage1_file = (
            OUTPUT_DIR
            / f"pid_{range_name}_stage1_coarse.csv"
        )

        stage1_df.to_csv(
            stage1_file,
            index=False,
        )

        best_stage1 = stage1_df.iloc[0]

        best_kp_stage1 = float(
            best_stage1["kp"]
        )

        best_ki_stage1 = float(
            best_stage1["ki"]
        )

        best_kd_stage1 = float(
            best_stage1["kd"]
        )

        print()
        print(
            f"{range_name} Stage-1 best:"
        )

        print(
            f"Kp = {best_kp_stage1:g} A/m"
        )

        print(
            f"Ki = {best_ki_stage1:g} A/(m*s)"
        )

        print(
            f"Kd = {best_kd_stage1:g} A*s/m"
        )

        print(
            f"Feasible = "
            f"{bool(best_stage1['constraint_feasible'])}"
        )

        # ----------------------------------------------------
        # Stage 2
        # ----------------------------------------------------

        fine_kp, fine_ki, fine_kd = (
            make_fine_grid(
                best_kp_stage1,
                best_ki_stage1,
                best_kd_stage1,
            )
        )

        stage2_df = evaluate_grid(
            fine_kp,
            fine_ki,
            fine_kd,
            frequencies,
            "STAGE 2 — FINE",
            range_name,
        )

        stage2_file = (
            OUTPUT_DIR
            / f"pid_{range_name}_stage2_fine.csv"
        )

        stage2_df.to_csv(
            stage2_file,
            index=False,
        )

        best = stage2_df.iloc[0]

        final_kp = float(best["kp"])
        final_ki = float(best["ki"])
        final_kd = float(best["kd"])

        # ----------------------------------------------------
        # Store all search results.
        # ----------------------------------------------------

        all_stage_results.extend(
            stage1_df.to_dict("records")
        )

        all_stage_results.extend(
            stage2_df.to_dict("records")
        )

        # ----------------------------------------------------
        # Validation.
        # ----------------------------------------------------

        validation_freqs = validation_frequencies(
            frequencies
        )

        validation_result = evaluate_gains(
            final_kp,
            final_ki,
            final_kd,
            validation_freqs,
            range_name,
        )

        validation_result["range"] = range_name
        validation_result["validation_frequency_points"] = (
            ", ".join(
                f"{f:g}"
                for f in validation_freqs
            )
        )

        validation_rows.append(
            validation_result
        )

        # ----------------------------------------------------
        # Summary.
        # ----------------------------------------------------

        summary_rows.append(
            {
                "frequency_range": range_name,

                "Kp_A_per_m":
                    final_kp,

                "Ki_A_per_m_s":
                    final_ki,

                "Kd_A_s_per_m":
                    final_kd,

                "mean_steady_rms_um":
                    float(
                        best[
                            "mean_steady_rms_um"
                        ]
                    ),

                "mean_peak_um":
                    float(
                        best["mean_peak_um"]
                    ),

                "mean_rms_current_A":
                    float(
                        best[
                            "mean_rms_current_a"
                        ]
                    ),

                "worst_actual_current_A":
                    float(
                        best[
                            "worst_actual_current_a"
                        ]
                    ),

                "worst_raw_command_current_A":
                    float(
                        best[
                            "worst_raw_command_current_a"
                        ]
                    ),

                "mean_saturation_fraction":
                    float(
                        best[
                            "mean_saturation_fraction"
                        ]
                    ),

                "command_excess_A":
                    float(
                        best["command_excess_a"]
                    ),

                "constraint_feasible":
                    bool(
                        best[
                            "constraint_feasible"
                        ]
                    ),

                "objective_score":
                    float(
                        best["score"]
                    ),
            }
        )

        print()
        print("-" * 78)
        print(
            f"FINAL {range_name}"
        )
        print("-" * 78)

        print(
            f"Kp = {final_kp:g} A/m"
        )

        print(
            f"Ki = {final_ki:g} A/(m*s)"
        )

        print(
            f"Kd = {final_kd:g} A*s/m"
        )

        print(
            f"Mean steady RMS = "
            f"{best['mean_steady_rms_um']:.5f} um"
        )

        print(
            f"Mean RMS current = "
            f"{best['mean_rms_current_a']:.5f} A"
        )

        print(
            f"Worst raw command = "
            f"{best['worst_raw_command_current_a']:.5f} A"
        )

        print(
            f"Constraint feasible = "
            f"{bool(best['constraint_feasible'])}"
        )

    # ========================================================
    # SAVE COMBINED RESULTS
    # ========================================================

    all_results_df = pd.DataFrame(
        all_stage_results
    )

    all_results_file = (
        OUTPUT_DIR
        / "pid_tuning_all_ranges_all_candidates.csv"
    )

    all_results_df.to_csv(
        all_results_file,
        index=False,
    )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_file = (
        OUTPUT_DIR
        / "pid_tuned_gains_by_frequency_range.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False,
    )

    validation_df = pd.DataFrame(
        validation_rows
    )

    validation_file = (
        OUTPUT_DIR
        / "pid_tuned_gains_validation.csv"
    )

    validation_df.to_csv(
        validation_file,
        index=False,
    )

    # ========================================================
    # SAVE JSON
    # ========================================================

    json_data = {
        "current_constraint_A": MAX_COMMAND_CURRENT_A,
        "steady_start_fraction":
            STEADY_START_FRACTION,

        "tuning_amplitudes_N":
            list(TUNING_AMPLITUDES_N),

        "phase_rad":
            TUNING_PHASE_RAD,

        "coarse_grid": {
            "kp_multipliers":
                list(COARSE_KP_MULTIPLIERS),
            "ki_multipliers":
                list(COARSE_KI_MULTIPLIERS),
            "kd_multipliers":
                list(COARSE_KD_MULTIPLIERS),
        },

        "fine_grid": {
            "kp_multipliers":
                list(FINE_KP_MULTIPLIERS),
            "ki_multipliers":
                list(FINE_KI_MULTIPLIERS),
            "kd_multipliers":
                list(FINE_KD_MULTIPLIERS),
        },

        "objective": {
            "displacement_weight":
                DISPLACEMENT_WEIGHT,
            "current_weight":
                CURRENT_WEIGHT,
            "saturation_weight":
                SATURATION_WEIGHT,
            "command_violation_weight":
                COMMAND_VIOLATION_WEIGHT,
        },

        "gains_by_frequency_range":
            summary_rows,
    }

    json_file = (
        OUTPUT_DIR
        / "pid_tuned_gains_by_frequency_range.json"
    )

    with json_file.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            json_data,
            f,
            indent=2,
        )

    # ========================================================
    # FINAL TABLE
    # ========================================================

    print()
    print("=" * 100)
    print("FINAL PID GAINS BY FREQUENCY RANGE")
    print("=" * 100)

    display_columns = [
        "frequency_range",
        "Kp_A_per_m",
        "Ki_A_per_m_s",
        "Kd_A_s_per_m",
        "mean_steady_rms_um",
        "mean_rms_current_A",
        "worst_raw_command_current_A",
        "constraint_feasible",
    ]

    print(
        summary_df[
            display_columns
        ].to_string(index=False)
    )

    print()
    print("=" * 78)
    print("FILES SAVED")
    print("=" * 78)

    print(
        f"All candidates:\n"
        f"{all_results_file}"
    )

    print(
        f"Final gain table:\n"
        f"{summary_file}"
    )

    print(
        f"Validation:\n"
        f"{validation_file}"
    )

    print(
        f"JSON:\n"
        f"{json_file}"
    )

    print()


if __name__ == "__main__":
    main()

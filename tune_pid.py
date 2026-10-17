
"""Grid-search PID gains across several simulated cutting conditions."""

from pathlib import Path
import math

import numpy as np
import pandas as pd

from config import PLANT
from controllers import PIDController
from models import SingleModePlant, SinusoidalCuttingForce
from config import DisturbanceConfig


# Search ranges. These are candidate values, not assumed optimal gains.
KP_VALUES = [0, 2500, 5000, 8000, 12000, 18000, 25000, 35000, 50000]
KD_VALUES = [0, 1, 2, 3, 5, 7, 10, 14, 20]

DURATION_S = 0.2
STEADY_START_FRACTION = 0.2


def evaluate_gains(kp, kd, conditions):
    """Return average vibration and effort across the test conditions."""
    rms_values_um = []
    rms_forces_n = []
    saturation_fractions = []

    for amplitude, frequency, phase in conditions:
        plant = SingleModePlant(PLANT)
        plant.reset()

        disturbance = SinusoidalCuttingForce(
            DisturbanceConfig(
                amplitude_n=amplitude,
                frequency_hz=frequency,
                phase_rad=phase,
            )
        )

        pid = PIDController(
            kp=kp,
            ki=0.0,
            kd=kd,
            plant_config=PLANT,
        )
        pid.reset()

        n_steps = round(DURATION_S / PLANT.dt_s)
        start = int(STEADY_START_FRACTION * n_steps)

        displacement = np.zeros(n_steps)
        control_force = np.zeros(n_steps)

        for i in range(n_steps):
            state = plant.state
            force = pid.act(
                state.displacement_m,
                state.velocity_m_s,
                PLANT.dt_s,
            )
            disturbance_force = disturbance(i * PLANT.dt_s)

            new_state = plant.step(disturbance_force, force)
            displacement[i] = new_state.displacement_m
            control_force[i] = force

        displacement_um = displacement[start:] * 1e6
        force_values = control_force[start:]

        rms_values_um.append(
            float(np.sqrt(np.mean(displacement_um**2)))
        )
        rms_forces_n.append(
            float(np.sqrt(np.mean(force_values**2)))
        )
        saturation_fractions.append(
            float(
                np.mean(
                    np.abs(force_values)
                    >= 0.99 * PLANT.max_control_force_n
                )
            )
        )

    mean_rms_um = float(np.mean(rms_values_um))
    mean_force_n = float(np.mean(rms_forces_n))
    mean_saturation = float(np.mean(saturation_fractions))

    # Lower is better. Strongly penalize actuator saturation.
    score = (
        mean_rms_um
        + 2.0 * mean_force_n
        + 50.0 * mean_saturation
    )

    return {
        "kp": kp,
        "ki": 0.0,
        "kd": kd,
        "mean_steady_rms_um": mean_rms_um,
        "mean_rms_force_n": mean_force_n,
        "mean_saturation_fraction": mean_saturation,
        "score": score,
    }


def main():
    # 18 conditions covering the assumed training frequency range.
    conditions = [
        (amplitude, frequency, phase)
        for amplitude in (0.5, 1.0, 1.5)
        for frequency in (150.0, 174.0, 200.0)
        for phase in (0.0, math.pi / 2)
    ]

    results = []

    total = len(KP_VALUES) * len(KD_VALUES)
    completed = 0

    for kp in KP_VALUES:
        for kd in KD_VALUES:
            results.append(evaluate_gains(kp, kd, conditions))
            completed += 1

        print(f"Tested {completed}/{total} gain combinations.")

    results_df = pd.DataFrame(results).sort_values("score")
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "pid_tuning_results.csv"
    results_df.to_csv(output_path, index=False)

    print("\nBest 10 PID settings:")
    print(results_df.head(10).to_string(index=False))
    print(f"\nFull results saved to: {output_path}")
    print("\nBest candidate gains:")
    print(f"kp = {results_df.iloc[0]['kp']:g}")
    print("ki = 0")
    print(f"kd = {results_df.iloc[0]['kd']:g}")


if __name__ == "__main__":
    main()
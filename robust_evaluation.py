"""Standalone robust evaluation using the same path as the App/API."""

from pathlib import Path

import pandas as pd

from deployment_service import (
    DEFAULT_MODEL_PATH,
    ROBUST_AMPLITUDES_N,
    ROBUST_FREQUENCIES_HZ,
    ROBUST_PHASES_RAD,
    ROBUST_DEFAULT_DURATION_S,
    run_robust_evaluation,
)


OUTPUT_DIR = Path("results")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result = run_robust_evaluation(
        model_path=DEFAULT_MODEL_PATH,
        amplitudes_n=ROBUST_AMPLITUDES_N,
        frequencies_hz=ROBUST_FREQUENCIES_HZ,
        phases_rad=ROBUST_PHASES_RAD,
        duration_s=ROBUST_DEFAULT_DURATION_S,
    )

    cases = pd.DataFrame(result["cases"])
    summary = pd.DataFrame(result["summary"])

    cases_path = OUTPUT_DIR / "robust_evaluation_cases.csv"
    summary_path = OUTPUT_DIR / "robust_evaluation_summary.csv"

    cases.to_csv(cases_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"Saved cases:   {cases_path.resolve()}")
    print(f"Saved summary: {summary_path.resolve()}")
    print("\nSummary:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6g}"))


if __name__ == "__main__":
    main()

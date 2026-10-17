
"""Train SAC and run the controller evaluation suite."""
import argparse

from sac_training import train_sac
from experiments import run_experiments


def main():
    parser = argparse.ArgumentParser(
        description="Train SAC and compare vibration-control baselines."
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=300_000,
        help="SAC training timesteps. Use a small value for a smoke test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for SAC training.",
    )
    parser.add_argument(
        "--model-path",
        default="results/sac_vibration_model",
        help="Output model path without the .zip suffix.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Evaluate an already-trained model instead of training.",
    )
    args = parser.parse_args()

    if args.timesteps <= 0:
        parser.error("--timesteps must be positive.")

    if not args.skip_training:
        train_sac(
            total_timesteps=args.timesteps,
            seed=args.seed,
            model_path=args.model_path,
        )

    model_file = args.model_path
    if not model_file.lower().endswith(".zip"):
        model_file += ".zip"

    run_experiments(model_path=model_file, output_dir="results")


if __name__ == "__main__":
    main()

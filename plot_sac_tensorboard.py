from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    raise SystemExit(
        "TensorBoard is required. Install it with:\n"
        "    pip install tensorboard pandas matplotlib"
    )


DEFAULT_MODELS = [
    "SAC_generalized_plant_1M_noNoise_incrDispReward.zip",
    "SAC_specified_plant_1M_0_50Hz.zip",
    "SAC_specified_plant_1M.zip",
    "SAC_generalized_plant_1M_0_600Hz.zip",
]


# Uniform naming schema for plot legends
MODEL_DISPLAY_NAMES = {
    "SAC_generalized_plant_1M_noNoise_incrDispReward": "Generalized (150–200 Hz)",
    "SAC_specified_plant_1M_0_50Hz": "Generalized (0–50 Hz)",
    "SAC_specified_plant_1M": "Specified (174 Hz)",
    "SAC_generalized_plant_1M_0_600Hz": "Generalized (0–600 Hz)",
}


METRIC_CANDIDATES = {
    "episode_reward": [
        "rollout/ep_rew_mean",
    ],
    "critic_loss": [
        "train/critic_loss",
    ],
    "actor_loss": [
        "train/actor_loss",
    ],
    "entropy_coefficient": [
        "train/ent_coef",
    ],
    "entropy_coefficient_loss": [
        "train/ent_coef_loss",
    ],
    "learning_rate": [
        "train/learning_rate",
    ],
}


def find_event_file(run_dir: Path) -> Path:
    """Find the TensorBoard event file inside a SAC_xx directory."""
    event_files = sorted(run_dir.glob("events.out.tfevents.*"))

    if not event_files:
        raise FileNotFoundError(
            f"No TensorBoard event file found in:\n{run_dir}"
        )

    if len(event_files) > 1:
        print(
            f"WARNING: {len(event_files)} event files found in {run_dir}. "
            f"Using {event_files[0].name}"
        )

    return event_files[0]


def load_scalars(event_file: Path) -> dict[str, pd.DataFrame]:
    """Read every scalar tag from a TensorBoard event file."""
    accumulator = EventAccumulator(
        str(event_file),
        size_guidance={"scalars": 0},
    )
    accumulator.Reload()
    scalar_tags = accumulator.Tags().get("scalars", [])

    data = {}
    for tag in scalar_tags:
        events = accumulator.Scalars(tag)
        data[tag] = pd.DataFrame(
            {
                "step": [event.step for event in events],
                "wall_time": [event.wall_time for event in events],
                "value": [event.value for event in events],
            }
        )

    return data


def choose_tag(
    all_tags: list[str],
    candidates: list[str],
) -> str | None:
    """Find the desired TensorBoard tag."""
    for candidate in candidates:
        if candidate in all_tags:
            return candidate

    if "critic_loss" in candidates[0]:
        keywords = ["critic_loss"]
    elif "actor_loss" in candidates[0]:
        keywords = ["actor_loss"]
    elif "ep_rew_mean" in candidates[0]:
        keywords = ["ep_rew_mean", "episode_reward"]
    elif "ent_coef_loss" in candidates[0]:
        keywords = ["ent_coef_loss"]
    elif "ent_coef" in candidates[0]:
        keywords = ["ent_coef"]
    elif "learning_rate" in candidates[0]:
        keywords = ["learning_rate"]
    else:
        keywords = []

    for tag in all_tags:
        tag_lower = tag.lower()
        if any(keyword in tag_lower for keyword in keywords):
            return tag

    return None


def safe_filename(name: str) -> str:
    """Make a Windows-safe filename."""
    invalid = '<>:"/\\|?*'
    for character in invalid:
        name = name.replace(character, "_")
    return name


def get_display_name(model_stem: str) -> str:
    """Get the uniform display name for a model stem."""
    return MODEL_DISPLAY_NAMES.get(model_stem, model_stem)


def save_metric_csv(
    df: pd.DataFrame,
    output_path: Path,
    model_filename: str,
    tensorboard_run: str,
    metric_name: str,
) -> None:
    output = df.copy()
    output.insert(0, "model_filename", model_filename)
    output.insert(1, "tensorboard_run", tensorboard_run)
    output.insert(2, "metric", metric_name)
    output.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )


def plot_curves(
    curves: list[tuple[str, pd.DataFrame]],
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    if not curves:
        print(f"No data available for: {title}")
        return

    plt.figure(figsize=(10, 5.5))

    for model_stem, df in curves:
        if df.empty:
            continue

        label = get_display_name(model_stem)

        plt.plot(
            df["step"],
            df["value"],
            label=label,
        )

    plt.xlabel("Training steps")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(
        fontsize=9,
        loc="best",
    )
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Saved plot: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and plot SAC TensorBoard curves."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results"),
        help="Path to the results folder.",
    )
    parser.add_argument(
        "--map",
        dest="map_path",
        type=Path,
        default=Path("sac_model_tensorboard_map.csv"),
        help="Path to the model-to-TensorBoard mapping CSV.",
    )
    parser.add_argument(
        "--out",
        dest="out_dir",
        type=Path,
        default=Path("sac_learning_curves"),
        help="Directory where CSVs and plots will be saved.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_MODELS,
        help="Model filenames to process.",
    )

    args = parser.parse_args()

    results_dir = args.results
    tensorboard_dir = results_dir / "tensorboard"

    if not results_dir.exists():
        raise FileNotFoundError(
            f"Results folder not found:\n{results_dir.resolve()}"
        )

    if not tensorboard_dir.exists():
        raise FileNotFoundError(
            f"TensorBoard folder not found:\n{tensorboard_dir.resolve()}"
        )

    if not args.map_path.exists():
        raise FileNotFoundError(
            f"Mapping CSV not found:\n{args.map_path.resolve()}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = args.out_dir / "csv"
    plot_dir = args.out_dir / "plots"
    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    mapping = pd.read_csv(args.map_path)
    required_columns = {"model_filename", "tensorboard_run"}
    missing_columns = required_columns - set(mapping.columns)

    if missing_columns:
        raise ValueError(
            "Mapping CSV is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    selected = mapping[mapping["model_filename"].isin(args.models)].copy()
    available_models = set(mapping["model_filename"].astype(str))
    missing_models = [m for m in args.models if m not in available_models]

    if missing_models:
        print("\nWARNING: These models are not present in the mapping CSV:")
        for model in missing_models:
            print(f"  - {model}")

    if selected.empty:
        raise ValueError(
            "None of the requested models were found in the mapping CSV."
        )

    metric_curves: dict[str, list[tuple[str, pd.DataFrame]]] = {
        metric: [] for metric in METRIC_CANDIDATES
    }

    for _, row in selected.iterrows():
        model = str(row["model_filename"])
        run = str(row["tensorboard_run"]).strip()

        print()
        print("=" * 70)
        print(f"Model:           {model}")
        print(f"TensorBoard run:  {run}")
        print("=" * 70)

        if not run or run.lower() == "nan":
            print("WARNING: No TensorBoard run is mapped to this model.")
            continue

        run_dir = tensorboard_dir / run
        if not run_dir.exists():
            print(f"WARNING: TensorBoard directory does not exist:\n  {run_dir.resolve()}")
            continue

        try:
            event_file = find_event_file(run_dir)
        except FileNotFoundError as error:
            print(f"WARNING: {error}")
            continue

        print(f"Event file:       {event_file.name}")

        try:
            scalar_data = load_scalars(event_file)
        except Exception as error:
            print(f"WARNING: Could not read TensorBoard data: {error}")
            continue

        all_tags = list(scalar_data.keys())
        print("\nAvailable scalar tags:")
        for tag in all_tags:
            print(f"  {tag}")

        if not all_tags:
            print("WARNING: No scalar data found.")
            continue

        all_scalar_frames = []
        for tag, df in scalar_data.items():
            temp = df.copy()
            temp.insert(0, "tag", tag)
            temp.insert(0, "tensorboard_run", run)
            temp.insert(0, "model_filename", model)
            all_scalar_frames.append(temp)

        model_stem = safe_filename(Path(model).stem)

        if all_scalar_frames:
            all_scalars = pd.concat(all_scalar_frames, ignore_index=True)
            all_scalars.to_csv(
                csv_dir / f"{model_stem}_all_scalars.csv",
                index=False,
                encoding="utf-8-sig",
            )

        for metric_name, candidates in METRIC_CANDIDATES.items():
            tag = choose_tag(all_tags, candidates)
            if tag is None:
                print(f"  {metric_name}: NOT AVAILABLE")
                continue

            df = scalar_data[tag].copy()
            print(f"  {metric_name}: {tag} ({len(df)} points)")

            metric_csv = csv_dir / f"{model_stem}_{metric_name}.csv"
            save_metric_csv(df, metric_csv, model, run, metric_name)

            metric_curves[metric_name].append((model_stem, df))

            ylabel = {
                "episode_reward": "Episode reward",
                "critic_loss": "Critic loss",
                "actor_loss": "Actor loss",
                "entropy_coefficient": "Entropy coefficient",
                "entropy_coefficient_loss": "Entropy coefficient loss",
                "learning_rate": "Learning rate",
            }[metric_name]

            individual_plot = plot_dir / f"{model_stem}_{metric_name}.png"
            plot_curves(
                [(model_stem, df)],
                f"{metric_name.replace('_', ' ').title()} — {get_display_name(model_stem)}",
                ylabel,
                individual_plot,
            )

    print()
    print("=" * 70)
    print("Creating combined plots")
    print("=" * 70)

    combined_names = {
        "episode_reward": "episode_reward_all_models.png",
        "critic_loss": "critic_loss_all_models.png",
        "actor_loss": "actor_loss_all_models.png",
        "entropy_coefficient": "entropy_coefficient_all_models.png",
        "entropy_coefficient_loss": "entropy_coefficient_loss_all_models.png",
        "learning_rate": "learning_rate_all_models.png",
    }

    ylabels = {
        "episode_reward": "Episode reward",
        "critic_loss": "Critic loss",
        "actor_loss": "Actor loss",
        "entropy_coefficient": "Entropy coefficient",
        "entropy_coefficient_loss": "Entropy coefficient loss",
        "learning_rate": "Learning rate",
    }

    for metric_name, curves in metric_curves.items():
        if not curves:
            continue

        plot_curves(
            curves,
            f"{metric_name.replace('_', ' ').title()} — Selected SAC Runs",
            ylabels[metric_name],
            plot_dir / combined_names[metric_name],
        )

    print()
    print("=" * 70)
    print("Finished.")
    print("=" * 70)
    print(f"\nCSV output:\n  {csv_dir.resolve()}")
    print(f"\nPlot output:\n  {plot_dir.resolve()}")


if __name__ == "__main__":
    main()
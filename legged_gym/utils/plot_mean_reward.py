#!/usr/bin/env python3
"""Plot the ``mean_reward`` scalar from TensorBoard event files."""
"""
"Usage:"
"python legged_gym/utils/plot_mean_reward.py \
  --log-dir logs/experiment_a \
  --log-dir logs/experiment_b \
  --title "Mean Reward Comparison" \
  --label "Initial State Method" \
   "
"""



import argparse
import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
sys.path = [
    path for path in sys.path if Path(path or ".").resolve() != script_dir
]

import matplotlib
import numpy as np

matplotlib.rcParams["toolbar"] = "toolbar2"
import matplotlib.pyplot as plt

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    event_accumulator = None


def find_event_files(paths):
    event_files = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if path.is_file() and path.name.startswith("events.out.tfevents"):
            event_files.append(path)
        elif path.is_dir():
            event_files.extend(sorted(path.rglob("events.out.tfevents*")))
        else:
            raise FileNotFoundError(f"No event file or directory found: {path}")
    return event_files


def load_scalar(event_file, tag):
    accumulator = event_accumulator.EventAccumulator(
        str(event_file), size_guidance={event_accumulator.SCALARS: 0}
    )
    accumulator.Reload()
    scalar_tags = accumulator.Tags().get("scalars", [])
    resolved_tag = tag
    if resolved_tag not in scalar_tags:
        suffix_matches = [
            scalar_tag for scalar_tag in scalar_tags if scalar_tag.rsplit("/", 1)[-1] == tag
        ]
        if len(suffix_matches) == 1:
            resolved_tag = suffix_matches[0]
        else:
            available_tags = ", ".join(scalar_tags) or "none"
            raise ValueError(
                f"Tag '{tag}' was not found in {event_file}. Available scalar tags: {available_tags}"
            )
    return accumulator.Scalars(resolved_tag)


def rolling_std(values, window):
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return np.zeros_like(values)

    window = max(1, min(window, values.size))
    if window % 2 == 0:
        window -= 1
    if window < 2:
        return np.zeros_like(values)

    pad = window // 2
    kernel = np.full(window, 1.0 / window)
    padded = np.pad(values, (pad, pad), mode="edge")
    mean = np.convolve(padded, kernel, mode="valid")
    mean_square = np.convolve(padded * padded, kernel, mode="valid")
    return np.sqrt(np.maximum(mean_square - mean * mean, 0.0))


def main():
    if event_accumulator is None:
        raise SystemExit(
            "TensorBoard is required to read event files. Install it with: pip install tensorboard"
        )

    parser = argparse.ArgumentParser(
        description="Read TensorBoard event files and plot the mean reward."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="TensorBoard event file(s) or directories containing them.",
    )
    parser.add_argument(
        "--log-dir",
        action="append",
        default=[],
        help="Log directory to search recursively. Repeat this option for multiple directories.",
    )
    parser.add_argument("--tag", default="mean_reward", help="Scalar tag to plot.")
    parser.add_argument("--title", help="Custom plot title. Defaults to the scalar tag.")
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="Custom legend label, matched to curves in event-file order.",
    )
    parser.add_argument(
        "--std-window",
        type=int,
        default=21,
        help="Odd-sized rolling window used to calculate the standard-deviation band.",
    )
    args = parser.parse_args()

    log_paths = args.paths + args.log_dir
    if not log_paths:
        parser.error("provide at least one event file, directory, or --log-dir")

    event_files = find_event_files(log_paths)
    if not event_files:
        raise FileNotFoundError("No events.out.tfevents files were found.")

    figure, axis = plt.subplots(figsize=(10, 5))
    plotted = 0
    for event_file in event_files:
        try:
            events = load_scalar(event_file, args.tag)
        except ValueError as error:
            print(f"Skipping {event_file}: {error}")
            continue

        events = [event for event in events if event.step > 0]
        if not events:
            print(f"Skipping {event_file}: no scalar data remains after step 0.")
            continue

        steps = np.asarray([event.step for event in events])
        rewards = np.asarray([event.value for event in events], dtype=float)
        reward_std = rolling_std(rewards, args.std_window)
        label = args.label[plotted] if plotted < len(args.label) else str(event_file.parent)
        line = axis.plot(steps, rewards, label=label, linewidth=3)[0]
        axis.fill_between(
            steps,
            rewards - reward_std,
            rewards + reward_std,
            color=line.get_color(),
            alpha=0.2,
        )
        plotted += 1

    if not plotted:
        raise RuntimeError(f"No '{args.tag}' scalar data was found.")

    axis.set_title(args.title or args.tag)
    axis.set_xlabel("Training iteration")
    axis.set_ylabel(args.tag)
    axis.grid(True, alpha=0.3)
    if plotted > 1 or args.label:
        axis.legend()
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

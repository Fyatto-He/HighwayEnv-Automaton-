"""Plot saved episode metrics without requiring TensorBoard."""
import argparse
import csv
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from baseline_waypoint.config import RESULTS_DIR, experiment_name


def plot_training(run_dir, window=100):
    run_dir = Path(run_dir)
    with (run_dir / "training_episodes.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return
    steps = [int(row["timesteps"]) for row in rows]

    def rolling(key):
        values = np.array([float(row[key]) for row in rows])
        cumulative = np.r_[0, values.cumsum()]
        ends = np.arange(1, len(values) + 1)
        starts = np.maximum(0, ends - window)
        return (cumulative[ends] - cumulative[starts]) / (ends - starts)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(steps, rolling("reward"), color="royalblue")
    axes[0].set_ylabel("Episode reward")
    axes[0].set_title(f"PPO training: rolling average of up to {window} episodes")
    axes[1].plot(steps, rolling("success"), label="Success", color="seagreen")
    axes[1].plot(steps, rolling("collision"), label="Collision", color="firebrick")
    axes[1].set(xlabel="Training transitions (all environments)", ylabel="Episode fraction", ylim=(-0.02, 1.02))
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(run_dir / "training_curves.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="ppo_seed7")
    args = parser.parse_args()
    plot_training(RESULTS_DIR / experiment_name(args.run_name))


if __name__ == "__main__":
    main()

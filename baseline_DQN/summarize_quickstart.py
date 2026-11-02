"""Compare all completed quickstart training seeds without selecting a best run."""
import argparse
import csv
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from baseline_DQN.config import RESULTS_DIR, QUICKSTART_REFERENCE, experiment_name


def rolling(values, window=100):
    values = np.asarray(values, dtype=float)
    totals = np.r_[0, values.cumsum()]
    ends = np.arange(1, len(values) + 1)
    starts = np.maximum(0, ends - window)
    return (totals[ends] - totals[starts]) / (ends - starts)


def read_rows(path):
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize(runs, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    summaries = []
    for name in runs:
        run_dir = RESULTS_DIR / experiment_name(name)
        metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        if metadata["preset"] != "quickstart" or metadata["status"] != "trained":
            raise ValueError(f"{name} is not a completed quickstart run.")
        evaluation = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
        episodes = read_rows(run_dir / "training_episodes.csv")
        progress = read_rows(run_dir / "progress.csv")
        steps = [int(row["timesteps"]) for row in episodes]
        label = f"Training seed {metadata['seed']}"
        axes[0].plot(steps, rolling([row["length"] for row in episodes]), label=label, linewidth=1.3)
        axes[1].plot(steps, rolling([row["reward"] for row in episodes]), linewidth=1.3)
        exploration = [row for row in progress if row.get("rollout/exploration_rate")]
        axes[2].plot([int(row["time/total_timesteps"]) for row in exploration],
                     [float(row["rollout/exploration_rate"]) for row in exploration], linewidth=1.3)
        summaries.append({"run": name, "training_seed": metadata["seed"],
                          "actual_timesteps": metadata["actual_timesteps"],
                          "training_seconds": metadata["training_seconds"],
                          "evaluation_first_seed": evaluation["first_seed"],
                          "dqn": evaluation["dqn"], "random": evaluation["random"],
                          "versions": metadata["versions"]})
    for axis, title, ylabel in zip(axes, ("Episode length", "Episode reward", "Exploration rate"),
                                  ("Policy decisions / simulated seconds", "Native episode return", "Random action probability")):
        axis.set(title=title, xlabel="Training transitions", ylabel=ylabel)
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Official highway-env DQN quickstart: independent training seeds\n"
                 "Length and reward: rolling means over up to 100 training episodes")
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=170)
    plt.close(fig)

    metrics = ("completion_rate", "collision_rate", "average_episode_reward", "average_episode_duration_seconds")
    aggregate = {policy: {key: {"mean": float(np.mean([row[policy][key] for row in summaries])),
                               "std_across_training_seeds": float(np.std([row[policy][key] for row in summaries],
                                                                         ddof=1 if len(summaries) > 1 else 0))}
                          for key in metrics} for policy in ("dqn", "random")}
    summary = {"reference": QUICKSTART_REFERENCE, "runs": summaries, "aggregate": aggregate,
               "interpretation": "Evaluation seeds are paired across policies and shared across training seeds. "
                                 "Uncertainty here is variation across training seeds, not independent traffic samples."}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# Official DQN quickstart reproduction", "",
             f"Reference: [highway-env quickstart]({QUICKSTART_REFERENCE}).", "",
             "Each run uses native `highway-fast-v0` traffic, observations, actions and reward, "
             "a `[256, 256]` MLP, target-network updates every 50 transitions, and one environment "
             "with one gradient update per transition after 200 warmup steps. The training budget "
             "is 20,000 transitions per seed. No slower traffic or forced nearby lead is injected.", "",
             "Training curves include exploration; evaluation uses deterministic actions on held-out "
             "traffic seeds starting at 10,000. Random actions use exactly the same initial traffic scenes.", "",
             "| Training seed | Steps | DQN completion | DQN collision | DQN return | Random completion | Random return |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summaries:
        dqn, random = row["dqn"], row["random"]
        lines.append(f"| {row['training_seed']} | {row['actual_timesteps']:,} | {dqn['completion_rate']:.0%} | "
                     f"{dqn['collision_rate']:.0%} | {dqn['average_episode_reward']:.2f} | "
                     f"{random['completion_rate']:.0%} | {random['average_episode_reward']:.2f} |")
    dqn, random = aggregate["dqn"], aggregate["random"]
    lines += ["", f"Across {len(summaries)} training seeds, mean DQN completion is "
              f"**{dqn['completion_rate']['mean']:.1%}** and mean return is "
              f"**{dqn['average_episode_reward']['mean']:.2f}**, versus "
              f"**{random['completion_rate']['mean']:.1%}** completion and "
              f"**{random['average_episode_reward']['mean']:.2f}** return for random actions.", "",
              f"Each policy has {summaries[0]['dqn']['episodes']} held-out episodes per training seed. "
              "These traffic seeds are shared across training seeds; the rows are not all independent "
              "traffic samples. Completion means surviving the 30-second episode without crashing. "
              "This is performance on the native task, separate from the earlier slower-traffic experiment.", "",
              "The website shows training curves and one demonstration, not a published held-out "
              "collision rate. These measurements reproduce the documented recipe and evaluate it "
              "locally; exact trajectories can vary with seeds and package versions.", "",
              "![All training seeds](training_curves.png)", "", "## Artifacts", ""]
    for row in summaries:
        name = row["run"]
        lines.append(f"- Seed {row['training_seed']}: [model](../../models/{name}.zip), "
                     f"[configuration](../../models/{name}.json), "
                     f"[evaluation](../{name}/evaluation.json), [episode rows](../{name}/evaluation_episodes.csv).")
    lines += ["", "## Videos", "",
              "The seed-0 recordings use the training seed chosen before evaluation and "
              "traffic seeds 10,000-10,002. Those episodes were not selected for their outcomes.", ""]
    best = max(summaries, key=lambda row: (row["dqn"]["completion_rate"], row["dqn"]["average_episode_reward"]))
    for row in summaries:
        video_folder = f"quickstart_seed{row['training_seed']}_videos"
        episodes_path = RESULTS_DIR / video_folder / "episodes.json"
        if not episodes_path.exists():
            continue
        if row["training_seed"] != 0 and row["run"] == best["run"]:
            lines += [f"Training seed {row['training_seed']} had the highest completion rate in the "
                      "50-episode evaluation. Its additional recordings use fresh traffic seeds "
                      "20,000-20,002. This model was selected after evaluation, so the five-seed "
                      "average above remains the primary result.", ""]
        for episode in json.loads(episodes_path.read_text(encoding="utf-8")):
            filename = f"episode_{episode['episode']:02d}.mp4"
            lines.append(f"- Training seed {row['training_seed']}, traffic seed {episode['seed']}: "
                         f"[video](../{video_folder}/{filename}), {episode['status']}, "
                         f"{episode['simulated_seconds']:.0f} simulated seconds.")
        lines.append("")
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", default=[f"dqn_quickstart_seed{seed}" for seed in range(5)])
    parser.add_argument("--output-name", default="quickstart_comparison")
    args = parser.parse_args()
    summarize(args.runs, RESULTS_DIR / experiment_name(args.output_name))


if __name__ == "__main__":
    main()

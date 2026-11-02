"""Train the official DQN quickstart or the historical slower-traffic experiment."""
import argparse
from collections import deque
import csv
from datetime import datetime, timezone
from functools import partial
import importlib.metadata
import json
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import torch

from baseline_DQN.config import (SEED, EVALUATION_EPISODES, MODELS_DIR, RESULTS_DIR,
                                 QUICKSTART_REFERENCE, experiment_name, training_settings)
from baseline_DQN.environment import TrafficHighwayEnv
from baseline_DQN.evaluate import compare


def make_training_env(seed, environment):
    """Create one independently seeded simulator inside its owning process."""
    env = Monitor(TrafficHighwayEnv(**environment))
    env.effective_env_config = dict(env.unwrapped.config)
    env.reset(seed=seed)
    return env


class EpisodeMetrics(BaseCallback):
    def __init__(self, run_dir, model_path, metadata):
        super().__init__()
        self.run_dir, self.model_path, self.metadata = run_dir, model_path, metadata
        self.recent = deque(maxlen=100)
        self.last_report = self.last_checkpoint = 0
        self.handle = None

    def _on_training_start(self):
        self.handle = (self.run_dir / "training_episodes.csv").open("w", newline="", encoding="utf-8")
        fields = ["timesteps", "reward", "length", "collision", "completed", "elapsed_time", "distance_travelled",
                  "encounter_count", "overtake_count", "lane_changes", "following_brake_steps",
                  "close_following_steps", "min_front_gap", "mean_speed"]
        self.writer = csv.DictWriter(self.handle, fieldnames=fields)
        self.writer.writeheader()

    def _on_step(self):
        for info in self.locals["infos"]:
            if "episode" not in info:
                continue
            row = {"timesteps": self.num_timesteps, "reward": info["episode"]["r"],
                   "length": info["episode"]["l"], "collision": int(info["is_collision"]),
                   "completed": int(info["is_completed"]), "elapsed_time": info["elapsed_time"],
                   "distance_travelled": info["distance_travelled"], **info["interaction_metrics"]}
            self.writer.writerow(row)
            self.handle.flush()
            self.recent.append(row)
            for key in ("reward", "collision", "completed", "overtake_count", "encounter_count", "mean_speed"):
                self.logger.record(f"episodes/{key}_mean_100", np.mean([r[key] for r in self.recent]))
        if self.num_timesteps - self.last_report >= 2_000:
            collisions = np.mean([r["collision"] for r in self.recent]) if self.recent else 0
            overtakes = np.mean([r["overtake_count"] for r in self.recent]) if self.recent else 0
            print(f"{self.num_timesteps:,} transitions; recent collisions={collisions:.1%}; overtakes/episode={overtakes:.2f}", flush=True)
            self.last_report = self.num_timesteps
        if self.num_timesteps - self.last_checkpoint >= 10_000:
            checkpoint = self.model_path.with_name(f"{self.model_path.stem}_{self.num_timesteps}.zip")
            self.model.save(checkpoint)
            snapshot = dict(self.metadata, actual_timesteps=self.num_timesteps, status="checkpoint")
            checkpoint.with_suffix(".json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
            self.last_checkpoint = self.num_timesteps
        return True

    def _on_training_end(self):
        if self.handle is not None:
            self.handle.close()


def plot_training(run_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with (run_dir / "training_episodes.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return
    steps = [int(row["timesteps"]) for row in rows]
    def rolling(key):
        values = np.array([float(row[key]) for row in rows])
        total = np.r_[0, values.cumsum()]
        ends = np.arange(1, len(values) + 1)
        starts = np.maximum(0, ends - 100)
        return (total[ends] - total[starts]) / (ends - starts)
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    axes[0].plot(steps, rolling("reward"), color="royalblue")
    axes[0].set_ylabel("Episode reward")
    axes[0].set_title("DQN: rolling averages over up to 100 episodes")
    axes[1].plot(steps, rolling("completed"), label="Time limit completed", color="seagreen")
    axes[1].plot(steps, rolling("collision"), label="Collision", color="firebrick")
    axes[1].set_ylabel("Episode fraction")
    axes[1].legend()
    axes[2].plot(steps, rolling("overtake_count"), label="Overtakes", color="purple")
    axes[2].plot(steps, rolling("encounter_count"), label="Close encounters", color="darkorange")
    axes[2].set(xlabel="Training transitions", ylabel="Per episode")
    axes[2].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(run_dir / "training_curves.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("quickstart", "slow-traffic"), default="quickstart")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=1,
                        help="1 matches the supplied single-env schedule; >1 uses spawned parallel simulators.")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--eval-episodes", type=int, default=EVALUATION_EPISODES)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-tensorboard", action="store_true")
    args = parser.parse_args()
    dqn_config, environment, default_timesteps = training_settings(args.preset)
    if args.seed is None:
        args.seed = 0 if args.preset == "quickstart" else SEED
    if args.timesteps is None:
        args.timesteps = default_timesteps
    if args.preset == "quickstart" and args.num_envs != 1:
        parser.error("Quickstart requires one simulator and one update per transition. Run separate seeds in parallel instead.")
    if args.timesteps <= 0 or args.num_envs <= 0 or args.eval_episodes < 0:
        parser.error("timesteps and num-envs must be positive; eval-episodes must be nonnegative.")
    name = experiment_name(args.run_name or f"dqn_{args.preset.replace('-', '_')}_seed{args.seed}")
    run_dir = RESULTS_DIR / name
    model_path = MODELS_DIR / f"{name}.zip"
    if model_path.exists() or (run_dir / "training_episodes.csv").exists():
        parser.error("Run already exists. Use a new --run-name to preserve its model and results.")
    run_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    set_random_seed(args.seed)
    factories = [partial(make_training_env, args.seed + index, environment) for index in range(args.num_envs)]
    env = DummyVecEnv(factories) if args.num_envs == 1 else SubprocVecEnv(factories, start_method="spawn")
    env.seed(args.seed)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "seed": args.seed,
        "requested_timesteps": args.timesteps, "num_envs": args.num_envs,
        "training_schedule": {
            "transitions_per_vector_step": args.num_envs,
            "gradient_updates_per_vector_step": dqn_config["gradient_steps"],
            "gradient_updates_per_transition_after_warmup": dqn_config["gradient_steps"] / args.num_envs,
        },
        "preset": args.preset, "reference": QUICKSTART_REFERENCE if args.preset == "quickstart" else None,
        "environment": environment,
        "effective_env_config": env.get_attr("effective_env_config")[0],
        "algorithm": "DQN", "policy": "MlpPolicy", "dqn_config": dqn_config,
        "versions": {package: importlib.metadata.version(package) for package in ("highway-env", "gymnasium", "stable-baselines3", "torch", "numpy")},
        "status": "running",
    }
    try:
        model = DQN("MlpPolicy", env, verbose=0 if args.quiet else 1, seed=args.seed, **dqn_config)
        metadata["sb3_defaults"] = {"exploration_fraction": model.exploration_fraction,
                                    "exploration_initial_eps": model.exploration_initial_eps,
                                    "exploration_final_eps": model.exploration_final_eps,
                                    "target_update_interval": model.target_update_interval,
                                    "net_arch": model.policy.net_arch}
        formats = ["csv"] + ([] if args.quiet else ["stdout"]) + ([] if args.no_tensorboard else ["tensorboard"])
        model.set_logger(configure(str(run_dir), formats))
        callback = EpisodeMetrics(run_dir, model_path, metadata)
        for path in (model_path.with_suffix(".json"), run_dir / "run.json"):
            path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Preset={args.preset}; DQN settings: {dqn_config}; num_envs={args.num_envs}", flush=True)
        print(f"One gradient update per vector step ({args.num_envs} collected transition(s)).", flush=True)
        started = time.perf_counter()
        status = "trained"
        try:
            model.learn(total_timesteps=args.timesteps, callback=callback)
        except KeyboardInterrupt:
            status = "interrupted"
            print("Interrupted; saving the current policy.", flush=True)
        except Exception:
            status = "failed"
            raise
        finally:
            try:
                model.save(model_path)
                metadata.update(status=status, actual_timesteps=model.num_timesteps,
                                training_seconds=time.perf_counter() - started)
                for path in (model_path.with_suffix(".json"), run_dir / "run.json"):
                    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            finally:
                if callback.handle is not None and not callback.handle.closed:
                    callback.handle.close()
    finally:
        # Close every worker even if model construction, learning or saving fails.
        env.close()
    plot_training(run_dir)
    print(f"Saved {model_path}; actual transitions={model.num_timesteps:,}.", flush=True)
    if args.eval_episodes and status == "trained":
        compare(model_path, run_dir, args.eval_episodes)


if __name__ == "__main__":
    main()

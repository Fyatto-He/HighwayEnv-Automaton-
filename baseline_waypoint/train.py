"""Train a small goal-conditioned PPO policy and evaluate it against random actions."""
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
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import torch

from baseline_waypoint.config import (ENV_CONFIG, GOAL_CONFIG, REWARD_WEIGHTS, PPO_CONFIG,
                             SEED, TRAINING_TIMESTEPS, NUM_ENVS, EVALUATION_EPISODES,
                             MODELS_DIR, RESULTS_DIR, experiment_name)
from baseline_waypoint.environment import GoalHighwayEnv
from baseline_waypoint.evaluate import compare
from baseline_waypoint.plot_results import plot_training


def make_training_env(seed):
    env = GoalHighwayEnv()
    env.reset(seed=seed)
    return Monitor(env)


class EpisodeMetrics(BaseCallback):
    """One readable CSV row per completed episode, plus rolling TensorBoard rates."""
    def __init__(self, run_dir):
        super().__init__()
        self.run_dir = run_dir
        self.recent = deque(maxlen=100)
        self.last_report = 0
        self.handle = None

    def _on_training_start(self):
        self.handle = (self.run_dir / "training_episodes.csv").open("w", newline="", encoding="utf-8")
        fields = ["timesteps", "reward", "length", "success", "collision", "timeout",
                  "elapsed_time", "final_distance", "goal_lane", "reward_goal",
                  "reward_collision", "reward_progress", "reward_time"]
        self.writer = csv.DictWriter(self.handle, fieldnames=fields)
        self.writer.writeheader()

    def _on_step(self):
        for info in self.locals["infos"]:
            if "episode" not in info:
                continue
            row = {
                "timesteps": self.num_timesteps, "reward": info["episode"]["r"],
                "length": info["episode"]["l"], "success": int(info["is_success"]),
                "collision": int(info["is_collision"]), "timeout": int(info["is_timeout"]),
                "elapsed_time": info["elapsed_time"], "final_distance": info["distance_to_goal"],
                "goal_lane": info["goal_lane"],
                **{f"reward_{name}": value for name, value in info["episode_reward_components"].items()},
            }
            self.writer.writerow(row)
            self.handle.flush()
            self.recent.append(row)
            for key in ("success", "collision", "reward", "final_distance"):
                self.logger.record(f"episodes/{key}_mean_100", np.mean([item[key] for item in self.recent]))
            for name, value in info["episode_reward_components"].items():
                self.logger.record(f"reward_components/{name}", value)
        if self.num_timesteps - self.last_report >= 10_000:
            success = np.mean([row["success"] for row in self.recent]) if self.recent else 0
            print(f"{self.num_timesteps:,} transitions; recent success: {success:.1%}", flush=True)
            self.last_report = self.num_timesteps
        return True

    def _on_training_end(self):
        if self.handle is not None:
            self.handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=TRAINING_TIMESTEPS)
    parser.add_argument("--num-envs", type=int, default=NUM_ENVS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--run-name", default="ppo_seed7")
    parser.add_argument("--eval-episodes", type=int, default=EVALUATION_EPISODES,
                        help="0 skips evaluation; full runs should use at least 50.")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.timesteps <= 0 or args.num_envs <= 0 or args.eval_episodes < 0:
        parser.error("Timesteps and num-envs must be positive; eval-episodes must be nonnegative.")
    name = experiment_name(args.run_name)
    run_dir = RESULTS_DIR / name
    model_path = MODELS_DIR / f"{name}.zip"
    if (run_dir / "training_episodes.csv").exists() or model_path.exists():
        parser.error("This run already exists. Choose a new --run-name to preserve its results.")
    run_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)  # Tiny MLPs are faster without a thread pool per inference.
    set_random_seed(args.seed)
    factories = [partial(make_training_env, args.seed + index) for index in range(args.num_envs)]
    vec_env = DummyVecEnv(factories) if args.num_envs == 1 else SubprocVecEnv(factories, start_method="spawn")
    vec_env.seed(args.seed)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "seed": args.seed,
        "requested_timesteps": args.timesteps, "num_envs": args.num_envs,
        "environment": {"env_config": ENV_CONFIG, "goal_config": GOAL_CONFIG, "reward_weights": REWARD_WEIGHTS},
        "ppo_config": PPO_CONFIG,
        "versions": {name: importlib.metadata.version(name) for name in ("highway-env", "gymnasium", "stable-baselines3", "torch", "numpy")},
        "status": "running",
    }
    print(f"Reward weights: {REWARD_WEIGHTS}", flush=True)
    print(f"Training {args.timesteps:,} transitions on {args.num_envs} environment(s).", flush=True)
    model = PPO("MultiInputPolicy", vec_env, seed=args.seed, verbose=0, **PPO_CONFIG)
    formats = ["csv"] + ([] if args.quiet else ["stdout"]) + ([] if args.no_tensorboard else ["tensorboard"])
    model.set_logger(configure(str(run_dir), formats))
    callback = EpisodeMetrics(run_dir)
    started = time.perf_counter()
    interrupted = False
    failed = False
    try:
        model.learn(total_timesteps=args.timesteps, callback=callback)
    except KeyboardInterrupt:
        interrupted = True
        print("Interrupted; saving the current policy and completed episode metrics.", flush=True)
    except Exception:
        failed = True
        raise
    finally:
        model.save(model_path)
        metadata.update(actual_timesteps=model.num_timesteps, training_seconds=time.perf_counter() - started,
                        status="failed" if failed else "interrupted" if interrupted else "trained")
        for path in (model_path.with_suffix(".json"), run_dir / "run.json"):
            with path.open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2)
        if callback.handle is not None and not callback.handle.closed:
            callback.handle.close()
        vec_env.close()
    plot_training(run_dir)
    print(f"Saved {model_path}; actual transitions: {model.num_timesteps:,}.", flush=True)
    if args.eval_episodes and not interrupted:
        compare(model_path, run_dir, episodes=args.eval_episodes)


if __name__ == "__main__":
    main()

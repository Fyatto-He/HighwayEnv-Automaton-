"""Evaluate DQN versus random actions on paired traffic seeds, including interactions."""
import argparse
import csv
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from stable_baselines3 import DQN
import torch

from baseline_DQN.config import MODELS_DIR, RESULTS_DIR, EVALUATION_EPISODES, EVALUATION_SEED, experiment_name
from baseline_DQN.environment import TrafficHighwayEnv


def load_experiment(model_path, render_mode=None):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"No saved model at {model_path}; run baseline_DQN/train.py first.")
    with model_path.with_suffix(".json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    environment = dict(metadata["environment"])
    if environment.get("scenario") == "quickstart" and "effective_env_config" in metadata:
        # Replay the recorded defaults even if the installed simulator changes.
        environment["env_config"] = metadata["effective_env_config"]
    env = TrafficHighwayEnv(render_mode=render_mode, **environment)
    try:
        model = DQN.load(model_path, device="cpu")
    except Exception:
        env.close()
        raise
    return model, env, metadata


def evaluate_policy(env, model, episodes, seed, label):
    rows = []
    for episode in range(episodes):
        episode_seed = seed + episode
        observation, _ = env.reset(seed=episode_seed)
        ego = env.highway.vehicle
        initial = {"initial_ego_x": float(ego.position[0]), "initial_lane": ego.lane_index[2],
                   "initial_lead_speed": float(env.lead_vehicle.speed),
                   "initial_lead_gap": env.closest_front_gap()}
        action_rng = np.random.default_rng(episode_seed + 1_000_000)
        terminated = truncated = False
        total_reward, steps = 0.0, 0
        while not (terminated or truncated):
            action = int(action_rng.integers(env.action_space.n)) if model is None else int(model.predict(observation, deterministic=True)[0])
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
        rows.append({"policy": label, "seed": episode_seed, "episode": episode + 1,
                     **initial, "reward": total_reward, "length": steps,
                     "collision": int(info["is_collision"]), "completed": int(info["is_completed"]),
                     "elapsed_time": info["elapsed_time"], "distance_travelled": info["distance_travelled"],
                     **info["interaction_metrics"]})
    return rows


def summarize(rows):
    mean = lambda key: float(np.mean([row[key] for row in rows]))
    return {
        "episodes": len(rows), "collision_rate": mean("collision"), "completion_rate": mean("completed"),
        "average_episode_reward": mean("reward"), "average_episode_duration_seconds": mean("elapsed_time"),
        "average_speed_mps": mean("mean_speed"), "average_distance_travelled_meters": mean("distance_travelled"),
        "episodes_with_close_encounter_rate": float(np.mean([row["encounter_count"] > 0 for row in rows])),
        "episodes_with_overtake_rate": float(np.mean([row["overtake_count"] > 0 for row in rows])),
        "average_distinct_close_encounters": mean("encounter_count"),
        "average_overtakes": mean("overtake_count"), "average_lane_changes": mean("lane_changes"),
        "average_close_following_steps": mean("close_following_steps"),
        "average_following_brake_steps": mean("following_brake_steps"),
        "average_minimum_front_gap_meters": mean("min_front_gap"),
    }


def save_rows(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare(model_path, output_dir, episodes=EVALUATION_EPISODES, seed=EVALUATION_SEED):
    if episodes < 1:
        raise ValueError("episodes must be positive.")
    torch.set_num_threads(1)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, env, metadata = load_experiment(model_path)
    try:
        trained = evaluate_policy(env, model, episodes, seed, "dqn")
        random = evaluate_policy(env, None, episodes, seed, "random")
    finally:
        env.close()
    save_rows(output_dir / "evaluation_episodes.csv", trained + random)
    summary = {"model": str(Path(model_path).resolve()), "first_seed": seed,
               "environment": env.configuration, "dqn": summarize(trained), "random": summarize(random)}
    with (output_dir / "evaluation.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps({"dqn": summary["dqn"], "random": summary["random"]}, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODELS_DIR / "dqn_quickstart_seed0.zip")
    parser.add_argument("--episodes", type=int, default=EVALUATION_EPISODES)
    parser.add_argument("--seed", type=int, default=EVALUATION_SEED)
    parser.add_argument("--output-name", default="evaluation")
    args = parser.parse_args()
    compare(args.model, RESULTS_DIR / experiment_name(args.output_name), args.episodes, args.seed)


if __name__ == "__main__":
    main()

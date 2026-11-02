"""Evaluate PPO and uniform random actions on exactly the same episode seeds."""
import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from stable_baselines3 import PPO
import torch

from baseline_waypoint.config import ENV_CONFIG, EVALUATION_EPISODES, EVALUATION_SEED, MODELS_DIR, RESULTS_DIR, experiment_name
from baseline_waypoint.environment import GoalHighwayEnv


def load_experiment(model_path, render_mode=None, traffic_mode="slow"):
    model_path = Path(model_path)
    metadata_path = model_path.with_suffix(".json")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run baseline_waypoint/train.py first.")
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    configuration = deepcopy(metadata["environment"])
    if traffic_mode == "slow":
        for key in ("traffic_speed_range", "lead_vehicle_gap"):
            configuration["env_config"][key] = deepcopy(ENV_CONFIG[key])
    elif traffic_mode != "saved":
        raise ValueError("traffic_mode must be 'slow' or 'saved'.")
    # Preserve old training settings in the sidecar; label the requested new traffic.
    env = GoalHighwayEnv(render_mode=render_mode, **configuration)
    metadata["evaluation_environment"] = configuration
    try:
        model = PPO.load(model_path, device="cpu")
    except Exception:
        env.close()
        raise
    return model, env, metadata


def evaluate_policy(env, model, episodes, seed, label):
    rows = []
    for episode in range(episodes):
        episode_seed = seed + episode
        obs, initial_info = env.reset(seed=episode_seed)
        initial_lane = env.highway.vehicle.lane_index[2]
        action_rng = np.random.default_rng(episode_seed + 1_000_000)
        total_reward, length = 0.0, 0
        terminated = truncated = False
        while not (terminated or truncated):
            action = int(action_rng.integers(env.action_space.n)) if model is None else int(model.predict(obs, deterministic=True)[0])
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            length += 1
        rows.append({
            "policy": label, "seed": episode_seed, "episode": episode + 1,
            "initial_lane": initial_lane,
            "lane_changes_required": abs(initial_info["goal_lane"] - initial_lane),
            "goal_lane": initial_info["goal_lane"],
            "goal_x": float(initial_info["goal_position"][0]),
            "goal_y": float(initial_info["goal_position"][1]),
            "initial_distance": initial_info["distance_to_goal"],
            "reward": total_reward, "length": length,
            "success": int(info["is_success"]), "collision": int(info["is_collision"]),
            "timeout": int(info["is_timeout"]), "elapsed_time": info["elapsed_time"],
            "final_distance": info["distance_to_goal"],
            **{f"reward_{name}": value for name, value in info["episode_reward_components"].items()},
        })
    return rows


def summarize(rows):
    mean = lambda key: float(np.mean([row[key] for row in rows]))
    successes = [row["elapsed_time"] for row in rows if row["success"]]
    return {
        "episodes": len(rows), "success_rate": mean("success"),
        "collision_rate": mean("collision"), "timeout_rate": mean("timeout"),
        "average_episode_reward": mean("reward"),
        "average_time_to_goal_seconds": float(np.mean(successes)) if successes else None,
        "average_final_distance_meters": mean("final_distance"),
        "success_rate_by_goal_lane": {
            str(lane): float(np.mean([row["success"] for row in rows if row["goal_lane"] == lane]))
            for lane in sorted({row["goal_lane"] for row in rows})
        },
        "success_rate_by_required_lane_changes": {
            str(count): float(np.mean([row["success"] for row in rows if row["lane_changes_required"] == count]))
            for count in sorted({row["lane_changes_required"] for row in rows})
        },
        "average_reward_components": {name: mean(f"reward_{name}") for name in ("goal", "collision", "progress", "time")},
    }


def save_rows(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare(model_path, output_dir, episodes=EVALUATION_EPISODES, seed=EVALUATION_SEED, traffic_mode="slow"):
    if episodes < 1:
        raise ValueError("Evaluation needs at least one episode.")
    torch.set_num_threads(1)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, env, metadata = load_experiment(model_path, traffic_mode=traffic_mode)
    try:
        trained = evaluate_policy(env, model, episodes, seed, "ppo")
        random = evaluate_policy(env, None, episodes, seed, "random")
    finally:
        env.close()
    save_rows(output_dir / "evaluation_episodes.csv", trained + random)
    summary = {"model": str(Path(model_path).resolve()), "first_seed": seed,
               "traffic_mode": traffic_mode, "environment": metadata["evaluation_environment"],
               "trained_environment": metadata["environment"],
               "ppo": summarize(trained), "random": summarize(random)}
    with (output_dir / "evaluation.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps({"ppo": summary["ppo"], "random": summary["random"]}, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODELS_DIR / "ppo_seed7.zip")
    parser.add_argument("--episodes", type=int, default=EVALUATION_EPISODES)
    parser.add_argument("--seed", type=int, default=EVALUATION_SEED)
    parser.add_argument("--output-name", default="evaluation")
    parser.add_argument("--traffic", choices=("slow", "saved"), default="slow",
                        help="Slower interactive traffic (default), or the checkpoint's original traffic.")
    args = parser.parse_args()
    compare(args.model, RESULTS_DIR / experiment_name(args.output_name), args.episodes, args.seed, args.traffic)


if __name__ == "__main__":
    main()

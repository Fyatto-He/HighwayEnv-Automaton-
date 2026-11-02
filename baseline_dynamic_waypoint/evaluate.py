"""Evaluate PPO, uniform random actions and the scripted reference on the same episode seeds."""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from stable_baselines3 import PPO
import torch

from baseline_dynamic_waypoint.config import (EVALUATION_EPISODES, EVALUATION_SEED, MODELS_DIR, RESULTS_DIR,
                                              SLOTS, experiment_name)
from baseline_dynamic_waypoint.environment import DynamicGoalHighwayEnv
from baseline_dynamic_waypoint.scripted import scripted_action
from baseline_waypoint.evaluate import save_rows, summarize as summarize_static


def load_experiment(model_path, render_mode=None):
    """Rebuild the exact task (including its hold time) from the model's JSON sidecar."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run baseline_dynamic_waypoint/train.py first.")
    with model_path.with_suffix(".json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    env = DynamicGoalHighwayEnv(render_mode=render_mode, **metadata["environment"])
    try:
        model = PPO.load(model_path, device="cpu")
    except Exception:
        env.close()
        raise
    return model, env, metadata


def evaluate_policy(env, model, episodes, seed, label):
    """model is an SB3 policy, None for uniform random actions, or "scripted"."""
    rows = []
    for episode in range(episodes):
        episode_seed = seed + episode
        obs, initial_info = env.reset(seed=episode_seed)
        ego = env.highway.vehicle
        initial_lane = ego.lane_index[2]
        anchor_gap = float(initial_info["anchor_position"][0] - ego.position[0])
        action_rng = np.random.default_rng(episode_seed + 1_000_000)
        total_reward, length, max_hold = 0.0, 0, 0
        terminated = truncated = False
        while not (terminated or truncated):
            if model is None:
                action = int(action_rng.integers(env.action_space.n))
            elif model == "scripted":
                action = scripted_action(env)
            else:
                action = int(model.predict(obs, deterministic=True)[0])
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            length += 1
            max_hold = max(max_hold, info["hold_steps"])
        rows.append({
            "policy": label, "seed": episode_seed, "episode": episode + 1,
            "slot": initial_info["slot"], "anchor_initial_gap": anchor_gap,
            "initial_lane": initial_lane,
            "lane_changes_required": abs(initial_info["goal_lane"] - initial_lane),
            "goal_lane": initial_info["goal_lane"],
            "initial_distance": initial_info["distance_to_goal"],
            "reward": total_reward, "length": length,
            "success": int(info["is_success"]), "collision": int(info["is_collision"]),
            "timeout": int(info["is_timeout"]), "anchor_lost": int(info["is_anchor_lost"]),
            "elapsed_time": info["elapsed_time"], "final_distance": info["distance_to_goal"],
            "max_hold_seconds": max_hold / env.env_config["policy_frequency"],
            **{f"reward_{name}": value for name, value in info["episode_reward_components"].items()},
        })
    return rows


def summarize(rows):
    mean = lambda key, subset=rows: float(np.mean([row[key] for row in subset]))
    summary = summarize_static(rows)
    summary.update({
        "anchor_lost_rate": mean("anchor_lost"),
        "average_max_hold_seconds": mean("max_hold_seconds"),
        "success_rate_by_slot": {slot: mean("success", [row for row in rows if row["slot"] == slot])
                                 for slot in SLOTS if any(row["slot"] == slot for row in rows)},
        "episodes_by_slot": {slot: sum(row["slot"] == slot for row in rows) for slot in SLOTS},
        "average_reward_components": {name: mean(f"reward_{name}") for name in ("goal", "collision", "progress", "time", "hold")},
    })
    return summary


def compare(model_path, output_dir, episodes=EVALUATION_EPISODES, seed=EVALUATION_SEED):
    if episodes < 1:
        raise ValueError("Evaluation needs at least one episode.")
    torch.set_num_threads(1)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, env, metadata = load_experiment(model_path)
    try:
        trained = evaluate_policy(env, model, episodes, seed, "ppo")
        random = evaluate_policy(env, None, episodes, seed, "random")
        scripted = evaluate_policy(env, "scripted", episodes, seed, "scripted")
    finally:
        env.close()
    save_rows(output_dir / "evaluation_episodes.csv", trained + random + scripted)
    summary = {"model": str(Path(model_path).resolve()), "first_seed": seed,
               "hold_seconds": metadata["environment"]["goal_config"]["hold_seconds"],
               "environment": metadata["environment"],
               "ppo": summarize(trained), "random": summarize(random), "scripted": summarize(scripted)}
    with (output_dir / "evaluation.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    for label in ("ppo", "random", "scripted"):
        result = summary[label]
        print(f"{label:>8}: success {result['success_rate']:.0%}, collision {result['collision_rate']:.0%}, "
              f"timeout {result['timeout_rate']:.0%}, anchor lost {result['anchor_lost_rate']:.0%}, "
              f"reward {result['average_episode_reward']:.2f}, by slot {result['success_rate_by_slot']}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODELS_DIR / "ppo_hold1s_seed7.zip")
    parser.add_argument("--episodes", type=int, default=EVALUATION_EPISODES)
    parser.add_argument("--seed", type=int, default=EVALUATION_SEED)
    parser.add_argument("--output-name", default="evaluation")
    args = parser.parse_args()
    compare(args.model, RESULTS_DIR / experiment_name(args.output_name), args.episodes, args.seed)


if __name__ == "__main__":
    main()

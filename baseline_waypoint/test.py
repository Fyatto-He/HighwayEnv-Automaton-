"""Watch a saved PPO policy, optionally recording annotated evaluation videos."""
import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import imageio.v2 as imageio
import torch

from baseline_waypoint.config import MODELS_DIR, RESULTS_DIR, EVALUATION_SEED, experiment_name
from baseline_waypoint.evaluate import load_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODELS_DIR / "ppo_seed7.zip")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=EVALUATION_SEED + 6)
    parser.add_argument("--goal-lane", type=int, help="Fix the target lane ID (0, 1 or 2).")
    parser.add_argument("--goal-distance", type=float, help="Place the goal this many meters ahead.")
    parser.add_argument("--cycle-goals", action="store_true", help="Visit lane IDs 0, 1, 2 in successive episodes.")
    parser.add_argument("--headless", action="store_true", help="Render offscreen, without opening a window.")
    parser.add_argument("--record", action="store_true", help="Save MP4s and initial/final PNGs inside baseline_waypoint/results/.")
    parser.add_argument("--output-name", default="visualization")
    parser.add_argument("--traffic", choices=("slow", "saved"), default="slow",
                        help="Use slower traffic by default; 'saved' reproduces the checkpoint's original traffic.")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive.")
    if args.cycle_goals and args.goal_lane is not None:
        parser.error("Use either --cycle-goals or --goal-lane.")
    torch.set_num_threads(1)
    # Human mode uses the same annotated WorldSurface; recording reads its pixels.
    model, env, _ = load_experiment(args.model, "rgb_array" if args.headless else "human", args.traffic)
    print(f"Traffic mode: {args.traffic}; desired speeds: {env.env_config.get('traffic_speed_range', 'original')}", flush=True)
    output_dir = RESULTS_DIR / experiment_name(args.output_name)
    if args.record:
        output_dir.mkdir(parents=True, exist_ok=True)
    try:
        for episode in range(args.episodes):
            options = {}
            if args.goal_lane is not None or args.cycle_goals:
                options["goal_lane"] = episode % env.env_config["lanes_count"] if args.cycle_goals else args.goal_lane
            if args.goal_distance is not None:
                options["goal_distance"] = args.goal_distance
            obs, info = env.reset(seed=args.seed + episode, options=options)
            print(f"Episode {episode + 1}: goal={info['goal_position']}, lane={info['goal_lane']}", flush=True)
            env.render()
            writer = None
            if args.record:
                prefix = output_dir / f"episode_{episode + 1:02d}"
                writer = imageio.get_writer(str(prefix.with_suffix(".mp4")), fps=env.metadata["render_fps"],
                                            codec="libx264", macro_block_size=1)
                frame = env.highway.viewer.get_image()
                writer.append_data(frame)
                imageio.imwrite(str(prefix) + "_start.png", frame)
            total_reward = 0.0
            try:
                terminated = truncated = False
                while not (terminated or truncated):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(int(action))
                    total_reward += reward
                    if env.highway.done:  # The user closed the interactive window.
                        return
                    if args.headless:
                        env.render()
                    if writer is not None:
                        writer.append_data(env.highway.viewer.get_image())
                if writer is not None:
                    frame = env.highway.viewer.get_image()
                    imageio.imwrite(str(prefix) + "_final.png", frame)
                    for _ in range(env.metadata["render_fps"]):
                        writer.append_data(frame)  # Hold the terminal outcome for one second.
                elif not args.headless:
                    import pygame
                    # Keep handling window events while showing the episode outcome.
                    for _ in range(10):
                        pygame.time.wait(100)
                        if any(event.type == pygame.QUIT for event in pygame.event.get()):
                            return
                print(f"  {info['termination_reason']}: reward={total_reward:.2f}, "
                      f"time={info['elapsed_time']:.1f}s, distance={info['distance_to_goal']:.2f}m", flush=True)
                print(f"  Reward totals: {info['episode_reward_components']}", flush=True)
            finally:
                if writer is not None:
                    writer.close()
    finally:
        env.close()


if __name__ == "__main__":
    main()

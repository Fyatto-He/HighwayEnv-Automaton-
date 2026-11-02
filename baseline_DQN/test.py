"""Watch native-task DQN and optionally record real 5-Hz simulation frames."""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import imageio.v2 as imageio
import torch

from baseline_DQN.config import MODELS_DIR, RESULTS_DIR, EVALUATION_SEED, experiment_name
from baseline_DQN.evaluate import load_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODELS_DIR / "dqn_quickstart_seed0.zip")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=EVALUATION_SEED)
    parser.add_argument("--headless", action="store_true", help="Draw offscreen without opening a window.")
    parser.add_argument("--record", action="store_true", help="Save annotated MP4s, PNGs and episode metrics.")
    parser.add_argument("--output-name", default="visualization")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive.")
    torch.set_num_threads(1)
    model, env, _ = load_experiment(args.model, render_mode="rgb_array" if args.headless else "human")
    output_dir = RESULTS_DIR / experiment_name(args.output_name)
    if args.record:
        output_dir.mkdir(parents=True, exist_ok=True)
    episodes = []
    try:
        for episode in range(args.episodes):
            seed = args.seed + episode
            obs, info = env.reset(seed=seed)
            env.render()
            viewer = env.highway.viewer
            fps = env.highway.config["simulation_frequency"]
            writer = None
            if args.record:
                prefix = output_dir / f"episode_{episode + 1:02d}"
                writer = imageio.get_writer(str(prefix.with_suffix(".mp4")), fps=fps,
                                            codec="libx264", macro_block_size=1)
                writer.append_data(viewer.get_image())
                imageio.imwrite(str(prefix) + "_start.png", viewer.get_image())
                viewer.frame_callback = writer.append_data
            total_reward = 0.0
            print(f"Episode {episode + 1}, seed {seed}, scenario={env.scenario}.", flush=True)
            try:
                terminated = truncated = False
                while not (terminated or truncated):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(int(action))
                    total_reward += reward
                    if env.highway.done:
                        return  # The user closed the interactive window.
                    if args.headless:
                        env.render()
                # Stop capture before holding/writing the final frame ourselves.
                viewer.frame_callback = None
                if writer is not None:
                    final_frame = viewer.get_image()
                    imageio.imwrite(str(prefix) + "_final.png", final_frame)
                    for _ in range(fps):
                        writer.append_data(final_frame)
                elif not args.headless:
                    import pygame
                    for _ in range(10):
                        pygame.time.wait(100)
                        if any(event.type == pygame.QUIT for event in pygame.event.get()):
                            return
                result = {"episode": episode + 1, "seed": seed, "reward": total_reward,
                          "status": env.status, "simulated_seconds": float(env.highway.time),
                          "collision": bool(env.highway.vehicle.crashed), **dict(env.metrics)}
                episodes.append(result)
                print(json.dumps(result, indent=2), flush=True)
            finally:
                viewer.frame_callback = None
                if writer is not None:
                    writer.close()
    finally:
        env.close()
        if args.record:
            with (output_dir / "episodes.json").open("w", encoding="utf-8") as handle:
                json.dump(episodes, handle, indent=2)


if __name__ == "__main__":
    main()

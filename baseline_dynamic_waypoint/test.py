"""Watch a saved dynamic-waypoint policy, optionally recording annotated videos."""
import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import imageio.v2 as imageio
import torch

from baseline_dynamic_waypoint.config import MODELS_DIR, RESULTS_DIR, EVALUATION_SEED, SLOTS, experiment_name
from baseline_dynamic_waypoint.evaluate import load_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODELS_DIR / "ppo_hold1s_seed7.zip")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=EVALUATION_SEED + 6)
    parser.add_argument("--slot", choices=SLOTS, help="Fix the slot (it must be on the road for the anchor's lane).")
    parser.add_argument("--anchor-index", type=int, help="Eligible anchor car, 0 = nearest.")
    parser.add_argument("--cycle-slots", action="store_true",
                        help="Try front, back, left, right in successive episodes (skipping off-road slots).")
    parser.add_argument("--headless", action="store_true", help="Render offscreen, without opening a window.")
    parser.add_argument("--record", action="store_true", help="Save MP4s and initial/final PNGs inside baseline_dynamic_waypoint/results/.")
    parser.add_argument("--output-name", default="visualization")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive.")
    if args.cycle_slots and args.slot is not None:
        parser.error("Use either --cycle-slots or --slot.")
    torch.set_num_threads(1)
    model, env, metadata = load_experiment(args.model, "rgb_array" if args.headless else "human")
    print(f"Hold time: {metadata['environment']['goal_config']['hold_seconds']:g} s", flush=True)
    output_dir = RESULTS_DIR / experiment_name(args.output_name)
    if args.record:
        output_dir.mkdir(parents=True, exist_ok=True)
    try:
        seed = args.seed
        for episode in range(args.episodes):
            options = {} if args.anchor_index is None else {"anchor_index": args.anchor_index}
            wanted = SLOTS[episode % len(SLOTS)] if args.cycle_slots else args.slot
            if wanted is not None:
                # Advance seeds until the chosen anchor can host this slot (LEFT/RIGHT need an adjacent lane).
                for _ in range(100):
                    try:
                        obs, info = env.reset(seed=seed, options={**options, "slot": wanted})
                        break
                    except ValueError:
                        seed += 1
                else:
                    raise ValueError(f"No seed near {seed} can host slot {wanted!r}.")
            else:
                obs, info = env.reset(seed=seed, options=options)
            seed += 1
            print(f"Episode {episode + 1}: {info['slot'].upper()} of car #{info['anchor_id']}, "
                  f"slot lane {info['goal_lane']}", flush=True)
            env.render()
            writer = None
            if args.record:
                prefix = output_dir / f"episode_{episode + 1:02d}_{info['slot']}"
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

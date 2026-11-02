"""Dynamic-waypoint settings: Marcos's baseline_waypoint settings plus a moving slot.

Distances are meters, speeds are m/s. Only the differences are listed here.
"""
from copy import deepcopy
from pathlib import Path

from baseline_waypoint.config import (ENV_CONFIG as STATIC_ENV_CONFIG, PPO_CONFIG as STATIC_PPO_CONFIG,
                                      REWARD_WEIGHTS as STATIC_REWARD_WEIGHTS, SEED, NUM_ENVS,
                                      EVALUATION_EPISODES, EVALUATION_SEED, experiment_name)  # noqa: F401

BASELINE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASELINE_DIR / "models"
RESULTS_DIR = BASELINE_DIR / "results"
TRAINING_TIMESTEPS = 300_000

SLOTS = ("front", "back", "left", "right")

ENV_CONFIG = deepcopy(STATIC_ENV_CONFIG)
# Reach a car up to 120 m ahead, then hold the slot for up to 5 s.
ENV_CONFIG["duration"] = 25.0
# FASTER/SLOWER set the target one grid step from the *current* speed, so the step
# size also caps acceleration (~step / 0.6 s). 2.5 m/s steps can roughly match
# 12-17 m/s traffic and still brake at ~4 m/s^2; 1 m/s steps brake too slowly
# (~1.7 m/s^2) and Marcos's 5 m/s steps match speeds too coarsely for long holds.
ENV_CONFIG["action"] = {"type": "DiscreteMetaAction", "target_speeds": [10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0]}

GOAL_CONFIG = {
    # Center-to-center gap of the anchor car at reset (it is chosen uniformly among eligible cars).
    "distance_range": [30.0, 120.0],
    # Center-to-center distance of the FRONT/BACK slots along the anchor's lane.
    # FRONT is farther because the anchor's IDM driver follows ego and keeps
    # about 10 m + 1.5 s of gap: a closer FRONT slot is pushed away by the anchor.
    # LEFT/RIGHT slots sit beside the anchor, centered in the adjacent lane.
    "slot_offsets": {"front": 35.0, "back": 15.0},
    # Success box in the slot lane's frame. "radius" is the longitudinal half-length; the
    # name is kept so the inherited check (no skipping the box between decisions) applies.
    "radius": 3.0,
    "lateral_tolerance": 1.5,
    # Consecutive in-slot time required for success; leaving the box resets the count.
    "hold_seconds": 1.0,
    "initial_speed": 20.0,
    "position_scale": [200.0, 12.0],
}

REWARD_WEIGHTS = {**STATIC_REWARD_WEIGHTS, "hold": 0.2}

PPO_CONFIG = deepcopy(STATIC_PPO_CONFIG)

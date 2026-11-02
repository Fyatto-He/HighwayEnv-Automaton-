"""All initial experiment settings. Distances are meters, speeds are m/s."""
from pathlib import Path

BASELINE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASELINE_DIR / "models"
RESULTS_DIR = BASELINE_DIR / "results"
SEED = 7
TRAINING_TIMESTEPS = 200_000
NUM_ENVS = 4
EVALUATION_EPISODES = 50
EVALUATION_SEED = 10_000

ENV_CONFIG = {
    "lanes_count": 3,
    "vehicles_count": 8,  # Traffic cars, in addition to the ego vehicle.
    "controlled_vehicles": 1,
    "vehicles_density": 0.7,
    "traffic_speed_range": [12.0, 17.0],  # Sustained IDM desired speeds, below ego's 20 m/s.
    "lead_vehicle_gap": 35.0,  # Center distance to a slower car in ego's starting lane.
    "ego_spacing": 2.0,
    "duration": 18.0,
    "simulation_frequency": 10,
    "policy_frequency": 10,
    "offroad_terminal": False,
    "action": {"type": "DiscreteMetaAction", "target_speeds": [15, 20, 25]},
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 7,  # Ego plus the six nearest traffic cars.
        "features": ["presence", "x", "y", "vx", "vy"],
        "absolute": False,
        "normalize": True,
        "clip": True,
        "see_behind": True,
        "include_obstacles": False,
        "order": "sorted",
        "features_range": {
            "x": [-200, 200], "y": [-12, 12],
            "vx": [-30, 30], "vy": [-10, 10],
        },
    },
    "screen_width": 1200,
    "screen_height": 360,
    "scaling": 5.0,
    "centering_position": [0.18, 0.55],
    "real_time_rendering": False,
}

GOAL_CONFIG = {
    "distance_range": [110.0, 170.0],
    # Lane centers are 4 m apart: a 5 m radius would accept an adjacent lane.
    "radius": 2.5,
    "initial_speed": 20.0,
    "position_scale": [200.0, 12.0],
}

REWARD_WEIGHTS = {"goal": 50.0, "collision": 50.0, "progress": 0.1, "time": 0.05}

PPO_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 256,  # Per environment: 4 * 256 = 1024 transitions per update.
    "batch_size": 256,
    "n_epochs": 5,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "policy_kwargs": {"net_arch": {"pi": [64, 64], "vf": [64, 64]}},
    "device": "cpu",
}


def experiment_name(value):
    """Keep generated outputs in baseline_waypoint/ even when a run name comes from a CLI."""
    if not value or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in value):
        raise ValueError("Run names may contain only letters, numbers, underscores and hyphens.")
    return value

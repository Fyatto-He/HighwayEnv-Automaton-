"""Official quickstart reproduction and the earlier slow-traffic experiment."""
from pathlib import Path

BASELINE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASELINE_DIR / "models"
RESULTS_DIR = BASELINE_DIR / "results"
SEED = 7
TRAINING_TIMESTEPS = 200_000
EVALUATION_EPISODES = 50
EVALUATION_SEED = 10_000

# Keep highway-fast-v0's default dynamics and native driving reward.
ENV_CONFIG = {
    "lanes_count": 3,
    "vehicles_count": 20,
    "duration": 30,
    "simulation_frequency": 5,
    "policy_frequency": 1,
    "vehicles_density": 1.0,
    "observation": {"type": "Kinematics", "vehicles_count": 5, "see_behind": True},
    "screen_width": 1200,
    "screen_height": 360,
    "scaling": 5.0,
    "centering_position": [0.18, 0.55],
    "real_time_rendering": False,
}

TRAFFIC_CONFIG = {
    "speed_range": [15.0, 20.0],
    "lead_distance": 35.0,  # Center-to-center; bumper gap is 30 m.
    "interaction_gap": 30.0,
}

# Historical slow-traffic settings; omitted options retain the SB3 defaults
# recorded beside those checkpoints. The official preset adds its two options below.
DQN_CONFIG = {
    "learning_rate": 5e-4,
    "buffer_size": 15_000,
    "learning_starts": 200,
    "batch_size": 32,
    "gamma": 0.8,
    "train_freq": 1,
    "gradient_steps": 1,
    "device": "cpu",
}

QUICKSTART_TIMESTEPS = 20_000
QUICKSTART_DQN_CONFIG = {
    **DQN_CONFIG,
    "policy_kwargs": {"net_arch": [256, 256]},
    "target_update_interval": 50,
}
QUICKSTART_REFERENCE = "https://highway-env.farama.org/quickstart/#training-an-agent"


def training_settings(preset):
    """Keep the reproduction independent of historical model configurations."""
    from copy import deepcopy
    if preset == "quickstart":
        return (deepcopy(QUICKSTART_DQN_CONFIG),
                {"scenario": "quickstart", "env_config": {}, "traffic_config": {}},
                QUICKSTART_TIMESTEPS)
    if preset == "slow-traffic":
        return (deepcopy(DQN_CONFIG),
                {"env_config": deepcopy(ENV_CONFIG), "traffic_config": deepcopy(TRAFFIC_CONFIG)},
                TRAINING_TIMESTEPS)
    raise ValueError(f"Unknown training preset: {preset}")


def experiment_name(value):
    if not value or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in value):
        raise ValueError("Run names may contain only letters, numbers, underscores and hyphens.")
    return value

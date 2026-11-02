"""Goal conditioning around the unchanged highway-v0 simulator."""
from copy import deepcopy

import gymnasium as gym
from gymnasium import spaces
import highway_env  # noqa: F401 -- registers highway-v0
import numpy as np

from baseline_waypoint.config import ENV_CONFIG, GOAL_CONFIG, REWARD_WEIGHTS


class GoalHighwayEnv(gym.Wrapper):
    """Standard Kinematics traffic observation plus a single positional goal.

    This is a PPO task, not a HER/GoalEnv implementation: the reward also depends
    on progress, collisions and time, so we deliberately do not expose compute_reward.
    """

    def __init__(self, render_mode=None, env_config=None, goal_config=None, reward_weights=None):
        self.env_config = deepcopy(ENV_CONFIG if env_config is None else env_config)
        self.goal_config = deepcopy(GOAL_CONFIG if goal_config is None else goal_config)
        self.reward_weights = deepcopy(REWARD_WEIGHTS if reward_weights is None else reward_weights)
        self._validate_config()
        self._render_mode = render_mode
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError("render_mode must be None, 'human' or 'rgb_array'.")
        native_config = deepcopy(self.env_config)
        native_config["offscreen_rendering"] = render_mode != "human"
        # Use our render() for both modes; the native env must not auto-render early.
        native_mode = "rgb_array" if render_mode is not None else None
        super().__init__(gym.make("highway-v0", config=native_config, render_mode=native_mode))
        self.highway = self.env.unwrapped
        self.metadata = dict(self.env.metadata, render_fps=self.env_config["policy_frequency"])
        n = self.env_config["observation"]["vehicles_count"]
        self.observation_space = spaces.Dict({
            "vehicles": spaces.Box(-1.0, 1.0, (n, 5), np.float32),
            "desired_goal": spaces.Box(-np.inf, np.inf, (2,), np.float32),
            "achieved_goal": spaces.Box(-np.inf, np.inf, (2,), np.float32),
            "goal_delta": spaces.Box(-np.inf, np.inf, (2,), np.float32),
        })
        self.position_scale = np.asarray(self.goal_config["position_scale"], dtype=np.float64)
        self.goal_position = np.zeros(2)
        self.episode_origin = np.zeros(2)
        self.status = "Driving"

    @property
    def render_mode(self):
        return self._render_mode

    def _validate_config(self):
        cfg, goal = self.env_config, self.goal_config
        if cfg["controlled_vehicles"] != 1 or cfg["lanes_count"] < 1:
            raise ValueError("Use exactly one ego vehicle and at least one lane.")
        if cfg["simulation_frequency"] % cfg["policy_frequency"] != 0:
            raise ValueError("Simulation frequency must be a multiple of policy frequency.")
        if goal["radius"] <= 0 or min(goal["position_scale"]) <= 0:
            raise ValueError("Goal radius and position scales must be positive.")
        low, high = goal["distance_range"]
        if not 0 < low <= high:
            raise ValueError("Goal distance_range must be positive and ordered.")
        # With these discrete speeds, a center crossing cannot jump over the disk.
        max_speed = max(max(cfg["action"]["target_speeds"]), goal["initial_speed"])
        if max_speed / cfg["policy_frequency"] >= 2 * goal["radius"]:
            raise ValueError("Increase policy_frequency or radius to avoid skipping a goal between decisions.")
        if cfg["observation"]["features"] != ["presence", "x", "y", "vx", "vy"]:
            raise ValueError("This small wrapper expects the five standard Kinematics features.")
        if "traffic_speed_range" in cfg:
            low, high = cfg["traffic_speed_range"]
            if not 0 < low <= high < goal["initial_speed"]:
                raise ValueError("Traffic speeds must be positive and below ego's initial speed.")
        if cfg.get("lead_vehicle_gap", 35) < 20:
            raise ValueError("Place the initial lead car at least 20 m ahead.")

    def reset(self, *, seed=None, options=None):
        options = {} if options is None else dict(options)
        # Goal options belong to this wrapper, not highway-env's config reset.
        raw, _ = self.env.reset(seed=seed)
        ego = self.highway.vehicle
        ego.speed = self.goal_config["initial_speed"]
        ego.target_speed = ego.speed
        ego.speed_index = ego.speed_to_index(ego.speed)
        ego.color = (60, 190, 255)
        for vehicle in self.highway.road.vehicles:
            if vehicle is not ego:
                vehicle.color = (235, 177, 88)
        self._configure_traffic()
        self.episode_origin = np.array([ego.position[0], 0.0])
        self._generate_goal(options)
        self.previous_distance = self.distance_to_goal()
        self.episode_totals = {name: 0.0 for name in self.reward_weights}
        self.status = "Driving"
        self.last_components = {name: 0.0 for name in self.reward_weights}
        # Re-observe because initial ego speed was changed after the native reset.
        observation = self._observation(self.highway.observation_type.observe())
        info = self._info(False, False, False)
        if self.render_mode == "human":
            self.render()
        return observation, info

    def _configure_traffic(self):
        """Keep traffic slower and put a visible lead car ahead of the ego.

        Older checkpoint configurations have neither key; 'saved' playback can
        therefore still reproduce their original traffic without changing weights.
        """
        traffic = [vehicle for vehicle in self.highway.road.vehicles if vehicle is not self.highway.vehicle]
        speeds = self.env_config.get("traffic_speed_range")
        if not traffic or speeds is None:
            return
        for vehicle in traffic:
            vehicle.speed = float(self.highway.np_random.uniform(*speeds))
            vehicle.target_speed = vehicle.speed
        if "lead_vehicle_gap" not in self.env_config:
            return
        ego = self.highway.vehicle
        lead = min(traffic, key=lambda vehicle: vehicle.position[0])
        shift = ego.position[0] + self.env_config["lead_vehicle_gap"] - lead.position[0]
        # Moving the entire traffic group preserves its original longitudinal gaps.
        for vehicle in traffic:
            vehicle.position[0] += shift
            vehicle.on_state_update()
        lead.position[1] = ego.position[1]
        lead.speed = lead.target_speed = float(speeds[0])
        lead.on_state_update()
        lead.target_lane_index = ego.lane_index

    def _generate_goal(self, options):
        rng = self.highway.np_random
        lane_id = int(options.get("goal_lane", rng.integers(self.env_config["lanes_count"])))
        if not 0 <= lane_id < self.env_config["lanes_count"]:
            raise ValueError("goal_lane is outside the road (lane IDs are zero based).")
        distance = float(options.get("goal_distance", rng.uniform(*self.goal_config["distance_range"])))
        lane = self.highway.road.network.get_lane(("0", "1", lane_id))
        ego_s, _ = lane.local_coordinates(self.highway.vehicle.position)
        target_s = ego_s + distance
        lane_changes = abs(lane_id - self.highway.vehicle.lane_index[2])
        min_speed = min(self.env_config["action"]["target_speeds"])
        if distance < max(20, lane_changes * 25) or min_speed <= 0:
            raise ValueError("Goal is too close for the required lane changes, or minimum speed is not positive.")
        if distance / min_speed + 2 * lane_changes >= self.env_config["duration"]:
            raise ValueError("Goal needs more driving/lane-change time than the episode allows.")
        if not self.goal_config["radius"] < target_s < lane.length - self.goal_config["radius"]:
            raise ValueError("Goal must lie inside the finite road, including its tolerance disk.")
        self.goal_lane = lane_id
        self.goal_position = lane.position(target_s, 0).astype(np.float64)

    def _observation(self, raw):
        achieved = (self.highway.vehicle.position - self.episode_origin) / self.position_scale
        desired = (self.goal_position - self.episode_origin) / self.position_scale
        vehicles = np.asarray(raw, dtype=np.float32).copy()
        # Native Kinematics keeps ego x absolute. Replace only that entry with
        # episode-relative progress, avoiding clipping at distant road positions.
        x_range = self.env_config["observation"]["features_range"]["x"]
        progress_x = self.highway.vehicle.position[0] - self.episode_origin[0]
        vehicles[0, 1] = np.clip(2 * (progress_x - x_range[0]) / (x_range[1] - x_range[0]) - 1, -1, 1)
        return {
            "vehicles": vehicles,
            "desired_goal": desired.astype(np.float32),
            "achieved_goal": achieved.astype(np.float32),
            # Redundant but makes the distance/direction directly available to an MLP.
            "goal_delta": (desired - achieved).astype(np.float32),
        }

    def distance_to_goal(self):
        return float(np.linalg.norm(self.highway.vehicle.position - self.goal_position))

    def step(self, action):
        raw, _, _, native_truncated, _ = self.env.step(int(action))
        distance = self.distance_to_goal()
        collision = bool(self.highway.vehicle.crashed)
        success = bool(distance < self.goal_config["radius"] and not collision)
        terminated = success or collision
        # A goal/collision at the deadline is a termination, not a timeout.
        truncated = bool(native_truncated and not terminated)
        weights = self.reward_weights
        self.last_components = {
            "goal": weights["goal"] * float(success),
            "collision": -weights["collision"] * float(collision),
            "progress": weights["progress"] * (self.previous_distance - distance),
            "time": -weights["time"],
        }
        for name, value in self.last_components.items():
            self.episode_totals[name] += value
        self.previous_distance = distance
        self.status = "GOAL REACHED" if success else "COLLISION" if collision else "TIMED OUT" if truncated else "Driving"
        info = self._info(success, collision, truncated)
        if self.render_mode == "human":
            self.render()
        return self._observation(raw), float(sum(self.last_components.values())), terminated, truncated, info

    def _info(self, success, collision, timeout):
        return {
            "is_success": success, "is_collision": collision, "is_timeout": timeout,
            "distance_to_goal": self.distance_to_goal(),
            "elapsed_time": float(self.highway.time),
            "goal_position": self.goal_position.copy(), "goal_lane": self.goal_lane,
            "ego_position": self.highway.vehicle.position.copy(),
            "termination_reason": self.status,
            "reward_components": dict(self.last_components),
            "episode_reward_components": dict(self.episode_totals),
        }

    def render(self):
        if self.render_mode is None:
            return None
        from baseline_waypoint.rendering import GoalViewer
        if self.highway.viewer is None:
            self.highway.viewer = GoalViewer(self.highway, self)
        self.highway.viewer.display()
        if not self.highway.viewer.offscreen:
            self.highway.viewer.handle_events()
        if self.render_mode == "rgb_array":
            return self.highway.viewer.get_image()
        return None

"""Measure native highway-fast-v0 or the historical slower-traffic experiment."""
from copy import deepcopy

import gymnasium as gym
import highway_env  # noqa: F401 -- register highway-fast-v0
import numpy as np

from baseline_DQN.config import ENV_CONFIG, TRAFFIC_CONFIG


class TrafficHighwayEnv(gym.Wrapper):
    """Native array observations/rewards, plus actual traffic interaction metrics."""

    def __init__(self, render_mode=None, env_config=None, traffic_config=None, scenario="slow-traffic"):
        if scenario not in ("quickstart", "slow-traffic"):
            raise ValueError("Scenario must be 'quickstart' or 'slow-traffic'.")
        self.scenario = scenario
        native = scenario == "quickstart"
        self.configuration = {
            "env_config": deepcopy(({} if native else ENV_CONFIG) if env_config is None else env_config),
            "traffic_config": deepcopy(({} if native else TRAFFIC_CONFIG) if traffic_config is None else traffic_config),
        }
        if native:
            self.configuration["scenario"] = scenario
        self.env_config = self.configuration["env_config"]
        self.traffic_config = self.configuration["traffic_config"]
        if not native:
            low, high = self.traffic_config["speed_range"]
            if not 0 < low <= high < 25:
                raise ValueError("Traffic speeds must be positive, ordered and below ego's 25 m/s initial speed.")
            if self.env_config["vehicles_count"] < 1:
                raise ValueError("This interaction scenario requires at least one traffic vehicle.")
            if self.traffic_config["lead_distance"] < 20:
                raise ValueError("The initial lead vehicle must be at least 20 m ahead.")
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError("Unsupported render mode.")
        self._render_mode = render_mode
        config = deepcopy(self.env_config)
        if native and render_mode is not None:
            # Display settings only; observations, rewards and dynamics stay native.
            config.update(screen_width=1200, screen_height=360, scaling=5.0,
                          centering_position=[0.18, 0.55], real_time_rendering=False)
        config["offscreen_rendering"] = render_mode != "human"
        native_render_mode = "rgb_array" if render_mode is not None else None
        super().__init__(gym.make("highway-fast-v0", config=config, render_mode=native_render_mode))
        self.highway = self.env.unwrapped
        self.metadata = dict(self.env.metadata, render_fps=self.highway.config["simulation_frequency"])
        self.metrics = {}
        self.status = "Driving"

    @property
    def render_mode(self):
        return self._render_mode

    def reset(self, *, seed=None, options=None):
        observation, native_info = self.env.reset(seed=seed, options=options)
        ego = self.highway.vehicle
        ego.color = (60, 190, 255)
        self.traffic = [car for car in self.highway.road.vehicles if car is not ego]
        self.lead_vehicle = min(self.traffic, key=lambda car: car.position[0])
        if self.scenario == "slow-traffic":
            self._set_slow_traffic(ego)
            observation = self.highway.observation_type.observe()
        for car in self.traffic:
            car.color = (235, 177, 88)
        self.lead_vehicle.color = (235, 110, 215)
        self.episode_start_x = float(ego.position[0])
        self.status = "Driving"
        self._encountered = set()
        self._overtaken = set()
        self._initially_ahead = {id(car) for car in self.traffic if car.position[0] > ego.position[0]}
        self._last_lane = ego.lane_index[2]
        self._speed_sum = 0.0
        self._decision_steps = 0
        self.metrics = {
            "encounter_count": 0, "overtake_count": 0, "lane_changes": 0,
            "following_brake_steps": 0, "close_following_steps": 0,
            "min_front_gap": self.closest_front_gap(), "mean_speed": float(ego.speed),
        }
        info = self._info(native_info)
        if self.render_mode == "human":
            self.render()
        return observation, info

    def _set_slow_traffic(self, ego):
        offset = ego.position[0] + self.traffic_config["lead_distance"] - self.lead_vehicle.position[0]
        # Shift every traffic car equally: no bunching or initial overlaps from
        # moving the lead alone into the following car's space.
        for car in self.traffic:
            car.position[0] += offset
            car.speed = float(self.highway.np_random.uniform(*self.traffic_config["speed_range"]))
            car.target_speed = car.speed  # Slow persistently, not just for one frame.
            car.on_state_update()
        self.lead_vehicle.position[1] = ego.position[1]
        self.lead_vehicle.on_state_update()
        self.lead_vehicle.target_lane_index = ego.lane_index

    def closest_front_gap(self):
        ego = self.highway.vehicle
        gaps = [max(0.0, float(car.position[0] - ego.position[0] - (car.LENGTH + ego.LENGTH) / 2))
                for car in self.highway.road.vehicles
                if car is not ego and car.lane_index == ego.lane_index and car.position[0] >= ego.position[0]]
        return min(gaps) if gaps else None

    def _update_interactions(self, previous_gap, previous_speed):
        ego = self.highway.vehicle
        threshold = self.traffic_config.get("interaction_gap", 30.0)
        current_gap = self.closest_front_gap()
        close_before = previous_gap is not None and 0 <= previous_gap < threshold
        close_after = current_gap is not None and 0 <= current_gap < threshold
        if close_before or close_after:
            self.metrics["close_following_steps"] += 1
        if close_before and previous_speed - ego.speed > 0.5 and not ego.crashed:
            self.metrics["following_brake_steps"] += 1
        if current_gap is not None:
            minimum = self.metrics["min_front_gap"]
            self.metrics["min_front_gap"] = current_gap if minimum is None else min(minimum, current_gap)
        for car in self.traffic:
            longitudinal = float(car.position[0] - ego.position[0])
            clearance = (car.LENGTH + ego.LENGTH) / 2
            bumper_gap = longitudinal - clearance
            if car.lane_index == ego.lane_index and 0 <= bumper_gap < threshold and not ego.crashed:
                self._encountered.add(id(car))
            lateral_clearance = abs(float(car.position[1] - ego.position[1])) >= (car.WIDTH + ego.WIDTH) / 2
            if (id(car) in self._initially_ahead and longitudinal <= -clearance
                    and lateral_clearance and not ego.crashed and ego.on_road):
                self._overtaken.add(id(car))
        lane = ego.lane_index[2]
        if lane != self._last_lane:
            self.metrics["lane_changes"] += abs(lane - self._last_lane)
        self._last_lane = lane
        self._decision_steps += 1
        self._speed_sum += float(ego.speed)
        self.metrics.update(encounter_count=len(self._encountered), overtake_count=len(self._overtaken),
                            mean_speed=self._speed_sum / self._decision_steps)

    def step(self, action):
        previous_gap, previous_speed = self.closest_front_gap(), self.highway.vehicle.speed
        observation, reward, terminated, truncated, native_info = self.env.step(int(action))
        self._update_interactions(previous_gap, previous_speed)
        crashed = bool(self.highway.vehicle.crashed)
        self.status = "COLLISION" if crashed else "TIME LIMIT COMPLETED" if truncated else "Driving"
        info = self._info(native_info)
        info["is_completed"] = bool(truncated and not crashed)
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, info

    def _info(self, native_info):
        return dict(native_info, **{
            "is_collision": bool(self.highway.vehicle.crashed),
            "elapsed_time": float(self.highway.time),
            "distance_travelled": float(self.highway.vehicle.position[0] - self.episode_start_x),
            "front_gap": self.closest_front_gap(),
            "interaction_metrics": dict(self.metrics),
            "termination_reason": self.status,
        })

    def render(self):
        if self.render_mode is None:
            return None
        from baseline_DQN.rendering import TrafficViewer
        if self.highway.viewer is None:
            self.highway.viewer = TrafficViewer(self.highway, self)
        self.highway.enable_auto_render = True
        self.highway.viewer.display()
        if not self.highway.viewer.offscreen:
            self.highway.viewer.handle_events()
        return self.highway.viewer.get_image() if self.render_mode == "rgb_array" else None

"""Behavioral checks for the task, including the world-coordinate goal overlay."""
from copy import deepcopy
import json
import unittest

import imageio.v2 as imageio
import numpy as np
from stable_baselines3.common.env_checker import check_env

from baseline_waypoint.config import ENV_CONFIG, MODELS_DIR, RESULTS_DIR
from baseline_waypoint.environment import GoalHighwayEnv
from baseline_waypoint.evaluate import load_experiment


class GoalEnvironmentTests(unittest.TestCase):
    def make_env(self, traffic=0, render_mode=None):
        config = deepcopy(ENV_CONFIG)
        config["vehicles_count"] = traffic
        env = GoalHighwayEnv(env_config=config, render_mode=render_mode)
        self.addCleanup(env.close)
        return env

    def place_at_goal(self, env, offset=0.0, crashed=False):
        ego = env.highway.vehicle
        ego.position = env.goal_position + np.array([offset, 0.0])
        ego.heading = ego.speed = ego.target_speed = 0.0
        ego.lane_index = ego.target_lane_index = ("0", "1", env.goal_lane)
        ego.crashed = crashed

    def test_sb3_contract_and_traffic(self):
        env = self.make_env(traffic=8)
        check_env(env, warn=True, skip_render_check=True)
        obs, _ = env.reset(seed=7)
        self.assertTrue(env.observation_space.contains(obs))
        self.assertGreater(obs["vehicles"][1:, 0].sum(), 0)
        for key in obs:
            self.assertEqual(obs[key].dtype, np.float32)
        np.testing.assert_allclose(obs["desired_goal"] - obs["achieved_goal"], obs["goal_delta"], atol=1e-6)

    def test_seed_replay_and_reachable_goals_cover_all_lanes(self):
        env = self.make_env(traffic=8)
        first_obs, first_info = env.reset(seed=123)
        second_obs, second_info = env.reset(seed=123)
        for key in first_obs:
            np.testing.assert_array_equal(first_obs[key], second_obs[key])
        np.testing.assert_array_equal(first_info["goal_position"], second_info["goal_position"])
        lanes = set()
        for seed in range(50):
            obs, info = env.reset(seed=seed)
            lane = env.highway.road.network.get_lane(("0", "1", info["goal_lane"]))
            s, lateral = lane.local_coordinates(info["goal_position"])
            self.assertAlmostEqual(lateral, 0.0)
            self.assertGreater(s, 0)
            self.assertLess(s, lane.length)
            self.assertGreater(info["goal_position"][0], env.highway.vehicle.position[0])
            self.assertTrue(env.observation_space.contains(obs))
            lanes.add(info["goal_lane"])
        self.assertEqual(lanes, {0, 1, 2})

    def test_traffic_speed_and_sustained_targets_are_slower_than_ego(self):
        env = self.make_env(traffic=8)
        env.reset(seed=7)
        ego = env.highway.vehicle
        low, high = env.env_config["traffic_speed_range"]
        traffic = [car for car in env.highway.road.vehicles if car is not ego]
        for car in traffic:
            self.assertGreaterEqual(car.speed, low)
            self.assertLessEqual(car.speed, high)
            self.assertLess(car.speed, ego.speed)
            self.assertEqual(car.target_speed, car.speed)
        targets = [car.target_speed for car in traffic]
        for _ in range(10):
            _, _, terminated, truncated, info = env.step(1)
            self.assertFalse(terminated or truncated)
        self.assertFalse(info["is_collision"])
        self.assertEqual([car.target_speed for car in traffic], targets)

    def test_initial_slow_lead_is_safe_and_deterministic(self):
        env = self.make_env(traffic=8)
        for seed in range(20):
            env.reset(seed=seed)
            ego = env.highway.vehicle
            cars = env.highway.road.vehicles
            traffic = [car for car in cars if car is not ego]
            lead = min(traffic, key=lambda car: car.position[0])
            self.assertEqual(lead.lane_index, ego.lane_index)
            self.assertAlmostEqual(lead.position[0] - ego.position[0], 35.0)
            self.assertEqual(lead.speed, env.env_config["traffic_speed_range"][0])
            self.assertEqual(lead.target_speed, lead.speed)
            for index, car in enumerate(cars):
                for other in cars[index + 1:]:
                    if car.lane_index == other.lane_index:
                        self.assertGreater(abs(car.position[0] - other.position[0]), (car.LENGTH + other.LENGTH) / 2)
            first_state = np.array([[*car.position, car.speed, car.target_speed] for car in cars])
            env.reset(seed=seed)
            second_state = np.array([[*car.position, car.speed, car.target_speed] for car in env.highway.road.vehicles])
            np.testing.assert_array_equal(first_state, second_state)

    def test_ego_approaches_moving_slow_lead_without_initial_collision(self):
        env = self.make_env(traffic=1)
        env.reset(seed=7)
        ego = env.highway.vehicle
        lead = next(car for car in env.highway.road.vehicles if car is not ego)
        initial_gap = float(lead.position[0] - ego.position[0])
        lead_start = float(lead.position[0])
        for _ in range(10):
            _, _, terminated, truncated, info = env.step(1)
            self.assertFalse(terminated or truncated)
        final_gap = float(lead.position[0] - ego.position[0])
        self.assertGreater(lead.position[0], lead_start)
        self.assertLess(final_gap, initial_gap - 5.0)
        self.assertGreater(final_gap, (ego.LENGTH + lead.LENGTH) / 2)
        self.assertFalse(info["is_collision"])

    def test_old_checkpoint_configuration_is_preserved_in_both_traffic_modes(self):
        model_path = MODELS_DIR / "ppo_seed7.zip"
        if not model_path.exists():
            self.skipTest("Historical PPO checkpoint is not present.")
        metadata_path = model_path.with_suffix(".json")
        original_bytes = metadata_path.read_bytes()
        original = json.loads(original_bytes)
        self.assertNotIn("traffic_speed_range", original["environment"]["env_config"])
        _, saved_env, saved_metadata = load_experiment(model_path, traffic_mode="saved")
        self.addCleanup(saved_env.close)
        self.assertEqual(saved_env.env_config, original["environment"]["env_config"])
        self.assertEqual(saved_metadata["environment"], original["environment"])
        saved_env.reset(seed=7)
        saved_traffic = [car for car in saved_env.highway.road.vehicles if car is not saved_env.highway.vehicle]
        self.assertGreater(max(car.target_speed for car in saved_traffic), ENV_CONFIG["traffic_speed_range"][1])
        _, slow_env, slow_metadata = load_experiment(model_path, traffic_mode="slow")
        self.addCleanup(slow_env.close)
        self.assertEqual(slow_metadata["environment"], original["environment"])
        self.assertEqual(slow_metadata["evaluation_environment"]["env_config"]["traffic_speed_range"], [12.0, 17.0])
        self.assertEqual(slow_metadata["evaluation_environment"]["env_config"]["lead_vehicle_gap"], 35.0)
        self.assertEqual(slow_env.observation_space, saved_env.observation_space)
        self.assertEqual(metadata_path.read_bytes(), original_bytes)

    def test_progress_reward_matches_distance_change(self):
        env = self.make_env()
        _, initial = env.reset(seed=7)
        _, reward, terminated, truncated, info = env.step(1)
        expected = env.reward_weights["progress"] * (initial["distance_to_goal"] - info["distance_to_goal"])
        self.assertAlmostEqual(info["reward_components"]["progress"], expected)
        self.assertAlmostEqual(reward, expected - env.reward_weights["time"])
        self.assertFalse(terminated or truncated)

    def test_goal_terminates_and_collision_takes_priority(self):
        env = self.make_env()
        env.reset(seed=7)
        self.place_at_goal(env)
        _, _, terminated, truncated, info = env.step(1)
        self.assertTrue(terminated and info["is_success"])
        self.assertFalse(truncated or info["is_collision"])
        self.assertEqual(info["reward_components"]["goal"], env.reward_weights["goal"])
        env.reset(seed=7)
        self.place_at_goal(env, crashed=True)
        _, _, terminated, truncated, info = env.step(1)
        self.assertTrue(terminated and info["is_collision"])
        self.assertFalse(truncated or info["is_success"])
        self.assertEqual(info["reward_components"]["collision"], -env.reward_weights["collision"])

    def test_strict_radius_and_adjacent_lane_do_not_count(self):
        env = self.make_env()
        env.reset(seed=7)
        self.place_at_goal(env, offset=env.goal_config["radius"])
        _, _, terminated, _, info = env.step(1)
        self.assertFalse(terminated or info["is_success"])
        env.reset(seed=7, options={"goal_lane": 0})
        self.place_at_goal(env)
        ego = env.highway.vehicle
        ego.position[1] = 4.0
        ego.lane_index = ego.target_lane_index = ("0", "1", 1)
        _, _, terminated, _, info = env.step(1)
        self.assertFalse(terminated or info["is_success"])

    def test_time_limit_is_truncation(self):
        env = self.make_env()
        env.reset(seed=7)
        env.highway.time = env.env_config["duration"]
        _, _, terminated, truncated, info = env.step(1)
        self.assertTrue(truncated and info["is_timeout"])
        self.assertFalse(terminated)

    def test_idle_crossing_detects_small_goal_at_decision_frequency(self):
        env = self.make_env()
        env.reset(seed=7)
        lane = env.highway.vehicle.lane_index[2]
        _, initial = env.reset(seed=7, options={"goal_lane": lane, "goal_distance": 40})
        for _ in range(40):
            _, _, terminated, truncated, info = env.step(1)
            if terminated or truncated:
                break
        self.assertTrue(info["is_success"])
        self.assertLess(info["distance_to_goal"], env.goal_config["radius"])
        expected_progress = env.reward_weights["progress"] * (initial["distance_to_goal"] - info["distance_to_goal"])
        self.assertAlmostEqual(info["episode_reward_components"]["progress"], expected_progress)

    def test_invalid_goal_is_rejected(self):
        env = self.make_env()
        for options in ({"goal_lane": 3}, {"goal_distance": 2}, {"goal_distance": 100_000}):
            with self.assertRaises(ValueError):
                env.reset(seed=7, options=options)

    def test_all_goal_lanes_are_reachable_with_native_actions(self):
        env = self.make_env()
        for lane in range(3):
            env.reset(seed=7, options={"goal_lane": lane, "goal_distance": 110})
            for _ in range(120):
                current_target = env.highway.vehicle.target_lane_index[2]
                action = 0 if lane < current_target else 2 if lane > current_target else 1
                _, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            self.assertTrue(info["is_success"], f"Could not reach target lane {lane}")

    def test_goal_pixels_follow_world_camera(self):
        env = self.make_env(render_mode="rgb_array")
        env.reset(seed=7, options={"goal_lane": 2, "goal_distance": 140})
        output_dir = RESULTS_DIR / "verification"
        output_dir.mkdir(parents=True, exist_ok=True)
        first_frame = env.render()
        surface = env.highway.viewer.sim_surface
        first_pixel = surface.pos2pix(*env.goal_position)
        green = np.array([65, 245, 125])
        np.testing.assert_array_equal(first_frame[first_pixel[1], first_pixel[0]], green)
        imageio.imwrite(output_dir / "goal_before_camera_move.png", first_frame)
        original_goal = env.goal_position.copy()
        # Leave the world target fixed and advance the camera with the ego.
        env.highway.vehicle.position[0] += 20
        second_frame = env.render()
        second_pixel = surface.pos2pix(*env.goal_position)
        self.assertEqual(second_pixel[0] - first_pixel[0], -surface.pix(20))
        np.testing.assert_array_equal(second_frame[second_pixel[1], second_pixel[0]], green)
        np.testing.assert_array_equal(env.goal_position, original_goal)
        imageio.imwrite(output_dir / "goal_after_camera_move.png", second_frame)


if __name__ == "__main__":
    unittest.main()

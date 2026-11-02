"""Behavioral checks for slower native traffic and measured ego interactions."""
from copy import deepcopy
import unittest

import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
import torch

from baseline_DQN.config import ENV_CONFIG, DQN_CONFIG, training_settings
from baseline_DQN.environment import TrafficHighwayEnv


class TrafficEnvironmentTests(unittest.TestCase):
    def make_env(self, cars=20, render_mode=None):
        config = deepcopy(ENV_CONFIG)
        config["vehicles_count"] = cars
        env = TrafficHighwayEnv(env_config=config, render_mode=render_mode)
        self.addCleanup(env.close)
        return env

    def test_native_array_observations_and_sb3_contract(self):
        env = self.make_env()
        check_env(env, warn=True, skip_render_check=True)
        obs, info = env.reset(seed=7)
        self.assertEqual(obs.shape, (5, 5))
        self.assertEqual(obs.dtype, np.float32)
        self.assertTrue(env.observation_space.contains(obs))
        self.assertEqual(env.action_space.n, 5)
        self.assertEqual(len(env.traffic), 20)
        self.assertEqual(info["interaction_metrics"]["encounter_count"], 0)

    def test_seeded_reset_slow_sustained_targets_and_safe_lead(self):
        env = self.make_env()
        for seed in range(20):
            first, _ = env.reset(seed=seed)
            positions = np.array([car.position.copy() for car in env.traffic])
            speeds = np.array([car.speed for car in env.traffic])
            for car in env.traffic:
                self.assertGreaterEqual(car.speed, 15)
                self.assertLessEqual(car.speed, 20)
                self.assertEqual(car.speed, car.target_speed)
            self.assertAlmostEqual(env.lead_vehicle.position[0] - env.highway.vehicle.position[0], 35)
            self.assertEqual(env.lead_vehicle.lane_index, env.highway.vehicle.lane_index)
            self.assertAlmostEqual(env.closest_front_gap(), 30)
            for i, car in enumerate(env.highway.road.vehicles):
                for other in env.highway.road.vehicles[i + 1:]:
                    if car.lane_index == other.lane_index:
                        self.assertGreater(abs(car.position[0] - other.position[0]), (car.LENGTH + other.LENGTH) / 2)
            second, _ = env.reset(seed=seed)
            np.testing.assert_array_equal(first, second)
            np.testing.assert_array_equal(positions, np.array([car.position for car in env.traffic]))
            np.testing.assert_array_equal(speeds, np.array([car.speed for car in env.traffic]))

    def test_idle_really_approaches_slow_lead_and_records_encounter(self):
        env = self.make_env(cars=1)
        env.reset(seed=7)
        initial_gap = env.closest_front_gap()
        _, reward, _, _, info = env.step(1)
        self.assertLess(env.closest_front_gap(), initial_gap)
        self.assertGreater(info["interaction_metrics"]["encounter_count"], 0)
        self.assertGreater(info["interaction_metrics"]["close_following_steps"], 0)
        # Reward remains the native highway task; no custom interaction bonus.
        self.assertAlmostEqual(reward, env.highway._reward(1))
        self.assertEqual(info["interaction_metrics"]["overtake_count"], 0)

    def test_slowing_while_following_is_measured_from_actual_speed(self):
        env = self.make_env(cars=1)
        env.reset(seed=7)
        env.step(1)
        _, _, _, _, info = env.step(4)
        self.assertGreater(info["interaction_metrics"]["following_brake_steps"], 0)

    def test_overtake_requires_real_clearance_and_counts_once(self):
        env = self.make_env(cars=1)
        env.reset(seed=7)
        ego, lead = env.highway.vehicle, env.lead_vehicle
        lead.position = ego.position + np.array([2.0, 4.0])
        lead.on_state_update()
        env._update_interactions(None, ego.speed)
        self.assertEqual(env.metrics["overtake_count"], 0)
        lead.position = ego.position + np.array([-8.0, 4.0])
        lead.on_state_update()
        env._update_interactions(None, ego.speed)
        env._update_interactions(None, ego.speed)
        self.assertEqual(env.metrics["overtake_count"], 1)

    def test_collision_and_timeout_use_native_endings(self):
        env = self.make_env(cars=1)
        env.reset(seed=7)
        env.highway.vehicle.crashed = True
        _, _, terminated, truncated, info = env.step(1)
        self.assertTrue(terminated and info["is_collision"])
        self.assertFalse(info["is_completed"])
        env.reset(seed=7)
        env.highway.time = 29.0
        _, _, terminated, truncated, info = env.step(1)
        self.assertFalse(terminated)
        self.assertTrue(truncated and info["is_completed"])

    def test_render_records_five_physics_frames_per_decision(self):
        env = self.make_env(cars=1, render_mode="rgb_array")
        env.reset(seed=7)
        first = env.render()
        frames = []
        env.highway.viewer.frame_callback = lambda frame: frames.append(frame.copy())
        env.step(1)
        env.render()
        self.assertEqual(len(frames), 5)
        self.assertEqual(first.shape, (360, 1200, 3))
        self.assertFalse(np.array_equal(first, frames[-1]))
        env.highway.viewer.frame_callback = None

    def test_requested_dqn_hyperparameters_are_exact(self):
        expected = {"learning_rate": 5e-4, "buffer_size": 15000, "learning_starts": 200,
                    "batch_size": 32, "gamma": 0.8, "train_freq": 1, "gradient_steps": 1}
        self.assertEqual({key: DQN_CONFIG[key] for key in expected}, expected)


class QuickstartTests(unittest.TestCase):
    def test_native_traffic_observations_rewards_and_trajectories_are_preserved(self):
        _, configuration, steps = training_settings("quickstart")
        self.assertEqual(steps, 20_000)
        env = TrafficHighwayEnv(**configuration)
        reference = gym.make("highway-fast-v0")
        self.addCleanup(env.close)
        self.addCleanup(reference.close)
        for seed in (0, 7, 42):
            observed, _ = env.reset(seed=seed)
            expected, _ = reference.reset(seed=seed)
            np.testing.assert_array_equal(observed, expected)
            for actual_car, native_car in zip(env.highway.road.vehicles, reference.unwrapped.road.vehicles):
                np.testing.assert_array_equal(actual_car.position, native_car.position)
                self.assertEqual(actual_car.speed, native_car.speed)
                self.assertEqual(actual_car.target_speed, native_car.target_speed)
            for action in (1, 3, 0, 1, 2, 4, 1, 2, 3, 1):
                observed, reward, terminated, truncated, _ = env.step(action)
                expected, native_reward, native_terminated, native_truncated, _ = reference.step(action)
                np.testing.assert_array_equal(observed, expected)
                self.assertEqual((reward, terminated, truncated),
                                 (native_reward, native_terminated, native_truncated))
                if terminated or truncated:
                    break

    def test_constructed_dqn_has_the_documented_network_and_update_schedule(self):
        torch.set_num_threads(1)
        parameters, configuration, _ = training_settings("quickstart")
        env = TrafficHighwayEnv(**configuration)
        self.addCleanup(env.close)
        model = DQN("MlpPolicy", env, seed=0, **parameters)
        linear = [layer for layer in model.q_net.q_net if isinstance(layer, torch.nn.Linear)]
        self.assertEqual([(layer.in_features, layer.out_features) for layer in linear],
                         [(25, 256), (256, 256), (256, 5)])
        self.assertEqual(model.target_update_interval, 50)
        self.assertEqual(model.n_envs, 1)
        self.assertEqual(model.train_freq.frequency, 1)
        self.assertEqual(model.gradient_steps, 1)

    def test_recording_does_not_change_the_native_policy_trajectory(self):
        _, configuration, _ = training_settings("quickstart")
        env = TrafficHighwayEnv(render_mode="rgb_array", **configuration)
        reference = gym.make("highway-fast-v0")
        self.addCleanup(env.close)
        self.addCleanup(reference.close)
        env.reset(seed=0)
        reference.reset(seed=0)
        first = env.render()
        self.assertEqual(first.shape, (360, 1200, 3))
        frames = []
        env.highway.viewer.frame_callback = lambda frame: frames.append(frame.copy())
        for action in (1, 3, 0):
            actual = env.step(action)
            expected = reference.step(action)
            env.render()
            np.testing.assert_array_equal(actual[0], expected[0])
            self.assertEqual(actual[1:4], expected[1:4])
        self.assertEqual(len(frames), 15)
        self.assertFalse(np.array_equal(first, frames[-1]))


if __name__ == "__main__":
    unittest.main()

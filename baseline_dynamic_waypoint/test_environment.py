"""Behavioral checks for the dynamic-waypoint task: slot geometry, motion, hold logic, rendering."""
from collections import Counter
from copy import deepcopy
import unittest

import numpy as np
from stable_baselines3.common.env_checker import check_env

from baseline_dynamic_waypoint.config import GOAL_CONFIG, RESULTS_DIR, SLOTS
from baseline_dynamic_waypoint.environment import DynamicGoalHighwayEnv
from baseline_dynamic_waypoint.scripted import scripted_action


class DynamicGoalTests(unittest.TestCase):
    def make_env(self, hold_seconds=1.0, render_mode=None):
        goal = deepcopy(GOAL_CONFIG)
        goal["hold_seconds"] = hold_seconds
        env = DynamicGoalHighwayEnv(goal_config=goal, render_mode=render_mode)
        self.addCleanup(env.close)
        return env

    def reset_with_slot(self, env, slot, first_seed=0):
        """The first seed whose sampled anchor can host this slot."""
        for seed in range(first_seed, first_seed + 50):
            try:
                return env.reset(seed=seed, options={"slot": slot, "anchor_index": 0})
            except ValueError:
                continue
        self.fail(f"No seed could host slot {slot}")

    def place_in_slot(self, env, longitudinal=0.0):
        """Teleport ego into the slot, moving at the anchor's speed."""
        lane = env.highway.road.network.get_lane(env.slot_lane_index)
        slot_s, _ = lane.local_coordinates(env.goal_position)
        ego = env.highway.vehicle
        ego.position = lane.position(slot_s + longitudinal, 0).astype(float)
        ego.heading = 0.0
        ego.speed = ego.target_speed = env.anchor.speed
        ego.lane_index = ego.target_lane_index = env.slot_lane_index

    def test_sb3_contract_and_new_observation_keys(self):
        env = self.make_env()
        check_env(env, warn=True, skip_render_check=True)
        obs, info = env.reset(seed=7)
        self.assertTrue(env.observation_space.contains(obs))
        for key in ("anchor", "slot", "hold_progress"):
            self.assertEqual(obs[key].dtype, np.float32)
        self.assertEqual(obs["slot"].sum(), 1.0)
        self.assertEqual(SLOTS[int(obs["slot"].argmax())], info["slot"])
        np.testing.assert_allclose(obs["desired_goal"] - obs["achieved_goal"], obs["goal_delta"], atol=1e-6)

    def test_seed_replay_and_every_slot_is_sampled_only_where_valid(self):
        env = self.make_env()
        first = env.reset(seed=123)[1]
        second = env.reset(seed=123)[1]
        self.assertEqual((first["slot"], first["anchor_id"]), (second["slot"], second["anchor_id"]))
        np.testing.assert_array_equal(first["goal_position"], second["goal_position"])
        counts = Counter()
        for seed in range(200):
            _, info = env.reset(seed=seed)
            anchor_lane = env.anchor.lane_index[2]
            gap = env.anchor.position[0] - env.highway.vehicle.position[0]
            low, high = env.goal_config["distance_range"]
            self.assertTrue(low <= gap <= high)
            self.assertFalse(info["slot"] == "left" and anchor_lane == 0)
            self.assertFalse(info["slot"] == "right" and anchor_lane == env.env_config["lanes_count"] - 1)
            counts[info["slot"]] += 1
        self.assertEqual(set(counts), set(SLOTS))

    def test_slot_geometry_relative_to_anchor(self):
        env = self.make_env()
        offsets = env.goal_config["slot_offsets"]
        expected = {"front": (offsets["front"], 0), "back": (-offsets["back"], 0), "left": (0, -1), "right": (0, 1)}
        for slot, (dx, lane_step) in expected.items():
            _, info = self.reset_with_slot(env, slot)
            anchor = env.anchor
            self.assertEqual(info["goal_lane"], anchor.lane_index[2] + lane_step)
            self.assertAlmostEqual(info["goal_position"][0] - anchor.position[0], dx)
            self.assertAlmostEqual(info["goal_position"][1] - anchor.position[1], 4.0 * lane_step)

    def test_waypoint_moves_with_anchor_which_keeps_its_lane(self):
        env = self.make_env()
        for slot in SLOTS:
            _, info = self.reset_with_slot(env, slot)
            anchor, lane = env.anchor, env.anchor.lane_index
            offset = info["goal_position"] - anchor.position
            start = info["goal_position"].copy()
            for _ in range(30):
                _, _, terminated, truncated, info = env.step(1)
                np.testing.assert_allclose(info["goal_position"] - anchor.position, offset, atol=1e-6)
                self.assertEqual(anchor.lane_index, lane)
                if terminated or truncated:
                    break
            self.assertGreater(info["goal_position"][0] - start[0], 5.0)
            self.assertFalse(anchor.enable_lane_change)

    def test_success_needs_exactly_the_hold_time_and_leaving_resets_it(self):
        env = self.make_env(hold_seconds=0.5)  # 5 decisions
        self.reset_with_slot(env, "back")
        for step in range(1, 5):
            self.place_in_slot(env)
            _, reward, terminated, _, info = env.step(1)
            self.assertEqual(info["hold_steps"], step)
            self.assertFalse(terminated or info["is_success"])
            self.assertEqual(info["reward_components"]["hold"], env.reward_weights["hold"])
        # Leave the box for one decision: the count restarts from zero.
        self.place_in_slot(env, longitudinal=env.goal_config["radius"] + 2.0)
        _, _, terminated, _, info = env.step(1)
        self.assertEqual(info["hold_steps"], 0)
        self.assertFalse(terminated)
        for step in range(1, 6):
            self.place_in_slot(env)
            _, reward, terminated, _, info = env.step(1)
        self.assertTrue(terminated and info["is_success"])
        self.assertEqual(info["termination_reason"], "GOAL REACHED")
        self.assertEqual(info["reward_components"]["goal"], env.reward_weights["goal"])
        self.assertAlmostEqual(reward, sum(info["reward_components"].values()))

    def test_tolerance_box_is_strict_and_lane_specific(self):
        env = self.make_env()
        self.reset_with_slot(env, "left")
        self.place_in_slot(env, longitudinal=env.goal_config["radius"])
        _, _, _, _, info = env.step(1)
        self.assertEqual(info["hold_steps"], 0)
        # The anchor's own lane is beside the LEFT slot, so it must not count.
        self.place_in_slot(env)
        ego = env.highway.vehicle
        ego.position[1] += 4.0
        ego.lane_index = ego.target_lane_index = env.anchor.lane_index
        _, _, _, _, info = env.step(1)
        self.assertEqual(info["hold_steps"], 0)

    def test_collision_beats_success_and_anchor_loss_truncates(self):
        env = self.make_env(hold_seconds=0.1)
        self.reset_with_slot(env, "back")
        self.place_in_slot(env)
        env.highway.vehicle.crashed = True
        _, _, terminated, truncated, info = env.step(1)
        self.assertTrue(terminated and info["is_collision"])
        self.assertFalse(truncated or info["is_success"])
        self.reset_with_slot(env, "back")
        env.anchor.crashed = True
        _, reward, terminated, truncated, info = env.step(1)
        self.assertTrue(truncated and info["is_anchor_lost"])
        self.assertFalse(terminated or info["is_timeout"] or info["is_success"])
        self.assertEqual(info["termination_reason"], "ANCHOR LOST")

    def test_time_limit_is_truncation(self):
        env = self.make_env()
        env.reset(seed=7)
        env.highway.time = env.env_config["duration"]
        _, _, terminated, truncated, info = env.step(1)
        self.assertTrue(truncated and info["is_timeout"])
        self.assertFalse(terminated or info["is_anchor_lost"])

    def test_progress_reward_tracks_distance_to_moving_slot(self):
        env = self.make_env()
        _, initial = env.reset(seed=7)
        _, _, _, _, info = env.step(1)
        expected = env.reward_weights["progress"] * (initial["distance_to_goal"] - info["distance_to_goal"])
        self.assertAlmostEqual(info["reward_components"]["progress"], expected)

    def test_scripted_driver_can_hold_every_slot_for_five_seconds(self):
        """Feasibility: each slot type is reachable and holdable with native actions."""
        env = self.make_env(hold_seconds=5.0)
        for slot in SLOTS:
            successes = 0
            for first_seed in range(0, 40, 4):
                self.reset_with_slot(env, slot, first_seed)
                terminated = truncated = False
                while not (terminated or truncated):
                    _, _, terminated, truncated, info = env.step(scripted_action(env))
                successes += info["is_success"]
            self.assertGreater(successes, 0, f"Scripted driver never held {slot}")

    def test_slot_box_is_drawn_at_the_moving_goal(self):
        env = self.make_env(render_mode="rgb_array")
        self.reset_with_slot(env, "front")
        frame = env.render()
        surface = env.highway.viewer.sim_surface
        x, y = surface.pos2pix(*env.goal_position)
        np.testing.assert_array_equal(frame[y, x], [65, 245, 125])
        for _ in range(10):
            env.step(1)
        frame = env.render()
        x, y = surface.pos2pix(*env.goal_position)
        np.testing.assert_array_equal(frame[y, x], [65, 245, 125])
        output_dir = RESULTS_DIR / "verification"
        output_dir.mkdir(parents=True, exist_ok=True)
        import imageio.v2 as imageio
        imageio.imwrite(output_dir / "front_slot.png", frame)


if __name__ == "__main__":
    unittest.main()

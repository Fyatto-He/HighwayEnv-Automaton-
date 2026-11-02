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

    def test_slot_error_input_matches_slot_geometry(self):
        env = self.make_env()
        self.assertIn("slot_error", env.observation_space.spaces)
        scales = env.goal_config["slot_error_scale"]
        self.reset_with_slot(env, "left")
        self.place_in_slot(env, longitudinal=-2.0)  # 2 m behind the slot center
        ego = env.highway.vehicle
        ego.position[1] += 0.5
        ego.speed = env.anchor.speed - 3.0
        obs = env._observation(env.highway.observation_type.observe())
        np.testing.assert_allclose(obs["slot_error"], [2.0 / scales[0], -0.5 / scales[1], 3.0 / scales[2]], atol=1e-5)
        self.assertTrue(env.observation_space.contains(obs))
        goal = deepcopy(GOAL_CONFIG)
        goal["slot_error_features"] = False
        plain = DynamicGoalHighwayEnv(goal_config=goal)
        self.addCleanup(plain.close)
        self.assertNotIn("slot_error", plain.reset(seed=7)[0])

    def test_shaping_terms_telescope_and_sum_to_the_reward(self):
        env = self.make_env()
        _, info = self.reset_with_slot(env, "right")
        first_lateral, first_speed, _ = env.shaping_errors()
        self.assertGreater(first_lateral, 0.0)  # Ego starts outside the slot lane in this episode.
        totals = {"lane": 0.0, "speed_match": 0.0}
        for _ in range(40):
            _, reward, terminated, truncated, info = env.step(2)  # LANE_RIGHT toward the slot lane
            self.assertAlmostEqual(reward, sum(info["reward_components"].values()))
            for name in totals:
                totals[name] += info["reward_components"][name]
            if terminated or truncated:
                break
        lateral, speed, _ = env.shaping_errors()
        self.assertAlmostEqual(totals["lane"], env.reward_weights["lane"] * (first_lateral - lateral))
        self.assertAlmostEqual(totals["speed_match"], env.reward_weights["speed_match"] * (first_speed - speed))
        self.assertGreater(totals["lane"], 0.0)
        # Far from the slot the speed mismatch is not penalized at all.
        env.reset(seed=7)
        longitudinal, _ = env.slot_error()
        if abs(longitudinal) >= env.goal_config["shaping_range"] and info["slot"] != "front":
            self.assertEqual(env.shaping_errors()[1], 0.0)

    def test_front_overtake_targets_the_passing_lane_then_the_slot_lane(self):
        env = self.make_env()
        self.reset_with_slot(env, "front")
        ego, anchor = env.highway.vehicle, env.anchor
        network = env.highway.road.network
        road, start, anchor_lane = anchor.lane_index
        passing = anchor_lane - 1 if anchor_lane > 0 else anchor_lane + 1
        # Behind the anchor in its own lane: one lane away from the passing lane.
        ego.position = network.get_lane(anchor.lane_index).position(
            network.get_lane(anchor.lane_index).local_coordinates(anchor.position)[0] - 20, 0).astype(float)
        lateral, _, lead = env.shaping_errors()
        self.assertAlmostEqual(lateral, 4.0)
        self.assertAlmostEqual(lead, -20.0)
        # Beside the anchor in the passing lane: no lateral error yet.
        ego.position = network.get_lane((road, start, passing)).position(
            network.get_lane((road, start, passing)).local_coordinates(anchor.position)[0], 0).astype(float)
        lateral, _, lead = env.shaping_errors()
        self.assertAlmostEqual(lateral, 0.0)
        # Past the anchor by more than pass_margin: back to the slot lane; lead is capped.
        lane = network.get_lane(anchor.lane_index)
        ego.position = lane.position(lane.local_coordinates(anchor.position)[0] + 50, 0).astype(float)
        lateral, _, lead = env.shaping_errors()
        self.assertAlmostEqual(lateral, 0.0)
        self.assertEqual(lead, env.goal_config["slot_offsets"]["front"])
        # The overtake term pays the change in lead, and is zero for other slots.
        _, info = self.reset_with_slot(env, "front")
        _, _, start_lead = env.shaping_errors()
        _, _, _, _, info = env.step(3)  # FASTER
        _, _, lead = env.shaping_errors()
        self.assertAlmostEqual(info["reward_components"]["overtake"], env.reward_weights["overtake"] * (lead - start_lead))
        self.reset_with_slot(env, "back")
        _, _, _, _, info = env.step(3)
        self.assertEqual(info["reward_components"]["overtake"], 0.0)

    def test_section_4_checkpoints_keep_their_reward_without_overtake(self):
        from baseline_dynamic_waypoint.config import MODELS_DIR
        from baseline_dynamic_waypoint.evaluate import load_experiment
        model_path = MODELS_DIR / "ppo_hold1s_1m_shaped_seed7.zip"
        if not model_path.exists():
            self.skipTest("Section 4 checkpoint is not present.")
        model, env, _ = load_experiment(model_path)
        self.addCleanup(env.close)
        self.assertNotIn("overtake", env.reward_weights)
        self.reset_with_slot(env, "front")
        lateral, _, lead = env.shaping_errors()
        self.assertEqual(lead, 0.0)
        self.assertAlmostEqual(lateral, abs(env.slot_error()[1]))  # Slot lane, as when it was trained.
        obs = env._observation(env.highway.observation_type.observe())
        _, _, _, _, info = env.step(int(model.predict(obs, deterministic=True)[0]))
        self.assertNotIn("overtake", info["reward_components"])

    def test_old_reward_weights_keep_five_terms(self):
        from baseline_dynamic_waypoint.config import REWARD_WEIGHTS
        weights = {name: value for name, value in REWARD_WEIGHTS.items() if name not in ("lane", "speed_match", "overtake")}
        env = DynamicGoalHighwayEnv(reward_weights=weights)
        self.addCleanup(env.close)
        env.reset(seed=7)
        _, reward, _, _, info = env.step(1)
        self.assertEqual(set(info["reward_components"]), {"goal", "collision", "progress", "time", "hold"})
        self.assertAlmostEqual(reward, sum(info["reward_components"].values()))

    def test_checkpoints_without_slot_error_still_load_and_run(self):
        from baseline_dynamic_waypoint.config import MODELS_DIR
        from baseline_dynamic_waypoint.evaluate import load_experiment
        model_path = MODELS_DIR / "ppo_hold1s_seed7.zip"
        if not model_path.exists():
            self.skipTest("First 300k checkpoint is not present.")
        model, env, _ = load_experiment(model_path)
        self.addCleanup(env.close)
        obs, _ = env.reset(seed=7)
        self.assertNotIn("slot_error", obs)
        env.step(int(model.predict(obs, deterministic=True)[0]))

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

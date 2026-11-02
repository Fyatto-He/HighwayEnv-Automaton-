# Dynamic waypoints: reach and hold a slot of a moving car

This extends Marcos's [`baseline_waypoint/`](../baseline_waypoint/README.md) PPO baseline.
There, the goal is a fixed world point. Here, the goal is a **slot relative to a moving
traffic car** (the *anchor*): its FRONT, BACK, LEFT or RIGHT. Success means staying in
that slot for a configurable hold time (1 s, 3 s or 5 s). Nothing in `baseline_waypoint/`
or the installed simulator is modified: `DynamicGoalHighwayEnv` subclasses
`GoalHighwayEnv`, and the viewer, plots and CSV helpers are imported from it.

## Start here

1. `config.py`: only the differences from `baseline_waypoint/config.py`, each with its reason.
2. `environment.py`: anchor/slot sampling -> per-step slot update -> hold counter -> reward.
3. `scripted.py`: a rule-based reference driver, used as a feasibility test and yardstick.
4. `train.py`, `evaluate.py`, `test.py`: same structure and flags as Marcos's scripts,
   plus `--hold-seconds`.

## How the dynamic waypoint works

At each reset:

- The **anchor** is drawn uniformly from the traffic cars 30-120 m ahead of ego.
- A **slot** is drawn uniformly from those that lie on the road: no LEFT in lane 0,
  no RIGHT in the last lane.
- The anchor's MOBIL lane changes are switched off (`enable_lane_change = False`), so
  its LEFT/RIGHT slot can never move off the road. IDM still controls its speed and
  following, including reacting to ego.

After every simulator step, the slot is **recomputed in the anchor's lane frame**:

```python
lane = road.network.get_lane(anchor.lane_index)
s, _ = lane.local_coordinates(anchor.position)
FRONT: lane.position(s + 35, 0)        BACK: lane.position(s - 15, 0)
LEFT/RIGHT: adjacent_lane.position(s, 0)   # lane ID -1 / +1, the same as LANE_LEFT / LANE_RIGHT
```

The same `goal_position` attribute as the static task is updated in place, so the
distance, progress reward and drawing code all keep working.

**Success box.** Ego's center must be within ±3 m along the slot lane and ±1.5 m across
it. It must stay there for `hold_seconds × 10` consecutive decisions, and leaving the box
resets the count. A collision on the same step takes priority. If the anchor crashes or
leaves the road, the episode is **truncated** as `ANCHOR LOST`; this is not counted as
ego's failure or as a timeout.

### Two settings forced by the simulator's physics

Both were measured with `scripted.py` before training. They differ from the first plan.

| Setting | Value | Why |
| --- | --- | --- |
| Target speeds | 10-25 m/s in **2.5 m/s steps** | `MDPVehicle` sets FASTER/SLOWER to one grid step from the *current* speed, so the step size also limits acceleration (about step / 0.6 s). With 1 m/s steps ego brakes at only ~1.7 m/s²: the scripted driver crashed in 41-46% of episodes and never held FRONT. Marcos's 5 m/s steps cannot match 12-17 m/s traffic closely enough for 5 s holds (33% vs 45% scripted success over all slots, measured with a 15 m FRONT offset). |
| FRONT offset | **35 m** center-to-center (BACK: 15 m) | Once ego is ahead, the anchor's IDM driver *follows* ego. It keeps a gap of about 10 m + 1.5 s × speed, so it brakes and pushes a closer FRONT slot away. Scripted FRONT success with a 5 s hold, by offset: 15 m 0%, 25 m 20%, 30 m 33%, 35 m 65%. |

The episode lasts 25 s for every hold time, so the variants differ only in the hold requirement.

## What the agent sees

All of Marcos's keys are kept. `desired_goal` and `goal_delta` now move with the anchor.

| Key | Shape | Meaning |
| --- | --- | --- |
| `vehicles` | `(7, 5)` | Unchanged Kinematics traffic rows |
| `achieved_goal`, `desired_goal`, `goal_delta` | `(2,)` each | Ego, slot, and slot minus ego, in the episode frame |
| `anchor` | `(4,)` | Anchor x, y, vx, vy **relative to ego**, normalized with the Kinematics ranges |
| `slot` | `(4,)` | One-hot of FRONT, BACK, LEFT, RIGHT |
| `hold_progress` | `(1,)` | Consecutive in-slot decisions ÷ required, in [0, 1] |
| `slot_error` | `(3,)` | Slot minus ego along the slot lane ÷ 10 m, across it ÷ 4 m, anchor minus ego speed ÷ 10 m/s, clipped to [-1, 1] |

Without the anchor's velocity, the policy would see a moving target only as noise in
`goal_delta`. With it, the policy can anticipate where the slot is going.

`slot_error` was added after the first runs failed. With `goal_delta`'s 200 m scale, the
±3 m success box spans only ±0.015 of the input. With `slot_error`'s local scale it spans
±0.3, and the speed mismatch is given directly. It is on by default (`slot_error_features`
in `config.py`); `train.py --no-slot-error` leaves it out. Model sidecars written before
the flag existed load without it, so the first 300k checkpoints still evaluate. The
policy is still SB3's default `MultiInputPolicy` (53 inputs, or 50 without
`slot_error`, with 64-64 MLPs), with no custom network.

## Reward

```text
reward = 50 * success - 50 * collision + 0.1 * (previous_distance - distance)
         - 0.05 + 0.2 * in_slot
         + 1.0 * (previous_lateral_error - lateral_error)
         + 0.5 * (previous_speed_error - speed_error)
```

The first four terms are Marcos's. Distance is measured to the *current* slot position,
and it still telescopes because each step uses the previous step's distance. `hold`
pays 0.2 per decision spent in the box; even a 5 s hold earns at most 10, well below the
50-point goal bonus.

The last two terms, `lane` and `speed_match`, were added after the first runs: PPO never
moved sideways into a LEFT/RIGHT slot lane.
- `lateral_error` is ego's distance in meters from the slot lane's center, so each lane
  closed earns +4.
- `speed_error` is `|anchor speed - ego speed|`, weighted by a proximity factor that fades
  from 1 at the slot to 0 at 30 m (`shaping_range`). Far from the slot, ego should catch
  up, not match speed.

Both terms are differences of consecutive errors, like the progress term. Over an episode
they sum to the start error minus the end error, so they move credit earlier but cannot be
farmed by oscillating. `train.py --no-shaping` drops them. Checkpoints trained earlier keep
their five-term reward from their sidecar. Every term is in `info["reward_components"]`,
the CSVs and TensorBoard.

## Commands

Run from the repository root with the project's Python environment (`.venv`):

```bash
# Environment contract, slot geometry, motion, hold logic, scripted feasibility, rendering.
.venv/bin/python -B -m unittest baseline_dynamic_waypoint.test_environment -v

# Train (run name defaults to ppo_hold<seconds>s_seed<seed>) and evaluate on 50 paired seeds.
.venv/bin/python -B baseline_dynamic_waypoint/train.py --hold-seconds 3 --timesteps 300000 --quiet

# Re-evaluate: PPO vs random vs scripted on identical seeds.
.venv/bin/python -B baseline_dynamic_waypoint/evaluate.py --model baseline_dynamic_waypoint/models/ppo_hold3s_seed7.zip --output-name eval_hold3s

# Watch or record; --cycle-slots shows front, back, left, right in turn.
.venv/bin/python -B baseline_dynamic_waypoint/test.py --model baseline_dynamic_waypoint/models/ppo_hold3s_seed7.zip --cycle-slots
.venv/bin/python -B baseline_dynamic_waypoint/test.py --model baseline_dynamic_waypoint/models/ppo_hold3s_seed7.zip --cycle-slots --headless --record --output-name videos_hold3s
```

The hold time is saved in each model's JSON sidecar, and `evaluate.py`/`test.py` rebuild
the task from it. Debug resets accept `options={"slot": "left", "anchor_index": 0}`
(eligible anchors are numbered nearest first). In the viewer, the anchor has a magenta
outline and the slot is a green box at the actual success tolerance; it thickens while
ego is holding. The HUD shows the slot, the relative speed and a hold-progress bar.

## Results

See [RESULTS.md](RESULTS.md).

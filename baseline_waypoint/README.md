# Goal-conditioned PPO with slower waypoint traffic

An independent, educational baseline using the installed highway-env simulator.
There is no DFA logic. All source, dependencies, models, metrics, images and
videos for this experiment live in `baseline_waypoint/`. The installed simulator and the
other project demos are unchanged.

This folder was renamed from `baseline/` to `baseline_waypoint/`. New environments
use slower traffic at **12-17 m/s** and a lead car **35 m ahead**, making ego catch
up to nearby vehicles. The lead starts at 12 m/s; ego starts at 20 m/s.

The included PPO checkpoint was trained for **200,704 transitions on the original,
faster traffic**. Its **94% success** versus **34% for random actions**, with 2%
collisions for both, applies only to that original scenario. PPO has not been
retrained on the slower traffic in this update. Default playback applies the
new slow traffic to that existing checkpoint; use `--traffic saved` to reproduce
its original task. See [the original measured results and verification report](results/ppo_seed7/REPORT.md),
[learning curves](results/ppo_seed7/training_curves.png), and
[recorded lane-change examples](results/visualization/).

The unchanged PPO checkpoint was also evaluated on the **new slower traffic**
over 50 paired episodes: **80% goal success and 20% collisions**, compared with
**22% success and 78% collisions for random actions**. Its average return was
39.43 versus -22.92. This measures transfer to a changed task, without PPO
retraining. See the [slower-traffic evaluation report](results/slow_traffic_evaluation/REPORT.md)
and [three slower-traffic recordings](results/slow_traffic_visualization/),
which all reached their goals.

## Start here

Read these files in this order:

1. `config.py`: road, goal, reward and PPO settings.
2. `environment.py`: reset -> goal generation -> observation -> step/reward.
3. `train.py`: the PPO setup, episode logger and model saving.
4. `rendering.py`, then `test.py`: the world-coordinate target and policy playback.
5. `evaluate.py`: paired evaluation against random actions.

The only task class is `GoalHighwayEnv`, a Gymnasium wrapper around `highway-v0`.
`GoalViewer` is a small extension of highway-env's existing viewer. Training uses
one callback to collect metrics; there is no custom policy or neural network.

## Installation

Run commands from the project root. These PowerShell commands use a separate
Python environment and do not install anything into the project's existing one:

```powershell
uv venv baseline_waypoint/.venv --python .venv/Scripts/python.exe
uv pip install --link-mode copy --python baseline_waypoint/.venv/Scripts/python.exe -r baseline_waypoint/requirements.txt
```

`--link-mode copy` avoids Windows/cloud-filesystem hardlink errors. Without uv:

```powershell
python -m venv baseline_waypoint/.venv
.\baseline_waypoint\.venv\Scripts\python.exe -m pip install -r baseline_waypoint/requirements.txt
```

`requirements-lock.txt` records the exact packages used for the saved experiment.
To reproduce that environment, install `-r baseline_waypoint/requirements-lock.txt` instead.
Use this environment's Python explicitly, or activate it:

```powershell
.\baseline_waypoint\.venv\Scripts\Activate.ps1
```

After activation the requested entry points work as `python baseline_waypoint/train.py`
and `python baseline_waypoint/test.py`. Commands below use the explicit interpreter so
activation is not required.

## The task and goal generator

- Three straight lanes, one ego vehicle, eight surrounding IDM traffic cars.
- Surrounding cars receive both initial speeds and sustained IDM target speeds
  of 12-17 m/s. Setting only initial speed would let them accelerate again.
- A 12 m/s lead car starts 35 m center-to-center ahead in ego's starting lane
  (about 30 m bumper clearance). The entire traffic group is shifted together
  to preserve its ordering and spacing, avoiding overlaps from moving one car.
  Native IDM following and MOBIL lane-changing behavior remain active.
- Five native discrete actions: `0=LANE_LEFT`, `1=IDLE`, `2=LANE_RIGHT`,
  `3=FASTER`, `4=SLOWER`. The existing steering/speed controllers execute them.
- Target speeds are 15, 20 and 25 m/s; ego starts at 20 m/s.
- Physics and policy decisions both run at 10 Hz. The time limit is 18 seconds.
- Each reset creates new seeded traffic, picks a uniformly random lane ID
  (0, 1 or 2), then picks a position 110-170 m ahead of ego in that lane.
- The generator uses `lane.position(s, 0)` and validates road bounds, space for
  lane changes and the time budget. Reachability is geometric/kinematic; it
  cannot guarantee passage through every future traffic interaction.

The goal is the fixed world point `goal_position = [x, y]`. Success is exactly:

```python
np.linalg.norm(ego.position - goal_position) < radius
```

The default radius is **2.5 m**. Lane centers are 4 m apart, so a 5 m radius would
also accept driving past in an adjacent lane. You can change the radius in
`config.py`; there is deliberately no additional hidden lane-success condition.
Success uses the ego center, not contact between the ego rectangle and a marker.
At 25 m/s and 10 decisions/s, the ego advances at most approximately 2.5 m per
decision, less than the 5 m goal diameter. This makes aligned crossings detectable
at decision times; this baseline does not claim continuous-time disk detection
for grazing trajectories. Invalid speed/frequency/radius combinations are rejected.

Success or a collision terminates an episode. A collision takes priority if both
would occur together. Reaching the time limit without either event truncates it.
The native high-speed/right-lane rewards are discarded and replaced below.

For debugging, reset accepts `options={"goal_lane": 2, "goal_distance": 140}`.
Lane IDs in the CLI and metrics are zero based.

## What the agent sees

`KinematicsGoalObservation` was inspected in highway-env 1.12.1. Its implementation
reads only the ego vehicle and `vehicle.goal`; it does not include surrounding
traffic. Instead, this wrapper preserves the standard `Kinematics` traffic
observation and adds three small vectors:

| Dictionary key | Shape | Meaning |
| --- | --- | --- |
| `vehicles` | `(7, 5)` | Ego and six nearest cars; columns are presence, x, y, vx, vy |
| `achieved_goal` | `(2,)` | Ego position in an episode-fixed coordinate frame, scaled |
| `desired_goal` | `(2,)` | Target in that same frame, scaled |
| `goal_delta` | `(2,)` | Desired minus achieved; explicit target direction and offset |

The common origin is `[ego_x_at_reset, 0]`, with position scales `[200 m, 12 m]`.
Thus `desired_goal` stays fixed throughout an episode, `achieved_goal` moves, and
`goal_delta` shrinks toward zero. World positions in meters remain available in
`info`; the success condition always uses those unnormalized positions.

For traffic rows, native Kinematics uses ego-relative position **and velocity**;
the ego row retains its own absolute y and velocity. All rows are normalized using
the explicit ranges in `config.py`. Only the native ego x entry is replaced with
normalized displacement from the episode's starting x, avoiding saturation from
an irrelevant world coordinate. Empty traffic rows have presence=0; nearby cars
behind the ego are included. The goal vector is separate from the traffic array.

PPO's `MultiInputPolicy` uses SB3's default `CombinedExtractor`: flatten each array
and concatenate the four entries into **41 numbers**. The actor and critic each
receive that vector through their own two-layer, 64-unit MLP. The actor produces
five action probabilities; the critic predicts state value. Explicit `goal_delta`
is redundant but saves the network from first having to learn subtraction.

This is plain PPO, with no HER or `compute_reward` GoalEnv interface. SB3's
environment checker emits an advisory about the 2D `vehicles` array; the default
MultiInputPolicy correctly flattens that vector data.

## Reward, in full

```text
reward = 50 * success - 50 * collision
         + 0.1 * (previous_distance - current_distance) - 0.05
```

| Component | Purpose |
| --- | --- |
| `goal` | +50 once on safe success |
| `collision` | -50 on a crash; no simultaneous success bonus |
| `progress` | +0.1 per meter of net Euclidean progress, negative when moving away |
| `time` | -0.05 per decision, including terminal decisions |

Progress is measured from consecutive observations, not from the episode start
on every step. It therefore telescopes rather than paying repeatedly for the same
distance. Euclidean progress provides only weak lateral guidance far from the goal;
the terminal reward supplies most of the lane-selection signal.

`info["reward_components"]` contains the four weighted step terms.
`info["episode_reward_components"]` contains their accumulated episode totals.
Training prints the weights; testing prints every episode's component totals.
The renderer displays the current step terms. CSV and TensorBoard preserve them
for analysis. All four weights are centralized in `config.py`.

## Verify, train and inspect logs

```powershell
# Environment contract, seeded goals, rewards, endings and camera alignment.
.\baseline_waypoint\.venv\Scripts\python.exe -B -m unittest baseline_waypoint.test_environment -v

# A small end-to-end training/save/load/evaluation check.
.\baseline_waypoint\.venv\Scripts\python.exe -B baseline_waypoint/train.py --timesteps 2048 --num-envs 1 --run-name my_smoke --eval-episodes 5 --quiet

# Optional new PPO training on slower traffic; uses a fresh run name.
.\baseline_waypoint\.venv\Scripts\python.exe -B baseline_waypoint/train.py --timesteps 200000 --run-name ppo_slow_seed7 --quiet
```

Choose a new `--run-name` for each experiment: the script rejects existing names
so it cannot silently overwrite a saved run. Training collects complete rollouts;
with four environments and 256 steps per environment, the 200,000-step request
actually collects **200,704 transitions**. Seed defaults to 7. CPU inference uses
one Torch thread. Four spawned simulation processes improve throughput; use
`--num-envs 1` when multiprocessing is unavailable.

PPO hyperparameters live in `PPO_CONFIG`, including learning rate, rollout size,
batch size, discount, entropy weight and MLP dimensions. Seed reproducibility is
intended within the recorded package/platform setup, not guaranteed bit-for-bit
across Torch/platform versions.

Artifacts for a run named `ppo_seed7`:

```text
baseline_waypoint/models/ppo_seed7.zip               trained SB3 policy
baseline_waypoint/models/ppo_seed7.json              matching task configuration and versions
baseline_waypoint/results/ppo_seed7/run.json         seed, requested/actual steps and runtime
baseline_waypoint/results/ppo_seed7/training_episodes.csv
baseline_waypoint/results/ppo_seed7/progress.csv     PPO optimizer and rolling episode metrics
baseline_waypoint/results/ppo_seed7/events.out.tfevents.*
baseline_waypoint/results/ppo_seed7/training_curves.png
baseline_waypoint/results/ppo_seed7/evaluation_episodes.csv
baseline_waypoint/results/ppo_seed7/evaluation.json
```

Keep each model beside its JSON sidecar. Testing/evaluation load the saved task
configuration and expose an explicit traffic choice: `--traffic slow` (default)
overrides only `traffic_speed_range` and `lead_vehicle_gap` with the new defaults;
`--traffic saved` uses the checkpoint's unchanged original traffic settings.
The model sidecar is never rewritten by this choice. Evaluation stores the
effective environment separately from the original training environment.
Ctrl+C during learning saves the current policy and marks the run interrupted.

```powershell
.\baseline_waypoint\.venv\Scripts\tensorboard.exe --logdir baseline_waypoint/results
.\baseline_waypoint\.venv\Scripts\python.exe -B baseline_waypoint/plot_results.py --run-name ppo_seed7
```

TensorBoard is enabled by default; `--no-tensorboard` disables it. Plots show
rolling reward, success and collision rate over up to 100 completed episodes.
Those curves describe training behavior, which includes exploratory actions.

## Watch the goal and policy

```powershell
# Interactive window, three target lanes with slower traffic (the default).
.\baseline_waypoint\.venv\Scripts\python.exe -B baseline_waypoint/test.py --episodes 3 --cycle-goals

# Reproduce the existing checkpoint's original, faster traffic.
.\baseline_waypoint\.venv\Scripts\python.exe -B baseline_waypoint/test.py --episodes 3 --cycle-goals --traffic saved

# Fix one target 140 m ahead in lane ID 2.
.\baseline_waypoint\.venv\Scripts\python.exe -B baseline_waypoint/test.py --goal-lane 2 --goal-distance 140

# Save slower-traffic videos without overwriting the original recordings.
.\baseline_waypoint\.venv\Scripts\python.exe -B baseline_waypoint/test.py --headless --record --episodes 3 --cycle-goals --output-name visualization_slow
```

The ego is blue and surrounding cars are amber. The goal has a bright green crosshair, a `GOAL` label and a green
ring at the **actual tolerance radius**. A line connects ego and target. The HUD
shows distance in meters, elapsed time, lane, reward terms and the final outcome.
The target is drawn with `WorldSurface.pos2pix` and `WorldSurface.pix`, the same
camera/scale transform highway-env uses for cars. It stays fixed in world space
while moving across the image as the camera follows ego.

The extension reuses the native road, traffic and road-object drawing routines.
It does not patch library files or install a global viewer callback. Annotated
MP4s (10 fps) and PNGs are written to `baseline_waypoint/results/visualization/`, or a
different directory selected with `--output-name`. Each video holds the final
outcome for one second. The verification tests also save a before/after camera
pair under `baseline_waypoint/results/verification/`.
Playback starts at seed 10,006 by default. The historical recordings under
`results/visualization/` use the saved original traffic and require lane changes.
Fresh recordings with default options use slower traffic; their results should
be assessed separately from those original examples.

## Evaluation and interpretation

```powershell
# Evaluate the existing policy on the changed, slower traffic.
.\baseline_waypoint\.venv\Scripts\python.exe -B baseline_waypoint/evaluate.py --episodes 50 --output-name evaluation_slow

# Reproduce original-traffic evaluation in a separate output directory.
.\baseline_waypoint\.venv\Scripts\python.exe -B baseline_waypoint/evaluate.py --episodes 50 --traffic saved --output-name evaluation_saved
```

The default evaluation seed starts at 10,000, separate from the training seeds.
PPO uses deterministic actions. The comparison samples all five actions uniformly
for the random baseline. Both policies reset with the exact same list of seeds,
so initial traffic and goals are paired. Evaluation writes individual episode
rows and the summary; the full training command runs this automatically too.

- **Success rate:** fraction reaching the radius safely, over all episodes.
- **Collision rate:** fraction ending in a crash, over all episodes.
- **Average reward:** complete task return, including failures and timeouts.
- **Average time to goal:** simulated seconds, **successful episodes only**;
  JSON `null` if there were no successes.
- **Final distance:** Euclidean distance in meters at episode end, averaged over
  **all** episodes. Failed agents can overshoot a goal and end far away.
- Per-lane success rates, success grouped by the number of required lane changes,
  and timeout rates help expose lane bias or a policy that only stays in its lane.
- Average component returns explain which terms account for the total reward.

The original checkpoint's results cover one training seed and a modest, sparse
highway task. Its historical 94% success rate applies to original traffic;
the measured result for the unchanged policy with `--traffic slow` is 80%.
Inspect lane-conditioned results and videos as well as aggregate success.

Official references: [highway-env observations](https://highway-env.farama.org/observations/)
and [SB3 PPO / MultiInputPolicy](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html).

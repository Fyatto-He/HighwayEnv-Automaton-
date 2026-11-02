# DQN highway baseline and official quickstart reproduction

## Reproduce the highway-env website example

The default training preset now follows the
[official DQN quickstart](https://highway-env.farama.org/quickstart/#training-an-agent):
native `highway-fast-v0`, default kinematics observations, an MLP with hidden
layers `[256, 256]`, target-network updates every **50** transitions, and
**20,000** training transitions. Each training run uses **one** simulator and
one gradient update per transition after the initial 200-step warmup.
The other documented settings are learning rate `5e-4`, replay capacity 15,000,
batch size 32, discount 0.8, `train_freq=1`, and `gradient_steps=1`.

The earlier slower-traffic experiment differed: it injected 15-20 m/s traffic
and a nearby lead, enabled observations behind ego, and used SB3's default
`[64, 64]` network and 10,000-step target update interval. Its four-environment
run also performed only one gradient update per four collected transitions.
Those checkpoints describe a different experiment; their sidecars still load
their original settings.

Run these commands from the repository root:

```powershell
# Behavioral tests, including exact native simulator trajectory comparisons.
.\baseline_waypoint\.venv\Scripts\python.exe -B -m unittest baseline_DQN.test_environment -v

# Train and evaluate one complete reproduction (50 paired episodes per policy).
.\baseline_waypoint\.venv\Scripts\python.exe -B baseline_DQN/train.py --preset quickstart --seed 0 --run-name dqn_quickstart_seed0 --quiet

# Repeat with seeds 1, 2, 3 and 4, using the corresponding run names.
# Independent training seeds can run concurrently; keep --num-envs 1 per run.

# Watch the completed seed-0 policy interactively.
.\baseline_waypoint\.venv\Scripts\python.exe -B baseline_DQN/test.py --episodes 3

# Record the preselected seed-0 model on evaluation seeds 10000-10002.
.\baseline_waypoint\.venv\Scripts\python.exe -B baseline_DQN/test.py --headless --record --episodes 3 --output-name quickstart_seed0_videos

# Independently evaluate that model without overwriting its original results.
.\baseline_waypoint\.venv\Scripts\python.exe -B baseline_DQN/evaluate.py --episodes 50 --output-name quickstart_seed0_repeat

# Aggregate all five training seeds into a comparable plot and report.
.\baseline_waypoint\.venv\Scripts\python.exe -B baseline_DQN/summarize_quickstart.py
```

Completed models are `models/dqn_quickstart_seed0.zip` through
`models/dqn_quickstart_seed4.zip`, with JSON configurations beside them.
Each `results/dqn_quickstart_seedN/` contains episode CSVs, evaluation,
TensorBoard events, run metadata with package versions, and learning curves.
The combined report and plot are under `results/quickstart_comparison/`.
The default playback and standalone evaluation model is seed 0, chosen before
evaluation rather than selected for its score.
See the [measured five-seed report and videos](results/quickstart_comparison/REPORT.md)
and [combined training curves](results/quickstart_comparison/training_curves.png).

Seed 3 scored highest in the 50-episode evaluation. Additional recordings of
that model use fresh traffic seeds 20,000-20,002; all three outcomes are retained:

```powershell
.\baseline_waypoint\.venv\Scripts\python.exe -B baseline_DQN/test.py --model baseline_DQN/models/dqn_quickstart_seed3.zip --headless --record --episodes 3 --seed 20000 --output-name quickstart_seed3_videos
```

Training leaves native traffic positions, speeds, behaviors, observation,
actions, reward and episode length intact. Playback changes only display size
and colors and adds measured interaction labels. The tests compare seeded
observations, rewards, termination and trajectories directly with the plain
native environment, including when recording.

The website shows training curves for five seeds and one video. It does not
specify a held-out collision rate; the local report supplies measured results
rather than assuming that its demonstration guarantees every episode succeeds.
Results depend on random seeds and installed package versions. Native-task
performance must be assessed separately from the earlier slower-traffic task.

## Historical slower-traffic experiment

This folder uses the native `highway-fast-v0` driving task and Stable-Baselines3
DQN. Ego learns to drive through moving traffic using the simulator's existing
observations, controls and reward. It has no waypoint objective. The separate
`baseline_waypoint/` folder contains the goal-conditioned PPO experiment.

All DQN source, checkpoints, models, metrics and recordings belong here. The
runtime can reuse `baseline_waypoint/.venv/`, which already contains highway-env,
Stable-Baselines3, Torch, TensorBoard, matplotlib and video dependencies.

## Read these files first

1. `config.py`: the exact DQN hyperparameters and traffic settings.
2. `environment.py`: slowing traffic, placing a nearby lead car and measuring interactions.
3. `train.py`: DQN setup, episode logging, checkpoints and model saving.
4. `evaluate.py`: deterministic policy versus uniform random actions on paired seeds.
5. `rendering.py` and `test.py`: annotated playback and video recording.

## Install and run

Use the project root as the working directory. The renamed waypoint environment
can run this independent baseline without another dependency installation:

```powershell
# Behavioral tests, including genuinely slower surrounding traffic.
.\baseline_waypoint\.venv\Scripts\python.exe -B -m unittest baseline_DQN.test_environment -v

# A short training/save/load check; use a fresh run name.
.\baseline_waypoint\.venv\Scripts\python.exe -B baseline_DQN/train.py --preset slow-traffic --timesteps 2048 --run-name my_smoke --eval-episodes 5 --quiet

# Full training with four parallel simulators, followed by evaluation.
.\baseline_waypoint\.venv\Scripts\python.exe -B baseline_DQN/train.py --preset slow-traffic --timesteps 200000 --num-envs 4 --run-name dqn_highway_baseline --quiet

# Original single-simulator update schedule from the supplied example.
.\baseline_waypoint\.venv\Scripts\python.exe -B baseline_DQN/train.py --preset slow-traffic --timesteps 200000 --num-envs 1 --run-name dqn_single --quiet

# Watch the historical 150,000-step checkpoint in a window.
.\baseline_waypoint\.venv\Scripts\python.exe -B baseline_DQN/test.py --model baseline_DQN/models/dqn_highway_baseline_150000.zip --episodes 3

# Save annotated checkpoint videos and episode metrics without a window.
.\baseline_waypoint\.venv\Scripts\python.exe -B baseline_DQN/test.py --model baseline_DQN/models/dqn_highway_baseline_150000.zip --headless --record --episodes 3 --output-name checkpoint_150000_videos

# Evaluate the historical checkpoint over 50 matched traffic scenes.
.\baseline_waypoint\.venv\Scripts\python.exe -B baseline_DQN/evaluate.py --model baseline_DQN/models/dqn_highway_baseline_150000.zip --episodes 50 --output-name checkpoint_150000
```

Choose another `--run-name` when a previous run already exists. The scripts use
paths relative to their own directory, so artifacts remain inside `baseline_DQN/`.
Models load the matching JSON task configuration rather than silently applying
new defaults from `config.py`.

To create a dedicated DQN Python environment instead:

```powershell
uv venv baseline_DQN/.venv --python .venv/Scripts/python.exe
uv pip install --link-mode copy --python baseline_DQN/.venv/Scripts/python.exe -r baseline_DQN/requirements.txt
```

Then substitute `.\baseline_DQN\.venv\Scripts\python.exe` in the commands above.
The `--link-mode copy` option avoids Windows/cloud-filesystem hardlink failures.

## Why ego will meet other cars

The fast environment keeps its native setup: three lanes, twenty surrounding
vehicles, a 30-second episode, physics at 5 Hz and one discrete decision per
second. Ego starts at 25 m/s and its native target speeds are 20, 25 and 30 m/s.

On every seeded reset, `TrafficHighwayEnv`:

1. Samples each surrounding car's speed uniformly between **15 and 20 m/s**.
2. Sets both its actual `speed` and IDM `target_speed`. Changing only initial
   speed would let traffic immediately accelerate back to its old target.
3. Places a lead vehicle **35 m ahead of ego in ego's lane**, approximately
   30 m of bumper clearance with the native 5 m vehicle lengths.
4. Shifts the other traffic by the same longitudinal offset, preserving its
   original longitudinal ordering and spacing instead of dropping a car on top
   of another one.

At the starting relative speed, ego will close on that lead unless it changes
lane or slows down. This creates an actual traffic decision early in an episode.
Surrounding IDM/MOBIL vehicles retain their native acceleration, following and
lane-changing behavior. Their paths are not scripted around the learned policy.

Blue is ego, amber is surrounding traffic, and magenta highlights the initial
slow lead car. That lead can move or change lane like any other traffic vehicle.
It is a visual reference, not a goal or an obstacle with special collision rules.

## DQN settings

The training code uses the requested setup:

```python
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=5e-4,
    buffer_size=15000,
    learning_starts=200,
    batch_size=32,
    gamma=0.8,
    train_freq=1,
    gradient_steps=1,
    verbose=1,
)
model.learn(total_timesteps=200_000)
```

The runnable script centralizes those values in `DQN_CONFIG`, sets a fixed random
seed and uses the CPU. By default `--num-envs 1` retains one gradient update per
collected decision, as in the supplied example. SB3's remaining DQN settings use
their standard defaults unless explicitly listed in `config.py`.

`--num-envs 4` runs four independently seeded simulators in spawned processes to
use more CPU cores. Every value in `DQN_CONFIG` stays unchanged, including
`train_freq=1` and `gradient_steps=1`. With multiple simulators, one vector step
collects one transition from **each** simulator, then performs **one** gradient
update. Four simulators therefore mean one update per four collected transitions
after warmup, instead of one update per transition. This is an explicit training
schedule difference, not an identical reproduction of the single-simulator run.
The selected simulator count and update-to-transition ratio are saved in each
model's JSON metadata. The timestep budget counts transitions across all workers;
200,000 transitions with four workers means 50,000 vector steps. All workers are
closed before evaluation starts. On restricted Windows setups, spawning workers
may require permission to create local process communication pipes; use
`--num-envs 1` if those are unavailable.

`Kinematics` supplies a `(5, 5)` matrix: ego plus four nearby cars, with columns
`presence, x, y, vx, vy`. Its native normalized traffic features are ego-relative;
the ego row contains its own position and velocity. `see_behind=True` also makes
nearby cars behind ego eligible for observation. Missing rows have presence=0.
`MlpPolicy` flattens these **25 numbers** before passing them to DQN's MLP, which
predicts a Q value for each of five actions:

```text
0 = LANE_LEFT    1 = IDLE    2 = LANE_RIGHT    3 = FASTER    4 = SLOWER
```

During training, epsilon-greedy exploration sometimes chooses a random action.
Testing and evaluation use deterministic actions from the learned Q values.

## Native task reward

This baseline preserves highway-env's reward; interaction counters do not add
bonuses. Native reward favors high forward speed and right lanes and penalizes
collisions. With the default weights, the unnormalized sum is:

```text
-1.0 * collision + 0.1 * right_lane_score + 0.4 * speed_score
```

`speed_score` scales forward speed from 20 to 30 m/s into `[0, 1]`.
`right_lane_score` ranges from 0 for the leftmost lane to 1 for the rightmost.
The native environment maps the weighted sum from `[-1, 0.5]` to `[0, 1]` and
multiplies by the on-road indicator. A crash terminates the episode; reaching
30 seconds truncates it. A collision penalty in this normalized reward is not
the same numeric return as the waypoint baseline's explicit -50 penalty.

## What counts as interaction

Metrics are sampled at policy decision boundaries. The viewer also draws actual
intermediate physics frames, but it does not invent extra metric samples.

| Metric | Definition |
| --- | --- |
| Encounter count | Distinct cars encountered ahead in ego's lane with bumper gap in `[0, 30)` m |
| Close-following steps | Decision samples with a same-lane forward bumper gap below 30 m |
| Overtake count | An initially ahead car becomes fully behind ego with at least 5 m center clearance, at least 2 m lateral separation, and no collision |
| Lane changes | Changes in ego's observed physical lane ID |
| Following brake steps | Ego loses more than 0.5 m/s during a decision whose pre-step forward gap was below 30 m, without a collision |
| Minimum front gap | Smallest sampled same-lane forward bumper gap |

An encounter alone shows proximity, not a safe response. Look at the combination
of encounters, overtakes or braking, speed, collision rate and completed
episodes. Overtakes are based on tracked longitudinal order changes, not merely
issuing a lane-change action or counting traffic that was already behind ego.

## Logs, evaluation and recordings

Training writes a reloadable model, JSON configuration and 10,000-step
checkpoints under `models/`. Per-episode CSV data, DQN optimizer logs,
TensorBoard events and learning plots go under `results/<run-name>/`. The metadata records
actual training steps, seed, package versions, environment settings and runtime.

Evaluation compares the saved DQN with uniform random actions on the same
episode seeds, starting at 10,000 by default. This pairs initial traffic
conditions without claiming the two policies produce the same later trajectories.
Here there are no waypoint goals: a completed episode means surviving the time
limit. The report should include reward, collision rate, completion rate, average
speed and the interaction metrics above. Aggregate results alone do not prove
that the policy reacts correctly to every nearby vehicle.

The viewer displays the front gap, ego speed, encounters, overtakes, lane changes,
following-brake count and episode outcome. Its labels and gap line use native
world-to-pixel coordinates, so they follow the camera and moving cars correctly.

With `--record`, `test.py` uses highway-env's existing intermediate rendering
calls to save **five real physics frames per simulated second**. The DQN still
chooses one action per second. It writes MP4s plus initial/final PNGs and
`episodes.json` under `baseline_DQN/results/visualization/`; the last frame is
held for one second. No installed library file is modified.

```powershell
.\baseline_waypoint\.venv\Scripts\tensorboard.exe --logdir baseline_DQN/results
```

Official references: [highway-env fast environment](https://highway-env.farama.org/environments/highway/)
and [Stable-Baselines3 DQN](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html).

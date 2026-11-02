# Measured results (2026-11-02, macOS, seed 7, one run each)

## 1. Check of Marcos's static-waypoint baseline

Run on this Mac with `baseline_waypoint/requirements-lock.txt`; his code was not changed.

- `unittest baseline_waypoint.test_environment`: **14/14 pass**. His REPORT.md
  mentions 10; tests have been added since.
- His `ppo_seed7` checkpoint, re-evaluated on seeds 10,000-10,049, reproduces his
  published numbers **exactly**. Original traffic: PPO 94% vs random 34% success,
  2% vs 2% collisions, return 55.39 vs 9.50. Slow traffic: 80% vs 22%, return
  39.43 vs -22.92. Outputs: `baseline_waypoint/results/evaluation_{saved,slow}_mac/`.
- **New: PPO retrained from scratch on slow traffic** (`ppo_slow_seed7`, 200,704
  transitions, 205 s). Deterministic evaluation: **12% success, 32% collisions, 56%
  timeouts** (random: 22% / 78% / 0%). That is far below the 80% his original-traffic
  policy reaches on the same slow traffic without retraining. The training curve was
  still rising: rolling training success went from about 25-35% to 50% in the last
  25k transitions. So 200k transitions is too few for slow traffic, rather than the
  task being unlearnable.

## 2. Dynamic waypoint: PPO, 300,288 transitions per hold time

Each run used 3 subprocess environments and took about 500 s. Evaluation covers
50 paired episodes (seeds 10,000-10,049), with the same traffic, anchor and slot
for every policy: 15 FRONT, 19 BACK, 8 LEFT, 8 RIGHT.

| Hold | Policy | Success | Collision | Timeout | Anchor lost | Mean return | Mean longest hold |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 s | PPO | 2% | 88% | 10% | 0% | -46.93 | 0.12 s |
| 1 s | Random | 2% | 86% | 12% | 0% | -43.92 | 0.05 s |
| 1 s | Scripted | **76%** | 4% | 20% | 0% | 37.78 | 0.76 s |
| 3 s | PPO | 0% | 10% | 90% | 0% | -18.03 | 0.00 s |
| 3 s | Random | 0% | 88% | 12% | 0% | -45.95 | 0.05 s |
| 3 s | Scripted | **76%** | 4% | 20% | 0% | 40.03 | 2.28 s |
| 5 s | PPO | 2% | 36% | 62% | 0% | -23.55 | 0.13 s |
| 5 s | Random | 0% | 88% | 12% | 0% | -45.95 | 0.05 s |
| 5 s | Scripted | **76%** | 4% | 20% | 0% | 42.30 | 3.80 s |

Scripted success by slot is the same for every hold time: FRONT 53%, BACK 100%,
LEFT 75%, RIGHT 62%.

**PPO did not learn the task at this budget, for any hold time.** The scripted
driver shows that the task can be done with the native actions. Its success does
not drop from a 1 s to a 5 s hold: once it is in the box, holding is easy. The hard
part is arriving with the right speed.

### What the policies do (videos in `results/videos_hold{1,5}s/`)

- **1 s hold:** the policy drives fast toward the slot and does not brake. In the
  BACK example it passes through the box at +8 m/s and rear-ends the anchor at 3.7 s.
  It has learned to collect the progress reward but not to stop.
- **5 s hold:** the policy learned to be cautious. It stays about 25 m behind the BACK
  slot at low speed until the 25 s limit, which explains the 62% timeouts. 3 s behaves
  the same way (90% timeouts).
- Training-time (stochastic) success peaked at about 10-20% for 1 s and about 1% for
  3 s and 5 s.

### Likely causes (hypotheses, not yet tested)

1. **The precise slot error is nearly invisible to the policy.** `goal_delta` uses
   Marcos's 200 m longitudinal scale, so the ±3 m box spans only ±0.015 of the input.
   Static goals only need to be crossed, while holding needs fine relative position and
   speed. Adding a local-scale slot error (Δs, Δlateral, Δspeed in the slot lane's
   frame) is the most direct fix.
2. **The training budget is too small.** Static goal-reaching on slow traffic was
   still undertrained at 200k (section 1). This task is strictly harder.
3. **The hold reward is sparse.** The +0.2 per decision only starts inside a 6 m box.
   A speed-matching term near the slot, or a curriculum (closer anchors or shorter
   holds first), would give the policy a gradient before its first success.

Recommended next experiment: hypothesis 1 combined with 1M transitions, all else
unchanged, so the effect can be attributed.

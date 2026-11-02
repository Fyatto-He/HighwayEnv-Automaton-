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

Hypotheses 1 and 2 were then tested separately (section 3).

## 3. Follow-up: 1M transitions, with and without the `slot_error` input

This tests hypotheses 1 and 2 separately. Four runs, everything else unchanged: seed 7,
3 subprocess environments each, 1,000,704 transitions, about 36 min per run with all
four in parallel. Evaluation uses the same 50 paired seeds; the random and scripted
rows are identical to section 2.

| Hold | Input | Success | Collision | Timeout | Mean return | Successes by slot (F/B/L/R) |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 s | 50 inputs (original) | 2% | 58% | 40% | -30.38 | 0/15, 1/19, 0/8, 0/8 |
| 1 s | + `slot_error` | **12%** | 24% | 64% | -9.99 | 0/15, 4/19, 2/8, 0/8 |
| 5 s | 50 inputs (original) | 2% | 10% | 88% | -9.80 | 0/15, 0/19, 1/8, 0/8 |
| 5 s | + `slot_error` | **8%** | 34% | 58% | -16.80 | 0/15, 4/19, 0/8, 0/8 |
| any | Scripted reference | 76% | 4% | 20% | 37.8-42.3 | 8/15, 19/19, 6/8, 5/8 |

- **More training alone does not help.** Going from 300k to 1M transitions leaves
  success at 2%. Collisions fall (88% → 58% for 1 s) and turn into timeouts instead.
  Every training curve flattens by about 200-400k transitions.
- **`slot_error` helps a little**, from 1 to 6 successes for 1 s and from 1 to 4 for
  5 s, out of 50. That is a single seed and few episodes, so treat it as a hint, not
  an established effect. The gains are mostly BACK, the only slot that never needs
  an overtake and often needs no lane change.
- **FRONT never succeeds** in any PPO run. It needs an overtake, a merge and a slow-down.
- **Lane selection is the visible failure for LEFT/RIGHT.** In `videos_hold1s_1m_slotobs/`
  episode 4, ego matches the anchor's speed about 12 m from a RIGHT slot but stays two
  lanes away until the time limit. The Euclidean progress reward gives almost no signal
  for moving 8 m sideways, a weakness Marcos's README already notes for static goals.
- The one successful 5 s BACK video (`videos_hold5s_1m_slotobs/episode_02`) is a genuine
  5 s hold at matched speed (−0.7 m/s relative).

**Conclusion so far:** the inputs and the budget were not the main bottleneck; the
reward is. PPO finds "approach and match speed", but nothing in the reward guides it
to the right lane or through an overtake before the first success. Next candidates,
all of which change the reward: shaping on the lane and speed errors near the slot,
or a curriculum (closer anchors, BACK/LEFT/RIGHT first, then FRONT).

## 4. Reward shaping: lane and speed-match terms

The two telescoping terms are described in the README: +1.0 per meter of lateral
distance to the slot lane closed, and +0.5 per m/s of speed mismatch removed within
30 m of the slot. They are added on top of section 3's `slot_error` input. Everything
else is unchanged: 1,000,704 transitions, 3 environments, about 30 min per run with
four in parallel. Two training seeds (7 and 8) estimate the run-to-run spread;
evaluation uses the same 50 paired seeds as before.

| Hold | Training seed | Success | Collision | Timeout | Mean return | Successes F/B/L/R |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 s | 7 | **54%** | 22% | 24% | 20.38 | 0/15, 17/19, 5/8, 5/8 |
| 1 s | 8 | **52%** | 14% | 34% | 22.00 | 0/15, 18/19, 6/8, 2/8 |
| 5 s | 7 | **56%** | 14% | 30% | 28.57 | 0/15, 17/19, 6/8, 5/8 |
| 5 s | 8 | **54%** | 28% | 18% | 21.63 | 0/15, 16/19, 6/8, 5/8 |
| 1 s / 5 s | section 3 (no shaping, seed 7) | 12% / 8% | 24% / 34% | 64% / 58% | | |
| any | Scripted reference | 76% | 4% | 20% | 37.8-42.3 | 8/15, 19/19, 6/8, 5/8 |

- **Shaping is the change that mattered.** Success rises from 8-12% to 52-56%, and both
  seeds and both hold times agree within 4 points. Training success climbs steadily
  from about 300k to 550k transitions, then plateaus at about 50-57%.
- **Hold length barely matters once the policy arrives with the right speed.** The 5 s
  policies succeed as often as the 1 s ones; their mean longest hold is 2.9 s versus
  0.55 s.
- **Without FRONT, PPO is close to the scripted driver.** On BACK/LEFT/RIGHT (35
  episodes) PPO succeeds 27-28 times in three runs and 26 times in the fourth; the
  scripted driver succeeds 30 times, and the script reads the true simulator state
  while PPO only sees its observation.
- **FRONT is still 0/15 in all four runs.** Traced on the 15 FRONT seeds of the 5 s,
  seed-7 policy: ego never gets ahead of the anchor. In 14 of 15 episodes it settles
  about 20-40 m *behind* the anchor at matched speed and times out; the other episode
  is a collision. That is the BACK behavior applied to the wrong slot. Passing means
  a lane change plus about 3 s at higher speed next to slower traffic, a long detour
  that no reward term pays for before the first success.
- `speed_match` nets about -0.1 to -0.3 per episode, so it contributes little directly.
  Being telescoping, that means the weighted speed error ends slightly larger than it
  starts. The lane term (+3.4 per episode) is the shaping term doing the work.

**Remaining failure modes:** FRONT overtakes, and collisions (14-28%) mostly during
lane changes into LEFT/RIGHT slots. Next candidates: a curriculum that starts FRONT
with anchors already beside or slightly behind ego, or a shaping term on the
longitudinal *order* (ahead of the anchor versus behind it) for FRONT.

## 5. FRONT: overtake shaping

**Diagnosis of section 4's FRONT failure.** For a FRONT slot the target lane *is* the
anchor's lane, so the lane term charged -4 for pulling out to pass and paid it back only
after the merge, discounted and through traffic. The shaping actively discouraged the
maneuver FRONT needs.

**Change.** This only applies when the new `overtake` weight is present, so section 4
checkpoints reproduce exactly (tested).
- Until ego leads the anchor by 10 m, a FRONT slot's lane target is the nearest lane
  beside the anchor.
- A new telescoping `overtake` term pays 0.3 per meter of lead gained on the anchor,
  capped at the 35 m offset.
- Everything else is as in section 4: same 2 seeds × 2 hold times and the same evaluation.

| Hold | Training seed | Success | Collision | Timeout | Successes F/B/L/R | Section 4 success |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| 1 s | 7 | 40% | 30% | 30% | 0/15, 12/19, 4/8, 4/8 | 54% |
| 1 s | 8 | 40% | 28% | 32% | **3/15**, 6/19, 6/8, 5/8 | 52% |
| 5 s | 7 | **64%** | 34% | 2% | **4/15**, 17/19, 5/8, 6/8 | 56% |
| 5 s | 8 | 58% | 36% | 6% | **2/15**, 16/19, 6/8, 5/8 | 54% |

- **FRONT is now learnable.** Three of the four runs succeed on FRONT in evaluation
  (2-4 of 15; it was 0 in every earlier run). Training FRONT success was still climbing
  at 1M in those runs (final sixths: 30%, 22%, 5%).
  `videos_hold5s_1m_overtake_front_success/` shows a genuine overtake: ego passes the
  anchor in the lane beside it at +8.5 m/s, merges back 35 m ahead, and holds the
  slot for 5 s.
- **But collisions rose to 28-36%**, and the overall effect is mixed: 5 s holds gained
  2-8 points, 1 s holds lost 12-14 points. The 1 s seed-8 run also shows a large BACK
  drop (6/19, 12 timeouts). With two seeds per setting this is within plausible
  run-to-run noise, so no overall gain or loss is established.
- **The FRONT crashes expose a flaw in the new term.** Traced on the 15 FRONT seeds of
  the 5 s, seed-7 policy: all 11 collisions are rear-end hits at 18-23 m/s. **5 hit
  the anchor itself from behind in its own lane.** The overtake term pays for any gain
  on the anchor, including driving straight at it in its lane, so it rewards ramming as
  much as passing. The other 6 rear-end other cars ahead in ego's lane.

**Proposed fix (not yet run).** While ego is behind the anchor *in the anchor's own
lane*, cap the lead at a safe following gap, e.g. `min(lead, -15 m)`. Closing to a
safe distance still pays and passing in the next lane pays in full, but driving into
the anchor earns nothing. The potential stays continuous when ego changes lanes at
the cap, so it still telescopes.

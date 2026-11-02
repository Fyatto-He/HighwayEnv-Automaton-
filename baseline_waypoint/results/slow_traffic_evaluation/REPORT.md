# Existing PPO evaluated on slower traffic

The existing `ppo_seed7.zip` checkpoint was evaluated with `--traffic slow`
on **50 episodes, seeds 10,000-10,049**. Uniform random actions were evaluated
on those same seeded traffic scenes and goals. **PPO was not retrained** for
this evaluation; its weights still come from the original 200,704-transition run.

The changed scenario uses traffic speeds and sustained IDM target speeds of
**12-17 m/s**, with a **12 m/s lead car 35 m ahead** in ego's starting lane.
Ego starts at 20 m/s, so it catches the slower lead and must respond while
continuing toward its waypoint. Native traffic behavior remains active.

| Metric | Existing PPO | Random actions |
| --- | ---: | ---: |
| Episodes | 50 | 50 |
| Goal successes | 40 (80%) | 11 (22%) |
| Collisions | 10 (20%) | 39 (78%) |
| Timeouts | 0 | 0 |
| Average episode reward | 39.43 | -22.92 |
| Average time to goal, successful episodes only | 6.18 s | 6.63 s |
| Average final distance, all episodes | 15.60 m | 57.64 m |

The checkpoint's **94% original-traffic success rate** is a separate historical
result, described in [the original report](../ppo_seed7/REPORT.md). Its measured
success with the new slower traffic is **80%**, with a higher 20% collision rate.
The current policy often reaches its waypoint but is not consistently safe in
this more interactive scenario. These measurements cover one fixed checkpoint
and 50 evaluation seeds.

Three additional slower-traffic recordings, using seeds **10,006-10,008** and
cycling target lanes 0, 1 and 2, all reached their goals. They are examples rather
than a substitute for the complete 50-episode evaluation.

Artifacts:

- [Exact evaluation summary and effective/training configurations](evaluation.json)
- [All paired episode rows](evaluation_episodes.csv)
- [Three annotated MP4s and initial/final PNGs](../slow_traffic_visualization/)
- [Saved PPO checkpoint](../../models/ppo_seed7.zip)

To repeat without overwriting these results:

```powershell
.\baseline_waypoint\.venv\Scripts\python.exe -B baseline_waypoint/evaluate.py --episodes 50 --traffic slow --output-name slow_traffic_evaluation_repeat
```

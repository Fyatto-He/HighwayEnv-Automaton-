# Fifteen driving scenarios with live DFAs

Run from this project directory:

```powershell
.\.venv\Scripts\python.exe mainV4.py
```

Choose a selector tab (**1–5**, **6–10**, or **11–15**) and click a scene.
Keys **1–9** and **0** select the original ten scenes. **F1–F5** select scenes
11–15 directly; **[ / ]** browse selector pages without resetting the scene.
Press **P** to play/pause, **Space** for a 0.2 s step, **R** to reset/replay,
**- / =** to change playback speed, and **Escape** to close. Selection starts
paused. `--scenario 3 --play` loads and plays a particular example.
The window can be resized; its minimum size is 1160 × 780.

| # | Example | Observed progress |
| --- | --- | --- |
| 1 | Overtake left | Behind A → left of A → return ahead of A |
| 2 | Overtake right | Behind A → right of A → return ahead of A |
| 3 | Left pass canceled, then completed | Car A back → Car A left → Car A back → Car A left → Car A front |
| 4 | Right pass canceled, then completed | Car A back → Car A right → Car A back → Car A right → Car A front |
| 5 | Lane 1 to lane 3 | Lane 1 → lane 2 → lane 3 |
| 6 | Brake and match velocity | Matching → A slows → mismatch → match again behind A |
| 7 | Avoid personal-space violation | Follow close → A slows → react within 3 m of personal space → stabilize behind A; entering personal space fails |
| 8 | Merge | Entry lane → waypoint → main lane → outside Z and outside B personal |
| 9 | Pass two cars | Left of A → return ahead → approach B → left of B → return ahead |
| 10 | Reach W while avoiding Z | Lane 1 → lane 2 detour → waypoint outside Z |
| 11 | Yield and merge behind a faster car | Car A left → Car A back → Car A velocity |
| 12 | Work-zone detour and return | Lane 1 → lane 2 around Z → waypoint back in lane 1 |
| 13 | Slalom overtakes around a work zone | A back → A left → A front → B back → B right → B front |
| 14 | Match speed beside A, then merge ahead | Car A left → Car A velocity → Car A front |
| 15 | Two waypoints around a forbidden zone | Lane 1 → Waypoint W1 in lane 2 → Waypoint W2 in lane 3 |

The left panel uses the existing highway-env renderer. Ego is green, A is blue,
and B is amber. Red rectangles are personal space, blue rectangles are the side
regions, green rectangles are front regions, and amber rectangles are back
regions. W and Z are axis-aligned rectangles. The close-following scene also
plots the personal-space boundary beside the measured following distance.
Every state card shows its exact entry vocabulary, including the initial and
failure states. A green dot means that formula is currently true. States must
have a vocabulary witness; timing-only and "begin retry" nodes are removed.
The DFA remembers task progress between observations, so the active state can
retain its history after its entry condition becomes false. This is shown by
the live dot rather than presenting a historical condition as a current fact.
Condition labels between states, a separate retry rail, and red failure branches
show the actual transitions. Double borders identify accepting states. Current
states are blue and visited states are green. Acceptance confirmation appears
on the incoming edge, with an amber progress bar. The latest transition flashes
yellow for 0.85 simulation seconds. True predicates appear as chips below the
graph. Hover a loop arrow to inspect its guard in the persistent inspector.
The graph follows the active state; use the mouse wheel over it to inspect a
longer path. The observed path remains visible below the scene selector.

Scenes 3 and 4 have exactly three nodes, named directly from the state
vocabulary: `Car A back`, `Car A left` (or `Car A right`), and `Car A front`.
Cancellation returns to the same `Car A back` node, and retry traverses the same
side node again. There are no separate cancellation, recovery, retry, or B-speed
nodes. B's braking and ego's reaction remain in the deterministic traffic
controller. The DFA back edge is taken when the measured `Car A back` predicate
becomes true. Transition guards use only this car-state vocabulary, and the
accepting edge retains its alignment and 1.2 s confirmation requirement.

`mainV4.py` reuses `mainV2.py` for the environment, snapshots and road camera,
`simulator_dashboard.py` for Pygame drawing helpers, and highway-env's existing
lane/speed controllers. Existing entry points are preserved. This file defines
only these fifteen example monitors and scene scripts; there is no policy training,
task planner, or DFA generator. The DFA state selects the scene controller's
next maneuver. Timers script traffic slowdowns and the finishing confirmation.

The vocabulary is evaluated from the measured state at **every 1/30 s physics
tick**, independently of rendering. Each DFA takes at most one transition per
tick. Transition labels use the supplied vocabulary. Guards are deterministic
for every Boolean valuation; unmatched valuations retain the state. The final
edge has an explicit execution gate, described below. Terminal states are
absorbing and stop playback.
The retry explicitly requires reaching A's side region again. Its cancellation
back edge permits another interruption before completing the pass.

There is no separate finishing state. Acceptance requires the final guard to
stay true for **1.2 uninterrupted seconds after alignment**: the ego center must be
within 0.15 m of the target lane center, with heading error at most 0.02 rad.
Scenes 6 and 7 also wait for A to finish its scripted speed change. Losing the
guard or alignment resets the timer. This is a timed execution confirmation on
the accepting edge, rather than another vocabulary predicate or graph state.
The edge shows the wait and progress explicitly. Other transitions remain
vocabulary-driven.

In scenes 3 and 4, B starts **ahead** in the passing lane, matching ego's initial
18 m/s and acceleration. After ego reaches A's side at matching velocity and
continues for 0.7 s, B brakes toward 8 m/s. The traffic controller reacts to
`Car B back AND NOT Car B velocity` when B is slower than ego. Ego brakes in the passing lane,
falls far enough behind A, then returns. B accelerates toward 30 m/s after the
recovery; ego waits behind A until the passing lane has room before retrying.
There is no braking atom in the vocabulary: the guard detects the scripted
velocity mismatch, while the traffic controller supplies the actual braking.

The five additional scenes use the same vocabulary and controllers:

- **11:** Ego starts beside a faster A, yields, merges behind it, and restores
  velocity matching. `Car A personal` leads to failure.
- **12:** A work zone occupies lane 1. Ego detours through lane 2 and returns to
  a waypoint in its starting lane. Entering Z leads to failure.
- **13:** Ego passes A on the left, returns, and passes B on the right. A work
  zone blocks the left lane during the second pass. Entering Z leads to failure.
- **14:** Ego first matches A's velocity while staying beside it, then
  accelerates and merges ahead into A's lane. `Car A personal` leads to failure.
- **15:** Ego visits two named waypoint rectangles in order. W1 is in lane 2;
  W2 is in lane 3. Z blocks staying in lane 2 after W1. Reaching W2 before W1
  does not advance the DFA, and entering Z leads to failure.

`Waypoint W1` and `Waypoint W2` are separate instances of the existing waypoint
predicate, evaluated with the same rectangle containment rule. They do not add
a new predicate type. Every new graph has vocabulary labels, an initial state,
an accepting state, a failure branch, live highlighting, and reset/replay.

For a named car with heading θ, `s = (p_ego-p_car) · (cos θ, sin θ)` and
`d = (p_ego-p_car) · (-sin θ, cos θ)`, positive right. The evaluator and drawn
rectangles share these definitions:

| Predicate | Condition |
| --- | --- |
| Car A personal | `abs(s) <= H AND abs(d) <= W/2` |
| Car A left | `abs(s) <= H AND -3W/2 <= d < -W/2` |
| Car A right | `abs(s) <= H AND W/2 < d <= 3W/2` |
| Car A front | `H < s <= H + L_f(v_A) AND abs(d) <= W/2` |
| Car A back | `-H - L_b(v_A) <= s < -H AND abs(d) <= W/2` |
| Car A velocity | `norm(v_ego - v_A) < epsilon_v` |
| In lane i | highway-env's `lane.on_lane(p_ego)` |
| Waypoint W | `abs(x-x_W) <= half_length_W AND abs(y-y_W) <= half_width_W` |
| Avoid zone Z | `abs(x-x_Z) > half_length_Z OR abs(y-y_Z) > half_width_Z` |

`Geometry` defaults to `d_0 = 0.8 m`, `T = 0.35 s`, and
`epsilon_v = 0.45 m/s`, with `H = L_A/2 + d_0 + T*norm(v_A)`.
The front and back rectangles begin exactly at the personal-space edges and
extend by **`L_f(v_A) = 0.9 * norm(v_A)`** and
**`L_b(v_A) = 0.75 * norm(v_A)`** metres. The horizons are in seconds and apply
to each reference car's measured vector speed. At 16 m/s, the extensions are
14.4 m and 12 m; at 8 m/s they are 7.2 m and 6 m. They shrink to zero at rest.
Rendering and predicate evaluation use the same lengths, with no gap between
personal space and the front/back regions.
Lane width comes from the referenced car's lane (4 m in these scenes).
The same six car predicates are instantiated for B. Names pin the same car
throughout a run. Lane numbers count left to right; `original` and `main` are
fixed scene lane references, not new predicate types.

At 16 m/s, personal-space half-length is 8.9 m (previously 20.5 m). Lane width
is unchanged. Scene 7 uses **`L_b(v_A) = 0.14 * norm(v_A)`** so `Car A back`
only becomes true very close to the personal rectangle. Ego starts five metres
outside personal space, continues its close-driving approach when A brakes,
then uses a stronger braking response and tracks a gap at 90% of that compact
back extension. These are deterministic illustrative driving settings.

Overtaking examples start only **3 m outside A's personal space**, at 18 m/s
behind A's 16 m/s. Ego accelerates after moving out of the original lane strip.
It returns when just beyond the personal front edge and adjusts toward A's
speed to stay inside the short front region while finishing the maneuver.
The two-car example places B 64 m ahead of A and waits for `Car B back` before
starting the second pass.

Scene 8 uses an entry lane ending at x=260 m, W centered at (145, 8) with size
20 × 3.6 m, and Z centered at (253, 8) with size 16 × 4 m. The main lane is lane
2. Scene 10 uses W centered at (300, 4) with size 36 × 3.6 m and Z centered at
(210, 0) with size 16 × 3.6 m, leaving lane 2 outside Z. Rectangle edges belong
to the rectangle: waypoint boundaries count as reached; forbidden boundaries
count as violations.

Scene 12 uses W at (290, 0), size 44 × 3.6 m, and Z at (210, 0), size
40 × 3.6 m. Scene 13 places a 64 × 3.6 m work zone at (350, 0). Scene 15 uses
W1 at (165, 4), size 36 × 3.6 m; W2 at (270, 8), size 44 × 3.6 m; and Z at
(218, 4), size 30 × 3.6 m. Scenes 11 and 14 start ego at 18 m/s beside A at
22 m/s, illustrating yielding behind versus synchronizing and merging ahead.

The supplied vocabulary expresses bounded spatial regions, point lane
membership and instantaneous vector-velocity matching. It cannot assert
physical collision absence, sustained matching, a settled lane change, or an
unbounded “has passed” relation. Scenario 7's failure is explicitly a
personal-space violation, as are failures in scenes 11 and 14. Scenes 8, 10,
12, 13, and 15 fail when `NOT Avoid zone Z`.
Safety and progress guards are disjoint. Scenario 6 confirms the restored match
while behind A after settling; its illustrative scalar comparison holds an
acceleration/braking command for one second between speed comparisons. The
comparison is a distance trace, not another traffic vehicle or vocabulary atom.

Verify all fifteen real simulations, including the required observed state paths:

```powershell
.\.venv\Scripts\python.exe -B mainV4.py --verify
.\.venv\Scripts\python.exe -B -m unittest test_mainV4
```

The verification command exits unsuccessfully if a scene fails to accept,
skips a required stage, takes an unsatisfied guard, or physically crashes.
The tests also cover exact geometric boundaries, rotated frames, vector
velocities, speed-scaled front/back boundaries, vocabulary on every state,
every Boolean guard valuation, injected failure conditions,
the braking-triggered cancellation, close reaction distance, completed lane
alignment, uninterrupted confirmation, deterministic replay, named waypoint
order, selector pagination, keyboard shortcuts, and UI behavior.
To render the initial scene, every observed transition, and confirmation using
dummy SDL:

```powershell
.\.venv\Scripts\python.exe -B mainV4.py --verify --screenshots "$env:TEMP\highway-v4-previews"
```

Other options: `--list` lists all fifteen demos; `--scenario 15 --headless-steps
200` runs the selected scene without a window and prints its observed edges.

# v21.x — a ship-like open-water ego, and the eleven defects it exposed

**What this directory is.** The record of a single line of work: give the
open-water suite the AIS suite's *dimensionless* dynamics, so that evidence
transfers between the two. This is the **closed** record — what was found, what
was fixed, what it cost. Anything still open lives in `../OPEN_PROBLEMS.md`;
nothing is filed in both places.

Investigated 2026-09-15 → 2026-09-19. Baseline throughout is **v20.7.8-2**.

**Why it was worth doing.** The open-water ego turned inside **0.60** of its own
length. Every AIS vessel needs **2.20**. Conclusions drawn on a vehicle that
pirouettes do not obviously apply to ships — and `OPEN_PROBLEMS.md` §7.3's
headline evidence, the escape that made an abort "a save", needs
`ω = −0.227 rad/s`, which exceeds a ship-like `max_yaw_rate` of 0.1894 outright.

**The result nobody predicted.** Rescaling the ego was the *goal*; what it
actually produced was a fault-finder. Eleven defects surfaced, **nine of them
pre-existing and present in every config back to v20.7.5**. The rescale did not
create them — it moved the configuration far enough from the tuned baseline that
they stopped cancelling out.

---

## 1. The lineage

| version | intent | outcome |
|---|---|---|
| `v21.0` | Match the AIS suite's dimensionless dynamics | **failed.** `max_delta_yaw_rate` derived from AIS's yaw *window fraction*, which at `dt = 0.1` forces 0.947 rad/s² — **5× `max_yaw_rate`**. Exposed three latent bugs |
| `v21.1` | Yaw derived from angular *acceleration* (20 s to full helm) + `omega_window_ramp_time` | 2 of 3 reach goal; `cross_stbd_090` **livelocked 11h49m** |
| `v21.2` | `catch_goal_dist` scaled to the turn radius | **3/3 reach goal.** Trajectories are bit-identical prefixes of v21.1's |
| `v21.3` | Make it affordable: `yaw_rate_resolution` 0.1° → 0.5°, endpoints pinned | **4.9–5.0× faster** |
| `v21.4` | Remove the speed cap v21.2 introduced | max v 0.3038 → **0.4850**, frames 2500 → 1584 |
| `v21.5` | Candidate grid on a fixed lattice | max v → **0.5000**, straight-line yaw dither → **0.000000** |

Cumulative on `owenc_head_on`: **149.9 min → 18.8 min, 8.0×.**

---

## 2. The defects, and which were pre-existing

Nine of eleven were already in the code. They are marked **P**.

### 2.1 Unit confusions

**1. `max_delta_yaw_rate` used as a rate (P).** The stuck-escape commanded
`-config.max_delta_yaw_rate` — an angular **acceleration** as an angular
**velocity**. Invisible because every prior config has the ratio at 1.00
(v20.7.8-2) or 0.05 (AIS). v21.0 was the first with a ratio above 1, at 5.0.

**2. ω never clamped to `max_yaw_rate` (P).** `base.motion` assigns
`x[4] = u[1]` outright so it cannot leave the band; `dwa_ramped.motion` slews
toward a setpoint and can. Observed `ω = −0.36482` against a limit of 0.18939.

**3. An empty window reported as Guard A (P).** Once `|ω| > max_yaw_rate +
Δω·dt` the window inverts, `np.arange` yields nothing, and the Guard A raise
fires — reporting "the braking test rejected everything" when nothing was ever
tested. **Opposite diagnoses, indistinguishable from the message.** This is a
*third* meaning for Guard A that `OPEN_PROBLEMS.md` §2 does not list.

### 2.2 Parameters that did not survive the rescale

**4. `catch_goal_dist` is coupled to the turn radius.** Held at 0.417 ship
lengths while the turn radius grew 3.7×, so the real tolerance fell from **0.70
turning circles to 0.19**. A vessel cannot manoeuvre into a circle five times
smaller than the one it is flying. `cross_stbd_090` livelocked **11h49m and
9,996 iterations with zero net displacement**. Fixed at 1.36 circles, AIS's
ratio — and both routes to that number agree:

```
match AIS catchGD/turnR = 1.36  ->  1.36 x 2.640 = 3.60
match AIS catchGD/L     = 3.00  ->  3.00 x 1.20  = 3.60
```

**5. `predict_time_to_goal` has an upper bound nobody had stated.**
`calc_to_goal_cost` is an *angle*; the goal rollout travels `v·T`, and once that
exceeds `dist_localgoal` the rollout overshoots, the bearing flips ~180° and the
cost saturates at `gain·π`:

```
0.6 * pi = 1.88496   vs measured to_goal at v=0.5 of 1.88134
speed cap = dist_localgoal / predict_time_to_goal = 7.00 / 19.69 = 0.3555
measured max speed                                              = 0.3038
```

The ego was held to **71% of `max_speed`** — and I had attributed it to ship
physics ("the braking test acts as a speed governor"). The obstacle-free
reference falsified that: with the nearest obstacle 49 u away the governor
cannot bind, and the ego still capped at 0.3038.

Both bounds I first quoted were wrong. Measured instead — argmin v at three
moving states, `max_speed` 0.50:

```
T_to_goal      3     4    5-12    13     14
argmin v    0.45  0.50    0.50  0.40   0.35
```

Safe window **[4, 12]**; the 13.0 I first proposed was already capped. A sweep at
T = 5/8/11 differs by **under 1%** on every measure — inside the window the exact
value does not matter, only the margin to its edges.

> ⚠ **Superseded 2026-09-20 — this window is wrong, and 8.0 was a bad choice.**
> The only criterion here is *"does the rollout overshoot and cap the speed?"*.
> `predict_time_to_goal` also sets the stiffness of `to_goal_cost` against the
> obstacle barrier, and on that axis 8.0 is **70× too stiff**: it is what causes
> the 0.000 grazes (item 12) and the overtake station-keeping (item 13). The
> working value is **2.0** — *below* this "safe window", whose lower edge of 4.0
> is the worst value measured. See `../OPEN_PROBLEMS.md` §5.1. The sweep showing
> "<1% on every measure" is consistent: 5, 8 and 11 are all far into the
> goal-dominated regime, so all three graze identically.

### 2.3 Geometry the planner could not express

**6. `ω = 0` was never sampleable (P).** `np.arange(dw[0], dw[1]+1e-6, res)`
anchors the grid at the window's **lower edge**, which moves with the state. Over
a 1584-step run:

```
max_speed (0.500) was a sample on    71 steps ( 4.5%)
omega = 0 ("hold course")             0 steps ( 0.0%)
```

**Not once in 1,584 steps.** On a straight transit the planner was choosing
between a slight port and a slight starboard turn every step. "Maintain course
and speed" — what COLREGs Rule 17 requires of a stand-on vessel — **was not a
representable action.** Fixing it took straight-line yaw from 0.000560 to
**exactly 0.000000**, for +1.2% wall-clock.

**7. Theta\* snaps both endpoints to grid nodes (P).** `calc_xy_index` rounds to
the nearest node, so the ego starts at the *declared* position while its path
begins at the *snapped* one. Scales with resolution:

| scenario | res | start offset | goal offset | `catch_goal_dist` |
|---|---|---|---|---|
| `owenc_head_on` | 1.0 | 0.707 | 0.707 | 0.5 |
| `ny_hist_215804000` | 50.0 | **26.259** | **31.987** | 57.0 |

At the AIS grid the goal offset is **56% of the catch radius** and the start
offset ~25 m — more than a ship length before the vessel moves. An **accuracy**
defect, not a termination one: the ego steers to the snapped point and the catch
is measured against that same point, so the loop is self-consistent.

### 2.4 Hull models that disagreed

**8. Seven hull sites, and the count was wrong twice (P).** `OPEN_PROBLEMS.md`
item 1 called `closest_obstacle_on_curve` "the one hull site still on the disc".
It was not. The complete inventory, audited 2026-09-19:

| # | site | status |
|---|---|---|
| 1 | `_swept_hull_extent` (Stage 1) | ✅ item 1, `194aacf` |
| 2 | `calc_trajectory_clearance_and_collision` | ✅ pre-existing |
| 3 | `check_collision_at_current_position...` | ✅ `b128c9b` |
| 4 | `plot_robot` | ✅ draws the true rectangle |
| 5 | `encounter_report.py` | ✅ fixed 2026-09-19 |
| 6 | `algorithm.py` `_build_collision_event` | ✅ fixed 2026-09-19 |
| 7 | `..._no_time_decay.py` | disc-only; v20.7.6 ablation, not in use |

Site 5 scored capsule runs against a disc hull, turning grazes into apparent
collisions: `head_on` read **−0.381** as a disc and **0.000** as a capsule. Site
6 logged the collision *threshold* using the disc radius — 3× the one that fired;
detection was already correct, so reporting only.

**The real fix is the inventory**, not the individual sites. The count had been
wrong twice because there was no list.

### 2.5 Measurement that was not measuring

**9. The suite was scoring free passes.** The manifest hard-codes
`ego_speed: 0.5` and places every target to meet the ego at (50.51, 40.61) at
t = 90 s. A slower ego arrives late and the encounter never happens —
`cross_stbd_090` at v21.2 had a minimum separation of **14.812 u** against a
1.15 u threshold. Fixing the speed cap recovered most of it (68.4% late → 5.8%),
and the suite was regenerated from a v21.5 reference. Which cases a delay breaks
depends on geometry:

```
miss = |v_e| * d * sin(angle between v_e and v_e - v_t)

collinear (head-on, overtaking)  0.00   unaffected -- still meet exactly
045 crossings                    0.87   still force action
090 and 135 crossings       1.60/2.05   NO ACTION NEEDED -- invalid
```

An earlier version of that arithmetic treated the delay as a perpendicular miss
for *every* geometry, giving `head_on` a 4.57 u CPA — nonsense for two vessels
on the same line. The sine factor is what makes collinear cases immune.

**10. `--t-meet` (new).** The shipped `t_meet` values were tuned for an ego at
0.5. A different ego reaches a different point at the same time, which pushed
`overtaken_ego_fast`'s target spawn to (50.0, 100.72) against a map edge of 99.5.
Max feasible is 43.1 s against the shipped 50.0.

**11. The generated scenarios inherited `group: "baseline"`.** `run_paths` uses
`group` to place output, so every generated encounter filed itself under
`runs/<version>/baseline/`. The generator now derives it from the output
directory.

---

## 3. Observability, added because its absence cost a night

`max_sim_steps` (default 0 = unlimited), `progress_every_s` (default 30 s), and
a circling detector. The control loop printed **nothing**, so a run was opaque
until it ended: the livelock burned 11h49m before anyone could see the ego had
not moved in 4,000 iterations, and two processes died leaving no record of where
they stopped.

The detector deliberately does **not** use distance-to-goal — with static
obstacles the ego often must move *away* from the goal, so a goal-relative metric
cries wolf on correct behaviour. It compares, over a trailing window, how far the
ego **travelled** against how large a **region** it stayed within:

```
travelled / extent  ~1    running
                    >2.0  covering ground inside a box it never leaves
```

Two mistakes made getting there, both caught by testing rather than reasoning:
net displacement is the wrong measure (a closed loop only returns `net ≈ 0` at a
*whole number* of laps; at 1.5 laps it reads 0.21 and passes), and extent must be
the bounding box's **larger side**, not its diagonal (which inflates by √2 for
anything circular — precisely the case being detected).

---

## 4. Performance

| change | gain | how |
|---|---|---|
| Shared rollout + preallocation | **2.0×** | the goal rollout was re-integrating a bit-exact **prefix** of the obstacle rollout to produce one row that already existed |
| `yaw_rate_resolution` 0.1° → 0.5° | **4.9×** | the yaw window is only ±5.43°, so 0.5° still leaves 22 choices |
| Speed cap removed | **1.6×** | fewer frames for the same journey |
| **cumulative** | **8.0×** | 149.9 min → 18.8 min |

A correction worth keeping: I proposed 5° for the yaw resolution "because helm
orders come in 5° increments". Wrong yardstick — the window is ±5.43°, so 5°
would give **three** candidates. Resolution must be set against the window width,
not helm practice.

---

## 5. What this cost, and the lesson

Two wasted nights: **11h49m** on the livelock (no step cap, no progress output)
and **2h03m** on a reference run whose premise a later fix invalidated.

But the pattern worth recording is a reasoning one. Across this work I proposed
mechanisms that measurement then contradicted:

| symptom | my explanation | verdict |
|---|---|---|
| ego capped at 0.30 | braking test as speed governor | **falsified** — obstacle-free it still capped |
| ego capped at 0.4850 | local goal closing near the goal | **falsified** — flat from step 300, nowhere near the goal |
| 0.000 grazes | barrier tuned for the disc hull | **falsified** — centre distance identical |
| 0.000 grazes | `side_cost_gain` too weak | **falsified** — 5.8× gain changed nothing |
| 0.000 grazes | Gaussian barrier flat at contact | **falsified** — quadratic gave 0.000 too |

Every one was plausible; every one was wrong. In each case the data was already
on disk. The specific failure mode worth naming: **a single-step cost
decomposition is not a prediction about a re-planning controller.** The
decomposition said `side_cost_gain = 2.11` would flip the `head_on` decision; the
closed loop said no, because whatever restores the graze at the next step
outweighs the one-step margin. `OPEN_PROBLEMS.md` §7.3's reachability grid has
exactly the same shape and deserves the same caution.

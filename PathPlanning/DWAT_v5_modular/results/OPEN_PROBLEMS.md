# Open problems: one feasibility test, seen from four directions

Four investigations converged on the same twelve lines of code:

- the open-water **"ego is overtaken"** work (`v20.7.9_ramped_window/`),
- the **New York AIS suite** (`ny_ais_suite/`), whose ships abort or ground,
- an old question about whether the cost function is **already doing CPA/TCPA**,
- the **stern-sector** question (slowing while overtaken goes unpenalised).

They are not four problems. They are four views of a single defect, and §1 is
the frame that makes that visible. **Read §1 and §3 and you have the whole
document**; the rest is evidence and bookkeeping.

**Resuming in a new conversation?** §4 is the current state of every known
failure. §5 is the backlog, grouped. §6 is what to do next.

---

## 1. The frame: two filters in series, and they model different physics

Every control step, `calc_control_and_trajectory_merged` runs the candidate grid
through **two independent tests**, in this order:

```
        451–891 sampled (v, ω) candidates
                     │
    ┌────────────────▼─────────────────────────────────┐
    │ STAGE 1 — braking admissibility                  │  closest_obstacle_on_curve
    │ "could I brake to rest before hitting something?"│
    │   reject if  v² + 2·a·v·dt > 2·a·dist            │
    │   · obstacles FROZEN at current position         │
    │   · hull = circumscribed DISC                    │
    │   · scans the whole arc, NO time limit           │
    └────────────────┬─────────────────────────────────┘
                     │ survivors only
    ┌────────────────▼─────────────────────────────────┐
    │ STAGE 2 — score; inf if it collides              │  calc_trajectory_clearance_
    │   · obstacles EXTRAPOLATED  ob_pos + ob_vel·t    │    and_collision
    │   · hull = CAPSULE when configured               │
    │   · bounded by predict_time_obstacle             │
    └────────────────┬─────────────────────────────────┘
                     │
              pick argmin cost
```

**Stage 1 and Stage 2 disagree about what a ship is and what the world does, on
four independent axes — and Stage 1 is the wrong one on all four.** One is now
closed (`194aacf`, 2026-09-15); the fourth was added 2026-09-19:

| axis | Stage 2 (cost) | Stage 1 (feasibility) | backlog |
|---|---|---|---|
| **obstacle motion** | `ob_pos + ob_vel·t`, relative motion | **frozen** — the function takes no `ob_vel` argument at all | item **5** |
| **ego hull** | capsule when configured, half-beam `W/2` | ✅ **fixed** — honours `robot_collision_shape` since `194aacf`. *Was* always the circumscribed disc `sqrt(L²+W²)/2`, **3.16×** wider abeam for a 19 × 6.33 m ego | item **1** ✅ |
| **look-ahead** | bounded by `predict_time_obstacle` | **unbounded** — "no span restriction", scans the full circle | item **9** (new) |
| **ego's own path** | the ramped rollout — the spiral the ego actually flies | **a constant-velocity arc at the SETPOINT**, radius `\|v/ω\|`. Across one window that is radii of 1.0–10.0 u, none of them the ego's | item **11** (new) |

That table *is* the problem statement. Everything in §5 is either repairing one
of those rows, replacing the whole of Stage 1, or cleaning up inputs so the
repair can be measured.

**A caution the closed row leaves behind.** The 3.16× figure is the gain *abeam*.
Along the bow — the direction a braking test cares most about — a capsule buys only
`sqrt(L²+W²)/2 − L/2` = **0.513 m**, because the bow still reaches `(L−W)/2` ahead
of the hull centre. Quoting 3.16× as the improvement to Stage 1's reach overstates
it by an order of magnitude in the direction that matters. See the ⚠ in §4.1.

The third row is worth a number, because it was unnamed until now: for
`ny_hist_366962130` the 120 s horizon covers **624 m**, while a gentle turn at
`ω = max_delta_yaw_rate` has radius 836 m and Stage 1 scans **5.25 km** of arc
around it. A candidate can be refused over an obstacle the ego would not reach
for minutes — and be required to be able to brake for it *now*.

---

## 2. The two guards, and the one thing they do not tell you

Which stage emptied the set determines which error you see. Collapsing these two
is where most historical confusion came from.

| | **Guard A** | **Guard B** |
|---|---|---|
| message | `No admissible (v, ω) pairs found in dynamic window` | `No best trajectory found` |
| raised at | [dwa_average_weighted_side.py:562][a] (`len(to_goal_costs) == 0`) | [:579][b] (`best_index == -1`) |
| means | **Stage 1** emptied the set | Stage 1 passed something, **Stage 2** emptied it |
| what makes it fire | `v² + 2·a·v·dt > 2·a·dist` for all `(v, ω)` | `is_collision` inside the horizon for all survivors |
| caught by `least_bad`? | **no** — still raises unconditionally | **yes** — this is the case the fallback handles |

[a]: ../../DWAT_v4_st/dwa_average_weighted_side.py#L562
[b]: ../../DWAT_v4_st/dwa_average_weighted_side.py#L579

**The fallback key was misleadingly named, and has been renamed.** It does not
catch the "no admissible pairs" guard. [dwa_ramped.py:296][c] gates on
`best_index == -1 and len(controls) > 0` — meaning admissible candidates
existed. Guard A at [dwa_ramped.py:294][d] still raises regardless of the
setting.

| | |
|---|---|
| was | `no_admissible_fallback` — echoed **Guard A's** message, caught **Guard B** |
| now | **`all_candidates_collide_fallback`** — named for the guard it catches |

The old key still loads (it is a property alias on `Config`), so every config
written before the rename keeps working. The **guard messages themselves are
unchanged**: they were never the ambiguity. The collision was between the key's
name and Guard A's message, and renaming the key removes it.

⚠ **Logs before and after the rename differ.** `output_display` writes
`config.__dict__`, and the alias is a property, so it is absent from it. Every
log written before the rename carries `no_admissible_fallback`; every log after
carries `all_candidates_collide_fallback`. Existing log files are untouched, but
anything that diffs *config blocks* across the rename will show this one key as
a spurious difference. Trajectory diffs are unaffected.

[c]: ../../DWAT_v4_st/dwa_ramped.py#L296
[d]: ../../DWAT_v4_st/dwa_ramped.py#L294

### A third meaning nobody listed

Guard A has a **third** cause, found 2026-09-19 and distinct from both above:
the dynamic window can be **empty**. Once `|ω| > max_yaw_rate + Δω·dt` the window
inverts, `np.arange` returns nothing, and the Guard A raise fires having tested
nothing at all. "The braking test rejected everything" and "nothing was ever
tested" are opposite diagnoses and the message does not distinguish them. Fixed
in `63033a1` with its own error; recorded here because the §2 table above still
reads as though there are only two.

**Scope note on `v = 0` always being admissible (§7.1).** That is a property of
any window wide enough to contain `v = 0`, however obtained — not of v20.7.9. It
is equally true of most `ais_v1.1` configs, where `max_accel · dt = 4.0` puts
`v = 0` in the window across 73–80% of the speed range, and unconditionally for
`ny_synth_01`. It is **not** true of v21.x, whose ramped floor is `v − 0.385`, so
Guard A is reachable there. This bears directly on **item 3**: halving `dt`
halves `max_accel · dt` and changes how much of the range admits `v = 0`.

### The inference to *not* make

Guard B proves Guard A did not fire, and therefore that Stage 1 passed at least
one candidate. **It proves nothing about whether Stage 1 was too aggressive.**

```
451 candidates ──[Stage 1]──> 44 ──[Stage 2]──> 0 ──> Guard B
                     │
                     └── 407 discarded here, never scored.
                         If any of them would have survived Stage 2,
                         Stage 1 caused this abort.
```

Which guard fires tells you **which filter emptied the set last**, not **which
filter was too aggressive**. Guard B rules out "Stage 1 rejected *everything*";
it does not rule out "Stage 1 rejected the *survivors*". §4 contains a case where
it demonstrably did, and an earlier draft of this document got exactly this
backwards.

---

## 3. What follows from the frame

Three consequences, in decreasing order of obviousness.

**(a) The natural fix is the reverse of "add CPA to the cost".** The cost
function already computes a sampled, time-discounted CPA (§7.2). The feasibility
test has no CPA anywhere in it. So:

> **Move the CPA that already exists in Stage 2 onto Stage 1, and retire the
> braking test.**

**(b) The stern-sector problem is a corollary, not a separate item.** Stopping
looks *safe* to a braking test built for frozen obstacles, and *fatal* to a CPA.
Only Stage 2 can currently see it, where it is one term among four rather than a
veto. Measured at the item-8 abort state: braking straight ahead scores
**−0.5060 u**, which any CPA-based feasibility test rejects outright. A bespoke
stern-sector term is probably unnecessary once Stage 1 is CPA-based.

**(c) Whatever replaces Stage 1 must stay O(1) in horizon.** v20.7.9's ramped
rollout works because it restricts multi-step manoeuvres to a one-parameter
family ("ramp to a setpoint at `max_accel`, then hold") — **linear** in sample
count, **constant** in horizon length. A true sequence search over `k` rounds
costs `891^k`:

| depth k | sequences | wall-clock for one case (~40 min/case today) |
|---|---|---|
| 1 (today) | 8.9e2 | 40 min |
| 2 | 7.9e5 | 25 days |
| 3 | 7.1e8 | 60 years |
| 300 (full horizon) | **1e885** | — |

There are ~1e80 atoms in the observable universe. Even depth 2 is off the table.
**Pick a closed-form or single-rollout criterion, never a tree.**

---

## 4. State of every known failure

This is the ledger. Anything not in it is passing.

| case | outcome | guard | **actual cause** | fix |
|---|---|---|---|---|
| `owenc_overtaken_ego` @ predT30 | abort, iter 0 | **B** | window couldn't represent accelerating away | ✅ **v20.7.9** |
| `owenc_overtaken_ego_fast` @ predT30 | abort, iter 0 | **B** | same | ✅ **v20.7.9** |
| `owenc_overtaken_ego` @ predT5 | abort, iter 875 | **A** | genuinely cornered — but the only escape needs *more* speed + turn, which Stage 1 forbids | item 6 |
| `ny_hist_366962130` | abort, iter 208 | **B** | phantom vessel **+** disc hull in Stage 1 | items **2 + 1** |
| `ny_synth_00` | abort, iter 51 | **B** | **unknown** — Stage 1 pruned 287 candidates nobody has scored | *measure first* |
| `ny_hist_367782880` | grounding, iter 59 | — | `dt` too coarse: 125 m/step vs 25 m shore cells | item 3 |
| `ny_hist_367791550` | grounding, iter 52 | — | same | item 3 |
| `v3crowd5` | fails 137.4 s | — | `_choose_local_goal` path-following divergence | **not this problem** |

Two things this table makes visible that earlier drafts did not:

- **The AIS suite has four failures, not two.** Two aborts *and* two groundings.
  Previous versions of this document discussed only the aborts, which made item 3
  (`ais_v1.2`) look like a nice-to-have when it is the fix for **half** the
  suite's failures.
- **Only one open case is squarely about the braking test.** `ny_synth_00`, and
  even there it is unproven. If items 1 and 2 resolve `ny_hist_366962130`, the
  open-water `predT5` case becomes the strongest evidence for item 5.

### 4.1 `ny_hist_366962130` — Stage 1's hull decides it

Instrumented at the failing call (both aborting AIS cases):

| | `ny_synth_00` | `ny_hist_366962130` |
|---|---|---|
| candidates in the dynamic window | 451 | 451 |
| passed Stage 1 | **164 (36%)** | **44 (10%)** |
| rejected by Stage 1, never scored | 287 (64%) | 407 (90%) |
| of the survivors, scored finite | **0** | **0** |
| would any *pruned* candidate have scored finite? | **untested** | **yes — item 1 admits one** |

The 44 survivors are essentially the whole `v = 0` row (at `v = 0` the braking
test always passes, since `closest_obstacle_on_curve` clamps its return at
`max(0, …)`). They score `inf` because 16 *live* vessels are extrapolated over
120 s and one closes on a stationary ego.

The blocking candidate is the slowest forward one, `v = 0.4`, and **Stage 1's
hull alone decides it**:

| hull used by `closest_obstacle_on_curve` | gap to the blocker | braking room needed at `v = 0.4` | verdict |
|---|---|---|---|
| disc, 10.013 m (today) | +2.43 m | 4.20 m | **rejected → abort** |
| capsule, half-beam 3.165 m (item 1) | +4.27 m | 4.20 m | **admissible** |

Commit `7e5fc81` put it plainly: `closest_obstacle_on_curve` "alone decides this
case". **Item 1 is a candidate fix for this abort, not merely a prerequisite** —
though the margin is 0.07 m, so it must be measured, not assumed.

> ⚠ **The `+4.27 m` in that table is now doubted.** Item 1 has since been
> implemented, and the capsule does **not** buy 6.85 m in the direction that
> matters. A capsule's reach is direction-dependent:
>
> | direction | disc | capsule | capsule buys |
> |---|---|---|---|
> | dead ahead | `sqrt(L²+W²)/2` = 10.0134 m | `L/2` = **9.5000 m** | **0.5134 m** |
> | abeam | 10.0134 m | `W/2` = 3.1650 m | 6.8484 m |
>
> The bow still sticks out `(L−W)/2` = 6.335 m in front of the hull centre, so
> substituting the half-beam 3.165 m as a *scalar* radius — which is what "capsule,
> half-beam 3.165 m" reads like — models the ego as a 3.165 m **disc** and
> silently discards the bow. The implemented test subtracts that bow lead
> explicitly.
>
> This table needs the gap to grow by **1.77 m** (2.43 → 4.20). The capsule
> delivers that only if the blocker is far enough off the bow; dead ahead it
> delivers 0.51 m and the abort stands. **Re-measure before claiming item 1
> fixes this case.**

And the thing it is refused against **is not there**: the blocker's AIS track does
not begin until 620 s *after* the run ends. 17 of the 78 vessels in this scene are
phantoms parked at entry points they never reach (item 2).

So this case is two input defects compounding, and says nothing about ship
physics: creeping at 0.4 m/s needs only 4.20 m of stopping room, which the true
hull clears.

### 4.2 `ny_synth_00` — the genuine braking case, still unproven

A 147 m ship at 3.30 m/s with `max_accel` 0.0612 m/s² needs 54 s and 90 m to
stop, and the dynamic window will not let it drop below 2.69 m/s in one step:

| | |
|---|---|
| along-curve distance available (`closest_obstacle_on_curve`) | 97–100 m |
| braking room required at 2.69 m/s (window floor) | 86.2 m — **passes** |
| braking room required at 3.34 m/s (window ceiling) | 124.8 m — **rejected** |

That is the 64% Stage 1 prunes: every candidate that does not slow down.
Requiring a 147 m ship to stop is an assumption the vehicles in this dataset do
not satisfy.

*(An earlier `ny_ais_suite/README.md` compared the 86.2 m requirement against an
18.0 m straight-line hull clearance. Those are incommensurable —
`closest_obstacle_on_curve` returns distance **along the candidate arc** to the
collision zone. Corrected.)*

**What is not known:** whether any of the 287 pruned candidates would have scored
finite. Until someone scores them, "the braking test caused this abort" and "the
braking test is incidental here" are both live. This is one instrumented call and
it gates the whole backlog — see §6, step 4.

---

## 5. The backlog, regrouped

The old grouping ("items 4, 5, 6, 7 are one problem") was close but misfiled the
two that matter most. Grouping by **which stage of §1 the item touches** is
cleaner and gives a working order for free.

### Group A — Stage 1 is built on the wrong physics *(the real work)*

| # | item | status | what it is |
|---|---|---|---|
| **1** | **Capsule in `closest_obstacle_on_curve`** | ✅ **implemented 2026-09-15** | Row 2 of the §1 table. All three branches (`v≈0`, straight, arc) now honour `robot_collision_shape`; disc verified bit-identical. Whether it fixes `ny_hist_366962130` is **now in doubt** — see the ⚠ in §4.1. |
| **5** | **Admissibility against *predicted* obstacle positions** | open, **upgraded** | Row 1 of the §1 table. Formerly "restore Guard A in a form a wide window cannot bypass" — same thing: give Stage 1 the obstacle velocities and it stops being a static-world rule. |
| **9** | **Bound Stage 1's look-ahead to the horizon** | **new — was unnamed** | Row 3 of the §1 table. Impact unmeasured; one instrumented run from being known. |
| **6** | **Replace braking feasibility with turn/CPA** | open | The synthesis of 1, 5 and 9. This is `ais_v2` and the stern-sector guard **at once** — one piece of work under two names. |
| **7** | **Stern-sector cost term** | open, **scope narrowed 2026-09-19** | Per §3(b), falls out of item 6. **Do not build it first.** Scope: item 7 is the *Stage 1* defect — the braking test rewards deceleration when a faster ship closes from astern. It is **not** the overtaking counterpart (item 13), which is a Stage 2 cost balance. Measured at v21.5: both `overtaken_ego*` cases pass safely because the ego sees the target 13 u out with ~40 s of warning and steps aside. **Item 7 needs the cornered state**, which only the short-horizon `predT5` regime produces — untested at v21.x. |

| **10** | **`closest_obstacle_on_curve` misreports an obstacle it is already touching** | **new — found 2026-09-15** | Pre-existing, independent of item 1, and it *pre-dates* the capsule. If the hull already overlaps an obstacle at `t = 0`, the arc branch reports the distance to the **far side** of the obstacle and the straight branch reports `inf` — "nothing ahead" — while in contact. |
| **11** | **Stage 1 scores every candidate against a path and a speed the ego will not have** | **new — found 2026-09-19** | Row 4 of the §1 table. `closest_obstacle_on_curve` and the braking inequality are both handed the *setpoint* `(v, ω)`, but under `ramped_rollout` the setpoint is a target the ego approaches at `max_accel` — it is not the state. So the ego is judged on a path it is not on, at a speed it does not have. **Both halves, not one.** |
| **12** | ~~The candidate set cannot express a wide pass~~ → **`predict_time_to_goal` is 4× too long** | ✅ **cause found 2026-09-20** | **Superseded.** The short menu was the *consequence* of reacting late, not the cause. `predict_time_to_goal` went 1.0 → 8.0 in the rescale, making `to_goal_cost` **70× stiffer** across the yaw window while `side_cost_gain` stayed at 0.6. Setting it to 2.0 restores v20.7.8-2's clearance **exactly**. See §5.1. |
| **13** | **Nothing penalises indefinite speed-matching** | ✅ **same cause as item 12, 2026-09-20** | `overtake_ego`: the ego closes to 3.07 u, locks to the target's speed (**exactly 0.200**) and station-keeps for 173 s — 263.5 s against a 153.3 s baseline, **72% longer**, never passing. "Unmoved by barrier gain or shape" was right: the barrier was never the lever. At `predict_time_to_goal = 2.0` the ego **overtakes**, in 176.6 s. Not a separate defect. |
| **14** | **A perfectly symmetric encounter has no side preference** | **new — found 2026-09-20** | On an exactly collinear head-on the planner chatters ±1 yaw cell — **149 sign flips in 199 steps, net turn exactly zero** — until float noise breaks the tie. **Not a cause of the grazes** (§5.1), but it is the missing COLREGs Rule 14 side rule made load-bearing. Masked until v21.3, when `pin_path_endpoints` removed the 0.707 u Theta\* snap that had been breaking the tie by accident. |
| **15** | **`dist_localgoal` is also the maximum admissible cross-track error** | **low priority — 2026-09-20** | The pure-pursuit follower intersects a circle of radius `dist_localgoal` with the path; stray further and there is no intersection and the run **aborts** (`No intersection points found…`). So `dist_localgoal = 7.0` silently caps evasion at 7 u. Both `predict_time_to_goal` ∈ {1.0, 4.0} runs died this way with `max |dev|` of 6.96 and 6.99 — **not collisions**. A `local_goal_recovery: "rejoin"` option exists (default `"abort"`, so nothing changes unless enabled). ⚠ **Never enable it without an oscillation check** — on a diverging controller it converts a visible abort into an invisible divergence, and turns 5 honest failures into 6 misleading "successes". Demoted from blocker to low priority: it was found while chasing item 16, and is not its cause. |
| **16** | **`predict_time_to_goal` supplies the path-following loop's phase lead** | **new — found 2026-09-20** | The goal rollout's *length* is what makes `to_goal_cost` anticipatory. Shorten it and the loop loses its lead while keeping a 20 s actuation lag, and the cross-track error diverges (growth 1.3–2.1× per half-cycle on every manoeuvring case). The parameter has **three conflicting roles** — see §5.1. Fixing item 12 must not go through it. |
| **17** | **A scenario silently overrides the algorithm config** | **new — found 2026-09-20** | `map_creation.py:306` lets a scenario's `enable_map_boundary_obstacles` win over the algorithm config's. Every `owenc215_*` run has had 396 boundary obstacles active while `v21.5`/`v21.6` declared `false`. **Inert here** — measured min distance to any boundary is 14.5 u against `max_obstacle_cost_dist` of 8.0, and runs with boundaries on and off are bit-identical — but the precedence is undocumented and the same mechanism applies to every shared key. |

Items 1, 5, 9 and 11 are the same defect in four fields of the same function.
Item 6 is what you get if you fix them at once and stop patching.

### 5.1 Items 12 and 13, solved: `predict_time_to_goal` was 4× too long

**Measured 2026-09-20.** `calc_to_goal_cost` is the heading error at the **final
row** of the goal rollout, so its sensitivity to `ω` scales with how far that
rollout travels. The rescale raised `predict_time_to_goal` from 1.0 to 8.0 and
left all four cost gains alone. Spread of `to_goal_cost` across the yaw window,
same state, same ship:

| `predict_time_to_goal` | 1.0 | 2.0 | 4.0 | **8.0 (shipped)** |
|---|---|---|---|---|
| goal-cost spread | 0.0039 | 0.0127 | 0.0531 | **0.2747** |

**70×.** The barrier's own spread is ~0.049, so the competition *inverts*: at
1.0 the barrier outweighs the goal term 12×, at 8.0 the goal term outweighs the
barrier 5.6×. Nothing else in the rescale matters — `time_decay_lambda` and
`predict_time_obstacle` move the chosen clearance by **zero**.

Closed loop on `owenc215_head_on` (clearance is surface-to-surface, so it is
comparable across hulls; centre distance is not):

| config | `T_goal` | goal | clearance | min centre | max dev |
|---|---|---|---|---|---|
| v21.5 shipped | 8.0 | ✅ | **0.000** | 0.751 | 1.603 |
| v21.5 | 4.0 | ❌ item 15 | 0.000 | 0.785 | 6.994 |
| **v21.5** | **2.0** | ✅ | **0.433** | 1.181 | 3.487 |
| v21.5 | 1.0 | ❌ item 15 | 0.880 | 1.663 | 6.963 |
| v21.5 `side_cost_gain=2.11` | 8.0 | ✅ | **0.000** | 0.752 | 1.612 |

On the offset geometry `T_goal = 2.0` reproduces the baseline **exactly** —
`0.482` against v20.7.8-2's `0.482`, the centre distances differing only by hull
(capsule 1.183, disc 1.632). Item 13 falls out at the same value: the ego stops
station-keeping and **overtakes**, 263.4 s → **176.6 s**, `max |dev|` 0.003 →
5.900.

**`T_goal` is not monotone** and 1.0 is not the answer. The old value cannot be
copied back because the ship changed: `max_accel` fell 5.2×, so at `T_goal = 1.0`
the *speed* term loses its authority too (spread 0.0074, and free-space `argmin
v` collapses to 0.12). `predict_time_to_goal` sets the stiffness of the heading
term **and** the authority of the speed term from one number, and the rescale
changed `max_yaw_rate` and `max_accel` by different factors (3.7× and 5.2×), so
no single value restores both. Splitting them is the principled fix.

> ### ⚠ `T_goal = 2.0` is NOT the fix — it buys the clearance with a control instability
>
> **Found 2026-09-20 by watching the videos, after the table above said it worked.**
> At `T_goal = 2.0` the cross-track error does not settle after the encounter: it
> oscillates with **growing** amplitude, and the heading error to the local goal
> reaches **+136.9°** — the ego ends up pointing nearly backwards. Peak |xte|
> per half-cycle, all 11 `owenc215_*` cases:
>
> | case | peaks | growth |
> |---|---|---|
> | `head_on` *(reached goal)* | 1.68 → 1.42 → 2.12 → **3.49** | 1.49, 1.65 |
> | `overtake_ego` *(reached goal)* | 1.86 → 1.71 → 2.99 → **5.90** | 1.74, 1.98 |
> | `cross_port_045` *(reached goal)* | 1.92 → 1.59 → 2.76 → **5.35** | 1.73, 1.94 |
> | `cross_port_135` *(failed)* | 4.20 → **6.47** | 1.54 |
> | `overtaken_ego_fast` *(failed)* | 3.28 → **6.79** | 2.07 |
>
> At `T_goal = 8.0` the same cases show **one peak and no oscillation**. So the
> "6 of 11 reached the goal" result was an artefact: all six were diverging too
> and merely finished before the amplitude passed `dist_localgoal`. The five
> "failures" were only the fastest divergers. **Reaching the goal is not a pass
> criterion on its own; the oscillation ratio must be checked with it.**
>
> **Why a shorter horizon destabilises.** `calc_to_goal_cost` scores the heading
> at the **end** of the goal rollout, i.e. the heading the ego *will have* after
> turning for `T_goal` seconds. That is **phase lead**. Shortening `T_goal`
> removes the anticipation while the actuation lag stays long (`ω` ramps at
> `max_delta_yaw_rate`, 20 s to full helm). Lag without lead is a divergent loop.
>
> So `predict_time_to_goal` has **three** roles, and they conflict:
>
> | role | wants |
> |---|---|
> | stiffness of the heading term vs the barrier | **small** |
> | authority of `speed_cost` (`max_accel · T`) | large |
> | **phase lead of the path-following loop** | large |
>
> No single value satisfies all three, which is why both 8.0 (grazes) and 2.0
> (diverges) are wrong. **The lever to use instead is `to_goal_cost_gain`**: it
> weakens the heading term's magnitude while leaving the rollout length — and
> therefore the anticipation — intact. Matching v20.7.8-2's goal/barrier spread
> ratio of 1.22 gives `to_goal_cost_gain ≈ 0.13` at `T_goal = 8.0`. **Under test;
> not yet established.**

**Where the v21.x derivation went wrong.** It bounded `predict_time_to_goal`
from above by *overshoot* (`v·T > dist_localgoal` saturates the bearing) and
concluded a "safe window [4, 12]", then took 8.0. The criterion was "does it cap
the speed?" — the competition against the barrier was never in it. The working
value, 2.0, lies **below** that window, and 4.0 — its lower edge — is the worst
value tested.

### Item 12's original evidence, and what it actually showed

At the `head_on` decision 12 s before CPA, evaluating **every** yaw candidate at
full speed:

| ω | min clearance | to_goal cost | rollout ends |
|---|---|---|---|
| −0.09659 | **0.578** | 0.28478 | 2.00 u off track |
| −0.06981 | 0.519 | 0.28191 | 3.51 u off track |
| +0.05236 | 0.346 | 0.24581 | 15.96 u off track |
| −0.00873 | −0.433 | 0.06359 | **COLLIDES** |
| 0.00000 | −0.680 | 0.00512 | **COLLIDES** |

**The best clearance any candidate can produce is 0.578 u.
`max_side_weight_dist` is 2.0.** The ego is not declining to pass clear —
passing clear is not on the menu. And the chosen candidate sits near the *top* of
what is available: the planner is doing the best it can with the set it is given.

> ⚠ **This measurement is correct and its conclusion was not.** It was taken
> **12 s before CPA**, by which time the menu really is that short. Evaluated
> earlier, the same window is rich: at a centre gap of 20 u the widest available
> candidate clears **4.742 u**, and at 30 u it clears **9.999 u**. The menu was
> ample and the planner spent it standing still. *The short menu is the
> consequence of reacting late, not the cause of it* — §5.1 has the cause. The
> general lesson: **measuring at the moment of failure finds the constraint that
> binds last, which is rarely the one that can be fixed.**

**Why the menu is that short.** The yaw window is ±5.43° and the rollout holds
one command for 78.76 s. At the window edge the ego curls into a circle of
radius `v/ω = 5.2 u` and comes back, so the maximum lateral offset any *single*
command can place at the CPA is ~0.58 u. "Turn now, straighten later" is not
representable. This is v20.7.9's original defect — a manoeuvre the window cannot
express — moved from the speed axis to the yaw axis.

**Three fixes this falsified**, all aimed at the cost function rather than the
candidate set:

| hypothesis | prediction | measured |
|---|---|---|
| barrier tuned for the disc hull | capsule passes closer | centre distance identical |
| `side_cost_gain` too weak | clearance opens up | **0.000 at every gain 0.6 → 3.5** |
| Gaussian barrier flat at contact | quadratic fixes it | **0.000; min ctr 0.750 vs 0.751** |

**All three verdicts stand, re-measured 2026-09-20.** `side_cost_gain = 2.11`
closed-loop gives clearance **0.000** and min centre **0.752**, against the
baseline's 0.000 / 0.751 — the gain really does nothing, and a single-step
decomposition predicting 1.218 at a 20 u gap really is wrong. The barrier
pathologies are real and worth recording — `A·exp(−λd²)` has **zero gradient at
`d = 0`**, peaks at `d = 0.577`, and is **non-monotonic below zero**
(`B(−0.5) = 1.37 < B(0) = 2.00`, so deeper overlap would score cheaper, reachable
only because `is_collision` forces `inf` first) — but they are not what causes
the grazes.

What was wrong was the *inference drawn from* the three falsifications: that
since no cost-function change helped, the candidate set must be at fault. Every
one of the three scales a term that competes against `to_goal_cost`, and at
`predict_time_to_goal = 8.0` that term outweighs all of them. They failed for one
shared reason, which is weaker evidence than three independent failures.

**Retracted:** the note that `side_cost_gain` needs a **1.90×** dilution
correction (matched value ≈ 1.14) because it is aggregated as a *mean* over a
longer horizon. The reasoning is sound but the effect is **nil**: holding
`λ·T` fixed while varying `T`, and separately restoring `T_obs = 30, λ = 0.1`,
each move the chosen clearance by **0.000**. The mean-dilution term is real and
irrelevant at this scale.

### Item 14, measured — symmetry changes the manoeuvre, not the outcome

The cost function is mirror-symmetric in port/starboard (the open-water suite
confirms it empirically: mirrored pairs agree to 2 d.p.). Until v21.3 every
"collinear" case was in fact offset — Theta\* snapped the ego's start by
**0.707 u** — and that accident was the only thing breaking the tie.
`pin_path_endpoints` removed it and the suite regeneration put the target on
`x = 50.0` exactly, so the encounter became genuinely symmetric for the first
time. On the axis the planner chatters:

| phase | sign flips | net turn | `|k|max` |
|---|---|---|---|
| barrier active, t = 50–70 | **149 / 199 steps** | **exactly 0 cells** | 1 |
| committed, t = 76–90 | 8 / 139 | −207 cells | 10 |

It commands a turn every step, alternates sides, and integrates to nothing.

**It is not what causes the grazes.** Sweeping the target's lateral offset with
every cost parameter untouched:

| offset | breaks tie at | centre dist then | max dev | **min centre** |
|---|---|---|---|---|
| 0.00 | 81.4 s | 7.8 u | 1.603 | **0.751** |
| 0.05 | 73.3 s | 15.1 u | 0.902 | **0.726** |
| 0.25 | 64.6 s | 23.0 u | 0.492 | **0.707** |
| 0.50 | 62.3 s | 25.1 u | 0.237 | **0.712** |

Breaking the tie 19 s earlier at 3× the range leaves the outcome **unchanged, and
marginally worse**. The planner converges on the same ~0.72 u equilibrium however
early it starts; reacting sooner only buys a smaller deviation. Symmetry sets the
manoeuvre's *shape*, not its endpoint.

So the exactly-symmetric suite is **better, not broken** — it stopped an
accidental offset from masking a missing COLREGs Rule 14 side rule. Do not
"fix" it by re-introducing an offset.

### Item 11, measured

`closest_obstacle_on_curve(x, ob, radii, v, y, config)` takes the **setpoint**
`v`, and `v` is what picks both the branch and the arc radius. With the ego at
`v = 0.470`, `ω = −0.050` — genuinely on an arc of `R = 9.40 u`:

| setpoint `v` | branch taken | path scanned | dist returned |
|---|---|---|---|
| 0.000 | `abs(v) < 1e-5` | **a straight ray along the heading** | 48.610 |
| 0.050 | arc | `R = 1.00 u` | inf |
| 0.200 | arc | `R = 4.00 u` | inf |
| 0.470 | arc | `R = 9.40 u` | inf |
| 0.500 | arc | `R = 10.00 u` | inf |

Across a single window the candidates are scored against arcs from 1.0 to
10.0 u — **and the ego is on none of them.** Under `ramped_rollout` its real
path is a spiral: `v` ramps toward the setpoint at `max_accel` while `ω` ramps
at `max_delta_yaw_rate`, so the radius `v/ω` changes throughout. The `v = 0`
candidate is the extreme case: it is scored along a straight ray the ship is
not on and, at `0.470 u/s`, cannot be on.

**The same setpoint confusion is in the inequality.** `v² + 2·a·v·dt > 2·a·dist`
reads `v` as the ship's current speed. At setpoint `v = 0` it demands zero
braking room while the ship is doing 0.470 and needs 2.9 u. That is the
mechanism behind §7.1's "`v = 0` is always admissible", stated as a property of
the candidate rather than of the window.

**Both halves are still setpoint-based.** A correction was designed and
validated numerically on 2026-09-17 — replace `v` with the speed actually
reached after one `dt`, `v₁ = v₀ + clamp(v − v₀, ±a·dt)` — and shown to restore
Guard A at the cornered state while remaining a no-op under the narrow window
(`max |classic − real| = 0.0010 u` across the whole v20.7.8-2 window). **It was
never implemented.** Fixing only the inequality would leave the path wrong, so
the two belong together.

**Why this is not item 5.** Item 5 is about the *obstacle's* motion being
frozen. This is about the *ego's* own path and speed being wrong. Two different
vessels, two different errors, same function.

### Item 10, measured

Found while validating item 1 against a brute-force hull sweep. Ego 215.1 × 36.83 m,
one obstacle (r = 20 m) centred 30.4 m away — inside the hull on any reading.
**The correct answer is 0.0 m in every row:**

| branch | hull | returned |
|---|---|---|
| arc (`R` = 164 m) | disc, pre-item-1 | **165.65 m** |
| arc | disc, post-item-1 | 165.65 m *(unchanged — this is not a regression)* |
| arc | capsule | 0.45 m |
| straight | disc, pre-item-1 | **`inf`** |
| straight | capsule | 0.00 m |

The cause is that both branches solve for where the path *enters* the obstacle's
inflated circle, measuring forward from the current heading. Starting already
inside, the forward entry is most of a lap away. Item 1's capsule improves the
numbers by a lot but only the straight branch is exactly right.

**Why this has stayed invisible:** `dist ≈ 0` makes Stage 1 reject every
candidate with `v > 0`, so a correct reading here produces Guard A — the very
"cornered" report §7.3 shows is a *save*. A 165 m reading instead says the way
ahead is clear. This interacts with items 5 and 9 and should be fixed as part of
item 6 rather than patched alone.

### Group B — the inputs are wrong *(must land before any measurement means anything)*

| # | item | status | what it is |
|---|---|---|---|
| **2** | **`on_start` for replayed tracks** | not started | 17 of 78 vessels are phantoms in one AIS scene; one of them costs `ny_hist_366962130` its goal. Formerly filed "independent / unrelated to feasibility" — it is **co-decisive with item 1** in that abort. |
| **3** | **`ais_v1.2` — per-ego `dt`** | evidenced, not adopted | Config-only, orthogonal to Group A — **but it is the fix for both groundings**, i.e. half the AIS suite's failures. Roughly doubles run cost and changes the reference case, which is why it stalled. |

### Group C — failure handling *(cheap, cosmetic, do it anyway)*

| # | item | status | what it is |
|---|---|---|---|
| **4** | **Guard B fallback for the AIS configs** | exists in `dwa_ramped`, absent from `dwa_average_weighted_side` | Formerly called "the smallest possible fix". It is **not a fix** — it handles Guard B's *symptom*. For `ny_hist_366962130` it would keep the ego creeping past a ship that does not exist. Port it for **legibility**: a degraded-but-continuing run says more than an abort. |

### Closed

| # | item | outcome |
|---|---|---|
| **8** | `v20.7.8-2 predT5 + least_bad`, narrow window | **✅ 2026-09-12.** The abort **was a save**. See §7.3. |

### The renaming that dissolved most of the mess

- "no admissible" named **two different guards** (§2).
- `ais_v2` and "the stern-sector guard" were **the same work** under two names.
- **Item 1 belongs *inside* Group A**, not as a chore preceding it — it is a
  Stage 1 defect, identical in kind to item 5.
- **Item 4 belongs *outside* Group A** — it touches neither stage.
- **Item numbers are unchanged** from earlier drafts, so older notes still
  resolve. Only the *grouping* is new; items 9 and 10 are the additions.

---

## 6. Order of work

**The principle: fix what is wrong with the measurement before designing against
it.** Anything measured on a scene containing a ship that is not there, or judged
against a hull three times too wide, is not evidence.

There are **two independent tracks**. The open-water track is where the cost
function is tuned; the AIS track is where Stage 1's physics is wrong. They share
no blocking dependency, so either can proceed.

### Track 1 — open water: finish what `predict_time_to_goal` opened

1. ~~**Validate `T_goal = 2.0` across all 11 `owenc215_*` cases.**~~ ✅ done
   2026-09-20, and it **failed**: 5 of 11 abort and the other 6 diverge. Do not
   ship it. `null_safe_pass` was clean (`max |dev|` exactly 0.000), so the
   over-reaction worry was unfounded; the problem is item 16 instead.
   **Next: `to_goal_cost_gain ≈ 0.13` at `T_goal = 8.0`**, which weakens the
   heading term without removing the loop's phase lead. Score every run on
   **two** criteria — clearance **and** oscillation growth < 1.0.
2. **Split `predict_time_to_goal` into two parameters.** It sets the stiffness of
   `to_goal_cost` *and* the authority of `speed_cost` from one number. The rescale
   moved `max_yaw_rate` and `max_accel` by different factors (3.7× and 5.2×), so
   no single value restores both — which is why the old 1.0 cannot simply be
   copied back (it reaches 0.880 clearance and then loses the path).
3. **Item 15** — `local_goal_recovery: "rejoin"` is implemented and defaults
   off. Low priority, and **do not enable it until item 16 is fixed**: on a
   diverging controller it hides the divergence instead of reporting it.
4. **Item 14** — a COLREGs Rule 14 side preference. The suite is now exactly
   symmetric, so this is load-bearing rather than cosmetic. **Do not** "fix" the
   symmetry by re-introducing a scenario offset; §5.1 shows the offset changes
   the manoeuvre's shape and not its outcome.

### Track 2 — AIS: Stage 1 is built on the wrong physics

5. ~~**Item 1** — capsule in `closest_obstacle_on_curve`.~~ ✅ done 2026-09-15.
   Next action is to **re-measure `ny_hist_366962130`**: §4.1 predicted this
   admits the blocking candidate, and the direction-dependence of a capsule's
   reach (§4.1 ⚠) makes that prediction look optimistic.
6. **Item 2** — `on_start`, defaulting to today's behaviour.
7. **Re-baseline the AIS suite** on the corrected scene and hull. Expect
   `ny_hist_366962130` to change outcome; if it does, it leaves the feasibility
   backlog entirely and `ny_synth_00` becomes the only AIS instance of it.
8. **Measure what Stage 1 prunes — before designing anything.** At the
   `ny_synth_00` abort, score the 287 candidates Stage 1 rejected. If any is
   collision-free, Stage 1 causes that abort too and item 6 is the whole job; if
   none is, item 4 really is the answer there. **One instrumented call, and it
   decides everything below it.** Take the item 9 (unbounded-span) number in the
   same pass.
9. **Item 4** — port `least_bad` to `dwa_average_weighted_side` (or run AIS on
   `dwa_ramped`). Do this regardless of how step 8 turns out, for legibility.
10. **Item 6** — design the replacement feasibility test, absorbing items 5 and 9.
    Only now are its inputs trustworthy. Item 7 probably falls out; do not build
    it separately.
11. **Item 3** (`ais_v1.2`) — orthogonal, so "whenever convenient". But it fixes
    half the suite's failures, so *convenient* should not mean *never*.

### Also open, unsequenced

- **Item 10** — `closest_obstacle_on_curve` misreports an obstacle it is already
  touching. Independent of everything above.
- **Cull obstacles outside the prediction horizon.** Should be behaviour-neutral;
  worth doing for cost.
- **`max` vs `mean` aggregation.** Deferred until parameter matching was done —
  which §5.1 has now done, so it is eligible. Compare, do not dive deep.

### Dropped

- ~~**Shorten the open-water basin.**~~ Dropped 2026-09-20 — it was a runtime
  optimisation worth ~23% of a run that now costs ~15 min. Not worth the churn of
  changing the reference case.
- ~~**Silent-death investigation.**~~ Parked unless it recurs.

Every step must stay **config-restorable**: new behaviour behind a key that
defaults to the current one, per `../configs/algorithm/README.md`.

---

## 7. Evidence and established results

Reference material. Nothing here is open; it is what the sections above stand on.

### 7.1 What v20.7.9 is, and three things it is not

Full record: [`v20.7.9_ramped_window/README.md`](v20.7.9_ramped_window/README.md).

**Built for the overtaken cases.** ✅ `owenc_overtaken_ego` and
`owenc_overtaken_ego_fast` were the only 2 of 11 open-water cases that failed,
both on iteration 0.

**The right answer was hidden behind several rounds of acceleration.** ✅ This is
the heart of it. The dynamic window admitted only what was reachable in one `dt`,
then scored each candidate by *holding that velocity for the whole 30 s horizon*.
From rest the window offered `v ∈ [0, 0.02]`, so the planner predicted covering
**0.60 u in 30 s** when it could really cover **14.40 u**. The escape —
accelerate away — was unrepresentable, so every candidate collided.

**They aborted with "no admissible pairs".** ❌ **Incorrect.** They aborted with
**`No best trajectory found`** (Guard B). Guard A appears only in the `predT5`
variant, at iteration 875. Getting these the right way round is what makes this
document hang together.

**v20.7.9 "is" the least-bad choice.** ❌ `least_bad` is one of *four* config
changes and is the backstop, not the mechanism. The fix is the **ramped window**:
a sampled `(v, ω)` becomes a *setpoint approached at `max_accel`* rather than a
velocity teleported to. That alone takes iteration 0 from **0 of 243**
collision-free candidates to **596 of 891**.

**The lost guard is Guard A, and `least_bad` did not cause the loss.** The widened
window always contains `v = 0`; at `v = 0` the admissibility test reduces to
`0 > 2·a·dist`, never true since `dist ≥ 0`; so `v = 0` is always admissible and
Guard A can never fire. That is a property of the *window*, not the fallback —
the two changes could be adopted separately.

> **⚠ Caveat, load-bearing and unenforced.** The window floor is
> `max(min_speed, v − max_accel · v_window_ramp_time)`. It reaches 0 from any
> state only because v20.7.9 sets `v_window_ramp_time = 2.5`, which is *exactly*
> `max_speed / max_accel = 0.5 / 0.2`. That is a config **convention**, not a code
> invariant. Set the ramp to 2.0 and the floor at cruise becomes 0.1, `v = 0`
> leaves the window, and Guard A becomes reachable at some speeds and not others —
> the worst of both worlds. **Item 5 must not rest on this equality holding by hand.**

**`predT5` is designed to fail**, and does under both versions — that is intended.
But the guard loss is **not `predT5`-only**: `v = 0` is always admissible under
any v20.7.9 config, `predT30` included. `predT5` is merely where the planner
reaches a cornered state and the loss becomes *observable*. In the shipping config
the guard is equally gone, has never been called on, and **no test exercises it**.

So: *the collision at predT5 is intended and uninteresting; the silent removal of
Guard A everywhere is unintended and is the actual regression.*

### 7.2 CPA/TCPA is already in the cost function

`calc_trajectory_clearance_and_collision` computes:

```
predicted_ob_pos = ob_pos + ob_vel · t          # constant-velocity extrapolation
clearance(t)     = |ego(t) − ob(t)| − (r_ob + r_ego)
cost_side_vec(t)   = barrier(clearance(t))      # gaussian or quadratic
cost_direct_vec(t) = exp(−λ·t)                  # time discount
cost = aggregate(cost_side_vec · cost_direct_vec)   # "mean" or "max"
```

That is the CPA construction exactly — relative motion under constant target
velocity. With `dynamic_ob_cost_aggregation: "max"` and `λ → 0`, `argmax` of the
product is `argmin` of `clearance(t)` — **the CPA, and its index is the TCPA**.
The intuition was right: this is a soft, sampled generalisation of CPA/TCPA, not
something missing from the design.

Four real differences remain:

1. **Sampled, not closed-form.** The true minimum can fall between samples. In the
   AIS suite at `dt = 10 s` this understates proximity by ~20% (planner sees
   20.7 m where the truth is 16.5 m).
2. **`"mean"` blurs it.** The AIS configs use `mean`, a time-weighted average over
   the horizon — not a CPA, but the average of a barrier. `"max"` is the CPA-like
   reduction. This is what `v20.7.8-3` changes; it is **not unevaluated** — it ran
   on the nine `synthetic/scenario_*` scenes on 2026-08-15 at `predT30`, all nine
   reaching goal. It has never been run on the encounter, open-water or AIS
   suites, which are the ones that discriminate.
3. **It is a barrier, not a constraint.** Nothing says "reject if CPA < d_safe".
   Only `clearance ≤ 0` — actual overlap — produces `inf`. There is no *standoff*
   that is forbidden, only penalised.
4. **The time discount points the wrong way for COLREGs.** `exp(−λ·t)`
   *down-weights* far-future encounters, so the planner reacts late. "Early and
   substantial action" means acting *while TCPA is still large*. A TCPA-gated
   urgency term would be the opposite sign to the current discount. **A genuine
   open design question, not a bug.**

### 7.3 Item 8 (closed): the baseline's abort was a save

Two configs isolate `least_bad` from the window widening
(`v20.7.9/predT5-narrow-control.json` and `…-leastbad.json`, one key apart), both
run against `owenc_overtaken_ego`.

**Result 1 — the control is exact.** `dwa_ramped` with `v_window_ramp_time: 0.0`
and `ramped_rollout: false` reproduces `v20.7.8-2/predT5` **bit-identically**: 876
steps, max |diff| = 0.000e+00, same Guard A abort at iteration 875. The ramped
module is a strict superset of the baseline — verified, not assumed — so it is a
usable vehicle for ablations.

**Result 2 — the experiment as designed cannot answer the question.** Enabling
`least_bad` changed *nothing*: also bit-identical, same abort. The baseline fails
on **Guard A**; `least_bad` only catches **Guard B**. The premise was wrong — which
is itself the empirical confirmation of §2.

**Result 3 — answered directly instead**, by backward reachability from the
cornered state, using the planner's own `motion()` (validated bit-exact against
the last five logged steps) and the target's actual motion (validated to 4.9e-13
against the recorded collision position):

| command from the abort state | min clearance over 3 s |
|---|---|
| **best of a 241 × 26 grid — `v = 0.500`, `ω = −0.227` (32% starboard)** | **+0.0173 u** |
| hard turn to starboard | +0.0034 u |
| hold course | −0.0004 u |
| brake straight ahead (what v20.7.9 does) | **−0.5060 u** |
| hard turn to port | −0.6694 u |

**An escape existed.** The abort was a *save* — fired at the last moment from
which the situation was recoverable, with 0.6 s and 0.0563 u of clearance left.
Thin (0.017 u against a 1.15 u threshold), and a lower bound: a re-planning
controller can do at least as well as the best constant command.

**And the escape is exactly what Stage 1 forbids.** At `v = 0.490` the braking
rule demands `dist > v²/(2a) + v·dt` = **0.649 u** while only **0.0563 u** is
available, so every candidate carrying speed is rejected and Guard A fires. But
the only surviving manoeuvre needs `v = 0.500` — *more* speed — plus a turn. The
ego is at 0.490 against a target closing at 0.600: it cannot outrun it and it
cannot stop. **Turning at full speed is the whole of its escape set, and Stage 1
discards that set first.**

This is the sharpest evidence for item 6. It is no longer an argument from ship
physics; it is a measured case in which the static-world braking rule throws away
the unique surviving manoeuvre, after which the planner correctly reports it has
none.

> **⚠ Reproducibility gap.** Commit `2ba4a54` is labelled `test:` but adds two
> configs and documentation only — the 241 × 26 reachability grid that produced
> the table above **is not in the repository**. The two ablation runs re-verify
> (bit-identical, 875 steps, Guard A), but the escape numbers cannot currently be
> re-derived from anything committed. If item 6 leans on them, that script should
> land.

**Consequence for item 5:** Guard A reported a real, recoverable emergency rather
than an inevitability. Restoring it in a form a wide window cannot bypass matters
more than previously judged — and `least_bad` is confirmed to be no substitute.

# Open problems: feasibility, aborts, and what the cost function already knows

Three investigations converged on the same piece of code, from different
directions, and it is no longer possible to reason about any one of them alone:

- the **open-water "ego is overtaken"** work (`v20.7.9_ramped_window/`),
- the **New York AIS suite** (`ny_ais_suite/`), whose large ships find no
  trajectory,
- an old question about whether the cost function is **already doing CPA/TCPA**.

They meet in `calc_control_and_trajectory_merged`, in the handful of lines that
decide whether a candidate is allowed at all. This document sorts out what is
established, what is a misreading, and what is genuinely open.

**Start here if you are resuming this work in a new conversation.** Section 7
says what is independent of what; section 8 suggests an order.

---

## 1. There are two aborts, not one, and almost everything hinges on which is which

This distinction is the single most important thing in this document. Most of
the confusion below comes from collapsing the two.

| | **Guard A** | **Guard B** |
|---|---|---|
| message | `No admissible (v, ω) pairs found in dynamic window` | `No best trajectory found` |
| raised at | [dwa_average_weighted_side.py:534][a] (`len(to_goal_costs) == 0`) | [:551][b] (`best_index == -1`) |
| fires when | the **braking-admissibility** `continue` rejected *every* candidate | candidates **were** admissible, but every `final_cost` came out `inf` |
| what makes it fire | `v² + 2·a·v·dt > 2·a·dist` for all `(v, ω)` | `is_collision` inside the prediction horizon for all survivors |
| caught by `least_bad`? | **no** — still raises unconditionally | **yes** — this is the case the fallback handles |

[a]: ../../DWAT_v4_st/dwa_average_weighted_side.py#L534
[b]: ../../DWAT_v4_st/dwa_average_weighted_side.py#L551

Two consequences that are easy to get backwards:

1. **`no_admissible_fallback` is misleadingly named.** It does not catch the
   "no admissible pairs" guard. [dwa_ramped.py:293][c] gates on
   `best_index == -1 and len(controls) > 0` — meaning admissible candidates
   existed. Guard A at [dwa_ramped.py:291][d] still raises regardless of the
   setting. The key would be better named `all_candidates_collide_fallback`.

2. **Guard B firing is proof that Guard A did not.** If admissibility had
   rejected everything, `to_goal_costs` would be empty and Guard A would raise
   first. So any run whose log says `No best trajectory found` had admissible
   candidates — the braking test is *not* what stopped it.

[c]: ../../DWAT_v4_st/dwa_ramped.py#L293
[d]: ../../DWAT_v4_st/dwa_ramped.py#L291

---

## 2. What v20.7.9 is, and three things it is not

Full record: [`v20.7.9_ramped_window/README.md`](v20.7.9_ramped_window/README.md).
Summarising, with the corrections that matter:

**It was built for the overtaken cases.** ✅ Correct.
`owenc_overtaken_ego` and `owenc_overtaken_ego_fast` were the only 2 of 11
open-water cases that failed, both on iteration 0.

**The reason they failed is that the right answer is hidden behind several
rounds of acceleration.** ✅ Correct, and this is the heart of it. The dynamic
window admits only what is reachable in one `dt`, then scores each candidate by
*holding that velocity for the whole 30 s horizon*. From rest the window offers
`v ∈ [0, 0.02]`, so the planner predicts covering **0.60 u in 30 s** when it can
really cover **14.40 u**. The escape — accelerate away — is unrepresentable, so
every candidate collides.

**They aborted with "no admissible pairs".** ❌ **Incorrect** — and this is the
correction that matters most. They aborted with **`No best trajectory found`**
(Guard B). The v20.7.9 README states this explicitly in its §1 table. Guard A
appears only in the `predT5` variant, at iteration 875. Getting these the right
way round is what makes the rest of this document hang together.

**v20.7.9 "is" the least-bad choice.** ❌ Not quite. `least_bad` is one of
*four* config changes, and it is the backstop, not the mechanism. The fix is the
**ramped window**: a sampled `(v, ω)` becomes a *setpoint approached at
`max_accel`* rather than a velocity teleported to. That alone takes iteration 0
from 0 of 243 collision-free candidates to **596 of 891**. `least_bad` only
matters when even that leaves nothing.

**The lost guard is Guard A, and `least_bad` did not cause the loss.** The
widened window always contains `v = 0`; at `v = 0` the admissibility test reduces
to `0 > 2·a·dist`, which is never true; so `v = 0` is always admissible and
Guard A can never fire again. That is a property of the *window*, not of the
fallback. The README says this at its §5 "It was the window widening, not
`no_admissible_fallback`, that did this" — consistent, and worth keeping
straight because the two changes could be adopted separately.

---

## 3. "Brute-force round-after-round search would be neatest but explodes" — verified

✅ **Both halves are correct**, and the numbers are worth having.

v20.7.9 samples **11 v × 81 ω = 891** candidates per step (matches the
documented 891), with a 30 s horizon at `dt = 0.1 s` = **300 control steps**. A
true sequence search over `k` rounds costs `891^k`:

| depth k | sequences | wall-clock for one case (at ~40 min/case today) |
|---|---|---|
| 1 (today) | 8.9e2 | 40 min |
| 2 | 7.9e5 | 25 days |
| 3 | 7.1e8 | 60 years |
| 4 | 6.3e11 | 54,000 years |
| 300 (full horizon) | **1e885** | — |

For scale, there are ~1e80 atoms in the observable universe. Even depth 2 is
already off the table for a suite that re-runs.

**The useful reframing:** v20.7.9's ramped rollout *is* a cheap approximation of
that search. It does not give up on multi-step manoeuvres — it restricts them to
a one-parameter family ("ramp toward a setpoint at `max_accel`, then hold"), so
the cost stays **linear** in sample count and **constant** in horizon length,
while still representing the accelerate-away escape that the one-step window
could not. That is why it works. Any future feasibility test should preserve that
property: pick a closed-form or single-rollout criterion, never a tree.

---

## 4. "predT5 is meant to fail, so it's not too bad" — right, with one important exception

✅ **`predT5` is designed to fail.** It exists to demonstrate that a short
prediction horizon is insufficient. It fails under both versions, and that is the
intended outcome. Nothing disputes this.

❌ **"Those cases were already unavoidable collisions when v20.7.8-2 aborted"** —
**tested 2026-09-12, and false.** An escape existed at the abort state: full
speed with a 32% starboard turn holds **+0.0173 u** of clearance, where braking
straight ahead — what v20.7.9 actually does — gives **−0.5060 u**. The
baseline's abort was a *save*, the last moment from which the situation was
still recoverable, not merely a report. So v20.7.9 removed a save, not only a
warning. Full command grid below.

❌ **"So it's not too bad" does not follow**, and the README pre-empts exactly
this inference: the missing Guard A is **not `predT5`-only**. `v = 0` is always
admissible under any v20.7.9 config, `predT30` included. `predT5` is merely the
setting where the planner actually reaches a cornered state, so it is the only
place the loss is *observable*. In the shipping config the guard is equally gone;
it has simply never been called on, and **no test in the suite exercises it**.

So the accurate statement is: *the collision at predT5 is intended and
uninteresting; the silent removal of Guard A everywhere is unintended and is the
actual regression.*

---

## 5. The New York aborts are the same code path v20.7.9 already handles

Both failing AIS cases report **`No best trajectory found`** — Guard B. By §1
that means admissible candidates existed and every one of them was predicted to
collide somewhere inside the horizon.

**This is precisely the case `no_admissible_fallback: "least_bad"` was written
for**, and it is already implemented in `dwa_ramped.py`. The AIS configs do not
get it only because they run `dwa_average_weighted_side`, where the fallback does
not exist.

Instrumented at the failing call, both cases:

| | `ny_synth_00` | `ny_hist_366962130` |
|---|---|---|
| candidates in the dynamic window | 451 | 451 |
| passed braking admissibility | **164 (36%)** | **44 (10%)** |
| rejected by it | 287 (64%) | 407 (90%) |
| of the survivors, scored finite | **0** | **0** |

In both, admissibility passed something and every survivor then collided inside
the horizon. `ny_hist_366962130` is the sharper case: at `v = 0` the braking
test always passes, so the 44 survivors are essentially the whole `v = 0` row —
and they score `inf` because 16 live vessels are predicted forward over 120 s
and one closes on a stationary ego. Going forward is blocked by a phantom,
holding station is unsafe, and the planner aborts rather than choosing the least
bad of the three. That is exactly the situation `least_bad` was written for.

The braking test prunes two thirds of the search — every candidate that does not
slow down — but it is not what ends the run. `ny_ais_suite/README.md` originally
attributed the abort to admissibility and has been corrected. It also compared
the braking requirement (86.2 m) against a straight-line hull clearance (18.0 m);
those are incommensurable. `closest_obstacle_on_curve` returns distance *along
the candidate arc* to the collision zone, which here is 97–100 m — so the window
floor at 2.69 m/s passes and the ceiling at 3.34 m/s does not.

This does **not** dissolve the braking problem — a test that demands a 147 m ship
be able to stop is still wrong, and it still prunes most of the search space. But
it relocates it: braking admissibility is a **cost/quality** problem (it throws
away good manoeuvres), not the **abort** problem. The abort problem is Guard B,
and it has a known fix.

---

## 6. CPA/TCPA — yes, it is already in there, but on the wrong side of the decision

**The short answer: yes, the dynamic-obstacle cost already computes a sampled,
time-discounted CPA. What has no CPA anywhere in it is the feasibility test.**

What `calc_trajectory_clearance_and_collision` does:

```
predicted_ob_pos = ob_pos + ob_vel · t          # constant-velocity extrapolation
clearance(t)     = |ego(t) − ob(t)| − (r_ob + r_ego)
```

That is the CPA construction exactly — relative motion under constant target
velocity. Then:

```
cost_side_vec(t) = barrier(clearance(t))         # gaussian or quadratic
cost_direct_vec(t) = exp(−λ·t)                   # time discount
cost = aggregate(cost_side_vec · cost_direct_vec)  # "mean" or "max"
```

With `dynamic_ob_cost_aggregation: "max"` and `λ → 0`, `argmax` of that product
is `argmin` of `clearance(t)` — **the CPA, and its index is the TCPA**. So the
intuition is right: this is a soft, sampled generalisation of CPA/TCPA, not
something missing from the design.

Four real differences remain:

1. **Sampled, not closed-form.** The true minimum can fall between samples. In
   the AIS suite at `dt = 10 s` this understates proximity by ~20% (planner sees
   20.7 m where the truth is 16.5 m).
2. **`"mean"` blurs it.** The AIS configs use `mean`, a time-weighted average
   over the horizon, which is *not* a CPA — it is the average of a barrier.
   `"max"` is the CPA-like reduction. (This is what `v20.7.8-3` changes, and it
   is still unevaluated.)
3. **It is a barrier, not a constraint.** Nothing says "reject if CPA < d_safe".
   Only `clearance ≤ 0` — actual overlap — produces `inf`. There is no
   *standoff* that is forbidden, only penalised.
4. **The time discount points the wrong way for COLREGs.** `exp(−λ·t)`
   *down-weights* far-future encounters, so the planner reacts late. "Early and
   substantial action" means acting *while TCPA is still large*. A TCPA-gated
   urgency term would be the opposite sign to the current discount. This is a
   genuine open design question, not a bug.

**And the point that ties this document together:** the *feasibility* gate
shares none of it. `closest_obstacle_on_curve`, which feeds both the braking
admissibility test and the static cost, treats **dynamic obstacles as stationary
at their current position** and checks the whole arc without time limits. So the
planner scores candidates with a relative-motion CPA and then admits or rejects
them with a static-world braking rule. The two halves disagree about what a
moving ship even is.

That makes the natural design move the reverse of "add CPA to the cost":

> **Move the CPA that already exists in the cost onto the feasibility side, and
> retire the braking test there.**

It also explains the stern-sector problem in one line: stopping looks safe to a
braking test built for static obstacles, and looks fatal to a CPA — and only the
cost side can see it, where it is merely one term among four rather than a veto.

---

## 7. How all of this relates to the existing backlog

Same problem seen from two ends, versus genuinely separate work:

| # | item | status | relationship |
|---|---|---|---|
| 1 | **Capsule in `closest_obstacle_on_curve`** | not started | **Prerequisite for 4, 5 and 6.** The one hull site still on the disc. Needs no new config key — `robot_collision_shape` already exists, default `disc`. |
| 2 | **`on_start` for replayed tracks** | not started | **Independent.** Scene-input correctness; 17 of 78 vessels are phantoms in one AIS scene. Unrelated to feasibility. |
| 3 | **`ais_v1.2` — per-ego `dt`** | evidenced, not adopted | **Independent** (config-only), but *measured against* 1 and 2, so re-baseline first. |
| 4 | **Guard B fallback for the AIS configs** | exists in `dwa_ramped`, absent from `dwa_average_weighted_side` | **Same problem as v20.7.9's.** Both AIS aborts are Guard B. Porting `least_bad` is the smallest possible fix. |
| 5 | **Restore Guard A in a form a wide window cannot bypass** | open, **upgraded** | **Same problem as `ais_v2`.** Evaluate admissibility against *predicted* obstacle positions and the braking test stops being a static-world rule. |
| 6 | **Replace braking feasibility with turn/CPA** | open | **The synthesis of 4, 5 and §6.** This is `ais_v2` and the stern-sector guard at once. |
| 7 | **Stern-sector cost term** | open | **Overlaps 6.** If feasibility becomes CPA-based, "slowing while overtaken" may fall out without a bespoke term. Do not build it first. |
| 8 | **`v20.7.8-2 predT5 + least_bad`, narrow window** | **✅ done 2026-09-12** | Answered, though not as intended — see below. The abort **was a save**. |

So: **items 4, 5, 6 and 7 are one problem, not four.** They are the same
feasibility test viewed from the open-water suite, the AIS suite and the COLREGs
angle. Items 1, 2, 3 and 8 are genuinely separate and much smaller.

The messiness was real but it was mostly naming: "no admissible" refers to two
different guards, and `ais_v2` and "the stern-sector guard" are the same piece of
work under two names.

---

### Item 8: run 2026-09-12 — and the answer is **no, it was avoidable**

Two configs isolate `least_bad` from the window widening
(`v20.7.9/predT5-narrow-control.json` and `…-leastbad.json`, differing in one
key). Both were run against `owenc_overtaken_ego`.

**Result 1 — the control is exact.** `dwa_ramped` with `v_window_ramp_time: 0.0`
and `ramped_rollout: false` reproduces `v20.7.8-2/predT5` **bit-identically**:
876 steps, max |diff| = 0.000e+00, same Guard A abort at iteration 875. The
ramped module is a strict superset of the baseline — now verified rather than
assumed, which makes it a usable vehicle for ablations of this kind.

**Result 2 — the experiment as designed cannot answer the question.** Enabling
`least_bad` changed *nothing*: also 876 steps, also bit-identical, same abort.
That is §1 confirmed empirically. The baseline aborts on **Guard A**, and
`least_bad` only ever catches **Guard B**, so adding it to a run that fails on
Guard A is a no-op. The premise of the experiment was wrong.

**Result 3 — the question was answered directly instead**, by backward
reachability from the cornered state rather than by running a planner. Take the
baseline's own state at iteration 875 and ask whether *any* constant command
escapes, using the planner's own `motion()` (validated bit-exact against the last
five logged steps) and the target's actual motion (validated to 4.9e-13 against
the recorded collision position):

| command from the abort state | min clearance over 3 s |
|---|---|
| **best of a 241 × 26 grid — `v = 0.500`, `ω = −0.227` (32% starboard)** | **+0.0173 u** |
| hard turn to starboard | +0.0034 u |
| hold course | −0.0004 u |
| brake straight ahead (what v20.7.9 does) | **−0.5060 u** |
| hard turn to port | −0.6694 u |

**An escape existed.** The abort was a *save* — it fired at the last moment from
which the situation was still recoverable, with 0.6 s and 0.0563 u of clearance
left. The margin is thin (0.017 u against a 1.15 u threshold), and it is a lower
bound: a re-planning controller can do at least as well as the best constant
command.

**And the escape is exactly what the feasibility test forbids.** At `v = 0.490`
the braking rule demands `dist > v²/(2a) + v·dt` = **0.649 u** while only
**0.0563 u** is available, so every candidate carrying speed is rejected and
Guard A fires. But the only manoeuvre that survives needs `v = 0.500` — *more*
speed — plus a turn. The ego is at 0.490 against a target closing at 0.600: it
cannot outrun it and it cannot stop. Turning at full speed is the whole of its
escape set, and the test discards that set first.

This is the sharpest available evidence for item 6 (§7). It is no longer an
argument from ship physics; it is a measured case in which the static-world
braking rule throws away the unique surviving manoeuvre, after which the planner
correctly reports that it has none.

**Consequences.** Item 8 is closed. Item 5 is upgraded: Guard A reports a real,
recoverable emergency rather than an inevitability, so restoring it in a form the
wide window cannot bypass matters more than previously judged — and `least_bad`
is confirmed to be no substitute for it.

## 8. Suggested order

1. **Item 8** — one run, settles whether the baseline's abort was a save. Cheap,
   and it changes how item 5 should be judged.
2. **Item 1** — capsule in `closest_obstacle_on_curve`. Small, no new config key,
   unblocks everything else.
3. **Item 2** — `on_start`, defaulting to today's behaviour.
4. **Re-baseline the AIS suite** on the corrected scene and hull.
5. **Item 4** — port `least_bad` to `dwa_average_weighted_side` (or run AIS on
   `dwa_ramped`). Turns both AIS aborts into degraded-but-continuing runs and
   makes the remaining failures legible.
6. **Item 6** — design the replacement feasibility test. Only now, with 1–5 done,
   are the inputs to that decision trustworthy.
7. **Item 3** (`ais_v1.2`) whenever convenient — it is orthogonal.

Every step must stay config-restorable: new behaviour behind a key that defaults
to the current one, per `../configs/algorithm/README.md`.

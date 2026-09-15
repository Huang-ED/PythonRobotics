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
three independent axes — and Stage 1 is the wrong one on all three:**

| axis | Stage 2 (cost) | Stage 1 (feasibility) | backlog |
|---|---|---|---|
| **obstacle motion** | `ob_pos + ob_vel·t`, relative motion | **frozen** — the function takes no `ob_vel` argument at all | item **5** |
| **ego hull** | capsule when configured, half-beam `W/2` | **always the circumscribed disc** `sqrt(L²+W²)/2` — **3.16×** wider for a 19 × 6.33 m ego | item **1** |
| **look-ahead** | bounded by `predict_time_obstacle` | **unbounded** — "no span restriction", scans the full circle | item **9** (new) |

That table *is* the problem statement. Everything in §5 is either repairing one
of those three rows, replacing the whole of Stage 1, or cleaning up inputs so the
repair can be measured.

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
| raised at | [dwa_average_weighted_side.py:534][a] (`len(to_goal_costs) == 0`) | [:551][b] (`best_index == -1`) |
| means | **Stage 1** emptied the set | Stage 1 passed something, **Stage 2** emptied it |
| what makes it fire | `v² + 2·a·v·dt > 2·a·dist` for all `(v, ω)` | `is_collision` inside the horizon for all survivors |
| caught by `least_bad`? | **no** — still raises unconditionally | **yes** — this is the case the fallback handles |

[a]: ../../DWAT_v4_st/dwa_average_weighted_side.py#L534
[b]: ../../DWAT_v4_st/dwa_average_weighted_side.py#L551

**`no_admissible_fallback` is misleadingly named.** It does not catch the "no
admissible pairs" guard. [dwa_ramped.py:293][c] gates on `best_index == -1 and
len(controls) > 0` — meaning admissible candidates existed. Guard A at
[dwa_ramped.py:291][d] still raises regardless of the setting. The key would be
better named `all_candidates_collide_fallback`.

[c]: ../../DWAT_v4_st/dwa_ramped.py#L293
[d]: ../../DWAT_v4_st/dwa_ramped.py#L291

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
| **1** | **Capsule in `closest_obstacle_on_curve`** | not started | Row 2 of the §1 table. The one hull site still on the disc. No new config key — `robot_collision_shape` exists, default `disc`. **Candidate fix for `ny_hist_366962130`.** |
| **5** | **Admissibility against *predicted* obstacle positions** | open, **upgraded** | Row 1 of the §1 table. Formerly "restore Guard A in a form a wide window cannot bypass" — same thing: give Stage 1 the obstacle velocities and it stops being a static-world rule. |
| **9** | **Bound Stage 1's look-ahead to the horizon** | **new — was unnamed** | Row 3 of the §1 table. Impact unmeasured; one instrumented run from being known. |
| **6** | **Replace braking feasibility with turn/CPA** | open | The synthesis of 1, 5 and 9. This is `ais_v2` and the stern-sector guard **at once** — one piece of work under two names. |
| **7** | **Stern-sector cost term** | open, **likely unnecessary** | Per §3(b), falls out of item 6. **Do not build it first**, and possibly not at all. |

Items 1, 5 and 9 are the same defect in three fields of the same function. Item 6
is what you get if you fix all three at once and stop patching.

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
  resolve. Only the *grouping* is new; item 9 is the one addition.

---

## 6. Order of work

**The principle: fix what is wrong with the measurement before designing against
it.** Items 1 and 2 both change what the AIS suite *reports*. Anything measured
before them is measured on a scene containing a ship that is not there, judged
against a hull three times too wide.

1. **Item 1** — capsule in `closest_obstacle_on_curve`. Small, no new config key,
   and a candidate fix for `ny_hist_366962130` in its own right.
2. **Item 2** — `on_start`, defaulting to today's behaviour.
3. **Re-baseline the AIS suite** on the corrected scene and hull. Expect
   `ny_hist_366962130` to change outcome; if it does, it leaves the feasibility
   backlog entirely and `ny_synth_00` becomes the only AIS instance of it.
4. **Measure what Stage 1 prunes — before designing anything.** At the
   `ny_synth_00` abort, score the 287 candidates Stage 1 rejected. If any is
   collision-free, Stage 1 causes that abort too and item 6 is the whole job; if
   none is, item 4 really is the answer there. **One instrumented call, and it
   decides everything below it.** Take the item 9 (unbounded-span) number in the
   same pass.
5. **Item 4** — port `least_bad` to `dwa_average_weighted_side` (or run AIS on
   `dwa_ramped`). Do this regardless of how step 4 turns out, for legibility.
6. **Item 6** — design the replacement feasibility test, absorbing items 5 and 9.
   Only now are its inputs trustworthy. Item 7 probably falls out; do not build it
   separately.
7. **Item 3** (`ais_v1.2`) — orthogonal, so "whenever convenient". But it fixes
   half the suite's failures, so *convenient* should not mean *never*.

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

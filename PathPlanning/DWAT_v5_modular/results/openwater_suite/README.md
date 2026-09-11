# Open-water encounter suite — results

The 11 single-target open-water cases (`owenc_*`), run on the featureless
100×100 basin so that a manoeuvre is attributable to the planner rather than to
a coastline. Baseline **v20.7.8-2**.

**Under v20.7.8-2: 9 of 11 reach the goal, 2 fail** — both the "ego is
overtaken" geometry, and both for a reason that is not about overtaking. The
cause and the fix are below; **under v20.7.9 all 11 reach the goal** with the
other nine effectively unchanged. v20.7.9 is a **prototype, not yet a
replacement** — it removes a safety guard without replacing it. Full record:
`../v20.7.9_ramped_window/README.md`.

Reference run for deviation and delay: `openwater_baseline` at **153.3 s**
(`runs/v20.7.8-2/baseline/openwater_baseline__20260827_184123/`). Note there is
a second, longer baseline run (167.3 s); scoring against that one silently
produces wrong deviations, so use the 153.3 s run.

## Results

Clearance is surface-to-surface: centre distance minus the target's radius and
the ego's half-diagonal. Deviation is the peak distance from the obstacle-free
baseline; delay is the extra time to goal.

| case | goal | clearance | t_cpa | side at CPA | deviation | delay | v_min | COLREGs sector |
|---|---|---|---|---|---|---|---|---|
| `head_on` | yes | **0.48** | 90.4 | starboard | 1.74 | 0.7 s | 0.50 | head-on (R14) |
| `cross_stbd_045` | yes | 0.58 | 89.8 | port | 1.79 | 0.7 s | 0.50 | crossing from starboard (R15, give-way) |
| `cross_port_045` | yes | 0.57 | 89.8 | starboard | 1.79 | 0.7 s | 0.50 | crossing from port (R17, stand-on) |
| `cross_stbd_090` | yes | 0.77 | 88.3 | port | 2.17 | 1.0 s | 0.50 | crossing from starboard (R15, give-way) |
| `cross_port_090` | yes | 0.76 | 88.3 | starboard | 2.15 | 1.0 s | 0.50 | crossing from port (R17, stand-on) |
| `cross_stbd_135` | yes | 0.94 | 83.8 | port | 2.97 | 1.9 s | 0.43 | crossing from starboard (R15, give-way) |
| `cross_port_135` | yes | 0.94 | 83.8 | starboard | 2.98 | 1.9 s | 0.43 | crossing from port (R17, stand-on) |
| `overtake_ego` | yes | 1.21 | 87.8 | starboard | 2.47 | 1.1 s | 0.50 | overtaking, ego overtakes (R13, give-way) |
| `null_safe_pass` | yes | 4.88 | 90.0 | port | **0.00** | 0.0 s | 0.50 | control — target was never a threat |
| `overtaken_ego` | **NO** | — | — | — | — | — | — | overtaken (R13/R17, stand-on) |
| `overtaken_ego_fast` | **NO** | — | — | — | — | — | — | overtaken (R13/R17, stand-on) |

Two readings worth keeping straight:

- **`null_safe_pass` inverts the pass condition.** The target passes 6 units
  clear, so a deviation of 0.00 is the correct result and a large manoeuvre
  would be the failure. It is the control against over-reaction.
- **`side at CPA` is recorded but not scored.** The dynamic-obstacle cost is
  mirror-symmetric in port/starboard, so the planner has no side preference to
  measure. It is the baseline to diff against once a COLREGs side rule exists.

The mirrored pairs agree to two decimal places — 0.58/0.57, 0.77/0.76,
0.94/0.94, with deviations 1.79/1.79, 2.17/2.15, 2.97/2.98. That is the
empirical confirmation of the cost function's mirror symmetry.

## The two failures

Both abort on **iteration 0** with `No best trajectory found`. The cause is the
prediction horizon, not the encounter:

At `t = 0` the ego is at rest, so the dynamic window admits only
`v ∈ [0, 0.02]` — a single acceleration step — yet every candidate is scored by
holding that velocity across the full `predict_time_obstacle = 30 s`. The
planner predicts itself crawling 0.6 units while the target, spawned directly
astern and closing, advances 18 and passes through it. Every candidate scores
as colliding, so there is nothing to choose.

Established by construction rather than argued:

| horizon | `owenc_overtaken_ego` |
|---|---|
| `predT30` (baseline) | fails, iteration 0 — `No best trajectory found` |
| **`predT10`** | **reaches goal** — clearance 0.91, deviation 2.94, v held at 0.50 |
| `predT5` | fails differently — `No admissible (v, ω) pairs found in dynamic window` |

So a long horizon can render infeasible a situation a shorter one handles.
This will bite *any* case where a target is inbound on the ego's present
position while the ego is slow; overtaking is merely the geometry that
guarantees it. Both variants are kept as the regression test.

**Shortening the horizon is not the fix.** It only lowers the spawn distance
that triggers the abort — the condition is
`predT >= (gap - 1.15) / (v_target - 0.02)`, so `predT10` still breaks on any
target spawning within ~7 u astern and `predT5` within ~4 u. Measured
iteration-0 survival, target dead astern at 0.6 u/s:

| spawn gap | predT5 | predT10 | predT15 | predT20 | predT30 |
|---|---|---|---|---|---|
| 4.0 u | 27% | 0% | 0% | 0% | 0% |
| 6.0 u | 100% | 0% | 0% | 0% | 0% |
| 8.0 u | 100% | 100% | 0% | 0% | 0% |
| 10.55 u | 100% | 100% | 100% | 0% | 0% |
| 14.0 u | 100% | 100% | 100% | 100% | 0% |

**The two cases are the same failure but not the same encounter.** At iteration
0 they are indistinguishable — 243/243 candidates colliding, predicted contact
at 16.21 s and 16.28 s — because their differences (10.55 vs 13.03 u astern,
0.58 vs 0.73 u/s closing) very nearly cancel. Their designed meeting points
differ though (t=100 s at y=35.6 vs t=50 s at y=60.6), so once the planner can
actually move they diverge: true contact at 88 s vs 45 s. While the bug stands
they are duplicate tests; only with it fixed do they probe different margins.

Until 2026-09-05 these two cases wrote no log at all — `save_log` returned
before writing whenever there were no log entries, so an iteration-0 abort left
nothing on disk. They now record the error and traceback.

Three figures in this directory: `overtaken_failure.png` (the mechanism and the
horizon sweep), `overtaken_frame0.png` (the first frame the aborted run would
have drawn — the reachable set is smaller than the ship), and
`overtaken_target_tracks.png` (both planned target legs, and what separates
them).

## The fix — v20.7.9

The dynamic window and a 30 s horizon are conceptually incompatible. Classic
DWA uses `predict_time` of 1–3 s, where "hold v constant" is a fair
approximation and one-`dt` reachability is nearly the whole reachable set. At
30 s neither holds: the window offers `v <= 0.02` while the ego can really
cover 14.4 u. The bug is in how candidates are **scored**, not in how the ship
moves — the executed trajectories are max-accel either way, because the
baseline's `v + a*dt` upper bound *is* the max-accel step.

`v20.7.9` (`PathPlanning/DWAT_v4_st/dwa_ramped.py`, config
`configs/algorithm/v20.7.9.json`) makes window and rollout agree:

- `v_window_ramp_time: 2.5` — the v window spans what the horizon can reach
  (`max_speed / max_accel`), so a sampled v is a *target speed*, not a
  next-step velocity. At rest the window becomes `[0, 0.50]` instead of
  `[0, 0.02]`, and the predicted 30 s reach 14.40 u instead of 0.60 u.
- `ramped_rollout: true` — a sampled `(v, omega)` is approached at
  `max_accel` / `max_delta_yaw_rate`, in the rollout *and* in execution.
  Widening the window without this would let the simulated ego jump
  0 -> 0.5 u/s in one `dt` and silently violate `max_accel`.
- `no_admissible_fallback: "least_bad"` — when every candidate scores `inf`,
  keep the one furthest from contact instead of raising. Iteration 0 now has
  596 of 891 candidates collision-free, so the widened window carries the fix
  on its own and this rarely fires. **It is not the only guard that changed:**
  widening the window also made the separate `No admissible (v, ω) pairs`
  abort unreachable — see the caveat below.
- `v_resolution: 0.05` — the v window went from 3 samples to the full range;
  coarsening keeps the count near the baseline (891 vs 243, ~3.7x slower).

The search stays a 2-D grid over `(v, omega)` — one setpoint per candidate,
with the acceleration profile generated by the rollout. Cost grows linearly
with the sample count, not exponentially; that only happens if you search
*sequences* of commands.

Results, scored against the v20.7.9 obstacle-free baseline (also 153.3 s,
within 0.0003 u of the v20.7.8-2 one). Columns are clearance / deviation /
delay:

| case | v20.7.8-2 | v20.7.9 | t_cpa | v_min | vs baseline |
|---|---|---|---|---|---|
| `head_on` | 0.48 / 1.74 / 0.7 | 0.48 / 1.74 / 0.7 | 90.4 | 0.50 | same |
| `cross_stbd_045` | 0.58 / 1.79 / 0.7 | 0.58 / 1.79 / 0.7 | 89.8 | 0.50 | same |
| `cross_port_045` | 0.57 / 1.79 / 0.7 | 0.57 / 1.79 / 0.7 | 89.8 | 0.50 | **moved** |
| `cross_stbd_090` | 0.77 / 2.17 / 1.0 | 0.77 / 2.17 / 1.0 | 88.3 | 0.50 | same |
| `cross_port_090` | 0.76 / 2.15 / 1.0 | 0.76 / 2.15 / 1.0 | 88.3 | 0.50 | same |
| `cross_stbd_135` | 0.94 / 2.97 / 1.9 | 0.94 / 3.00 / 1.8 | 83.8 | 0.45 | **moved** |
| `cross_port_135` | 0.94 / 2.98 / 1.9 | 0.94 / 3.00 / 1.8 | 83.8 | 0.45 | **moved** |
| `overtake_ego` | 1.21 / 2.47 / 1.1 | 1.21 / 2.47 / 1.1 | 87.8 | 0.50 | same |
| `null_safe_pass` | 4.88 / 0.00 / 0.0 | 4.88 / 0.00 / 0.0 | 90.0 | 0.50 | same |
| `overtaken_ego` | **failed** | 1.15 / 3.15 / 1.5 | 109.5 | 0.45 | — |
| `overtaken_ego_fast` | **failed** | 0.99 / 2.44 / 1.1 | 52.0 | 0.50 | — |

All 11 reach the goal, none collide. Seven of the nine previously-working cases
are unchanged to two decimals, and their trajectories match v20.7.8-2 to
**0.0003 u** over identical step counts — the change is invisible in normal
operation. `null_safe_pass` holds at 0.00 deviation, so the 25x wider speed
window bought no over-reaction. The `135` pair moved slightly (deviation
2.97/2.98 -> 2.995/2.995, delay 1.9 -> 1.8 s, v_min 0.43 -> 0.45) and in doing
so became exactly symmetric: mirrored-pair |delta dev| went from 0.01 to 0.000.

The two repaired cases now diverge as designed — `t_cpa` 109.5 s vs 52.0 s,
tracking the manifest's 100 s and 50 s meeting times — so they have finally
become the two distinct regression tests the suite always claimed they were.
`overtaken_ego` also beats the `predT10` workaround (clearance 1.15 vs 0.91)
without shortening the horizon for every other case.

Two approximations remain in the prototype, both noted in the module docstring:
the yaw window is still one `dt` (unlike v, a yaw rate genuinely *is* reachable
within one `dt`, so it is already self-consistent), and
`closest_obstacle_on_curve` still assumes a constant-velocity arc for
admissibility and static-obstacle cost, so it under-estimates travel during the
ramp. Neither bites in open water; both could in a crowded scene.

Baseline reproducibility was re-verified after the change: `owenc_head_on` at
v20.7.8-2 still reproduces its stored trajectory bit-exactly (1541 points,
max abs diff 0.0). The three new `Config` attributes default to the classic
behaviour and are read only by `dwa_ramped`.

### The caveat — v20.7.9 removes an abort guard

`predT5` is *meant* to fail — it exists to show a short horizon cannot see a
0.1 u/s overtake coming. It fails under both versions, as intended. What
changed is the **manner**: v20.7.8-2 aborted at iteration 875 with
`No admissible (v, ω) pairs` and 0.065 clearance; v20.7.9 **collides** at
iteration 881 (clearance −0.034).

The admissibility test rejects `(v, ω)` when
`v² + 2·a·v·dt > 2·a·dist`. At `v = 0` that is `0 > 0.4·dist`, never true — so
`v = 0` is always admissible. The baseline's narrow window at cruise offered
only `v ∈ [0.48, 0.50]`, all of which failed; v20.7.9's window always contains
`v = 0`, so **the abort has become unreachable**. The planner then slows
(0.50 → 0.36) while a 0.6 u/s ship closes from dead astern, and is run down.

Slowing is the worst available response to being overtaken, and nothing in the
cost function knows that. This is latent in v20.7.8-2 too; the narrow window
merely hides it.

Two qualifications keep this the right size. The baseline's abort was **not a
save** — it fired a fraction of a second from contact and reported that no
certifiable move remained, rather than avoiding anything; what v20.7.9 removed
is the *warning*. But the missing guard is **not** confined to `predT5`:
`v = 0` is always admissible under the widened window, so the abort is
unreachable at `predT30` too — it has simply never been called on there. None
of the 11 cases here would catch it, because at `predT30` the planner never
chooses to slow. Full analysis: `../v20.7.9_ramped_window/README.md` §5.

The crowded scenes were checked for the same reason and show no regression:
`v3crowd6` and `v3crowd7` still reach goal with clearances within 0.01 of
baseline, and `v3crowd5` fails identically (a `_choose_local_goal` divergence,
unrelated). See `../v3crowd_suite/README.md`.

## Where the runs are

`runs/` is git-ignored, so these paths are local.

| case | run directory (under `runs/v20.7.8-2/`) | artifacts |
|---|---|---|
| `head_on` | `encounters/owenc_head_on__20260904_170445/` | log, frames, video |
| `cross_stbd_045` | `encounters/owenc_cross_stbd_045__20260904_170445/` | log, frames, video |
| `cross_port_045` | `encounters/owenc_cross_port_045__20260904_170445/` | log, frames, video |
| `cross_stbd_090` | `encounters/owenc_cross_stbd_090__20260904_170445/` | log, frames, video |
| `cross_port_090` | `encounters/owenc_cross_port_090__20260904_172157/` | log, frames, video |
| `cross_stbd_135` | `encounters/owenc_cross_stbd_135__20260904_172158/` | log, frames, video |
| `cross_port_135` | `encounters/owenc_cross_port_135__20260904_172159/` | log, frames, video |
| `overtake_ego` | `encounters/owenc_overtake_ego__20260904_172308/` | log, frames, video |
| `null_safe_pass` | `encounters/owenc_null_safe_pass__20260904_173905/` | log, frames, video |
| `overtaken_ego` | `encounters/owenc_overtaken_ego__20260904_170339/` | log only (aborted) |
| `overtaken_ego_fast` | `encounters/owenc_overtaken_ego_fast__20260904_170341/` | log only (aborted) |
| `overtaken_ego` @ predT10 | `predT10/encounters/owenc_overtaken_ego__20260904_173918/` | log, frames, video |

And the v20.7.9 runs (under `runs/v20.7.9/`), baseline
`baseline/openwater_baseline__20260909_040103/`:

| case | run directory |
|---|---|
| `head_on` | `encounters/owenc_head_on__20260909_031126/` |
| `cross_stbd_045` | `encounters/owenc_cross_stbd_045__20260909_173857/` |
| `cross_port_045` | `encounters/owenc_cross_port_045__20260909_173857/` |
| `cross_stbd_090` | `encounters/owenc_cross_stbd_090__20260909_031126/` |
| `cross_port_090` | `encounters/owenc_cross_port_090__20260909_031126/` |
| `cross_stbd_135` | `encounters/owenc_cross_stbd_135__20260909_173857/` |
| `cross_port_135` | `encounters/owenc_cross_port_135__20260909_173857/` |
| `overtake_ego` | `encounters/owenc_overtake_ego__20260909_173857/` |
| `null_safe_pass` | `encounters/owenc_null_safe_pass__20260909_031126/` |
| `overtaken_ego` | `encounters/owenc_overtaken_ego__20260909_031126/` |
| `overtaken_ego_fast` | `encounters/owenc_overtaken_ego_fast__20260909_031126/` |

Every run above reproduces its original 2026-08-27 trajectory bit-exactly.

## Reproducing

```bash
# one case
python -m PathPlanning.DWAT_v5_modular.main \
    --scenario owenc_head_on --algorithm-config v20.7.8-2

# score the suite (use the 153.3 s baseline)
python PathPlanning/DWAT_v5_modular/adapters/encounter_report.py \
    --baseline-log runs/v20.7.8-2/baseline/openwater_baseline__20260827_184123/log_details.json \
    --manifest PathPlanning/DWAT_v5_modular/configs/scenarios/encounters/owenc_manifest.json \
    --runs "runs/v20.7.8-2/encounters/owenc_*/log_details.json"

# the same case under the fix
python -m PathPlanning.DWAT_v5_modular.main \
    --scenario owenc_overtaken_ego --algorithm-config v20.7.9

# score v20.7.9 against its own obstacle-free baseline
python PathPlanning/DWAT_v5_modular/adapters/encounter_report.py \
    --baseline-log runs/v20.7.9/baseline/openwater_baseline__20260909_040103/log_details.json \
    --manifest PathPlanning/DWAT_v5_modular/configs/scenarios/encounters/owenc_manifest.json \
    --runs "runs/v20.7.9/encounters/owenc_*/log_details.json"
```

The contact sheet for these cases is
`../encounter_suite/openwater_encounters.png`; the design rationale is
`../encounter_suite/dwat_encounter_suite_rationale.html`.

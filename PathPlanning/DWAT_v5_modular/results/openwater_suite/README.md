# Open-water encounter suite — results

The 11 single-target open-water cases (`owenc_*`), run on the featureless
100×100 basin so that a manoeuvre is attributable to the planner rather than to
a coastline. Baseline **v20.7.8-2**.

**9 of 11 reach the goal. 2 fail**, both the "ego is overtaken" geometry, and
both for a reason that is not about overtaking — see below.

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

Until 2026-09-05 these two cases wrote no log at all — `save_log` returned
before writing whenever there were no log entries, so an iteration-0 abort left
nothing on disk. They now record the error and traceback.

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
```

The contact sheet for these cases is
`../encounter_suite/openwater_encounters.png`; the design rationale is
`../encounter_suite/dwat_encounter_suite_rationale.html`.

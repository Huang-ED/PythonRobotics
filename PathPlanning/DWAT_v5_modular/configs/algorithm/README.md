# Algorithm configs — lineage and conventions

What each `v20.*` config was trying to do, why it exists, and which ones are
safe to build on. The lineage below is the developer's own development notes,
**cross-checked against the files** — where a note and the file disagree, that
is called out rather than smoothed over.

**A higher version number does not mean better.** The numbers are
chronological. `v20.8.x` is an abandoned branch; `v20.9.1`–`v20.9.3` are
undocumented; `v20.7.9` is an unadopted prototype. See *Which to use*.

## Layout

```
configs/algorithm/<version>.json             -> version "<version>"
configs/algorithm/<version>/<variant>.json   -> version "<version>/<variant>"
```

A config's **location is its version identifier**, and `runs/` mirrors it
exactly (`runs/v20.7.8-2/`, `runs/v20.7.8-2/predT10/`). Subdirectories hold
ablation sweeps over one parameter, so the sweep reads as a single experiment:

| directory | sweep |
|---|---|
| `v20.7.8-2/` | `predict_time_obstacle` ∈ {5, 10, 12, 15, 18, 20, 30} |
| `v20.7.9/` | `predT5-ramped` — the 5 s point of the same sweep, under v20.7.9 |

**Do not rename or move a config.** `run_paths.version_from_config` derives the
run directory from the path, so renaming silently orphans every existing run in
`runs/<old name>/`. Add a new file instead. (Bare names must also stay unique:
`--algorithm-config predT5` resolves by filename, so two `predT5.json` under
different versions would be ambiguous — hence `predT5-ramped`.)

## Lineage

Notes are the developer's; parameters are verified from the JSON.

| version | intent (developer's notes) | verified distinguishing params |
|---|---|---|
| `v20.7.1` | Exponential weighting — time | *file not in repo* |
| `v20.7.2` | Polynomial weighting — distance | *file not in repo* |
| `v20.7.3` | The same time weighting (test) | *file not in repo* |
| `v20.7.4` | predT_obstacle 30; time weighting: same; distance: parabolic | *file not in repo* |
| `v20.7.5` | predT_obstacle 30; time: exponential λ=0.1; distance: parabolic | λ=0.1, safe_r 3.0; **barrier type not set — see hazard below** |
| `v20.7.6` | **Ablation of v20.7.5: time decay removed.** Multi movable obstacles implemented from here | module `dwa_average_weighted_side_no_time_decay` |
| `v20.7.7` | Time decay back on; distance weighting quadratic → **gaussian, decay 3.0** | barrier `exponential`, decay 3.0 |
| `v20.7.8` | Gaussian decay **6.0** | decay 6.0, safe_r 3.0 |
| `v20.7.8-2` | Decay 6.0 + **distance safety range 2.0** (soft boundary). Works for simple maps, *not* for more realistic ones | decay 6.0, **safe_r 2.0** |
| `v20.7.8-3` | Based on -2, but **max** of distance×time cost instead of integration | `dynamic_ob_cost_aggregation: "max"` |
| `v20.7.9` | Fix the open-water "ego is overtaken" failures — reconcile the dynamic window with the 30 s horizon | module `dwa_ramped`; `v_window_ramp_time` 2.5, `ramped_rollout`, `least_bad` |
| `v20.8.1` | **Static obstacles made consistent with the dynamic-obstacle cost.** Does not work on `video5` | module `dwa_consistent`, barrier `quadratic` |
| `v20.8.2` | Same branch as v20.8.1 | module `dwa_consistent`, barrier `exponential` decay 6.0 |
| `v20.9` | Tests on scene export | **parameter-identical to v20.7.8-2** — consistent with a scenario-export test rather than an algorithm change |
| `v20.9.1` | *not in the notes* | side_cost_gain **2.4** (from 0.6), safe_r **0.5**, decay **3.0** |
| `v20.9.2` | *not in the notes* | identical to v20.9.1 in these keys |
| `v20.9.3` | *not in the notes* | as v20.9.1 but `predict_time_obstacle` **3.0** |

`ais_ship_si_v1.json` belongs to the New York / AIS import work, not this
lineage.

## v20.8.1 — recorded insights

Why the "static obstacles consistent with dynamic" branch was abandoned, in the
developer's words, with their own confidence estimates:

- **The spatial domain of threat (threat severity) of obstacles needs to be
  reduced for static obstacles — the buffer range needs to be narrowed.**
  *70% confidence, a bit of trouble to implement.*
- **The last section of the ego's own prediction could be a straight line
  instead of a circular curve.** Intuitively this might cause new problems.
  *30% confidence, not difficult to implement.*

Both are unimplemented and untested. They are hypotheses, not conclusions.

## Gaps in the record

- **`v20.7.1`–`v20.7.4` have no config files.** The notes describe them, but
  nothing in this folder corresponds. They cannot be re-run.
- **`v20.9.1`–`v20.9.3` have no notes.** They are a real parameter change, not
  a test harness: `side_cost_gain` 0.6 → 2.4 with `max_side_weight_dist`
  2.0 → 0.5 — a much narrower but far stronger side-clearance barrier, plus
  gaussian decay back to 3.0. `v20.9.3` additionally cuts
  `predict_time_obstacle` to 3.0. **What these were testing, and whether any
  succeeded, is unrecorded — ask before treating them as evaluated.**

## Reproducibility hazard: v20.7.5 and v20.7.6

These two are the only configs that rely on implicit defaults, and the defaults
have moved since they were written. **Re-running them today does not reproduce
what they originally were.**

| config | omits | resolves today to | the notes say |
|---|---|---|---|
| `v20.7.5` | `dwa_module` | **`dwa_consistent`** (main.py's fallback) — the *v20.8 branch* module | predates v20.8 entirely |
| `v20.7.5` | `side_clearance_barrier_type` | `exponential`, decay 3.0 | **parabolic** (i.e. quadratic) |
| `v20.7.6` | `side_clearance_barrier_type` | `exponential`, decay 3.0 | parabolic, inherited from v20.7.5 |

So `--algorithm-config v20.7.5` currently runs the v20.8.x algorithm with a
gaussian barrier — neither of which is v20.7.5. Any comparison against it is
invalid until the keys are pinned explicitly.

**Not fixed here on purpose:** pinning them changes what the configs do, and
the correct values depend on what `runs/v20.7.5/` and `runs/v20.7.6/` were
actually produced with. That is the developer's call, not a guess to be made
silently. Every other config in the folder sets both keys explicitly.

## Which to use

| | status |
|---|---|
| **`v20.7.8-2`** | **the baseline.** Copy this for any new experiment, ablation or comparison |
| `v20.7.8-3` | unevaluated — ask whether it beat v20.7.8-2 |
| `v20.7.9` | prototype, **not a drop-in replacement**: it removes a safety guard. See `../../results/v20.7.9_ramped_window/README.md` |
| `v20.8.1`, `v20.8.2` | abandoned branch — do not derive new configs from these |
| `v20.9.1`–`v20.9.3` | undocumented, unevaluated |
| `v20.7.5`, `v20.7.6` | historical; see the hazard above before running |

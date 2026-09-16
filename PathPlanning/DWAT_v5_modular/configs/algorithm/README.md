# Algorithm configs — lineage and conventions

What each `v20.*` config was trying to do, why it exists, and which ones are
safe to build on. The lineage is the developer's own development notes,
**cross-checked against the files and against git history** — where a note and
the code disagreed, that is recorded rather than smoothed over.

**A higher version number does not mean better.** The numbers are
chronological. See *Which to use* at the bottom.

## Layout

```
<version>.json             -> version "<version>"          e.g. v20.7.8-2
<version>/<variant>.json   -> version "<version>/<variant>" e.g. v20.7.8-2/predT10
```

A config's **location is its version identifier**, and `runs/` mirrors it
exactly. Only live configs sit at the top level; the rest are filed by status:

| folder | holds | runs land in |
|---|---|---|
| *(top level)* | the baseline, its sweep, and whatever is currently being worked on | `runs/v20.7.8-2/` … |
| `_superseded/` | the lineage that led to the baseline (v20.7.5 → v20.7.8) | `runs/_superseded/` |
| `_abandoned/` | the `dwa_consistent` branch (v20.8.x) — do not build on these | `runs/_abandoned/` |
| `_scene_export/` | v20.9.x — attempts at a config for the `scene_export` dataset. **None succeeded yet** | `runs/_scene_export/` |

Subdirectories of a *version* mean something different from these status
folders: they hold a sweep over **one experimental factor**, so the sweep reads
as a single experiment.

The requirement is that the variants be generated from one stated factor by a
rule you can write in a sentence — so that a difference in outcome is
attributable to that factor. It is *not* that exactly one key changes. A sweep
may move several keys at once when they are all derived from the same thing:
the per-ego configs in `ais_v1.1/` each set ten keys, but all ten follow from
one vessel's hull and service speed, so the directory is still one experiment
with one independent variable.

If you cannot state the factor in a sentence, the directory is a grab-bag and
the configs belong at the top level under their own names.

| directory | factor |
|---|---|
| `v20.7.8-2/` | `predict_time_obstacle` ∈ {5, 10, 12, 15, 18, 20, 30} |
| `v20.7.9/` | `predT5-ramped` — the 5 s point of the same sweep, under v20.7.9 |
| `ais_v1.1/` | the ego vessel — hull and service speed, per AIS case |

**Bare names still resolve** (`--algorithm-config v20.7.5` works from anywhere)
because the filename is unique across the tree; a directory-qualified name
(`_superseded/v20.7.5`) works too. Filenames must therefore stay unique — hence
`predT5-ramped` rather than a second `predT5.json`.

## Changes to shared code must stay config-restorable

Any earlier behaviour can be recovered by selecting a different config, which is
why this tree — not a git branch — is how versions are kept. That only works
while the code still *implements* the old behaviour, so a change to shared code
(`DWAT_v4_st/*.py`, `map_creation.py`, `output_display.py`) must either:

- be behaviour-preserving, and be verified as such; or
- sit behind a key whose **default reproduces the previous behaviour**.

A straight replacement breaks the property silently: no config can bring back a
formula that is no longer in the code, and the only route back becomes git
archaeology.

Worked examples, all verified:

| change | how the old behaviour survives |
|---|---|
| Theta* obstacle map vectorised | same `d <= rr` predicate; bit-identical output |
| `global_planner_resolution` | absent from a scenario ⇒ `1.0`, the historical value |
| large occupancy maps drawn as an image | threshold above the largest existing map |
| capsule ego hull | `robot_collision_shape` absent ⇒ `"disc"` |
| capsule in `closest_obstacle_on_curve` (item 1) | same key; `"disc"` verified bit-identical over 4000 randomised calls **and** `owenc_head_on` re-run |
| `no_admissible_fallback` → `all_candidates_collide_fallback` | old key still loads, as a property alias on `Config` |

**Check it by re-running, not by reading.** Re-run an existing scenario on the
new code and diff the trajectory against its recorded run — `log_details.json`
stores the full resolved config and the trajectory, so this is a few lines:

```python
a = json.load(open(f'{old_run}/log_details.json'))['trajectory']
b = json.load(open(f'{new_run}/log_details.json'))['trajectory']
assert np.abs(np.array(a) - np.array(b)).max() == 0
```

Last verified 2026-09-12 on `openwater_baseline` (v20.7.8-2, 1534 steps) and
`scene_export` (v20.9.1, 260 steps): both bit-identical.

### One rename that does change logs

`no_admissible_fallback` became **`all_candidates_collide_fallback`** on
2026-09-15. It was named after Guard A's message (`No admissible (v, ω) pairs`)
but only ever caught Guard B (`No best trajectory found`) — see
`../../results/OPEN_PROBLEMS.md` §2.

Old configs keep working: `Config` exposes the old name as a property that
proxies to the new one, which is what `apply_algorithm_overrides`'s
`hasattr`/`setattr` gate needs. **But logs change.** `output_display` serialises
`config.__dict__`, and a property is not in `__dict__`, so:

| log written | carries |
|---|---|
| before 2026-09-15 | `no_admissible_fallback` |
| after | `all_candidates_collide_fallback` |

Existing runs are not rewritten. The trajectory-diff recipe above is unaffected;
only a *config-block* diff spanning the rename sees it.

**Do not rename or move a config without moving its `runs/` directory too.**
`run_paths.version_from_config` derives the run directory from the config path,
so a move without the matching `runs/` move orphans every existing run.

## Lineage

Notes are the developer's; parameters verified from the JSON; commits from git.

| version | intent (developer's notes) | verified |
|---|---|---|
| `v20.7.1` | Exponential weighting — time | commit `eb7563f` |
| `v20.7.2` | Polynomial weighting — distance | commit `eb7563f` |
| `v20.7.3` | The same time weighting (test) | commit `2affe18` |
| `v20.7.4` | predT_obstacle 30; time weighting: same; distance: parabolic | **no commit or config found** |
| `v20.7.5` | predT_obstacle 30; time: exponential λ=0.1; distance: parabolic | λ=0.1, safe_r 3.0, quadratic barrier |
| `v20.7.6` | **Ablation of v20.7.5: time decay removed.** Multi movable obstacles implemented from here | module `…_no_time_decay` |
| `v20.7.7` | Time decay back on; distance weighting quadratic → **gaussian, decay 3.0** | barrier `exponential`, decay 3.0 |
| `v20.7.8` | Gaussian decay **6.0** | decay 6.0, safe_r 3.0 |
| `v20.7.8-2` | Decay 6.0 + **distance safety range 2.0** (soft boundary). Works for simple maps, *not* for more realistic ones | decay 6.0, **safe_r 2.0** |
| `v20.7.8-3` | Based on -2, but **max** of distance×time cost instead of integration | `dynamic_ob_cost_aggregation: "max"` |
| `v20.7.9` | Fix the open-water "ego is overtaken" failures — reconcile the dynamic window with the 30 s horizon | module `dwa_ramped`; `v_window_ramp_time` 2.5, `ramped_rollout`, `least_bad` |
| `v21.0` | Give the open-water ego the AIS suite's **dimensionless** dynamics, so conclusions transfer between the two suites | **failed.** `max_delta_yaw_rate` was derived by matching AIS's yaw *window fraction*, which at `dt=0.1` forces 0.947 rad/s² — **5× `max_yaw_rate`**. Exposed three latent bugs (unclamped ω, the stuck-escape's rate/acceleration confusion, an empty window reported as Guard A) |
| `v21.1` | v21.0 with the yaw axis derived from **angular acceleration** (20 s to full helm, as AIS) plus `omega_window_ramp_time`, which lets one config match AIS's yaw acceleration *and* window width at once — impossible with a one-`dt` window | `head_on` and `overtaken_ego` reach goal; `cross_stbd_090` **livelocked 11h49m**. `catch_goal_dist` was still 0.417 `L` — only **0.19 turning circles**, unsatisfiable for a ship |
| **`v21.2`** | v21.1 with `catch_goal_dist` scaled to the turn radius (1.36 turning circles, AIS's ratio) | **3/3 reach goal.** Trajectories are bit-identical prefixes of v21.1's — `catch_goal_dist` is purely a termination test |
| `v20.8.1` | **Static obstacles made consistent with the dynamic-obstacle cost.** Does not work on `video5` | module `dwa_consistent`, barrier `quadratic` |
| `v20.8.2` | Same branch as v20.8.1 | module `dwa_consistent`, barrier `exponential` decay 6.0 |
| `v20.9` | Tests on scene export | **parameter-identical to v20.7.8-2** apart from `max_speed` 1.0 — the first attempt |
| `v20.9.1` | scene_export tuning | side_cost_gain **2.4** (from 0.6), safe_r **0.5**, decay **3.0** |
| `v20.9.2` | scene_export tuning | identical to v20.9.1 in these keys |
| `v20.9.3` | scene_export tuning | as v20.9.1 but `predict_time_obstacle` **3.0**, `max_accel` **0.8** |

**`v20.7.1`–`v20.7.4` have no config files, and that is not a loss.** They
predate the convention of recording a change as a config file — at the time
changes were recorded by committing the implementation itself, so the code for
v20.7.1–v20.7.3 is recoverable from the commits above (`git show eb7563f`).
They cannot be *run* through `--algorithm-config`, and re-creating configs for
them would misrepresent history. `v20.7.4` is the one genuinely unrecorded
step: no commit, no config, notes only.

`ais_v1.json` and `ais_v1.1.json` belong to the New York / AIS import work, not this
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

Both are unimplemented and untested — hypotheses, not conclusions.

## Fixed: the abandoned branch had captured the default module

`v20.7.5` was the only config omitting `dwa_module`, so it took whatever
`main.py` defaulted to. That default was **switched to `dwa_consistent` in
commit `0278417`**, the same commit that introduced the v20.8.1 trial — the one
the notes record as *"not working well"*. The branch was abandoned; the default
was never switched back. From then until 2026-09-11, `--algorithm-config
v20.7.5` silently ran the **abandoned v20.8.x algorithm**.

The commit that introduced the `dwa_module` key (`66c054a`) had set the default
to `dwa_average_weighted_side` and explicitly aimed at "backward compatibility",
so this was a regression, not a decision.

Fixed in both places (`main.py`, `algorithm.py`), and `v20.7.5` now pins
`dwa_module` explicitly so it cannot drift again.

`v20.7.5` and `v20.7.6` also omitted `side_clearance_barrier_type`, which today
defaults to `exponential`/3.0 — but the notes say both were **parabolic**. Both
now pin `"quadratic"`. Three independent lines of evidence agree that this
restores the original behaviour:

1. the notes: v20.7.5 "Distance weighting: Parabolic";
2. the notes: v20.7.7 changed "Distance Weighting **from quadratic to**
   gaussian", so everything before it was quadratic;
3. the key did not exist until `b78c9a5` (v20.7.8); the code before it computed
   `max_side_weight_dist · (1 − d/M)²` unconditionally — algebraically
   identical to what `"quadratic"` selects today.

Every config in the tree now sets both keys explicitly.

## `_scene_export/` — an unsolved problem

**The planner still cannot complete the `scene_export` dataset.** v20.9 through
v20.9.3 are successive attempts to find a workable config; none succeeded, and
they are kept as the record of what has already been tried. The folder is named
for the goal, not for a status, because the goal is still open.

| config | outcome |
|---|---|
| `v20.9` | collision at 33.8 s (v ≈ 0.02 at the time) |
| `v20.9.1` | `No best trajectory found` at 25.9 s, **at cruise** (v = 0.99) |
| `v20.9.2` | `No best trajectory found` at 12.5 s, **at cruise** (v = 1.00) |
| `v20.9.3` | `No intersection points found and robot not within local-goal radius` at 29.9 s |

### A lead worth testing (hypothesis, not a result)

Two of the four fail with `No best trajectory found` — the same error as the
open-water `owenc_overtaken_ego` cases that `v20.7.9` fixes. **But the
signature is different and v20.7.9 as it stands probably will not help:** the
overtaken cases fail at *rest* on iteration 0, where the speed window is
crushed to `[0, 0.02]`; these fail at *cruise*, where the speed window is fine.

What is squeezed at cruise is the **yaw** window. These configs allow
`max_yaw_rate` ±1.396 rad/s, but the one-`dt` window offers only ±0.1396 — a
**tenth** of the available turning authority. Every candidate is therefore a
near-constant-curvature arc, and at `max_speed` 1.0 with `predict_time_obstacle`
30 s each arc is **30 units long**. In a confined imported scene it is
plausible that every such arc hits something, so every cost is `inf` and the
planner aborts — the same window/rollout mismatch as the overtaken bug, but
expressed in yaw instead of speed.

`v20.7.9` deliberately left the yaw window at one `dt` (documented as a known
limitation, on the grounds that a yaw rate *is* reachable within one `dt`).
Extending its `v_window_ramp_time` idea to the yaw window is the obvious next
experiment. **Untested.**

`v20.9.3` is the odd one out: it already tried the opposite tack — a short
3 s horizon with 4× the acceleration — and failed differently, in the
local-goal selector rather than the cost loop.

## Which to use

| | status |
|---|---|
| **`v20.7.8-2`** | **the baseline.** Copy this for any new experiment, ablation or comparison |
| `v20.7.8-3` | unevaluated — ask whether it beat v20.7.8-2 |
| `v20.7.9` | prototype, **not a drop-in replacement**: it removes a safety guard. See `../../results/v20.7.9_ramped_window/README.md` |
| `v21.0` | **derivation error — do not use.** File to `_abandoned/` once the v21 line settles |
| `v21.1` | superseded by v21.2; correct dynamics, unusable goal tolerance |
| **`v21.2`** | ship-like open-water ego. 3/3 encounters reach goal. Not yet a baseline — only 3 of 11 cases run |
| `_abandoned/*` | do not derive new configs from these |
| `_scene_export/*` | v20.9.x — an open problem, not a solved one. See below |
| `_superseded/*` | historical; runnable and now pinned, but superseded by v20.7.8-2 |

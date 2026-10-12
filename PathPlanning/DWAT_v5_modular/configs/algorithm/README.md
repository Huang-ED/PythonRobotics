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
| `v21.13/` | the COLREGs layer on v21.12: `side_only` (side term alone), `port_turn` (+ no alteration to port), `port_turn_release6` (+ release hysteresis) -- each adds one term to the one before |
| `ais_v1.2/` | the same egos; `obstacle_radius`, `dt` and `global_planner_rr` derived from the map cell and each ego by `new_york_cases.derive_ais_v12` |

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
| **`v21.3`** | Make v21.2 affordable. Two changes, both measured: `yaw_rate_resolution` 0.1° → 0.5°, and `pin_path_endpoints` on | **4.9–5.0× faster** (872 → 176 candidates; 3.503 → 0.710 s/iter open water, 6.821 → 1.361 s/iter v3crowd). The yaw window is only ±5.43°, so 0.5° still leaves 22 choices across it. **Re-run the `cross_*_135` symmetry check** — that pair was resolution-sensitive once before |
| **`v21.4`** | Remove the speed cap v21.2 introduced. `predict_time_to_goal` 19.69 → 8.0 | The ego was held to 0.30 of a 0.50 `max_speed` — not by ship physics but because the goal rollout travelled `v·T` past `dist_localgoal`, flipping the bearing and saturating `to_goal_cost` at `gain·π`. Safe window measured as **T ∈ [4, 12]**; 8.0 is its midpoint. Result: max v 0.3038 → **0.4850**, frames 2500 → **1584**. Sweep in `v21.4/` shows T = 5/8/11 differ by <1% — the margin matters, the exact value does not |
| **`v21.5`** | Sample the candidate grid on a fixed lattice instead of the window edge | The grid was `np.arange` from `dw[0]`, which **moves with the state** — so which *values* were considered depended on where the ego happened to be. Over a 1584-step run `max_speed` was a sample on **4.5%** of steps and `ω = 0` — "hold course" — on **0.0%**: the planner could never command a straight line. Anchoring at a constant and adding both window edges gives 98.1% and 100.0%. Result: max v 0.4850 → **0.5000**, frames 1584 → **1534**, and mean \|ω\| on a straight transit **0.000560 → 0.000000** (the per-step yaw dither was this artefact). Costs 16% more candidates, net **+1.2%** wall-clock |
| `v21.6` | v21.5 with `predict_time_to_goal` 8.0 -> 2.0, to remove the graze | **diverged** (x1.49 per cycle). Config deleted; runs kept in `runs/v21.6/`. `mu` = 0.80, below the stability boundary -- see `../../results/OSCILLATION.md` 4.1 |
| **`v21.7`** | v21.5 + `dist_localgoal` 7 -> 20 + `predict_time_to_goal` 2.0 | **11/11 open water**, worst clearance 0.463. Commit `03d4266` |
| `v21.8` | Faster rates to avoid v21.7's 20 u look-ahead | **design error**: moved the rates, `dist_localgoal` and `T_goal` together, so its failure could not be attributed. Superseded by v21.9 |
| `v21.9` | Faster actuation: full helm 20 -> 9.10 s, `max_accel` 0.03849 -> 0.059917; `dist_localgoal` back to 7; `side_cost_gain` 1.2 | open water mean clearance 0.703, `predict_time_obstacle` 54.62 (6 x `T_act`) |
| **`v21.10`** | v21.9 with the horizon at `4 x T_act`: `predict_time_obstacle` 36.41, `side_cost_gain` 0.98 | **11/11 open water, worst clearance 0.628, mean 0.764**, every delay < 1.7 s, 33% less compute. Commit `db9f11f`. v3crowd from an undeclared start heading: **0/3** -- it loses its own path. (Its 3/3 with `start_yaw` declared was against targets mostly not on a collision course; `../../results/OPEN_PROBLEMS.md` item 19) |
| `v21.11` | v21.10 + `dist_localgoal` 7 -> 9 + full helm 9.10 -> 6.74 s (`max_delta_yaw_rate` 0.028099), horizon re-derived (`predict_time_obstacle` 33.38, `lambda` 0.029958). Byte copy of `v21.10/bal-mix-T33.json` | **Open water 13/13**, worst clearance 0.522 against v21.10's 0.628, every single-encounter delay shorter, null control 0.000. **v3crowd from an undeclared heading, every target a verified collision course (`v3crowd?_v2111`): 2/3** -- one grounding while evading (item 20), one pass at 0.367. Derivation and results: `../../results/OSCILLATION.md` 4.4-4.7. *(An earlier "3/3 in both start conditions" was measured on suites whose targets had gone stale.)* Note `catch_turning_point_dist` is still 7.0 against `dist_localgoal` 9 -- unintended, item 18 |
| **`v21.12`** | v21.11 + `admissibility_mode: "braking_rollout_static"`: Stage 1's braking test with OPEN_PROBLEMS items 9, 10, 11 corrected (real ramped speed, bounded look-ahead, contact at t = 0), against **static obstacles only** -- the braking test is a static-world rule, so moving traffic stays with Stage 2. Byte copy of `v21.11/braking_rollout_static.json`; its runs are under `runs/v21.11/braking_rollout_static/` | **Open water 13/13, 12 bit-identical to v21.11** (no land to brake for). **v3crowd `_v2111`: 3/3** -- v3crowd6 no longer grounds (item 20's grounding, clearance 0.57), v3crowd7 0.36 -> 0.62. Guard A is now a real grounding alarm (reachable again, item 11). Not for AIS yet: `ais_v1.1` + this mode hit a phantom in `ny_hist_366962130` (item 2, now fixed by `track_on_start: "extrapolate"`), and with that fixed it hits a real ship it planned to pass at 0.4 m -- no hard margin around ships (item 6). Promoted 2026-09-30 |
| **`v21.13`** | v21.12 + basic COLREGs (`colregs_mode: "side_preference"`, `../../../DWAT_v4_st/colregs.py`). Each risky target (DCPA <= 3.0, 0 < TCPA <= horizon) gets a role -- head-on R14, crossing give-way R15 / stand-on R17, overtaking/overtaken R13 -- locked until it is opening beyond `colregs_release_dist` 6.0. For head-on and give-way crossing only: a penalty on passing with the target to starboard at the candidate's own CPA, and `colregs_port_turn_gain` 1.0 on any alteration to port of the heading held when the role locked (read 10 s into the candidate). Stand-on and R13's side are roles in the log, never costs. Byte copy of `v21.13/port_turn_release6.json`; its runs are under `runs/v21.13/port_turn_release6/` | **16/16 reach the goal** (open water 13 + v3crowd 3). **R14/R15 encounters on the correct side: 18/18, vs v21.12's 7/20** (`adapters/colregs_report.py`). Min clearance 0.539 (v21.12: 0.524); mean delay 5.7 s (5.6). The sweep, measured on the same 16: `side_only` 17 right / 3 wrong -- a port turn curls away in a 33 s rollout, so the faded side term misses it (`cross_stbd_045`); `port_turn` 17 / 0 but a parallel target re-forms as a new encounter when the lock releases at the gate (v3crowd5 t3, 0.499); `port_turn_release6` fixes that, open-water results within 0.31 u deviation of `port_turn`. With `colregs_mode` "off" the code is bit-identical to v21.12 (`owenc215_head_on`, 1519 steps, max diff 0.0). Promoted 2026-10-12 |
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

`ais_v1.json`, `ais_v1.1.json` and `ais_v1.2.json` belong to the New York / AIS
import work, not this lineage.

**`ais_v1.2` (2026-10-02) is the AIS config to run: 7/8 cases reach the goal,
against 5/8 for `ais_v1.1`, and no run touches land.** It is `ais_v1.1` plus:
`track_on_start: "extrapolate"` (item 2), `global_los_check: "supercover"` and a
metric `global_planner_rr` (item 21: the global path no longer crosses land or
the contact zone), `obstacle_radius` = res/√2 = 35.36 m (item 21: land circles
now cover their cells, so the rim has no gaps to sail through), and per-ego
`dt` ≤ half a cell of travel per step (item 3). Stage 1 is still `static_arc`
in the classic module -- with `dwa_ramped` it would test setpoints (item 11),
and `braking_rollout_static` for AIS waits on a ship margin (item 6). The
per-case files are byte copies of `ais_v1.1/*-landfix-dtrule.json`, whose runs
are under `runs/ais_v1.1/<case>-landfix-dtrule/`; regenerate with
`new_york_cases --version ais_v1.2`. Remaining failure: `ny_synth_00` (147 m
ship, Guard A at 9 min).

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
| `v21.2` | ship-like open-water ego, faithful but slow. Superseded by v21.3 for routine use; keep for fidelity comparisons |
| `v21.3` | superseded by v21.4; fast, but still carries the speed cap |
| `v21.4` | superseded by v21.5; still cannot sample `ω = 0` |
| `v21.5` | superseded by v21.7; grazes to 0.000 clearance (`predict_time_to_goal` 8.0 makes the heading term too stiff) |
| `v21.6`-`v21.9` | superseded; see the lineage table for why each was dropped |
| `v21.10` | validated on open water only (0.628). Cannot sail the v3 route from an undeclared start heading (0/3), which is the case that matters -- turning onto the first leg is part of the task |
| `v21.11` | superseded by v21.12, identical except Stage 1. Open water 13/13; v3crowd 2/3 (grounds in v3crowd6, item 20). Its braking test cannot fire (item 11) |
| `v21.12` | superseded by v21.13, identical with `colregs_mode` "off". v21.11 with a working Stage 1 against land (`braking_rollout_static`). Open water 13/13, v3crowd 3/3; 7 of 20 R14/R15 encounters on the correct side |
| **`v21.13`** | **the config to run.** v21.12 + basic COLREGs (R13-15, R17 roles; starboard side preference and no port alteration for head-on and give-way crossing). 16/16 reach the goal, 18/18 R14/R15 encounters on the correct side. Future versions keep an explicit `admissibility_mode` -- `braking_rollout_static`, or item 6's option once it exists; never the code default `static_arc`. Open issues: item 18 (switch radius), item 6; stand-on behaviour (R17) is not costed |
| `_abandoned/*` | do not derive new configs from these |
| `_scene_export/*` | v20.9.x — an open problem, not a solved one. See below |
| `_superseded/*` | historical; runnable and now pinned, but superseded by v20.7.8-2 |

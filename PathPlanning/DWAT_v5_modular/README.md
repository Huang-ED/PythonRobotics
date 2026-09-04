# DWAT_v5_modular

Modular split of DWA + Theta* into four responsibilities:

- `main.py`: orchestrates setup, simulation run, and output/post-processing
- `map_creation.py`: loads scenario/map and computes global path
- `algorithm.py`: runs local planning and simulation loop
- `output_display.py`: plotting, frame saving, and log serialization

## Config Layout

- Scenario config: `configs/scenarios/<group>/*.json` (a bare name also works)
- Algorithm tuning config: `configs/algorithm/default.json`
- Runtime/display config: `configs/run/default.json`

Key runtime option:

- `show_global_planner_search`: if `true`, shows Theta* search expansion plotting.
  Keep this `false` (default) to avoid creating a separate/incomplete global-planner figure.

## EnvData Policy

`EnvData/` holds input data, sorted by what a file *is* rather than where it
came from:

```
EnvData/
  maps/        static maps a scenario points at directly
               simple*.png, land_shapes_*.png, openwater_100.npy
  datasets/    raw source data, one directory per dataset
               ais_20240827/, new_york/, scene_export/, case_discovery/
  derived/     adapter output, regenerable from datasets/
               scene_export/{occupancy_100.npy, dynamic_tracks.npz}
  tools/       scripts that produce maps
```

The split that matters is `datasets/` versus `derived/`: everything in
`derived/` can be rebuilt by re-running an adapter, so it is the part that is
safe to delete.

- Scenario JSON references `image_path` / `occupancy_path` / `track_file`
  (for example `EnvData/maps/simple2.png`).
- The loader resolves these paths in this order:
  1. absolute path
  2. repo-root-relative path
  3. scenario-file-relative path

## Imported Scenes

Scenarios can also come from an external environment dataset instead of a
hand-drawn image. An **adapter** under `PathPlanning/DWAT_v5_modular/adapters/` converts one dataset
into a scenario JSON plus a track bundle `.npz`; nothing downstream needs to
know which dataset it came from.

Two additive scenario keys enable this:

- `occupancy_path` — static map from a `.npy` occupancy grid rather than
  `image_path` (no resize, no re-thresholding).
- `dynamic_obstacles[].motion_type: "trajectory"` — an obstacle that replays a
  recorded, time-stamped track, so it can accelerate, turn and stop. Position is
  interpolated, so the recording's sample rate need not match `Config.dt`.

Existing scenarios are unaffected. Algorithm tuning stays in
`configs/algorithm/*.json` — an imported scenario describes the world, never the
planner. Full spec: `configs/README_scene_format.md`.

```bash
python PathPlanning/DWAT_v5_modular/adapters/scene_export_adapter.py     # dataset -> scenario + tracks

python -m PathPlanning.DWAT_v5_modular.main \
  --scenario PathPlanning/DWAT_v5_modular/configs/scenarios/imported/scene_export.json \
  --algorithm-config PathPlanning/DWAT_v5_modular/configs/algorithm/v20.7.8-2.json
```

## Config layout

Configs are grouped so no listing gets long, and the grouping mirrors `runs/`:

```
configs/
  algorithm/
    v20.7.5.json ... v20.9.3.json      one file per version
    v20.7.8-2/                         variants of v20.7.8-2
      predT5.json ... predT30.json     the prediction-horizon sweep
  scenarios/
    baseline/  combined/  encounters/  imported/  synthetic/  video/
  run/
    default.json  video.json
```

A scenario's folder is its `"group"` key, and an algorithm config's folder is
its version, so `configs/algorithm/v20.7.8-2/predT10.json` produces runs under
`runs/v20.7.8-2/predT10/`.

**You can pass a bare name.** These are equivalent:

```bash
--scenario ow_sequence
--scenario PathPlanning/DWAT_v5_modular/configs/scenarios/combined/ow_sequence.json
--scenario PathPlanning/DWAT_v5_modular/configs/scenarios/ow_sequence.json   # pre-grouping path
```

All three resolve to the same file; the lookup falls back to searching by
filename, and errors rather than guessing if a name is ambiguous. The old flat
variant names still work too (`v20.7.8-2_predT10` finds
`v20.7.8-2/predT10.json`), and they derive the same version, so old and new
invocations write into the same run directory.

## Where output goes

One run produces one directory:

```
runs/<version>/<group>/<scenario>__<timestamp>/
    run.json          what was run: config paths and the resolved parameters
    log_details.json  trajectory, per-iteration log, outcome
    frames/           frame_0.png, ... (only with frame saving enabled)
    video.mp4         assembled afterwards by ffmpeg
```

- **version** comes from the `--algorithm-config` filename. A single underscore
  marks a variant, so `v20.7.8-2_predT10.json` lands in `v20.7.8-2/predT10/`
  and the ablation sweep reads as one experiment instead of seven peers.
- **group** comes from the `"group"` key in the scenario JSON
  (`encounters`, `combined`, `synthetic`, `imported`, `video`, `baseline`).
  It exists to keep any one listing short.
- Older generations live under `runs/_archive/<generation>/` in the same shape.

`python tools/build_run_index.py` regenerates `runs/INDEX.md`, a table of every
run with its outcome -- use it to answer "which runs collided", which the
folder tree cannot.

Previously frames went to `DWAT_v5_modular/results/figs_<algo>-modular-<scenario>/`
with no timestamp while the log went to `Logs/<same>_<timestamp>/`. Reruns
therefore overwrote frames but accumulated logs, and pairing a video with the
numbers behind it was guesswork. `python tools/migrate_runs.py` moved the
history into the layout above; `tools/migration_manifest.csv` records every move.

`PathPlanning/DWAT_v5_modular/results/` now holds only small, committed
deliverables -- the encounter-suite figures and videos -- not run output.

## Run

From repository root:

Use scenario_3.json for a lightweight super simple test.
```bash
python -m PathPlanning.DWAT_v5_modular.main --scenario PathPlanning/DWAT_v5_modular/configs/scenarios/synthetic/scenario_3.json --algorithm-config PathPlanning/DWAT_v5_modular/configs/algorithm/v20.7.5.json
```

Example for video3 scenario:

Use video3.json for a real-world map test. 
(Currently without optimization, this might run for a while.)
```bash
python -m PathPlanning.DWAT_v5_modular.main \
  --scenario PathPlanning/DWAT_v5_modular/configs/scenarios/video/video3.json
```

Disable animation or frame saving:

```bash
python -m PathPlanning.DWAT_v5_modular.main --no-animation --no-save-frames
```

Mass Running Example:
```
for scenario in scenario_234 scenario_234-1 scenario_5 scenario_6; do
    python -m PathPlanning.DWAT_v5_modular.main \
    --scenario PathPlanning/DWAT_v5_modular/configs/scenarios/$scenario.json \
    --algorithm-config PathPlanning/DWAT_v5_modular/configs/algorithm/v20.7.7.json
done
```

## Encounter Test Suite

Controlled ship-encounter scenarios for exercising the local planner against
moving traffic: head-on, crossing from either side, overtaking in both
directions, and a no-action control.

### Why it is generated rather than hand-placed

Hand-placed obstacles rarely produce an actual encounter. Re-running
`video3_lrshift.json` with `dynamic_obstacles: []` reproduces the ego
trajectory to within **0.01 units over the full 206 s** — its three circling
obstacles never influence the ego at all.

So `adapters/encounter_suite.py` places targets by construction. Given an
obstacle-free reference run, the ego at time `t_meet` is at `M` with heading
`h` and speed `v_e`; a target with course `chi` and speed `v_t` spawned at

    P0 = M - v_t * chi_hat * t_meet

reaches `M` at the same instant. Both hold constant velocity, so the relative
position is `(t_meet - t) * (v_e*h_hat - v_t*chi_hat)` — constant in direction,
shrinking to zero. That is a genuine constant-bearing, decreasing-range
collision course, not a near miss, and its fixed direction is the target's
relative bearing, which is what decides the applicable COLREGs rule.

Targets use `motion_type: waypoint` with one long leg. Constant velocity in a
straight line is exactly what the planner's linear extrapolation assumes;
circular motion would instead measure prediction error, since DWA re-linearises
a curving target every step.

### Course angle vs. relative bearing

A crossing can be specified two ways, and they are **not** interchangeable:

- `course_rel_deg` — the angle between the two tracks.
- `bearing_rel_deg` — where the target sits off the ego's bow, clockwise. This
  is the form the COLREGs rules are written in.

They coincide only for head-on and overtaking. A 90-degree *course* crossing at
`v_t = 0.4` puts the target at a *bearing* of only 38.7 degrees. Specifying the
bearing instead constrains the speed: holding a bearing abeam requires
`v_t >= v_e`, and anything abaft the beam requires `v_t > v_e` — a target
slower than you, behind you, can never close. The generator solves for the
course and refuses geometries that do not exist.

### Maps

`openwater_baseline.json` is a featureless 100x100 basin (all-free occupancy
grid; the blocked outer ring comes from `enable_map_boundary_obstacles`) with a
straight north-south route. It exists because **video3 cannot support a fair
port/starboard comparison**: along its route one side is open to 45 units while
the other closes to 8, so a turn to starboard there may be the land's doing
rather than the planner's. Symmetric water is what makes a mirrored pair mean
something.

video3 keeps its own tier, for confinement rather than symmetry:

| slot | t (s) | port | stbd | purpose |
|---|---|---|---|---|
| A | ~45 | 10 | 45 | the only genuinely open stretch |
| B | ~133 | 5 | 17 | narrow both sides |

The `t = 160-175` pinch is deliberately excluded: the obstacle-free run clears
it with **1.21 units** of margin against a ~1.22 collision threshold, so
anything placed there tests the pinch, not the encounter. That zero margin is
also why plain `video3.json` fails on v20.7.8-2 (`No best trajectory found` at
t~169.6 s) while `video3_lrshift.json` completes.

### Running

```bash
# 1. obstacle-free reference run (the suite is placed against this)
python -m PathPlanning.DWAT_v5_modular.main \
  --scenario PathPlanning/DWAT_v5_modular/configs/scenarios/baseline/openwater_baseline.json \
  --algorithm-config PathPlanning/DWAT_v5_modular/configs/algorithm/v20.7.8-2.json \
  --no-animation --no-save-frames

# 2. generate the scenarios
python PathPlanning/DWAT_v5_modular/adapters/encounter_suite.py \
  --reference-log runs/v20.7.8-2/baseline/openwater_baseline__<ts>/log_details.json \
  --base-scenario PathPlanning/DWAT_v5_modular/configs/scenarios/baseline/openwater_baseline.json \
  --suite openwater

# 3. run them
for s in PathPlanning/DWAT_v5_modular/configs/scenarios/encounters/owenc_*.json; do
  python -m PathPlanning.DWAT_v5_modular.main --scenario "$s" \
    --algorithm-config PathPlanning/DWAT_v5_modular/configs/algorithm/v20.7.8-2.json \
    --no-animation --no-save-frames
done

# 4. score them
python PathPlanning/DWAT_v5_modular/adapters/encounter_report.py \
  --baseline-log runs/v20.7.8-2/baseline/openwater_baseline__<ts>/log_details.json \
  --manifest PathPlanning/DWAT_v5_modular/configs/scenarios/encounters/owenc_manifest.json \
  --runs 'runs/v20.7.8-2/encounters/owenc_*/log_details.json'
```

### Reading the results

`encounter_report.py` reports closest approach (surface-to-surface, negative
means hulls overlapped), which side the target was on at CPA, deviation from
the obstacle-free baseline, and delay to goal.

Two cautions. The **null case inverts the pass condition**: there the target
passes 6 units clear, so deviation near zero is correct and a large manoeuvre
is the failure. And **pass side is recorded but not scored** — the planner's
dynamic-obstacle cost (`dwa_average_weighted_side`) is a spatio-temporal
clearance that is mirror-symmetric in port/starboard, so it has no side
preference to score. It is logged because it is precisely the baseline needed
to show what changes once a COLREGs side rule is added.

### Known result: the overtaken cases abort at predT30

`owenc_overtaken_ego` and `owenc_overtaken_ego_fast` both fail on iteration 0
with `No best trajectory found` under `v20.7.8-2`, and the cause is not the
encounter. At `t=0` the ego is at rest, so the dynamic window admits only
`v in [0, 0.02]` -- a single acceleration step -- but every candidate is then
scored by holding that velocity constant across the full
`predict_time_obstacle = 30 s`. The planner predicts itself crawling 0.6 units
over 30 s while the target advances 18 and passes straight through it, so every
candidate is scored as colliding and there is nothing left to choose.

Verified by construction: the identical scenario **reaches the goal at
`predT10`** (clearance 0.91, deviation 2.94, speed held at 0.50). At `predT5`
it fails differently, with `No admissible (v, omega) pairs found in dynamic
window`.

So this is a property of the long horizon interacting with the one-step dynamic
window, not of overtaking. It will bite any encounter where a target is inbound
on the ego's present position while the ego is slow. Both variants are kept as
the regression test for it.

### Combined scenes: several targets in one scenario

`--combine` emits a **single** scenario holding every target of a set, the way
`video3.json` ships three obstacles at once — one run, one video.

```bash
python PathPlanning/DWAT_v5_modular/adapters/encounter_suite.py \
    --reference-log runs/v20.7.8-2/baseline/openwater_baseline__<ts>/log_details.json \
    --base-scenario PathPlanning/DWAT_v5_modular/configs/scenarios/baseline/openwater_baseline.json \
    --combine ow_sequence
```

| set | targets | what it is |
|---|---|---|
| `v3_sequence` | 3 | video3 route: head-on (t=45), starboard crossing (t=85), port crossing (t=133) |
| `ow_sequence` | 3 | basin: head-on, then the mirrored 090 crossing pair back to back |
| `ow_simultaneous` | 3 | three vessels converging within 20 s — no single manoeuvre clears all of them |

Two things are checked that do not arise for single-target scenarios, and both
caught real problems while these sets were being written:

- **Targets colliding with each other.** Each target is generated independently
  against the ego, so nothing stops two of them occupying the same water. The
  first `ow_simultaneous` aimed all three at one point at one instant and they
  collided (gap -1.00). A scene where the traffic piles up is not a harder test;
  the ego would be reacting to the pile-up rather than to an encounter. The
  meetings are now staggered 10 s, which keeps all three inside the ~50 s
  detection horizon while leaving them 7.32 units of mutual clearance.
- **Spawn distance against map size.** Spawn distance is `speed * t_meet`, so a
  late meeting throws the target a long way back up its own course.
  `cross_port_090` at `t=130` needed 52 units of run-up, wider than the basin's
  half-width, and spawned off-map until its speed was dropped to 0.3.

The tool also reports meeting spacing and warns when gaps fall under ~50 s
(`predict_time_obstacle` x closing speed), because below that the encounters
overlap instead of resolving one at a time. On video3 that is unavoidable:
excluding the t=160-175 pinch leaves a ~90 s usable window, and three cleanly
separated meetings would need ~100 s. Two would be clean; three overlap.

### Rendering a video

Run with frames (`configs/run/video.json` keeps the global-planner figure off),
then assemble with ffmpeg:

```bash
MPLBACKEND=Agg python -m PathPlanning.DWAT_v5_modular.main \
    --scenario PathPlanning/DWAT_v5_modular/configs/scenarios/combined/ow_sequence.json \
    --algorithm-config PathPlanning/DWAT_v5_modular/configs/algorithm/v20.7.8-2.json \
    --run-config PathPlanning/DWAT_v5_modular/configs/run/video.json
# prints:  Run directory: runs/v20.7.8-2/combined/ow_sequence__<timestamp>

RUN=runs/v20.7.8-2/combined/ow_sequence__<timestamp>
ffmpeg -y -framerate 30 -start_number 0 \
    -i "$RUN/frames/frame_%d.png" \
    -c:v libx264 -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
    "$RUN/video.mp4"
```

Frame saving is much slower than a headless run, and the cost scales with the
static map: the basin renders ~270 frames/min against video3's ~116, because
video3 rasterises roughly 3,200 obstacle circles every frame. Budget ~6 min for
a basin scene and ~18 min for a video3 one. Headless runs (`--no-animation
--no-save-frames`) take ~8 min regardless and are what the suite normally uses.

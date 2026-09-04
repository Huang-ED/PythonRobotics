# Imported-scene scenario format (format_version 1)

An additive superset of `configs/scenarios/*.json`. Every existing scenario
still loads unchanged; the new keys only take effect when present.

The motivation is that external environment datasets (`EnvData/scene_export`,
`EnvData/new york`, ...) each have their own idea of what a map and an obstacle
are. Rather than teach `map_creation.py` about each one, an **adapter** under
`PathPlanning/DWAT_v5_modular/adapters/` resolves a dataset down to two artefacts, and the planner
only ever reads those:

    dataset -> [adapter] -> scenario JSON + track bundle NPZ -> map_creation -> MapManager

Algorithm tuning stays where it already lives, in `configs/algorithm/*.json`.
An imported scenario describes the world, never the planner.

## What the format adds

| key | purpose |
|---|---|
| `occupancy_path` | static map from a `.npy` occupancy grid instead of `image_path` |
| `occupancy_resolution` | metres per cell |
| `occupancy_origin` | world coords of the `(0,0)` cell's lower-left corner |
| `occupancy_cell_anchor` | `"center"` or `"corner"` — where in the cell the obstacle point sits |
| `dynamic_obstacles[].motion_type: "trajectory"` | obstacle replays a recorded, time-stamped track |

`image_path` / `map_size` / `waypoint` / `circle` obstacles all behave exactly
as before.

## Static map: image vs occupancy

`image_path` resizes and thresholds a PNG. That is fine for hand-drawn maps,
but it resamples — for `scene_export`, resizing the bundled 1339x1339 PNG down
to 50x50 reproduces only ~97% of the exported grid's cells. When a dataset
ships both an image and a grid, and its obstacle trajectories are certified
against the grid, load the grid.

Occupancy convention, matching the exported data:

```
occupancy[i, j] != 0   <=>   cell (i, j) is blocked        # axis 0 is x, axis 1 is y
cell (i, j) spans      [i*res, (i+1)*res] x [j*res, (j+1)*res]  (+ origin)
```

`occupancy_cell_anchor` decides where the obstacle point for a blocked cell
goes. **`"center"` is almost always what you want** — anchoring at the corner
shifts every static obstacle half a cell relative to the trajectories.

As with the image path, `MapManager` keeps the full occupied set in
`static_obstacles` / `astar_obstacles` and the eroded rim in
`boundary_obstacles` (what DWA actually costs against).

If the grid already marks its outer ring as blocked — `scene_export` does —
leave `enable_map_boundary_obstacles` at `false` so no second border is added.

## Dynamic obstacles: `motion_type: "trajectory"`

For obstacles whose future is *recorded* rather than *generated*. Unlike
`waypoint` (constant speed, loops forever) and `circle`, a trajectory obstacle
is a lookup against a real timeline, so it can accelerate, turn, and stop.

```json
{
    "motion_type": "trajectory",
    "track_file": "EnvData/scene_export/converted/dynamic_tracks.npz",
    "track_index": 0,
    "radius": 0.3,
    "start_time": 0.0,
    "on_finish": "hold"
}
```

- `start_time` — simulation-clock offset, for staggering entries.
- `on_finish` — `"hold"` parks the obstacle at its last sample with zero
  velocity; `"loop"` restarts the recording.
- Small tracks may be inlined with `"trajectory": [[x,y],...]` and
  `"times": [...]` instead of `track_file`/`track_index`.

Sampling is decoupled from `Config.dt`: position is linearly interpolated
between samples, so a 0.25 s recording drives a 0.1 s simulation correctly.

**Velocity.** `velocities` is optional; when absent it is derived as
`v[k] = (p[k+1] - p[k]) / (t[k+1] - t[k])`, `v[-1] = 0`. This piecewise-constant
form is exactly the velocity implied by the interpolation, so the constant-velocity
prediction inside `calc_trajectory_clearance_and_collision` agrees with the
recording at every sample.

Note this is also the format's main *limitation*: DWA extrapolates each obstacle
linearly at its current velocity over `predict_time_obstacle`. For a curved
recording the prediction is only accurate for roughly one sample interval; over
a 30 s horizon it will diverge from the true track.

## Track bundle NPZ

| key | shape | notes |
|---|---|---|
| `positions` | `(n_obs, K, 2)` | padded by repeating the last real sample |
| `lengths` | `(n_obs,)` | real sample count per track |
| `times` | `(K,)` or `(n_obs, K)` | seconds; shared or per-track timeline |
| `velocities` | `(n_obs, K, 2)` | optional |
| `radii` | `(n_obs,)` | collision radius |
| `ids` | `(n_obs,)` | optional label |

Padding past `lengths[k]` is never read — `on_finish` governs that region.

## Map resolution

`occupancy_resolution` and the grid's cell count are independent of the world
size: the world spans `occupancy_origin + shape * resolution`, and obstacle
trajectories live in continuous world coordinates. So a scene's map can be
re-rasterised finer **without moving anything else** — only the sampling of the
land changes.

`scene_export` ships a 50x50 grid over a 50 m world (1 m cells) derived from a
1339x1339 image, i.e. ~27x the linear detail is discarded. That is coarser than
the ego itself (1.2 m long), so corridor widths are quantised to 1 m. The
adapter can refine it:

```bash
python PathPlanning/DWAT_v5_modular/adapters/scene_export_adapter.py     --map-cells 100     --config-out PathPlanning/DWAT_v5_modular/configs/scenarios/scene_export_100.json     --name scene_export_100
```

The adapter refuses to emit a refined grid that would place any recorded
trajectory point inside a blocked cell, and re-asserts the blocked outer ring
that the dataset relies on instead of an explicit bounds check.

**`obstacle_radius` must track the resolution.** It is the disk radius applied to
*every* static cell, and at the shipped 1 m grid it is 0.5 — exactly half a cell,
so the disks inscribe their cells. Refine the grid without changing it and every
wall inflates by `0.5 - resolution/2` per side. Set
`obstacle_radius = resolution / 2` in the algorithm config. (It is also the
default radius for dynamic obstacles that do not declare their own; imported
scenes always declare theirs, so only the static side is affected here.)

Cost scales with the rim-cell count, which is what the DWA obstacle term is
evaluated against — 326 cells at 50x50, 740 at 100x100, ~1400 at 200x200 — and
`closest_obstacle_on_curve` is linear in that for every candidate control.

| grid | m/cell | rim cells | note |
|---|---|---|---|
| 50x50 | 1.0 | 326 | as shipped; cell ~ the length of the ego |
| 100x100 | 0.5 | 740 | good balance |
| 200x200 | 0.25 | ~1400 | diminishing returns for this scene |

## Adding a dataset

Write an adapter in `PathPlanning/DWAT_v5_modular/adapters/` that calls `write_track_bundle` and
`write_scene_config` from `scene_format.py`. See `scene_export_adapter.py`.

## Running

```bash
# dataset -> scenario JSON + track bundle
python PathPlanning/DWAT_v5_modular/adapters/scene_export_adapter.py

# run it like any other scenario
python -m PathPlanning.DWAT_v5_modular.main \
    --scenario PathPlanning/DWAT_v5_modular/configs/scenarios/scene_export.json \
    --algorithm-config PathPlanning/DWAT_v5_modular/configs/algorithm/v20.7.8-2.json
```

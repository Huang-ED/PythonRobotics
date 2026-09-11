# `ais_*` — SI tuning for the New York AIS scenarios

Companion to `configs/algorithm/ais_v1.json`, `ais_v1.1.json` and the scenarios
written by `adapters/new_york_ais_adapter.py`. Config JSON is validated strictly
against `Config`'s attributes, so the reasoning lives here instead of in
comments.

Results for this lineage: `../results/ny_ais_suite/README.md`.

## Lineage

| version | change | status |
|---|---|---|
| `ais_v1` | first SI tuning, hand-checked on one 29 m towing vessel. Ego modelled as the circumscribed **disc** | the reference run and its video |
| `ais_v1.1` | **capsule** ego hull; per-ego kinematics derived rather than inherited (`ais_v1.1/` holds one config per case) | current |
| `ais_v1.2` | **proposed, not adopted**: derive `dt` per ego. See *The `dt` ceiling* below | open |

`ais_v1.json` was reconstructed from `log_details.json` of the run that produced
the video, not re-typed from memory, because the file had already drifted to
capsule geometry by the time the version was pinned. The two differ in exactly
one key, `robot_collision_shape`.

**Reserve `ais_v2` for a change of approach** — a different cost formulation, or
replacing the braking-admissibility test. Everything in v1.x is a correction
within one design: constants becoming ratios, and a hull model being fixed.

## This is a separate lineage from v20.7.x

`v20.7.8-2` and its `predT*` sweep are in **map units** on a 50-unit world with a
1.2-unit ego. The New York scenarios are in **metres and seconds** over a 12 km
region with a real vessel. The two are not comparable and the configs are not
interchangeable — this one was derived from ship kinematics, not by rescaling.

The reason a rescale is not enough: the four DWA cost terms are summed with
different dimensions.

| Term | Expression | Dimension |
|---|---|---|
| `to_goal_cost` | gain × \|heading error\| | dimensionless (rad) |
| `speed_cost` | gain × (`max_speed` − v) | m/s |
| `static_ob_cost` | gain × (`max_obstacle_cost_dist` − d) | m |
| `dynamic_ob_cost` | gain × `max_side_weight_dist` × exp(…) | m |

So the gains carry units. Copying them from a 50 m world into a 12 km one leaves
the goal term spanning 0–1.88 while the obstacle terms span 0–100, and the ego
stops steering towards its goal. Each gain here is set so its term spans a
comparable range: `to_goal` 0–1.88, `speed` 0–1.0, `static` 0–0.4, `dynamic`
0–1.2. That is a starting point, not a tuned result.

## Why `dt = 10 s`

`dt` is a control period, not an integration step. The dynamic window is
`±max_delta_yaw_rate · dt` wide, so at the v20.7.x value of 0.1 s a vessel with
realistic rudder response gets a window of ±0.0004 rad/s and every candidate is
indistinguishable — the planner has no authority to exercise. At 10 s the window
is about a third of the full turn-rate range.

The upper bound is collision sampling. `calc_trajectory_clearance_and_collision`
tests the predicted trajectory only at `dt` intervals, so the check can pass
between samples: at 5.5 m/s and dt = 10 s the samples are 55 m apart, which is
already wider than the ego's own 15 m collision radius.

**This config sits on the wrong side of that ratio, deliberately, and it costs
accuracy.** Re-checking the reference run at 1 s resolution, the true minimum
hull-to-hull clearance was **16.5 m** where the planner saw **20.7 m** — a 20%
understatement of how close it came. No collision was missed, but the margin the
planner reports is optimistic, so do not read the logged clearance as exact.

The reason for accepting that: `dt` sets the dynamic window width as well
(`±max_delta_yaw_rate · dt`), so shrinking it to tighten collision sampling also
shrinks the planner's control authority. At dt = 3 s the sampling is 16.5 m —
tight enough — and the yaw window is still ~15% of the full range, at roughly
3x the runtime. That is the right trade if clearance figures matter.

Raising `dt` goes the other way fast: at 30 s the scenario looks *better*
behaved only because it has largely stopped detecting collisions.

## `max_accel` is braking authority, not hydrodynamics

The admissibility test in `dwa_average_weighted_side.py` keeps a candidate only
when `dist > v²/(2a) + v·dt` — it requires the ego to be able to *stop*. At
0.3 m/s² that is 79 m, which a 29 m towing vessel can hold in these channels.

A deep-draft ship cannot. True surge deceleration for a 248 m vessel is around
0.05 m/s², demanding 400 m of clear water; measured on the Anchorage Channel
approach that rejects every candidate and the run aborts with "No best
trajectory found". **If you change the ego to a large vessel, this parameter has
to be justified explicitly, or the admissibility test has to change.**

## The `dt` ceiling — the open issue

`dt` is bounded below by control authority and above by collision sampling, and
in v1.1 it is a single fixed 10 s for every ego. That is wrong in the same way
the other constants were: a fast vessel covers 125 m per control step against
25 m shore cells and cannot see the coastline finely enough to turn away.

Measured on `ny_hist_367782880`, changing nothing but `dt`:

| `dt` | travel/step | outcome | duration | path | shore clearance |
|---|---|---|---|---|---|
| 10 s | 125 m | grounding | 10.0 min | 7.50 km | 1.3 m |
| 2 s | 25 m | **goal** | 18.9 min | 14.32 km | 6.9 m |
| *human* | — | — | 28.0 min | 14.98 km | — |

To reproduce, copy the case config and set `dt = 2.0`, scaling `v_resolution`
and `yaw_rate_resolution` by `2/10` and `predict_time_to_goal` to `3*dt`. The
probe is deliberately not kept in `ais_v1.1/`: that directory sweeps the ego,
and a `dt` variant would make it two factors at once.

Adopting this as `ais_v1.2` means `dt ≈ L / v` floored so the dynamic window
stays usable — and it roughly doubles the cost of every run.

## Ego footprint must match the scenario

`robot_length` / `robot_width` describe the ego vessel the adapter promoted out
of the AIS traffic (`source.ego_mmsi` in the scenario JSON). Beam is absent from
AIS everywhere in this dataset and is taken as L/3 for a tug. Changing `--ego`
in the adapter means changing these too.

One caveat for large vessels: when `robot_type` is `rectangle` the dynamic
collision check models the ego as a disc of radius `sqrt(L² + B²)/2`, ignoring
`robot_radius`. The error grows with aspect ratio — negligible for a 29 × 10 m
tug, but it inflates a 248 × 32 m ship to a 250 m-diameter disc. The static
check uses `robot_radius` instead, so the two disagree.

## `obstacle_radius` is tied to the map

Every occupied cell becomes a disc of `obstacle_radius`, so it must be half the
occupancy cell size or the shoreline is inflated. The adapter defaults to a 50 m
grid, hence 25.0. Re-running the adapter with `--resolution` means changing this.

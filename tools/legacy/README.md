# Legacy log-analysis scripts

**Reference material, not working tools.** Every script here predates the
`runs/` layout and most predate the current cost formulation. They are kept
because they show *how* a piece of analysis was done — which quantities were
pulled out of a log, and how they were recomputed to match the planner — and
that is the expensive part to reconstruct. The plumbing around them is cheap to
rewrite and is expected to be rewritten rather than repaired.

None of them are imported by anything. Do not wire them into the pipeline.

## The one worth reading first

### `log_v20_dwa_cost_calc.py`

Reconstructs the **full (v, ω) cost surface** for chosen iterations of a logged
run, decomposed into the terms the planner actually sums:

| term | what it is |
|---|---|
| `to_goal_cost` | heading/position error at the end of the predicted trajectory |
| `speed_cost` | `max_speed - v` at the end of the trajectory |
| `static_ob_cost` | `max(0, max_obstacle_cost_dist - d_static)` |
| `dynamic_ob_cost_total` | the spatio-temporal dynamic term |
| `dynamic_ob_cost_side` | its lateral (`D_side`) component |
| `dynamic_ob_cost_direct` | its direct-distance component |

It also builds the matching physical-distance matrices, and renders one figure
per iteration. Inadmissible `(v, ω)` pairs are `nan` rather than a large number,
so they stay visually distinct from merely expensive ones.

The valuable part is not the script, it is the set of rules it encodes for
reproducing the planner's own arithmetic — it carries comments like *"We must
do the same here, or costs will mismatch"* at the points where the
recomputation has to mirror an implementation detail exactly. Any new
cost-breakdown tool should start by reading those.

**Why it no longer runs:** the log path is hardcoded to
`Logs/figs_v20.3.4-video1.1_20260115_200511/log_details.csv`. `Logs/` was
folded into `runs/` on 2026-09-04, and the pipeline stopped emitting `.csv`
before that. It imports fine; it just cannot find its input.

### `log_v20_dynamic_obs_velocity.py`

Replays the dynamic obstacles of a scenario over an iteration range and exports
their per-step position and velocity, pre-rolling the simulation to reach the
requested start iteration. Useful as a model for "what were the obstacles doing
at step N" without re-running the planner.

**Why it no longer runs:** `from map_manager import MapManager` — a bare import
that only resolved when the file sat inside `DWAT_v4_st/`. It now raises
`ModuleNotFoundError`. The fix is a one-line change to
`from PathPlanning.DWAT_v4_st.map_manager import MapManager`, but nothing else
has been checked, so treat it as unverified.

## Older still

Moved out of `Logs/` and `Logs_split/` during the 2026-09-04 reorganisation,
where they sat loose among the run folders. They target the v10/v11 split-cost
generation, whose code is in `PathPlanning/_archive/DWAT_v3_split/`.

| script | what it did |
|---|---|
| `log_to_df.py`, `log_to_df_split.py` | log → pandas DataFrame |
| `log_to_figures.py`, `log_to_figures_split.py` | per-iteration cost figures |
| `plot_velocity_spaces.py` | velocity-space visualisation |
| `v10_log_to_df.py` | the v10 variant of the DataFrame loader |

## Not here, but moved at the same time

`DWAT_v4_st/dwa_astar.py` — the v4-era standalone runner — went to
`PathPlanning/_archive/DWAT_v4_st/` rather than into this folder: it is dead
*algorithm* code, not analysis tooling, so it belongs with the other retired
generations. Nothing imports it, and it looks like an entry point; the real one
is `PathPlanning/DWAT_v5_modular/main.py`.

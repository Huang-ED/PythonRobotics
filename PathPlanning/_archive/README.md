# Retired generations

Code that nothing in the current pipeline imports. Kept because the results it
produced are still in `runs/_archive/`, and a result is only interpretable next
to the code that made it.

The import closure from `PathPlanning/DWAT_v5_modular/main.py` reaches exactly
two directories: `DWAT_v5_modular` and `DWAT_v4_st`. Everything below is
outside it.

| directory | first commit | produced the runs under | notes |
|---|---|---|---|
| `AStar/` | 2017-12-10 | — | upstream (23 Sakai commits vs 9 Huang-ED) |
| `DynamicWindowApproach/` | 2017-12-14 | `runs/_archive/dwa_baseline/` | upstream folder, but 5 of 6 files are Huang-ED |
| `LocalGlobal/` | 2024-07-12 | `runs/_archive/local_global/` | 100% Huang-ED; the ancestor of the DWAT lineage |
| `DWAT/` | 2025-06-30 | `runs/_archive/v1/` | v7.4.x |
| `DWAT_v2/` | 2025-07-02 | `runs/_archive/v2/` | v8.1 - v8.6 |
| `DWAT_v3/` | 2025-07-16 | `runs/_archive/v3/` | v8.4 - v8.5 |
| `DWAT_v2_split/` | 2025-11-05 | `runs/_archive/v2_split/` | v8.2.3.x, split cost terms |
| `DWAT_v3_split/` | 2025-11-10 | `runs/_archive/v3_split/` | v9.x - v12.x |
| `dwa_cost_debug/` | — | — | scratch cost-visualisation scripts |

`DWAT_v4_st/` here holds only `dwa_astar.py`, the v4-era standalone runner:
nothing imports it and it looks like an entry point, so it was moved out of the
live directory on 2026-09-05. The rest of `DWAT_v4_st/` is still live.

Still live, and therefore **not** here:

- `PathPlanning/DWAT_v4_st/` — five modules the v5 runner depends on:
  `dwa_average_weighted_side.py` (15 algorithm configs select it, including
  v20.7.8-2), `dwa_average_weighted_side_no_time_decay.py` (1 config),
  `dwa_consistent.py` (supplies `Config`; 2 configs), `map_manager.py` and
  `theta_star.py` (both imported by `map_creation.py`).
- `PathPlanning/DWAT_v5_modular/` — the current pipeline.

Retired *analysis* scripts live in `tools/legacy/`, not here — see its README.

## On the upstream directories

This repository is a fork of AtsushiSakai/PythonRobotics. `AStar/` and
`DynamicWindowApproach/` came from it, which is why they predate everything
else. They are archived rather than deleted because `DynamicWindowApproach/`
is not really upstream code any more: only `dynamic_window_approach.py` is
Sakai's, while the other five files are Huang-ED's — including
`dwa_paper_with_width.py`, which `DWAT/`, `DWAT_v2/` and `DWAT_v3/` each
copied. It is the root of the DWAT lineage sitting inside an upstream folder.

Archiving diverges from upstream, which costs nothing in practice: the last
commit from the upstream author was 2024-05-25, and most upstream content
(Localization, SLAM, PathTracking, Mapping, Control, AerialNavigation, and
~35 other PathPlanning algorithms) had already been removed before this.

`EnvData/` asset paths in this directory were repointed to the current
`EnvData/maps/...` layout, so the retired code remains runnable.

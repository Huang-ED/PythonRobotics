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

Still live, and therefore **not** here:

- `DWAT_v4_st/` — `dwa_consistent.py` and `map_manager.py` are imported by v5.
- `DWAT_v5_modular/` — the current pipeline.

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

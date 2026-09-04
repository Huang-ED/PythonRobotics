# Retired generations

Code for DWAT generations that nothing imports any more. Kept because the
results they produced are still in `runs/_archive/`, and a result is only
interpretable next to the code that made it.

| directory | produced the runs under | notes |
|---|---|---|
| `DWAT/` | `runs/_archive/v1/` | v7.4.x |
| `DWAT_v2/` | `runs/_archive/v2/` | v8.1 - v8.6 |
| `DWAT_v2_split/` | `runs/_archive/v2_split/` | v8.2.3.x, split cost terms |
| `DWAT_v3/` | `runs/_archive/v3/` | v8.4 - v8.5 |
| `DWAT_v3_split/` | `runs/_archive/v3_split/` | v9.x - v12.x |
| `dwa_cost_debug/` | — | scratch cost-visualisation scripts |

Still live, and therefore **not** here:

- `DWAT_v4_st/` — `dwa_consistent.py` is imported by the v5 runner.
- `DWAT_v5_modular/` — the current pipeline.
- `LocalGlobal/`, `AStar/`, `DynamicWindowApproach/` — still imported.

Their `EnvData/` asset paths were repointed to the current
`EnvData/maps/...` layout, so these remain runnable.

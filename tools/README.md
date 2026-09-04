# Repository tools

Maintenance scripts for the run-output tree. None of them are part of the
planner; they exist to keep results findable.

## The layout they maintain

```
runs/<version>/<group>/<scenario>__<timestamp>/
    run.json          config paths and resolved parameters for this run
    log_details.json  trajectory, per-iteration log, outcome
    frames/           frame_0.png, ...
    video.mp4
```

- **version** — the `--algorithm-config` filename. `v20.7.8-2_predT10.json`
  nests as `v20.7.8-2/predT10`, so an ablation sweep reads as one experiment.
- **group** — the `"group"` key in the scenario JSON: `encounters`, `combined`,
  `synthetic`, `imported`, `video`, `baseline`. Purely to keep listings short.
- Pre-v5 generations sit under `runs/_archive/<generation>/` in the same shape.

`runs/` is git-ignored in full. Committed deliverables (the encounter-suite
figures and videos) live in `PathPlanning/DWAT_v5_modular/results/` instead.

## Scripts

| script | what it does |
|---|---|
| `run_naming.py` | Parses the historical `figs_*` names into version / scenario / timestamp. Imported by the other two; not run directly. |
| `migrate_runs.py` | One-shot move of the old split layout into `runs/`. `--dry-run` prints the plan; `--apply` performs it and writes `migration_manifest.csv`. |
| `build_run_index.py` | Regenerates `runs/INDEX.md`: every run with its outcome, artifacts and size. Run it after a batch of simulations. |

```bash
python tools/build_run_index.py          # refresh runs/INDEX.md
python tools/migrate_runs.py --dry-run   # (already applied once)
```

## Why the naming parser is fussy

Two schemes were in use, and the older one is genuinely ambiguous because both
halves may contain a hyphen:

```
figs_v11.10.4-scenario1-1   ->  version v11.10.4,   scenario scenario1-1
figs_v20.7.8-2-video3       ->  version v20.7.8-2,  scenario video3
```

They are separated by what follows the first hyphen: a bare integer is a
revision suffix belonging to the version, anything else begins the scenario.
A naive `([^-]+)-(.+)` split gets `v20.7.8-2_predT10` wrong, silently filing
those runs under `v20.7.8` — which is why the rule lives in one tested place
rather than being re-derived per script.

## legacy/

Log-processing helpers that used to sit loose among the run folders in `Logs/`
and `Logs_split/`. Moved here so those directories contain only runs. They
reference the pre-migration layout and are kept for reference, not for use.

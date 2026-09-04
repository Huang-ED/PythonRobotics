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

- **version** — where the algorithm config sits under `configs/algorithm/`.
  `v20.7.8-2/predT10.json` gives `v20.7.8-2/predT10`, so an ablation sweep reads
  as one experiment. (The older flat name `v20.7.8-2_predT10.json` maps to the
  same version, so pre-grouping commands still land in the same place.)
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
| `check_names.py` | Finds path names OneDrive / Windows cannot store. `--fix` renames them. Worth running after any bulk move. |

```bash
python tools/build_run_index.py          # refresh runs/INDEX.md
python tools/check_names.py              # exit 1 if any name is unsafe
python tools/migrate_runs.py --dry-run   # (already applied once)
```

## Why names are checked

This tree sits inside OneDrive, so a name that is fine on Linux can still break
sync -- or be silently changed. Both happened here, from a single historical
folder named `figs_temp - Compare`, which parsed into version `"temp "` and
scenario `" Compare"`:

- The **trailing** space was silently dropped by NTFS, so `temp /` and `temp/`
  became one directory while the migration still believed they were two.
  Nothing errored; the recorded destination in `migration_manifest.csv` simply
  did not match the disk.
- The **leading** space survived, and OneDrive will not sync an item whose name
  starts with one.

`run_naming.safe_component` and `run_paths.safe_component` now strip both at
every point a path component is built, so neither can recur.

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

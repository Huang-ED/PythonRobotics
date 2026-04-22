# DWAT_v5_modular

Modular split of DWA + Theta* into four responsibilities:

- `main.py`: orchestrates setup, simulation run, and output/post-processing
- `map_creation.py`: loads scenario/map and computes global path
- `algorithm.py`: runs local planning and simulation loop
- `output_display.py`: plotting, frame saving, and log serialization

## Config Layout

- Scenario config: `configs/scenarios/*.json`
- Algorithm tuning config: `configs/algorithm/default.json`
- Runtime/display config: `configs/run/default.json`

Key runtime option:

- `show_global_planner_search`: if `true`, shows Theta* search expansion plotting.
  Keep this `false` (default) to avoid creating a separate/incomplete global-planner figure.

## EnvData Policy

`EnvData/` remains a data asset folder.

- Keep map images and source assets there.
- Scenario JSON should reference `image_path` (for example `EnvData/simple2.png`).
- The loader resolves image paths in this order:
  1. absolute path
  2. repo-root-relative path
  3. scenario-file-relative path

## Run

From repository root:

Use scenario_3.json for a lightweight super simple test.
```bash
python -m PathPlanning.DWAT_v5_modular.main \
  --scenario PathPlanning/DWAT_v5_modular/configs/scenarios/scenario_3.json
```

Example for video3 scenario:

Use video3.json for a real-world map test. 
(Currently without optimization, this might run for a while.)
```bash
python -m PathPlanning.DWAT_v5_modular.main \
  --scenario PathPlanning/DWAT_v5_modular/configs/scenarios/video3.json
```

Disable animation or frame saving:

```bash
python -m PathPlanning.DWAT_v5_modular.main --no-animation --no-save-frames
```

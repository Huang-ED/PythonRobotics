# Encounter suite figures

Rendered results for the generated encounter scenarios, baseline **v20.7.8-2**.

| file | what it shows |
|---|---|
| `video3_route_and_slots.png` | The video3 route with the reference trajectory at 20 s ticks, and the encounter slots. Port/starboard arrows show which side has water. |
| `openwater_encounters.png` | Contact sheet, 9 scored open-water cases. Ego vs obstacle-free baseline, target track, closest-approach marker. |
| `video3_encounters.png` | Contact sheet, all 10 video3 cases. |

These live here rather than in a `figs_*` folder because `.gitignore` line 77
(`figs*`) silently swallows anything with that prefix -- including every run log
under `Logs/`. Small committed deliverables need a name that does not start with
`figs`.

## Regenerating

The contact sheets come from the run logs, not from saved frames, so they can be
rebuilt without re-simulating:

```bash
python PathPlanning/DWAT_v5_modular/adapters/encounter_plot.py \
    --baseline-log Logs/figs_v20.7.8-2-modular-openwater_baseline_<ts>/log_details.json \
    --manifest PathPlanning/DWAT_v5_modular/configs/scenarios/owenc_manifest.json \
    --runs "Logs/*owenc*/log_details.json" \
    --out PathPlanning/DWAT_v5_modular/results/encounter_suite/openwater_encounters.png
```

Add `--static-map` with a `.npy` of static obstacle points to draw land, as the
video3 sheet does.

## Note on frames

The suite was run with `--no-animation --no-save-frames`. Per-frame PNGs were
deliberately not kept: ~1700-2100 steps per run across 21 runs is roughly 40,000
images, and `savefig` dominates the runtime. The logs retain the full trajectory
and per-iteration `dynamic_obstacles_pos`, which is enough for any static
visualisation. Re-run a specific scenario without those two flags if you want an
animation or an `.mp4` of it.

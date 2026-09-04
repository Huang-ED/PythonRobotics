# Encounter suite figures

Rendered results for the generated encounter scenarios, baseline **v20.7.8-2**.

| file | what it shows |
|---|---|
| `video3_route_and_slots.png` | The video3 route with the reference trajectory at 20 s ticks, and the encounter slots. Port/starboard arrows show which side has water. |
| `openwater_encounters.png` | Contact sheet, 9 scored open-water cases. Ego vs obstacle-free baseline, target track, closest-approach marker. |
| `video3_encounters.png` | Contact sheet, all 10 video3 cases. |
| `ow_sequence.mp4` | Basin, three targets in sequence: head-on, then the mirrored 090 crossing pair. |
| `ow_simultaneous.mp4` | Basin, three vessels converging within 20 s -- no single manoeuvre clears all of them. |
| `v3seq.mp4` | video3 route, three targets: head-on (t=45), starboard crossing (t=85), port crossing (t=133). |

The three videos are the combined-scene runs (`--combine`); each is one run of
one scenario holding every target, not a montage. Their source runs are
`runs/v20.7.8-2/combined/{ow_sequence,ow_simultaneous,v3seq}__<ts>/`, which also
hold the logs the contact sheets were built from.

These live in `results/` rather than under `runs/`, which is git-ignored in
full: `results/` is now reserved for small, committed deliverables, and run
output no longer lands here at all.

(The blanket `figs*` ignore rule that used to make this folder's naming
awkward has been removed -- see `.gitignore`.)

## Regenerating

The contact sheets come from the run logs, not from saved frames, so they can be
rebuilt without re-simulating:

```bash
python PathPlanning/DWAT_v5_modular/adapters/encounter_plot.py \
    --baseline-log runs/v20.7.8-2/baseline/openwater_baseline__<ts>/log_details.json \
    --manifest PathPlanning/DWAT_v5_modular/configs/scenarios/owenc_manifest.json \
    --runs "runs/v20.7.8-2/encounters/owenc_*/log_details.json" \
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

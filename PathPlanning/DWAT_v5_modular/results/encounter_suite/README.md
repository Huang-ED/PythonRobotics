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

## Video of the individual cases

The suite was originally run headless, so only logs survived. On 2026-09-04 the
scored cases were re-rendered with frames and encoded to mp4. They live under
`runs/` (git-ignored) rather than here, because 11 videos is deliverable
material only if you want all of it:

| run directory | case |
|---|---|
| `runs/v20.7.8-2/encounters/owenc_head_on__20260904_170445/` | head-on, tightest basin clearance 0.48 |
| `runs/v20.7.8-2/encounters/owenc_cross_{stbd,port}_045__*/` | mirrored pair, 0.58 / 0.57 |
| `runs/v20.7.8-2/encounters/owenc_cross_{stbd,port}_090__*/` | mirrored pair, 0.77 / 0.76 |
| `runs/v20.7.8-2/encounters/owenc_cross_{stbd,port}_135__*/` | mirrored pair, 0.94 / 0.94 |
| `runs/v20.7.8-2/encounters/owenc_overtake_ego__20260904_172308/` | ego overtakes, 1.21 |
| `runs/v20.7.8-2/encounters/owenc_null_safe_pass__20260904_173905/` | the control: deviation 0.00 |
| `runs/v20.7.8-2/predT10/encounters/owenc_overtaken_ego__20260904_173918/` | Finding 03 falsified: reaches goal at predT10 |
| `runs/v20.7.8-2/encounters/v3enc_v3B_head_on__20260904_174030/` | confinement contrast, 0.43 — tightest in the suite |

Each holds `frames/`, `video.mp4`, `log_details.json` and `run.json`. Every
re-run reproduced its original trajectory **bit-exactly** and every scored
number matches the table in `dwat_encounter_suite_rationale.html`, so these are
the same results, now watchable.

The two `overtaken_ego` cases at predT30 still abort on iteration 0 and are not
worth rendering — the video would be a single frame. They now write a
`log_details.json` recording the error, which they previously did not.

## Note on frames

The suite was run with `--no-animation --no-save-frames`. Per-frame PNGs were
deliberately not kept: ~1700-2100 steps per run across 21 runs is roughly 40,000
images, and `savefig` dominates the runtime. The logs retain the full trajectory
and per-iteration `dynamic_obstacles_pos`, which is enough for any static
visualisation. Re-run a specific scenario without those two flags if you want an
animation or an `.mp4` of it.

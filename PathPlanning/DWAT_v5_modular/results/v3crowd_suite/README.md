# Crowded multi-ship scenes on the video3 route

Dense successors to `v3seq`. Where `v3seq` deliberately spaced its three
encounters so they would not interfere, these pack 5–7 ships with meetings
**16–30 s apart** — well inside the ~50 s detection horizon — so the ego is
handling several encounters at once by construction.

Baseline **v20.7.8-2**. Obstacle-free reference: 206.0 s
(`runs/v20.7.8-2/misc/_tmp_video3_static__20260826_172529/`).

| scene | ships | outcome | duration | min clearance | max ships in cost horizon at once |
|---|---|---|---|---|---|
| `v3crowd5` | 5 | **fails at 137.4 s** | — | 0.57 | 5 |
| `v3crowd6` | 6 | goal | 214.5 s | 0.68 | 5 |
| `v3crowd7` | 7 | **goal** | 214.0 s | 0.68 | **7** |

Seven ships cost the ego **8 seconds** against the obstacle-free run, with
clearances (0.68–1.17) inside the range the single-target suite already reports
(0.43–1.22). It is not scraping through.

## More ships is not harder

The headline result is that **`v3crowd5` fails while `v3crowd6` and `v3crowd7`
succeed**. Two explanations were tested and both are wrong:

- **Not ship count.** Seven succeeds.
- **Not the slow overtaking target.** All three carry one that lingers beside
  the ego for 38–56 s. `v3crowd6` has the *worst* case — 56 s within 4 units,
  minimum gap 1.04, still 1.14 units away when the ego reaches the turn — and
  it completes.

The discriminator is the **heading change across the route's sharp eastward
turn** (y ≈ 46, t ≈ 95–120 s):

| scene | heading swing | outcome |
|---|---|---|
| `v3crowd5` | **29.6°** | failed |
| `v3crowd6` | 71.6° | goal |
| `v3crowd7` | 81.7° | goal |

`v3crowd5`'s ego barely turned. Its two close targets there were
`T3_cross_stbd_090` and `T4_overtake` — and the ego needed to turn **to
starboard**, the side the starboard-crossing target occupied. In `v3crowd7` the
close pair was a head-on plus the overtake, neither of which penalises a
starboard turn the same way. That last step is interpretation; the 29.6°-vs-72°/82°
measurement is the solid part.

## The failure mode is new

`v3crowd5` does not fail like the documented `owenc_overtaken_ego` cases. It is
neither `No best trajectory found` nor `No admissible (v, ω) pairs`:

```
No intersection points found and robot not within local-goal radius
```

That is `algorithm.py`'s local-goal selection. The proximate cause is exact:

```
ego at (46.63, 31.41)
nearest point of the global path :  7.60 units
dist_localgoal (look-ahead)      :  7.00 units
```

The look-ahead circle can no longer reach the Theta\* path, and the ego is
further than `dist_localgoal` from the turning point, so there is no local goal
to steer to. This is a **path-tracking** failure, not a velocity-selection one —
a third distinct mode.

Worth noting for any investigation: nothing in the cost function pulls the ego
back toward the global path. The local planner avoids obstacles and seeks the
local goal, so lateral error has no restoring term and can accumulate until the
path leaves look-ahead range.

`final_frame.png` in the v3crowd5 run directory shows this directly — the ego
pose, the look-ahead circle, the global path just out of reach, and every
target's position at the moment of failure. It is reconstructed from the log,
since the run aborted before any such frame was rendered.

## How the scenes were built

Geometry on this coastline is tightly constrained, so the sets were searched
rather than hand-placed:

1. **Room measured along the route.** Lateral clearance to port and starboard
   at 10 s intervals. Usable window is t ≈ 36–150 s; `t = 155–180` is excluded
   because the obstacle-free run clears that pinch by 1.21 units against a
   ~1.22 collision threshold, so anything there tests the pinch, not the
   encounter.
2. **Navigability sweep.** Every `(t_meet, course, speed)` triple checked for
   land crossing and spawn-on-map: **973 legal targets** out of a much larger
   grid. `t=35–65` admits nearly any geometry; `t=75–115` only head-on and
   overtaking; `t=125–155` head-on and port-135.
3. **Search, not hand-picking.** Combinations scored for meeting spacing
   (16–30 s), geometry variety, and target-to-target clearance (> 1.5 units).
   51 feasible at 5 ships, 37 at 6, and **only 2 at 7** — seven is close to the
   geometric limit of this coastline before the planner is even involved.

Hand-designed candidates were tried first and all three were rejected by the
static checks — targets crossing islands, and two colliding with each other.
The cause was using a higher speed than the geometry had been verified at:
spawn distance is `speed × t_meet`, so raising speed pushes the spawn backwards
onto land.

## Regenerating

The three sets are entries in `COMBINED_SUITES` in `adapters/encounter_suite.py`:

```bash
python PathPlanning/DWAT_v5_modular/adapters/encounter_suite.py \
    --reference-log runs/v20.7.8-2/misc/_tmp_video3_static__20260826_172529/log_details.json \
    --base-scenario PathPlanning/DWAT_v5_modular/configs/scenarios/combined/v3seq.json \
    --combine v3crowd7

python -m PathPlanning.DWAT_v5_modular.main \
    --scenario v3crowd7 --algorithm-config v20.7.8-2 --run-config video
```

## Where the runs are

`runs/` is git-ignored, so these are local paths.

| scene | run directory (under `runs/v20.7.8-2/combined/`) | artifacts |
|---|---|---|
| `v3crowd5` | `v3crowd5__20260907_121547/` | log, frames, video, `final_frame.png` |
| `v3crowd6` | `v3crowd6__20260907_114539/` | log, frames, video |
| `v3crowd7` | `v3crowd7__20260907_114539/` | log, frames, video |

Comparison figure and per-scene track plots from the headless trials are in
`runs/v20.7.8-2/_trials/`. Every rendered run reproduces its trial bit-exactly.

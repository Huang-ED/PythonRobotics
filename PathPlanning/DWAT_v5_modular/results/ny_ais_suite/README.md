# New York AIS suite — results

First runs of the planner on recorded traffic rather than a synthetic scene:
NOAA AIS for January 2021 plus a clipped coastline shapefile, in the Narrows /
Upper Bay / Kill Van Kull. Real metres and seconds, real vessels, real
shoreline. Tuning lineage **`ais_v1.1`** (`../../configs/README_ais.md`).

The point of the dataset is that it supplies geometry nobody designed. Eight
cases produced four clean transits and four failures, and **every one of the
four failures identified something real** — two a `dt` that should have been
derived from the ego and was not, one an assumption the algorithm makes that
ships do not satisfy, one a replay artefact that puts vessels on the map before
they exist.

## Reproducing

`runs/` and `EnvData/derived/` are both git-ignored, so the occupancy grid and
track bundles have to be rebuilt before anything here will run:

```bash
# reference case (ais_v1) -- the one in the video
python -m PathPlanning.DWAT_v5_modular.adapters.new_york_ais_adapter
python -m PathPlanning.DWAT_v5_modular.main \
    --scenario ny_narrows_338862000 --algorithm-config ais_v1 \
    --no-animation --no-save-frames

# the eight-case suite (ais_v1.1)
python -m PathPlanning.DWAT_v5_modular.adapters.new_york_cases \
    --historical 4 --synthetic 4          # writes scenarios + per-ego configs
# ...run the commands it prints, however you like...
python -m PathPlanning.DWAT_v5_modular.adapters.new_york_cases --summarise
```

The first adapter run indexes all 10,162 track CSVs (a few minutes) and caches
the result; later runs read only the segments overlapping the window.

**The planner is deterministic.** Re-running a case reproduces step counts and
clearances exactly (verified on two cases, identical to 0.1 m), so a changed
result always means a changed input, never run-to-run noise.

## The reference case — `ais_v1`

MMSI 338862000, a 29 m towing vessel, 2021-01-28 18:35 UTC, 71 vessels in the
scene. Video: `runs/ais_v1/imported/ny_narrows_338862000__*/video.mp4`.

| | planner | human (the vessel's own AIS track) |
|---|---|---|
| duration | 27.8 min | 45.0 min |
| path | 8.37 km | 9.45 km |
| mean speed | 9.7 kn | 6.8 kn |
| collision | none | — |

Faster and shorter, but it **cuts straight across the Upper Bay** where the
recorded vessel followed the Anchorage Channel with the rest of the traffic.
The shortcut is open water and collision-free; it simply ignores the routeing
every other vessel observes. Nothing in the cost function represents a traffic
separation scheme, so this is the expected result, not a defect — but it is the
clearest single illustration of what the dataset tests that a synthetic scene
cannot.

## The suite — `ais_v1.1`, eight cases

`historical` promotes one recorded vessel to ego and removes it from the
obstacle set, so its own track is a human baseline for the same problem.
`synthetic` invents a start and goal in navigable water and runs it through the
same traffic with nothing removed — plausible hull and speed sampled from a real
vessel, but a voyage that never happened, and therefore no baseline.

| case | ego | outcome | steps | duration | path | ship clear | shore clear | human |
|---|---|---|---|---|---|---|---|---|
| `ny_hist_215804000` | 15 m @ 5.5 m/s | **goal** | 194 | 32.2 min | 9.76 km | 84.5 m | — | 54.0 min / 9.98 km |
| `ny_hist_366962130` | 19 m @ 5.2 m/s | no-admissible | 209 | 34.7 min | 8.61 km | **2.9 m** | 7.2 m | 38.0 min / 8.84 km |
| `ny_hist_367782880` | 15 m @ 12.8 m/s | grounding | 61 | 10.0 min | 7.50 km | 365.1 m | **1.3 m** | 28.0 min / 14.98 km |
| `ny_hist_367791550` | 24 m @ 14.7 m/s | grounding | 54 | 8.8 min | 4.02 km | 109.8 m | **2.7 m** | 26.0 min / 15.47 km |
| `ny_synth_00` | 147 m @ 3.3 m/s | no-admissible | 52 | 8.5 min | 1.61 km | 18.4 m | 100.9 m | — |
| `ny_synth_01` | 25 m @ 1.0 m/s | **goal** | 540 | 89.8 min | 3.88 km | 551.4 m | — | — |
| `ny_synth_02` | 30 m @ 3.8 m/s | **goal** | 96 | 15.8 min | 3.40 km | 56.8 m | — | — |
| `ny_synth_03` | 29 m @ 4.1 m/s | **goal** | 183 | 30.3 min | 7.30 km | 56.0 m | 16.1 m | — |

**4 of 8 reach the goal, and no case strikes another vessel.** The two failure
modes left are grounding and abort. Every case beats its human baseline on time,
which says less than it looks twice over: the speed cost rewards running at
`max_speed` and the recorded vessels had their own reasons to go slower, and in
any case three of the four `hist` runs stop early without finishing, so their
durations are not comparable to a completed passage at all.

### Reading the outcome column

`collision` and `grounding` are different failures with different fixes, and
the planner's own log does not distinguish them — both are just "collision",
because land and vessels are both circular obstacles to it.

- **ship** = a *dynamic* obstacle, another vessel replayed from AIS, radius its
  own `Length/2`
- **shore** = a *static* obstacle, one land cell of the rasterised coastline,
  radius exactly `obstacle_radius` (half a grid cell, 25 m here)

The summariser tells them apart by that radius: a struck obstacle carrying
exactly `obstacle_radius` was a map cell, anything else was a vessel. Without
that split the table is incoherent — a ferry 365 m clear of every ship reads as
a "collision" when it has actually run onto a pier.

`no-admissible` is the planner finding no acceptable trajectory at all and
aborting, not a contact.

### Clearances are re-measured, not read from the log

The planner samples its own trajectory only every `dt`, so it can step over a
contact between samples. An earlier round produced a run reporting
`reached_goal: True` with **no collision event** that had in fact passed through
a vessel at −0.92 m. The summariser therefore re-measures at 1 s using the same
hull model the planner used, and flags any such case in an `undetected` column.
**That column is False for all eight cases above.**

## What the four failures found

**1 & 2 — two groundings: `dt` is a constant that should be a ratio.** Both are
fast small craft covering ~125 m per 10 s control step against 25 m shore cells;
they cannot resolve the coastline finely enough to turn away. Changing nothing
but `dt` on `ny_hist_367782880`:

| `dt` | travel/step | outcome | duration | path | shore clear |
|---|---|---|---|---|---|
| 10 s | 125 m | grounding | 10.0 min | 7.50 km | 1.3 m |
| 2 s | 25 m | **goal** | 18.9 min | 14.32 km | 6.9 m |
| *human* | — | — | 28.0 min | 14.98 km | — |

Straight-line start→goal is 12.24 km, so at `dt = 2 s` the planner is 17% over
direct against the human's 22%. **This is the proposed `ais_v1.2`** and is not
adopted: it roughly doubles every run's cost and changes the reference case.

**3 — `ny_synth_00`, a 147 m ship, finds no admissible trajectory: the
algorithm assumes the ego can stop.** The admissibility test keeps a candidate
only if `dist > v²/(2a) + v·dt` — the ego must be able to brake to rest short of
the nearest obstacle on the curve. At the abort it is making 3.30 m/s with
`max_accel` 0.0612 m/s², so it needs 54 s and 90 m to stop, and the dynamic
window will not let it drop below 2.69 m/s in a single step:

| | |
|---|---|
| braking room required at the slowest speed the window allows (2.69 m/s) | 86.2 m |
| braking room required at the fastest (3.34 m/s) | 124.8 m |
| clearance actually available | 18.0 m |

Every candidate is rejected and the run aborts after 1.61 km. This is the one
finding that is not a tuning matter: it is an assumption in the algorithm that
does not hold for the vehicles the dataset contains. See
`../../configs/README_ais.md`. **Open, and the subject of `ais_v2`.**

**4 — `ny_hist_366962130` aborts 168.5 m from its goal, and neither cause is
the algorithm.** It covers 8.61 km of an 8.68 km trip, stops, holds station for
22 control steps while the cost climbs, and gives up. Two artefacts compound.

*A vessel that is not there.* The blocker is a 25 m craft sitting directly
between the ego and the goal, 147.7 m from it. Its AIS track does not begin
until t = 2700 s — **620 s after the run ends**. `DynamicObstacle._query_time`
clamps simulation time into each recording's own `[t_start, t_end]`, so a track
that has not started yet is replayed as an obstacle parked at its first fix.
**17 of the 78 vessels in this scene are parked at entry points they never
reach during the run**, five of them within 200 m of the ego's path.

*A hull the admissibility test does not share.* Even taking the phantom at face
value, the ego could have crept past it. The margin turns entirely on which
hull `closest_obstacle_on_curve` uses — and it is the one hull site still on
the disc:

| hull used | gap to the blocker | slowest forward candidate needs | verdict |
|---|---|---|---|
| disc, 10.013 m | +2.43 m | 4.20 m | rejected → abort |
| capsule, half-beam 3.165 m | +4.27 m | 4.20 m | admissible |

So this case says nothing about braking: creeping at 0.4 m/s requires only
4.20 m of stopping room, which the true hull clears. It is worth keeping
precisely because it failed for two reasons that both present as "the planner
gave up", and neither is the planner.

## Fixed along the way

Four defects, all found because the suite varied the ego, and all the same
mistake — a number correct for one vessel treated as a constant when it is
really a ratio to the ego's speed or length:

| | was | now |
|---|---|---|
| `dist_localgoal` | fixed 500 m; two ferries aborted outright | `v × 90 s` |
| `max_yaw_rate` | `v / 2.2L` uncapped; implied 22 °/s for a 15 m hull | capped at 10 °/s |
| `catch_goal_dist` | could equal `dist_localgoal`; success declared 300 m short | `≤ ½ dist_localgoal` |
| ego hull | circumscribed **disc**, `sqrt(L²+W²)/2` | oriented **capsule** |

The hull one is the largest. The disc treats a vessel as being as wide as it is
long: for a 248 × 32 m ship an obstacle 100 m abeam reads as −35 m — a
collision — where the capsule correctly reads 74 m of clear water. The error
scales with aspect ratio, so it is invisible on a 1.2 × 0.5 m lab robot and
disabling on a ship. It is behind `robot_collision_shape`, default `disc`, so
every pre-existing scenario is unaffected.

The change has to be made in **both** collision checks. Applying it only to the
planner (`calc_trajectory_clearance_and_collision`) and leaving the contact
monitor (`check_collision_at_current_position_circle_approximation`) on the disc
made the two disagree about the same ship: the planner correctly certified a
9.7 m abeam pass as clear, since the true half-beam is 3.165 m, and the monitor
then declared a collision against the 10.013 m circumscribed radius. The
`collision` this table used to record for `ny_hist_366962130` was that
disagreement and not a contact — the ego never came within 9.7 m of anything.
The two functions' disc expressions were algebraically identical
(`sqrt(L²+W²)/2` and `sqrt((L/2)²+(W/2)²)`, both 10.0134 m), which is why they
had always agreed before the capsule existed. Fixed in `b128c9b`; both
regression scenarios re-run bit-identical.

## Dataset notes worth keeping

- **Track segments are not contiguous**, despite the dataset readme. 92.8% of
  the 10,162 segments contain a gap over 90 s; the median worst gap is 79
  minutes. But only 3.6% of gaps are real discontinuities — 77.8% are under 5
  minutes and 18.6% are long gaps the vessel did not move across (berth dwell
  compressed out, where holding position is correct). The adapter splits on
  duration **and** displacement, so it cuts the real jumps without shattering
  every track at a pier.
- **Speed filtering must use positions, not `SOG`.** Max reported SOG is 102 kn;
  filtering on SOG alone still left obstacles moving at 37 kn, because SOG can
  look ordinary while the fixes jump. Splitting on speed recomputed from
  positions brings the maximum to 24.2 kn.
- **"Not land" is not good enough for a start or goal.** The ROI contains
  genuinely disconnected water — inland ponds, and pier slips the coastline
  polygon seals off. The first reference attempt failed in the global planner
  because the tug began alongside a berth: navigable on paper, unreachable in
  fact. Start and goal are now required to be in the *largest connected* body of
  water.
- **Downsampling the map is not free.** Rim cell count scales as 1/res (a
  perimeter, not an area), so 20 m → 100 m buys only ~4×; and `obstacle_radius`
  is half a cell, so a 100 m grid adds 50 m of phantom shoreline to every bank.
  50 m is the working compromise.
- **Obstacles exist before their tracks start.** Simulation time is clamped
  into each recording's own `[t_start, t_end]`, so a vessel whose AIS track
  begins mid-window is replayed as parked at its first fix from t = 0, and one
  whose track has ended stays parked at its last. The second is deliberate
  (`on_finish: "hold"`); the first is not, and it fabricates stationary traffic
  — 17 of 78 vessels in `ny_hist_366962130`, one of which cost that run its
  goal. A fix has to stay config-restorable, so it belongs as an `on_start`
  option defaulting to today's behaviour.
- 5.6% of AIS samples fall on land cells, but 82% of those are moored vessels at
  piers the polygon covers. Among *moving* samples it is ~1%, which is berth
  geometry rather than a projection error.

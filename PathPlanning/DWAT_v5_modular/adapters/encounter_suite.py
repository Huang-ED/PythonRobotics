"""Generate controlled ship-encounter scenarios against a reference ego run.

The problem this solves is that hand-placed dynamic obstacles rarely produce an
actual encounter. The three circling obstacles shipped in `video3_lrshift.json`
are the cautionary example: replaying that scenario with `dynamic_obstacles: []`
reproduces the ego trajectory to within 0.01 units over the full 206 s, i.e. the
obstacles never influence the ego at all.

So instead of placing obstacles and hoping, we place them by construction:

    Take a reference (obstacle-free) ego run. Pick a meeting time `t_meet`; the
    ego is then at M with heading h and speed v_e. Choose the target's course
    `chi` and speed `v_t`, and spawn it at

        P0 = M - v_t * chi_hat * t_meet

    Both vessels now hold constant velocity and arrive at M simultaneously, so
    absent any avoidance the encounter is a collision, not a near miss.

Because both are on straight constant-velocity tracks, their relative position
is

    target(t) - ego(t) = (t_meet - t) * (v_e * h_hat - v_t * chi_hat)

whose direction is *constant*: a genuine constant-bearing, decreasing-range
collision course. That fixed direction is the target's relative bearing, which
is what decides the applicable COLREGs rule -- so each case is emitted together
with its bearing and sector, and the geometry can be checked rather than
assumed. Note the bearing is a consequence of (chi, v_t), not a free parameter.

Offsetting the spawn by `cpa` units perpendicular to that relative direction
turns a collision course into a pass at exactly that CPA, which is how the
no-action control case is built.

Targets are emitted as `motion_type: waypoint` with one long leg: constant
velocity in a straight line, which is exactly what the planner's linear
extrapolation assumes. Circular motion would instead measure prediction error,
since DWA re-linearises a curving target every step.

Usage
-----
    python PathPlanning/DWAT_v5_modular/adapters/encounter_suite.py \
        --reference-log runs/v20.7.8-2/baseline/openwater_baseline__<ts>/log_details.json \
        --base-scenario PathPlanning/DWAT_v5_modular/configs/scenarios/openwater_baseline.json \
        --suite openwater
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


# --- COLREGs sectors, as relative bearings in degrees clockwise from the bow ---
# Rule 13 defines overtaking as approaching from more than 22.5 deg abaft the
# beam, i.e. the 112.5..247.5 sector. Rule 14 (head-on) is a narrow band dead
# ahead; the rest splits into crossing-from-starboard (ego gives way, Rule 15)
# and crossing-from-port (ego stands on, Rule 17).
HEAD_ON_HALF_WIDTH = 6.0
ABAFT_THE_BEAM = 112.5


def classify(rel_bearing_deg: float, course_diff_deg: float) -> str:
    """Label the encounter from the target's relative bearing and course delta.

    Bearing alone is not enough: a slower target dead ahead on the *same* course
    is an overtaking case, while a target dead ahead on a reciprocal course is
    head-on. Both sit at a relative bearing of 0.
    """
    b = rel_bearing_deg % 360.0
    d = abs((course_diff_deg + 180.0) % 360.0 - 180.0)

    if b < HEAD_ON_HALF_WIDTH or b > 360.0 - HEAD_ON_HALF_WIDTH:
        # Dead ahead: reciprocal course is head-on, same course is an overtake.
        return "head-on (R14)" if d > 135.0 else "overtaking, ego overtakes (R13, give-way)"
    if b > 180.0 - HEAD_ON_HALF_WIDTH and b < 180.0 + HEAD_ON_HALF_WIDTH and d < 45.0:
        return "overtaken, ego is overtaken (R13/R17, stand-on)"
    if b < ABAFT_THE_BEAM:
        return "crossing from starboard (R15, ego give-way)"
    if b <= 360.0 - ABAFT_THE_BEAM:
        return "overtaking sector (R13)"
    return "crossing from port (R17, ego stand-on)"


def load_static_obstacles(base_scenario: dict, base_path: Path, config):
    """Static obstacle points of the base scenario's map, for navigability checks.

    Loads the map only -- no global planning -- so this stays cheap enough to run
    on every generated case.
    """
    import sys
    sys.path.insert(0, str(_repo_root()))
    from PathPlanning.DWAT_v4_st.map_manager import MapManager

    mm = MapManager(config)
    mm.config.enable_map_boundary_obstacles = base_scenario.get(
        "enable_map_boundary_obstacles", False)
    if "occupancy_path" in base_scenario:
        occ_file = _resolve(base_scenario["occupancy_path"], base_path)
        mm.load_map_from_occupancy(
            np.load(str(occ_file)),
            resolution=base_scenario.get("occupancy_resolution", 1.0),
            origin=tuple(base_scenario.get("occupancy_origin", (0.0, 0.0))),
            cell_anchor=base_scenario.get("occupancy_cell_anchor", "center"),
        )
    else:
        img = _resolve(base_scenario["image_path"], base_path)
        mm.load_map_from_image(str(img), tuple(base_scenario["map_size"]))
    return mm.static_obstacles


def _resolve(rel: str, near: Path) -> Path:
    p = Path(rel)
    if p.is_absolute() and p.exists():
        return p
    for cand in ((_repo_root() / p), (near.parent / p)):
        if cand.exists():
            return cand
    raise FileNotFoundError(rel)


def check_navigable(obstacle, static_obs, t_meet, radius, obstacle_radius=0.5,
                    horizon_pad=20.0, step=0.5):
    """Does this target's track stay in navigable water up to (and past) the meeting?

    A target that sails across an island is not a traffic scenario -- it is a
    geometry the route simply cannot produce. On a real coastline some COLREGs
    encounters are impossible at a given point, because the other vessel would
    have to come from over land. Better to detect that than to ship it.

    Returns (n_blocked_samples, first_blocked_time or None).
    """
    if static_obs is None or len(static_obs) == 0:
        return 0, None
    w = np.asarray(obstacle["waypoints"], dtype=float)
    heading = w[1] - w[0]
    heading = heading / np.linalg.norm(heading)
    times = np.arange(0.0, t_meet + horizon_pad, step)
    pts = w[0][None, :] + heading[None, :] * (obstacle["speed"] * times[:, None])
    d = np.hypot(pts[:, None, 0] - static_obs[None, :, 0],
                 pts[:, None, 1] - static_obs[None, :, 1]).min(axis=1)
    blocked = d < (obstacle_radius + radius)
    first = float(times[blocked][0]) if blocked.any() else None
    return int(blocked.sum()), first


def check_target_separation(obstacles, radii, sim_duration, step=0.5):
    """Closest approach between every pair of targets in a combined scene.

    Targets in a multi-vessel scenario are generated independently against the
    ego, so nothing stops two of them from occupying the same water. A scene
    where the traffic collides with itself is not a harder test, it just looks
    broken -- and it would corrupt the ego's readings too, since it would be
    reacting to a pile-up rather than to an encounter.

    Returns a list of (i, j, min_surface_gap, time_of_closest_approach).
    """
    out = []
    times = np.arange(0.0, sim_duration + step, step)
    tracks = []
    for ob in obstacles:
        w = np.asarray(ob["waypoints"], dtype=float)
        d = w[1] - w[0]
        d = d / np.linalg.norm(d)
        tracks.append(w[0][None, :] + d[None, :] * (ob["speed"] * times[:, None]))
    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            gap = np.hypot(tracks[i][:, 0] - tracks[j][:, 0],
                           tracks[i][:, 1] - tracks[j][:, 1]) - radii[i] - radii[j]
            k = int(gap.argmin())
            out.append((i, j, float(gap[k]), float(times[k])))
    return out


def load_reference(log_path: Path):
    """Return (trajectory, dt) from an obstacle-free run's log.

    The log does not record whether obstacles were present, so supplying an
    obstacle-free run is the caller's responsibility; placing encounters against
    a run that already contains avoidance manoeuvres would compound them.
    """
    with log_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    traj = np.asarray(data["trajectory"], dtype=float)
    dt = float(data["config"]["dt"])
    if not data.get("reached_goal", False):
        raise ValueError(
            f"reference run did not reach the goal ({data.get('error')!r}); "
            "encounters placed against it would not be reproducible"
        )
    return traj, dt


def ego_state_at(traj: np.ndarray, dt: float, t: float):
    """Ego (position, heading, speed) at simulation time t, by interpolation."""
    idx = t / dt
    i = int(math.floor(idx))
    if i < 0 or i + 1 >= traj.shape[0]:
        raise ValueError(
            f"t_meet={t}s is outside the reference run (0..{(traj.shape[0] - 1) * dt:.1f}s)"
        )
    frac = idx - i
    pos = traj[i, 0:2] + frac * (traj[i + 1, 0:2] - traj[i, 0:2])
    # Heading is an angle: interpolate on the shortest arc, not linearly.
    h0, h1 = traj[i, 2], traj[i + 1, 2]
    heading = h0 + frac * ((h1 - h0 + math.pi) % (2 * math.pi) - math.pi)
    speed = traj[i, 3] + frac * (traj[i + 1, 3] - traj[i, 3])
    return pos, heading, float(speed)


def solve_course_for_bearing(h, v_e, bearing_rel_deg, speed):
    """Course that puts the target at a given *relative bearing* on a collision course.

    The two ways to describe a crossing are not interchangeable. Specifying the
    course difference fixes the angle between the two tracks; specifying the
    relative bearing fixes where the target sits in the ego's field of view,
    which is what the COLREGs rules are actually written in terms of. The two
    coincide only for head-on and overtaking.

    Solving the second form: we need `v_e*h_hat - speed*chi_hat = k*u` for the
    unit bearing direction `u` and some `k > 0` (the closing speed). Taking
    norms of `speed*chi_hat = v_e*h_hat - k*u` gives

        k^2 - 2*k*v_e*(h_hat . u) + (v_e^2 - speed^2) = 0

    whose discriminant is `speed^2 - v_e^2 * sin^2(bearing)`. So a target cannot
    hold a bearing abeam unless it is at least as fast as the ego -- a real
    constraint, not a numerical one, and the reason a true 90-degree-bearing
    crossing needs `speed >= v_e`.
    """
    beta = math.radians(bearing_rel_deg)
    # Ship convention: bearing is clockwise from the bow, math angles are CCW.
    u = np.array([math.cos(h - beta), math.sin(h - beta)])
    h_hat = np.array([math.cos(h), math.sin(h)])

    disc = speed ** 2 - (v_e * math.sin(beta)) ** 2
    if disc < 0:
        raise ValueError(
            f"no collision course exists at bearing {bearing_rel_deg} deg for a "
            f"target at speed {speed}: it must be at least "
            f"{v_e * abs(math.sin(beta)):.3f} to hold that bearing against an ego at {v_e}"
        )
    root = math.sqrt(disc)
    cos_term = v_e * float(h_hat @ u)
    # Prefer the larger root: the faster closing solution, i.e. the target
    # approaching rather than trailing away on the same bearing line.
    k = cos_term + root
    if k <= 1e-9:
        k = cos_term - root
    if k <= 1e-9:
        raise ValueError(f"no closing solution at bearing {bearing_rel_deg} deg, speed {speed}")

    chi_hat = (v_e * h_hat - k * u) / speed
    return math.atan2(chi_hat[1], chi_hat[0])


def build_target(traj, dt, t_meet, speed, radius, course_rel_deg=None,
                 bearing_rel_deg=None, cpa=0.0,
                 sim_duration=None, detection_range=25.0):
    """Construct one constant-velocity target that meets the ego at t_meet.

    The encounter geometry is given either as `course_rel_deg` (the angle
    between the two tracks) or as `bearing_rel_deg` (where the target sits
    relative to the ego's bow, clockwise). Exactly one must be supplied; see
    `solve_course_for_bearing` for why they are different questions.

    Either way the angle is measured against the ego's heading at the meeting
    point, so a case reads the same regardless of which way the route runs.
    """
    if (course_rel_deg is None) == (bearing_rel_deg is None):
        raise ValueError("supply exactly one of course_rel_deg or bearing_rel_deg")

    M, h, v_e = ego_state_at(traj, dt, t_meet)
    h_hat = np.array([math.cos(h), math.sin(h)])

    if bearing_rel_deg is not None:
        chi = solve_course_for_bearing(h, v_e, bearing_rel_deg, speed)
        course_rel_deg = math.degrees(chi - h)
    else:
        chi = h + math.radians(course_rel_deg)
    chi_hat = np.array([math.cos(chi), math.sin(chi)])

    # Direction from ego to target, constant along a collision course.
    rel = v_e * h_hat - speed * chi_hat
    rel_norm = float(np.hypot(*rel))
    if rel_norm < 1e-9:
        raise ValueError("degenerate encounter: ego and target velocities are identical")
    rel_hat = rel / rel_norm

    # Offsetting perpendicular to the relative direction sets the CPA exactly,
    # because the along-track component is what collapses to zero at t_meet.
    perp = np.array([-rel_hat[1], rel_hat[0]])
    spawn = M - speed * chi_hat * t_meet + cpa * perp

    rel_bearing = math.degrees(h - math.atan2(rel_hat[1], rel_hat[0])) % 360.0

    if sim_duration is None:
        sim_duration = (traj.shape[0] - 1) * dt
    # One leg long enough that the target never reaches its second waypoint and
    # doubles back (waypoint mode wraps around the list).
    leg = speed * sim_duration * 1.5 + 10.0

    # Range closes linearly at rel_norm, so detection time follows directly.
    t_detect = max(0.0, t_meet - detection_range / rel_norm)

    obstacle = {
        "motion_type": "waypoint",
        "waypoints": [
            [round(float(spawn[0]), 4), round(float(spawn[1]), 4)],
            [round(float(spawn[0] + chi_hat[0] * leg), 4),
             round(float(spawn[1] + chi_hat[1] * leg), 4)],
        ],
        "speed": speed,
        "radius": radius,
    }
    diagnostics = {
        "meeting_point": [round(float(M[0]), 2), round(float(M[1]), 2)],
        "t_meet_s": t_meet,
        "ego_heading_deg": round(math.degrees(h), 1),
        "ego_speed": round(v_e, 3),
        "spawn": [round(float(spawn[0]), 2), round(float(spawn[1]), 2)],
        "target_course_deg": round(math.degrees(chi), 1),
        "course_rel_deg": round(course_rel_deg, 1),
        "target_speed": speed,
        "closing_speed": round(rel_norm, 3),
        "rel_bearing_deg": round(rel_bearing, 1),
        "sector": classify(rel_bearing, course_rel_deg),
        "planned_cpa": cpa,
        "t_detect_s": round(t_detect, 1),
    }
    return obstacle, diagnostics


# Course is relative to the ego's heading at the meeting point:
#   +90 means the target runs 90 deg to port of the ego's heading, which puts it
#   *coming from* the ego's starboard side.
# Speeds are in the same units as the ego's 0.5 cruise.
OPENWATER_CASES = [
    dict(name="head_on",          course_rel_deg=180.0, speed=0.40, t_meet=90.0),
    dict(name="cross_stbd_090",   course_rel_deg=90.0,  speed=0.40, t_meet=90.0),
    dict(name="cross_stbd_045",   course_rel_deg=135.0, speed=0.40, t_meet=90.0),
    dict(name="cross_stbd_135",   course_rel_deg=45.0,  speed=0.40, t_meet=90.0),
    dict(name="cross_port_090",   course_rel_deg=-90.0, speed=0.40, t_meet=90.0),
    dict(name="cross_port_045",   course_rel_deg=-135.0, speed=0.40, t_meet=90.0),
    dict(name="cross_port_135",   course_rel_deg=-45.0, speed=0.40, t_meet=90.0),
    # Ego (0.5) runs down a slower target on the same course: ego is give-way.
    dict(name="overtake_ego",     course_rel_deg=0.0,   speed=0.20, t_meet=90.0),
    # A faster target runs the ego down from astern. This is the one case whose
    # geometry constrains the map rather than the reverse: the target must spawn
    # `(v_t - v_e) * t_meet` astern of the ego's start, so the ego cannot begin
    # hard against the top edge. That is why the basin starts the ego at y=85
    # rather than y=92 -- at y=92 no usable (speed, t_meet) pair exists.
    # Both variants abort on iteration 0 at the default predict_time_obstacle
    # of 30 s, with "No best trajectory found" -- and the reason is not the
    # overtaking itself. At t=0 the ego is at rest, so the dynamic window admits
    # only v in [0, 0.02] (one acceleration step), yet each candidate is scored
    # by holding that v constant across the whole 30 s horizon. The planner
    # therefore predicts itself crawling 0.6 units while the target advances 18
    # and passes through it, so every candidate scores as colliding.
    #
    # Confirmed by construction: the same scenario reaches the goal at predT10
    # (clearance 0.91, deviation 2.94). The horizon, not the encounter, is what
    # makes it unsolvable. Keep both variants -- they are the regression test for
    # that interaction between the one-step dynamic window and a long horizon.
    dict(name="overtaken_ego_fast", course_rel_deg=0.0,  speed=0.75, t_meet=50.0),
    # A 0.1 speed margin keeps the gap open through the ego's spin-up, so this
    # one actually reaches the overtaking and can be scored.
    dict(name="overtaken_ego",    course_rel_deg=0.0,   speed=0.60, t_meet=100.0),
    # Control: same geometry as cross_stbd_090 but passing well clear. The
    # correct response is no deviation at all.
    dict(name="null_safe_pass",   course_rel_deg=90.0,  speed=0.40, t_meet=90.0, cpa=6.0),
]

# video3 confinement tier. Slot A is the only stretch of the route with open
# water to starboard (~45 units, against ~10 to port); Slot B is the narrow
# reach where both sides close to 5-17 units. The t=160-175 pinch is excluded:
# the obstacle-free run clears it with 1.21 units of margin against a ~1.22
# collision threshold, so anything placed there tests the pinch, not the encounter.
#
# Every case below was selected by sweeping (course, speed) against the actual
# coastline and keeping only what floats -- see `check_navigable`. That filter
# is not a formality. On this route a real vessel cannot approach Slot A from
# port at speed (it would have to sail off the land mass to the east), and Slot
# B admits no starboard crossing at all, because starboard there *is* the shore.
# Which encounters a route can even produce is itself a property of the route.
VIDEO3_CASES = [
    # --- Slot A, t=45s: open to starboard, land ~10 units to port ---
    dict(name="v3A_head_on",         course_rel_deg=180.0,  speed=0.40, t_meet=45.0),
    dict(name="v3A_cross_stbd_045",  course_rel_deg=135.0,  speed=0.40, t_meet=45.0),
    dict(name="v3A_cross_stbd_090",  course_rel_deg=90.0,   speed=0.40, t_meet=45.0),
    dict(name="v3A_cross_stbd_135",  course_rel_deg=45.0,   speed=0.40, t_meet=45.0),
    # Mirrors v3A_cross_stbd_135 exactly: bearing 307.5 against 52.5.
    dict(name="v3A_cross_port_045",  course_rel_deg=-45.0,  speed=0.40, t_meet=45.0),
    # Only realisable slowly -- at 0.3+ the spawn point is inland.
    dict(name="v3A_cross_port_090",  course_rel_deg=-90.0,  speed=0.20, t_meet=45.0),
    dict(name="v3A_overtake_ego",    course_rel_deg=0.0,    speed=0.20, t_meet=45.0),
    # --- Slot B, t=133s: narrow both sides, no starboard approach exists ---
    dict(name="v3B_head_on",         course_rel_deg=180.0,  speed=0.30, t_meet=133.0),
    dict(name="v3B_cross_port_135",  course_rel_deg=-135.0, speed=0.30, t_meet=133.0),
    dict(name="v3B_overtake_ego",    course_rel_deg=0.0,    speed=0.20, t_meet=133.0),
]

SUITES = {"openwater": OPENWATER_CASES, "video3": VIDEO3_CASES}


# Combined scenes: several targets in ONE scenario, the way video3 ships three
# at once. Two different things are worth building here and they are not the
# same test.
#
# A *sequence* staggers encounters along the route so the ego meets one vessel
# at a time -- readable as a single continuous run, and the closest analogue to
# the original video3. Spacing matters: detection range is roughly
# predict_time_obstacle * closing speed (~25 units, ~50 s), so meetings closer
# together than that stop being independent and the ego is reacting to two
# things at once whether or not that was the intent.
#
# A *simultaneous* scene deliberately collapses that spacing to zero: several
# targets converging on one point at one instant. That is the genuinely hard
# case, because the planner's obstacle cost aggregates over all of them and the
# manoeuvre that clears one can close another.
COMBINED_SUITES = {
    # video3, 206 s route: Slot A -> a feasible mid-route slot -> Slot B.
    # Geometries are restricted to what that coastline can actually produce.
    "v3_sequence": [
        dict(name="A_head_on",        course_rel_deg=180.0,  speed=0.40, t_meet=45.0),
        dict(name="M_cross_stbd",     course_rel_deg=45.0,   speed=0.30, t_meet=85.0),
        dict(name="B_cross_port",     course_rel_deg=-135.0, speed=0.30, t_meet=133.0),
    ],
    # Open water, 153 s route: head-on, then a mirrored crossing pair, so one
    # run shows the port/starboard symmetry back to back.
    "ow_sequence": [
        dict(name="head_on",          course_rel_deg=180.0,  speed=0.40, t_meet=42.0),
        dict(name="cross_stbd_090",   course_rel_deg=90.0,   speed=0.40, t_meet=88.0),
        # Speed drops to 0.3 for the late meeting: spawn distance is
        # speed * t_meet, and 0.4 * 130 = 52 units would put this target off the
        # side of a 100-unit basin before the run even starts.
        dict(name="cross_port_090",   course_rel_deg=-90.0,  speed=0.30, t_meet=130.0),
    ],
    # Three vessels converging at once, from starboard bow, port bow, and dead
    # ahead. No single manoeuvre clears all three.
    # Meetings are staggered by 10 s rather than set equal. Aiming three targets
    # at one point at one instant makes them collide with each other -- the ego
    # would then be reacting to a pile-up, not to traffic. A 10 s stagger keeps
    # all three well inside the ~50 s detection horizon, so the ego still faces
    # them at once, while leaving the targets room to clear one another.
    # Crowded video3 scenes: meetings 16-30 s apart, i.e. deliberately inside the
    # ~50 s detection horizon, so encounters overlap instead of resolving one at
    # a time. Geometry is heavily constrained by the coastline -- a navigability
    # sweep over (t_meet, course, speed) found only 973 legal targets, and of
    # those only 2 seven-ship combinations keep the targets clear of each other.
    # The t=155-180 pinch is excluded: the obstacle-free run clears it by 1.21
    # units against a ~1.22 threshold, so anything placed there tests the pinch.
    #
    # v3crowd6 and v3crowd7 reach the goal. v3crowd5 does NOT -- it is kept as a
    # regression case for a path-tracking failure (see results/v3crowd_suite).
    "v3crowd5": [
        dict(name="T1_cross_stbd_045",    course_rel_deg=  45.0, speed=0.40, t_meet=38.0),
        dict(name="T2_head_on",           course_rel_deg= 180.0, speed=0.40, t_meet=60.0),
        dict(name="T3_cross_stbd_090",    course_rel_deg=  90.0, speed=0.20, t_meet=90.0),
        dict(name="T4_overtake",          course_rel_deg=   0.0, speed=0.40, t_meet=114.0),
        dict(name="T5_cross_port_135",    course_rel_deg=-135.0, speed=0.20, t_meet=142.0),
    ],
    "v3crowd6": [
        dict(name="T1_cross_stbd_090",    course_rel_deg=  90.0, speed=0.25, t_meet=34.0),
        dict(name="T2_cross_stbd_135",    course_rel_deg= 135.0, speed=0.35, t_meet=58.0),
        dict(name="T3_head_on",           course_rel_deg= 180.0, speed=0.30, t_meet=78.0),
        dict(name="T4_overtake",          course_rel_deg=   0.0, speed=0.45, t_meet=108.0),
        dict(name="T5_cross_port_135",    course_rel_deg=-135.0, speed=0.20, t_meet=130.0),
        dict(name="T6_cross_port_135",    course_rel_deg=-135.0, speed=0.25, t_meet=146.0),
    ],
    "v3crowd7": [
        dict(name="T1_cross_stbd_045",    course_rel_deg=  45.0, speed=0.35, t_meet=36.0),
        dict(name="T2_cross_stbd_135",    course_rel_deg= 135.0, speed=0.25, t_meet=52.0),
        dict(name="T3_cross_stbd_135",    course_rel_deg= 135.0, speed=0.40, t_meet=72.0),
        dict(name="T4_head_on",           course_rel_deg= 180.0, speed=0.20, t_meet=92.0),
        dict(name="T5_overtake",          course_rel_deg=   0.0, speed=0.40, t_meet=114.0),
        dict(name="T6_cross_port_135",    course_rel_deg=-135.0, speed=0.30, t_meet=132.0),
        dict(name="T7_cross_port_135",    course_rel_deg=-135.0, speed=0.20, t_meet=148.0),
    ],
    "ow_simultaneous": [
        dict(name="stbd_bow",         course_rel_deg=135.0,  speed=0.40, t_meet=80.0),
        dict(name="ahead",            course_rel_deg=180.0,  speed=0.40, t_meet=90.0),
        dict(name="port_bow",         course_rel_deg=-135.0, speed=0.40, t_meet=100.0),
    ],
}


def _apply_t_meet(args, cases, suite_name):
    """Return `cases` with any --t-meet overrides applied.

    Shared by the --suite and --combine paths. It used to live only in the
    --suite path, below an early `return _emit_combined(...)`, so --t-meet was
    accepted and then silently ignored for every combined set -- which is the
    only way to de-conflict targets that collide with EACH OTHER.
    """
    overrides = {}
    for spec in args.t_meet:
        name, _, val = spec.partition("=")
        if not val:
            raise SystemExit(f"--t-meet expects NAME=SECONDS, got {spec!r}")
        overrides[name.strip()] = float(val)
    if not overrides:
        return cases
    known = {c["name"] for c in cases}
    unknown = set(overrides) - known
    if unknown:
        raise SystemExit(
            f"--t-meet names not in {suite_name!r}: {sorted(unknown)}\n"
            f"  available: {sorted(known)}")
    for n, v in sorted(overrides.items()):
        was = next(c["t_meet"] for c in cases if c["name"] == n)
        print(f"t_meet override: {n} {was} -> {v} s")
    return [dict(c, t_meet=overrides.get(c["name"], c["t_meet"])) for c in cases]


def _emit_combined(args, base, base_path, traj, dt, sim_duration, static_obs,
                   out_dir, root, prefix) -> int:
    """Write a single scenario containing every target of a combined set."""
    cases = _apply_t_meet(args, COMBINED_SUITES[args.combine], args.combine)
    obstacles, diags = [], []

    for case in cases:
        obstacle, diag = build_target(
            traj, dt,
            t_meet=case["t_meet"],
            course_rel_deg=case.get("course_rel_deg"),
            bearing_rel_deg=case.get("bearing_rel_deg"),
            speed=case["speed"],
            radius=args.radius,
            cpa=case.get("cpa", 0.0),
            sim_duration=sim_duration,
            detection_range=args.detection_range,
        )
        n_blocked, first_blocked = check_navigable(
            obstacle, static_obs, case["t_meet"], args.radius)
        diag.update(name=case["name"], land_samples=n_blocked,
                    first_land_t=first_blocked, navigable=n_blocked == 0)
        sx, sy = diag["spawn"]
        diag["spawn_in_map"] = bool(0 <= sx <= 100 and 0 <= sy <= 100)
        obstacles.append(obstacle)
        diags.append(diag)

    scenario = dict(base)
    scenario["dynamic_obstacles"] = obstacles
    out_path = out_dir / f"{prefix}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(scenario, f, indent=2)
        f.write("\n")

    print(f"\ncombined scene: {args.combine}  ->  {len(obstacles)} targets in one scenario")
    print(f"file: {_display(out_path, root)}\n")
    header = (f"{'target':<16}{'t_meet':>7}{'spawn':>18}{'v_t':>6}"
              f"{'bearing':>9}{'t_det':>7}  sector")
    print(header)
    print("-" * len(header))
    for d in diags:
        flag = "" if d["spawn_in_map"] else "  [spawn off-map]"
        if not d["navigable"]:
            flag += f"  [CROSSES LAND from t={d['first_land_t']:.0f}s]"
        spawn = f"({d['spawn'][0]},{d['spawn'][1]})"
        print(f"{d['name']:<16}{d['t_meet_s']:>7.0f}{spawn:>18}{d['target_speed']:>6.2f}"
              f"{d['rel_bearing_deg']:>8.1f}\u00b0{d['t_detect_s']:>7.1f}  {d['sector']}{flag}")

    # How far apart the meetings are, in the ego's own timeline.
    meets = sorted(d["t_meet_s"] for d in diags)
    gaps = [round(b - a, 1) for a, b in zip(meets, meets[1:])]
    if gaps:
        tight = [g for g in gaps if g < 50.0]
        print(f"\nmeeting spacing: {gaps} s")
        if tight and args.combine != "ow_simultaneous":
            print("  note: gaps under ~50 s are inside the detection horizon, so "
                  "those encounters overlap rather than resolve one at a time.")

    # Targets must also not run into each other.
    radii = [args.radius] * len(obstacles)
    pairs = check_target_separation(obstacles, radii, sim_duration)
    worst = min(pairs, key=lambda r: r[2]) if pairs else None
    if worst is not None:
        i, j, gap, t = worst
        print(f"closest target-to-target gap: {gap:.2f} units between "
              f"{diags[i]['name']} and {diags[j]['name']} at t={t:.0f}s")
        if gap <= 0.0:
            print("  WARNING: two targets collide with each other. The scene would "
                  "show traffic piling up, and the ego's readings would be "
                  "reacting to that rather than to an encounter.")
        elif gap < 2.0:
            print("  note: the targets pass close to one another.")

    manifest = out_dir / f"{prefix}_manifest.json"
    with manifest.open("w", encoding="utf-8") as f:
        json.dump(diags, f, indent=2)
        f.write("\n")
    print(f"\nmanifest: {_display(manifest, root)}")
    return 0


def _display(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference-log", required=True,
                        help="log_details.json from an obstacle-free run of the base scenario")
    parser.add_argument("--base-scenario", required=True,
                        help="scenario JSON to clone; its dynamic_obstacles are replaced")
    parser.add_argument("--suite", choices=sorted(SUITES),
                        help="emit one scenario per case")
    parser.add_argument("--combine", choices=sorted(COMBINED_SUITES),
                        help="emit ONE scenario holding every target of this set")
    parser.add_argument("--prefix", default=None,
                        help="output filename prefix (default: the suite name)")
    parser.add_argument("--radius", type=float, default=0.5,
                        help="target collision radius")
    parser.add_argument("--out-dir", default="PathPlanning/DWAT_v5_modular/configs/scenarios")
    parser.add_argument("--detection-range", type=float, default=25.0,
                        help="range at which the planner starts costing a target; "
                             "predict_time_obstacle * closing speed")
    parser.add_argument("--t-meet", action="append", default=[], metavar="NAME=SECONDS",
                        help="override a case's t_meet, e.g. --t-meet overtaken_ego_fast=40. "
                             "Repeatable. The shipped t_meet values were tuned for an ego "
                             "cruising at 0.5; a different ego reaches a different point at "
                             "the same time, which can push an astern target's spawn off the "
                             "map. This retunes without forking the case list.")
    parser.add_argument("--no-map-check", action="store_true",
                        help="skip the check that each target's track stays in water")
    args = parser.parse_args()
    if (args.suite is None) == (args.combine is None):
        parser.error("supply exactly one of --suite or --combine")

    root = _repo_root()
    ref_path = Path(args.reference_log)
    if not ref_path.is_absolute():
        ref_path = root / ref_path
    base_path = Path(args.base_scenario)
    if not base_path.is_absolute():
        base_path = root / base_path
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    # The emitted scenario is a CLONE of the base scenario, so without this it
    # inherits the base's "group" -- "baseline" for openwater_baseline.json.
    # run_paths uses group to decide where a run's output lands, so every
    # generated encounter would file itself under runs/<version>/baseline/
    # instead of runs/<version>/encounters/. The group must match where the
    # scenario lives, which is what out_dir names.
    _group = out_dir.name

    traj, dt = load_reference(ref_path)
    sim_duration = (traj.shape[0] - 1) * dt
    with base_path.open("r", encoding="utf-8") as f:
        base = json.load(f)

    prefix = args.prefix or args.suite or args.combine
    summary = []

    static_obs = None
    if not args.no_map_check:
        class _Cfg:
            enable_map_boundary_obstacles = False
            obstacle_radius = 0.5
            robot_radius = 0.5
        try:
            static_obs = load_static_obstacles(base, base_path, _Cfg())
        except Exception as exc:      # a missing map must not block generation
            print(f"  [warn] map check unavailable: {exc}")

    if args.combine:
        return _emit_combined(args, base, base_path, traj, dt, sim_duration,
                              static_obs, out_dir, root, prefix)

    _cases = _apply_t_meet(args, SUITES[args.suite], args.suite)

    for case in _cases:
        name = case["name"]
        obstacle, diag = build_target(
            traj, dt,
            t_meet=case["t_meet"],
            course_rel_deg=case.get("course_rel_deg"),
            bearing_rel_deg=case.get("bearing_rel_deg"),
            speed=case["speed"],
            radius=args.radius,
            cpa=case.get("cpa", 0.0),
            sim_duration=sim_duration,
            detection_range=args.detection_range,
        )

        scenario = dict(base)
        scenario["dynamic_obstacles"] = [obstacle]
        scenario["group"] = _group
        out_path = out_dir / f"{prefix}_{name}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(scenario, f, indent=2)
            f.write("\n")

        n_blocked, first_blocked = check_navigable(
            obstacle, static_obs, case["t_meet"], args.radius)
        diag["land_samples"] = n_blocked
        diag["first_land_t"] = first_blocked
        diag["navigable"] = n_blocked == 0

        diag["name"] = name
        diag["file"] = str(out_path.relative_to(root))
        # A spawn outside the map is not fatal -- dynamic obstacles are not
        # bounds-checked -- but it means the target is invisible at t=0 and the
        # encounter will not look right on the animation.
        sx, sy = diag["spawn"]
        diag["spawn_in_map"] = bool(0 <= sx <= 100 and 0 <= sy <= 100)
        summary.append(diag)

    print(f"\nreference: {ref_path.relative_to(root)}  ({sim_duration:.1f}s, dt={dt})")
    print(f"suite: {args.suite}  ->  {len(summary)} scenarios in {out_dir.relative_to(root)}\n")
    header = f"{'case':<18}{'t_meet':>7}{'spawn':>18}{'v_t':>6}{'closing':>9}{'bearing':>9}{'t_det':>7}  sector"
    print(header)
    print("-" * len(header))
    for d in summary:
        flag = "" if d["spawn_in_map"] else "  [spawn off-map]"
        if not d.get("navigable", True):
            flag += f"  [CROSSES LAND from t={d['first_land_t']:.0f}s]"
        spawn = f"({d['spawn'][0]},{d['spawn'][1]})"
        print(f"{d['name']:<18}{d['t_meet_s']:>7.0f}{spawn:>18}{d['target_speed']:>6.2f}"
              f"{d['closing_speed']:>9.3f}{d['rel_bearing_deg']:>8.1f}°{d['t_detect_s']:>7.1f}"
              f"  {d['sector']}{flag}")

    manifest = out_dir / f"{prefix}_manifest.json"
    with manifest.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"\nmanifest: {manifest.relative_to(root)}")

    bad = [d["name"] for d in summary
           if not d.get("navigable", True) or not d["spawn_in_map"]]
    if bad:
        print(f"\n{len(bad)} case(s) are not physically realisable on this map: "
              f"{', '.join(bad)}")
        print("The target would have to sail over land or start outside the map. "
              "On a real coastline not every encounter geometry can occur at a "
              "given point; pick a different slot, speed, or geometry.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

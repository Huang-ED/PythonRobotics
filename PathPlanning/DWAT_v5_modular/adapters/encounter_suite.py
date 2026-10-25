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

The `openwater_turning` suite measures prediction error on purpose, with one
abrupt course alteration `turn_s` seconds before the meeting (see
`build_target`): `turn_away` is a collision course that alters clear -- a
threat the constant-velocity forecast reports and that never arrives --
and `turn_in` passes clear until it alters onto a collision course. They
test the two halves of the case for discounting the far horizon: that it
does not over-commit to forecasts that turn out wrong, and that it does not
react too late to the ones that turn out right.

Usage
-----
    python PathPlanning/DWAT_v5_modular/adapters/encounter_suite.py \
        --reference-log runs/v20.7.8-2/baseline/openwater_baseline__<ts>/log_details.json \
        --base-scenario PathPlanning/DWAT_v5_modular/configs/scenarios/openwater_baseline.json \
        --suite openwater \
        --check-log runs/v21.11/baseline/openwater_baseline__<ts>/log_details.json

Every generated target is replayed against a do-nothing ego -- the reference
run, plus each `--check-log` -- and the output says which intended collision
courses would actually hit it. Aim at one config and test another, and targets
can miss by metres (OPEN_PROBLEMS.md item 19); `--strict` refuses to write a
suite with any miss. See do_nothing_check.py.
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

try:
    from . import do_nothing_check as dnc
except ImportError:                     # run as a script
    import do_nothing_check as dnc


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
    times = np.arange(0.0, t_meet + horizon_pad, step)
    pts = dnc.target_track(obstacle, times)
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
    tracks = [dnc.target_track(ob, times) for ob in obstacles]
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
                 sim_duration=None, detection_range=25.0,
                 turn=None, turn_s=None, turn_deg=None):
    """Construct one constant-velocity target that meets the ego at t_meet.

    The encounter geometry is given either as `course_rel_deg` (the angle
    between the two tracks) or as `bearing_rel_deg` (where the target sits
    relative to the ego's bow, clockwise). Exactly one must be supplied; see
    `solve_course_for_bearing` for why they are different questions.

    Either way the angle is measured against the ego's heading at the meeting
    point, so a case reads the same regardless of which way the route runs.

    ``turn`` makes the target alter course once, by ``turn_deg`` (positive =
    to its starboard, i.e. clockwise), at ``t_meet - turn_s``. The geometry
    above always describes the COLLISION leg:

      "away"  sails the collision leg, then alters at the turn point and
              passes clear. Up to the turn a constant-velocity forecast says
              "collision at t_meet"; afterwards it is wrong.
      "in"    arrives at the turn point on a course ``turn_deg`` to port of
              the collision leg, then alters onto it. Up to the turn the
              forecast says "passes clear"; afterwards it is a collision.
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

    if turn not in (None, "away", "in"):
        raise ValueError(f"turn must be 'away' or 'in', got {turn!r}")
    if turn is not None:
        if cpa != 0.0:
            raise ValueError("a turning target's collision leg must have cpa 0")
        if not (0.0 < turn_s < t_meet):
            raise ValueError(f"turn_s={turn_s} must lie in (0, t_meet={t_meet})")
        t_turn = t_meet - turn_s
        P_turn = M - speed * chi_hat * turn_s
        # Ship convention: + is to starboard, i.e. clockwise in math angles.
        chi_other = chi - math.radians(turn_deg) if turn == "away" else chi + math.radians(turn_deg)
        other_hat = np.array([math.cos(chi_other), math.sin(chi_other)])
        if turn == "away":
            first_hat, last_hat = chi_hat, other_hat
        else:
            first_hat, last_hat = other_hat, chi_hat
        spawn = P_turn - speed * first_hat * t_turn

    rel_bearing = math.degrees(h - math.atan2(rel_hat[1], rel_hat[0])) % 360.0

    if sim_duration is None:
        sim_duration = (traj.shape[0] - 1) * dt
    # One leg long enough that the target never reaches its second waypoint and
    # doubles back (waypoint mode wraps around the list).
    leg = speed * sim_duration * 1.5 + 10.0

    # Range closes linearly at rel_norm, so detection time follows directly.
    t_detect = max(0.0, t_meet - detection_range / rel_norm)

    if turn is None:
        waypoints = [spawn, spawn + chi_hat * leg]
    else:
        waypoints = [spawn, P_turn, P_turn + last_hat * leg]
    obstacle = {
        "motion_type": "waypoint",
        "waypoints": [[round(float(p[0]), 4), round(float(p[1]), 4)] for p in waypoints],
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
    if turn is not None:
        diagnostics.update(
            turn=turn, turn_s=turn_s, t_turn_s=round(t_turn, 1), turn_deg=turn_deg,
            turn_point=[round(float(P_turn[0]), 2), round(float(P_turn[1]), 2)],
            expect_threat=(turn == "in"),
        )
    return obstacle, diagnostics


def build_arc_target(traj, dt, t_meet, speed, radius, course_rel_deg, arc, turn_rate_deg,
                     tau=25.0, window=(45.0, 15.0), sim_duration=None, sample_dt=0.5):
    """A target that turns steadily, so a constant-velocity forecast drifts.

    At look-ahead tau, a forecast from a vessel turning at omega misses its true
    position by about 0.5 * v * omega * tau^2 -- small up close, large far out,
    which is the claim being tested ("long-range predictions are less
    accurate"). The geometry is fixed at t_ref = t_meet - tau, on the collision
    leg of ``course_rel_deg``:

      "phantom"  at t_ref the target is ON the collision leg, heading along it:
                 its forecast says collision at t_meet. It keeps turning, and
                 passes clear.
      "hidden"   the target is at the meeting point M at t_meet, heading along
                 the collision leg there: a true collision. Its forecast at
                 t_ref points elsewhere and says it passes clear.

    It turns at ``turn_rate_deg`` (> 0 counter-clockwise, i.e. to port) only
    during [t_meet - window[0], t_meet + window[1]] and runs straight, along its
    tangent, before and after -- so it cannot circle back into the scene. A
    ``turn_rate_deg`` of 0 is the straight-target control. Emitted as an
    inline ``trajectory`` sampled every ``sample_dt`` s; the simulator's
    velocity is that of each segment, as with replayed AIS.
    """
    M, h, v_e = ego_state_at(traj, dt, t_meet)
    chi = h + math.radians(course_rel_deg)
    w = math.radians(turn_rate_deg)
    if sim_duration is None:
        sim_duration = (traj.shape[0] - 1) * dt
    t_end = sim_duration * 1.5
    ta, tb = t_meet - window[0], t_meet + window[1]

    if arc == "phantom":
        t0, psi0 = t_meet - tau, chi
        p0 = M - speed * np.array([math.cos(chi), math.sin(chi)]) * tau
    elif arc == "hidden":
        t0, psi0, p0 = t_meet, chi, M.copy()
    else:
        raise ValueError(f"arc must be 'phantom' or 'hidden', got {arc!r}")

    def heading(t):
        return psi0 + w * (min(max(t, ta), tb) - t0)

    def turned(t1, t2):
        """Displacement over [t1, t2] while turning (both inside the window)."""
        if abs(w) < 1e-12:
            return speed * (t2 - t1) * np.array([math.cos(psi0), math.sin(psi0)])
        a, b = heading(t1), heading(t2)
        return (speed / w) * np.array([math.sin(b) - math.sin(a), -math.cos(b) + math.cos(a)])

    def pos(t):
        tc = min(max(t, ta), tb)
        p = p0 + turned(t0, tc)
        if t != tc:                        # straight beyond the window, along the tangent
            ps = heading(tc)
            p = p + speed * (t - tc) * np.array([math.cos(ps), math.sin(ps)])
        return p

    times = np.arange(0.0, t_end + sample_dt / 2, sample_dt)
    pts = np.array([pos(t) for t in times])
    obstacle = {
        "motion_type": "trajectory",
        "trajectory": [[round(float(x), 4), round(float(y), 4)] for x, y in pts],
        "times": [round(float(t), 3) for t in times],
        "radius": radius,
        "on_finish": "hold",
    }

    # What a constant-velocity forecast made at t_ref says about the meeting,
    # against what happens.
    tr = t_meet - tau
    p_ref = pos(tr)
    v_ref = (pos(tr + sample_dt) - p_ref) / sample_dt
    forecast = p_ref + v_ref * tau
    rel_bearing = math.degrees(h - math.atan2(*(p_ref - M)[::-1])) % 360.0
    # Of the collision leg at t_ref, as build_target reports it.
    closing = float(np.hypot(*(v_e * np.array([math.cos(h), math.sin(h)])
                               - speed * np.array([math.cos(chi), math.sin(chi)]))))
    diagnostics = {
        "meeting_point": [round(float(M[0]), 2), round(float(M[1]), 2)],
        "t_meet_s": t_meet,
        "ego_heading_deg": round(math.degrees(h), 1),
        "ego_speed": round(v_e, 3),
        "spawn": [round(float(pts[0][0]), 2), round(float(pts[0][1]), 2)],
        "target_speed": speed,
        "course_rel_deg": course_rel_deg,
        "arc": arc,
        "turn_rate_deg": turn_rate_deg,
        "tau_s": tau,
        "turn_window_s": [ta, tb],
        "forecast_at_t_ref_misses_M_by": round(float(np.hypot(*(forecast - M))), 3),
        "actual_at_t_meet_misses_M_by": round(float(np.hypot(*(pos(t_meet) - M))), 3),
        "sector": classify(rel_bearing, course_rel_deg),
        "closing_speed": round(closing, 3),
        "rel_bearing_deg": round(rel_bearing, 1),
        "t_detect_s": round(max(0.0, t_meet - 25.0 / closing), 1),
        "planned_cpa": 0.0,
        "expect_threat": arc == "hidden" or abs(w) < 1e-12,
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

# video3 bend tier: the ego overtakes a slightly slower ship just before the
# route's double port bend (+47 deg at (44,46), +41 deg at (47,39), ~124 s).
# The ship holds a straight course south down the channel while the route
# turns east, so it sits on the INSIDE of the ego's turn. This is the single-
# ship core of the v3crowd5 (T4: 0.40 u/s, t_meet 115) and v3crowd7 (T5: 0.45,
# 116) encounters in which binary severity and frozen targets lost the route.
# Run under v21.13 (P5, 2026-10-26) it is not an overtake: the ego approaches
# on the -123 deg leg, so at ~72 s the ship is a crossing on its starboard bow
# and the ego gives way astern (~100 s) before the bend. No arm loses the route.
# Speed sets how long the overtake lasts (closing 0.05-0.15 u/s); t_meet sets
# where the ego draws level and the ship's course, which is the ego's heading
# at the meeting (-79 deg at 115 s, -69 at 118). The channel admits little
# else: on the ego's line only t_meet 115-118 floats (at 110-114 the ship runs
# onto the corner, at 120 onto the east shore), and a ship 0.5 u to port, on
# the inside of the turn, touches land at the corner for every t_meet 112-118.
# 0.5 u to starboard floats throughout; every case here still hits a
# do-nothing v21.13 ego and a do-nothing waypoint-switching (A1) ego.
VIDEO3_BEND_CASES = [
    dict(name=f"overtake_s{int(round(_v * 100)):02d}_t{int(_t)}{_tag}",
         course_rel_deg=0.0, speed=_v, t_meet=_t, cpa=_cpa)
    for _t, _cpa, _tag in ((115.0, 0.0, ""), (118.0, 0.0, ""), (115.0, -0.5, "_stbd05"))
    for _v in (0.35, 0.40, 0.45)
]

# Course-altering targets on the open-water basin. Same collision legs as
# head_on / cross_stbd_090 / cross_port_090 above, so each pairs with an
# existing straight case. The alteration is 45 deg, 20 s before the meeting:
# deep enough into the 33 s horizon that a planner reacting to the forecast has
# already begun to manoeuvre.
#
# The direction is the one that puts the CLEAR leg (after the turn for "away",
# before it for "in") well off a do-nothing ego: 3.7-4.9 u surface clearance.
# The other direction leaves a crossing target only 1.1 u off -- a near miss,
# not a false alarm. So a crossing target does not always alter the way
# COLREGs would ask of it; this suite tests prediction, not the target's
# seamanship. turn_deg > 0 is to the target's starboard.
OPENWATER_TURNING_CASES = [
    dict(name=f"{base}_{mode}", course_rel_deg=c, speed=0.40, t_meet=90.0,
         turn=mode, turn_s=20.0, turn_deg=deg)
    for base, c, away_deg in (("head_on", 180.0, 45.0),
                              ("cross_stbd_090", 90.0, -45.0),
                              ("cross_port_090", -90.0, 45.0))
    for mode, deg in (("away", away_deg), ("in", away_deg if base == "head_on" else -away_deg))
]

# Steadily turning targets: the constant-velocity forecast is wrong by an
# amount that grows with look-ahead (build_arc_target). A mechanism test of
# the horizon weighting, meant to be run with COLREGs off. Head-on is mirror
# symmetric without COLREGs, so it turns one way only; the starboard crossing
# turns both ways when hidden; its phantom turns to port only, because a
# phantom turning to starboard stays in the ego's path (a do-nothing ego still
# hits it at 0.5 and 1 deg/s, and passes only 0.54 u off at 2 deg/s). Rate 0
# is the straight control on each base.
OPENWATER_ARC_CASES = []
for _base, _c, _signs in (("head_on", 180.0, (1,)), ("cross_stbd_090", 90.0, (1, -1))):
    OPENWATER_ARC_CASES.append(dict(name=f"{_base}_straight", course_rel_deg=_c, speed=0.40,
                                    t_meet=90.0, arc="hidden", turn_rate_deg=0.0))
    for _sgn in _signs:
        for _arc in ("phantom", "hidden"):
            if _arc == "phantom" and _base == "cross_stbd_090" and _sgn < 0:
                continue
            for _r in (0.5, 1.0, 2.0):
                _side = "" if len(_signs) == 1 else ("_port" if _sgn > 0 else "_stbd")
                OPENWATER_ARC_CASES.append(dict(
                    name=f"{_base}_{_arc}{_side}_r{int(_r * 10):02d}", course_rel_deg=_c, speed=0.40,
                    t_meet=90.0, arc=_arc, turn_rate_deg=_sgn * _r))

# Target-speed sweep: the same collision legs as head_on / cross_stbd_090,
# meeting the ego at the same point at 90 s, with the target at 0.2-1.2 u/s.
# Speed sets how long the target is close: under "mean" aggregation a brief
# danger is diluted over the 33 s horizon, under "max" it is not. At 0.8 and
# 1.2 u/s the spawn lies outside the 100 u basin; targets are not
# bounds-checked, so they sail in. Mechanism test, COLREGs off.
OPENWATER_SPEED_CASES = [
    dict(name=f"{_b}_v{int(round(_v * 10)):02d}", course_rel_deg=_c, speed=_v, t_meet=90.0)
    for _b, _c in (("head_on", 180.0), ("cross_stbd_090", 90.0))
    for _v in (0.2, 0.4, 0.8, 1.2)
]

# Long exposure: a shallow-angle convergence at the ego's own speed, so the
# relative velocity is small and the target stays close for a long time
# (closing 0.22 u/s at 25 deg, 0.38 u/s at 45 deg, against 0.9 head-on).
# "mean" weighs how LONG the ego is close, "max" only the closest point, so
# the two should trade depth against duration differently. +course_rel = the
# target heads to port of the ego, i.e. converges from starboard. Mechanism
# test, COLREGs off -- where the planner is mirror-symmetric, so the port-side
# copies would only repeat these.
OPENWATER_EXPOSURE_CASES = [
    dict(name=f"converge_stbd_{int(_c):03d}", course_rel_deg=_c, speed=0.50, t_meet=90.0)
    for _c in (25.0, 45.0)
]

SUITES = {"openwater": OPENWATER_CASES, "video3": VIDEO3_CASES,
          "video3_bend": VIDEO3_BEND_CASES,
          "openwater_turning": OPENWATER_TURNING_CASES,
          "openwater_arcs": OPENWATER_ARC_CASES,
          "openwater_speed": OPENWATER_SPEED_CASES,
          "openwater_exposure": OPENWATER_EXPOSURE_CASES}


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

# Staggered slalom: two threats at different times, where clearing one means
# closing on the other. Two slow vessels ahead on the ego's own course (0.2
# against its 0.5, so it overtakes both: Rule 13, which carries no COLREGs
# cost -- the dynamic cost alone decides). A is met at 70 s, offset `cpa` to
# one side of the path; B is met `delta` s later, offset the same amount to
# the OTHER side. Alone, each forces a sidestep; together, every sidestep that
# clears A heads for B.
#
# This is the one situation where a time weight can rank candidates
# differently from a gain. With a single target every candidate's danger falls
# at about the same time, so any time weight acts on all of them alike. Here a
# candidate trades a closer pass of A (soon) against a closer pass of B (delta
# s later): a discount resolves A first and B later, uniform averaging hedges
# between them. delta sets how much time there is to do them one at a time.
#
# cpa 0.5: centre-line distance 0.5 against a contact distance of 0.7
# (0.5 target radius + 0.2 half-beam), so a do-nothing ego touches both.
for _d in (5, 10, 20, 30):
    for _first, _sgn in (("stbd", -1.0), ("port", 1.0)):
        COMBINED_SUITES[f"ow_slalom_{_first}_d{_d:02d}"] = [
            dict(name="A", course_rel_deg=0.0, speed=0.20, t_meet=70.0, cpa=0.5 * _sgn),
            dict(name="B", course_rel_deg=0.0, speed=0.20, t_meet=70.0 + _d, cpa=-0.5 * _sgn),
        ]

# Near-vs-far dilemma: acting early on a far threat worsens a near one. A is a
# slow vessel the ego overtakes, 1.2 u to starboard of the path (a do-nothing
# ego passes it 0.5 u clear), met at 70 s. B is head-on, met delta s later,
# 0.5 u to PORT of the path, so a do-nothing ego hits it and starboard is the
# cheap side to pass -- by geometry, with no COLREGs rule. Moving to
# starboard early for B closes on A while A is still abeam; waiting for A to
# pass means a later, sharper move for B. Meant for the mech_* configs
# (COLREGs off). cpa < 0 is starboard for both geometries.
for _d in (10, 15, 20, 25):
    COMBINED_SUITES[f"ow_dilemma_d{_d:02d}"] = [
        dict(name="A_near", course_rel_deg=0.0, speed=0.20, t_meet=70.0, cpa=-1.2,
             expect_threat=False),
        dict(name="B_far", course_rel_deg=180.0, speed=0.40, t_meet=70.0 + _d, cpa=0.5),
    ]


def _apply_t_meet(args, cases, suite_name):
    """Return `cases` with any --t-meet overrides applied.

    Shared by the --suite and --combine paths. It used to live only in the
    --suite path, below an early `return _emit_combined(...)`, so --t-meet was
    accepted and then silently ignored for every combined set -- which is the
    only way to de-conflict targets that collide with EACH OTHER.
    """
    known = {c["name"] for c in cases}
    out = [dict(c) for c in cases]
    # --speed works the same way. A meeting time alone cannot always keep a
    # target in water: on the v3 route the v3crowd7 overtake target crosses land
    # for every t_meet within +-25 s at its shipped 0.40, and clears at 0.45.
    for flag, key, unit in (("t_meet", "t_meet", "s"), ("speed", "speed", "u/s")):
        overrides = {}
        for spec in getattr(args, flag, None) or []:
            name, _, val = spec.partition("=")
            if not val:
                raise SystemExit(f"--{flag.replace('_', '-')} expects NAME=VALUE, got {spec!r}")
            overrides[name.strip()] = float(val)
        unknown = set(overrides) - known
        if unknown:
            raise SystemExit(
                f"--{flag.replace('_', '-')} names not in {suite_name!r}: {sorted(unknown)}\n"
                f"  available: {sorted(known)}")
        for c in out:
            if c["name"] in overrides:
                print(f"{key} override: {c['name']} {c[key]} -> {overrides[c['name']]} {unit}")
                c[key] = overrides[c["name"]]
    return out


def _load_checks(args, ref_path: Path, root: Path):
    """[(label, traj, dt, cfg)]: the reference run, then every --check-log."""
    checks = []
    for p in [ref_path] + [Path(c) for c in args.check_log]:
        p = p if p.is_absolute() else root / p
        traj, dt, cfg = dnc.load_obstacle_free_log(p)
        checks.append((_display(p, root), traj, dt, cfg))
    return checks


def _do_nothing(checks, names, obstacles, diags) -> int:
    """Run the do-nothing check for every log; record it in the manifests.

    Returns the number of (log, target) pairs meant to collide that miss.
    """
    expected = [dnc.expected_threat(d) for d in diags]
    misses = 0
    for i, (label, traj, dt, cfg) in enumerate(checks):
        results = dnc.check_targets(obstacles, traj, dt, cfg)
        misses += dnc.report(names, results, expected,
                             label + ("  (reference)" if i == 0 else ""))
        for d, r in zip(diags, results):
            d.setdefault("do_nothing", []).append(dict(log=label, **r))
    return misses


def _refuse(args, misses: int) -> bool:
    if not misses:
        return False
    print(f"\n{misses} intended collision course(s) miss a do-nothing ego: those "
          "targets test traffic passing clear, not avoidance. Retime (--t-meet), "
          "re-speed (--speed), or re-aim at the tested config's own obstacle-free run.")
    if args.strict:
        print("--strict: nothing written.")
        return True
    return False


def _emit_combined(args, base, base_path, traj, dt, sim_duration, static_obs,
                   out_dir, root, prefix, checks) -> int:
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
            turn=case.get("turn"),
            turn_s=case.get("turn_s"),
            turn_deg=case.get("turn_deg"),
        )
        n_blocked, first_blocked = check_navigable(
            obstacle, static_obs, case["t_meet"], args.radius)
        diag.update(name=case["name"], land_samples=n_blocked,
                    first_land_t=first_blocked, navigable=n_blocked == 0)
        if "expect_threat" in case:
            diag["expect_threat"] = bool(case["expect_threat"])
        sx, sy = diag["spawn"]
        diag["spawn_in_map"] = bool(0 <= sx <= 100 and 0 <= sy <= 100)
        obstacles.append(obstacle)
        diags.append(diag)

    misses = _do_nothing(checks, [d["name"] for d in diags], obstacles, diags)
    if _refuse(args, misses):
        return 1

    scenario = dict(base)
    scenario["dynamic_obstacles"] = obstacles
    # Same rule as the --suite path: the scene files itself under the group it
    # is written to, not the base's. Cloning baseline/v3_baseline_step.json
    # otherwise sends every combined run to runs/<version>/baseline/.
    scenario["group"] = out_dir.name
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
    parser.add_argument("--speed", action="append", default=[], metavar="NAME=U_PER_S",
                        help="override a case's target speed, e.g. --speed T5_overtake=0.45. "
                             "Repeatable. For when no --t-meet keeps a target's track in water.")
    parser.add_argument("--check-log", action="append", default=[], metavar="LOG",
                        help="obstacle-free log_details.json of a config the suite will be "
                             "run with; every target is replayed against it as a do-nothing "
                             "ego. Repeatable. The reference log is always checked.")
    parser.add_argument("--strict", action="store_true",
                        help="write nothing if any intended collision course misses a "
                             "do-nothing ego in any checked log")
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
    checks = _load_checks(args, ref_path, root)
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
                              static_obs, out_dir, root, prefix, checks)

    _cases = _apply_t_meet(args, SUITES[args.suite], args.suite)
    pending = []

    for case in _cases:
        name = case["name"]
        if "arc" in case:
            obstacle, diag = build_arc_target(
                traj, dt, t_meet=case["t_meet"], speed=case["speed"], radius=args.radius,
                course_rel_deg=case["course_rel_deg"], arc=case["arc"],
                turn_rate_deg=case["turn_rate_deg"], sim_duration=sim_duration)
        else:
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
                turn=case.get("turn"),
                turn_s=case.get("turn_s"),
                turn_deg=case.get("turn_deg"),
            )

        scenario = dict(base)
        scenario["dynamic_obstacles"] = [obstacle]
        scenario["group"] = _group
        out_path = out_dir / f"{prefix}_{name}.json"
        pending.append((out_path, scenario, obstacle))

        n_blocked, first_blocked = check_navigable(
            obstacle, static_obs, case["t_meet"], args.radius)
        diag["land_samples"] = n_blocked
        diag["first_land_t"] = first_blocked
        diag["navigable"] = n_blocked == 0

        diag["name"] = name
        diag["file"] = _display(out_path, root)
        # A spawn outside the map is not fatal -- dynamic obstacles are not
        # bounds-checked -- but it means the target is invisible at t=0 and the
        # encounter will not look right on the animation.
        sx, sy = diag["spawn"]
        diag["spawn_in_map"] = bool(0 <= sx <= 100 and 0 <= sy <= 100)
        summary.append(diag)

    misses = _do_nothing(checks, [d["name"] for d in summary],
                         [p[2] for p in pending], summary)
    if _refuse(args, misses):
        return 1
    for out_path, scenario, _ in pending:
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(scenario, f, indent=2)
            f.write("\n")

    print(f"\nreference: {_display(ref_path, root)}  ({sim_duration:.1f}s, dt={dt})")
    print(f"suite: {args.suite}  ->  {len(summary)} scenarios in {_display(out_dir, root)}\n")
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
    print(f"\nmanifest: {_display(manifest, root)}")

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

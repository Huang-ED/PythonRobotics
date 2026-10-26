"""A slower ship sailing the ego's own route ahead of it, caught up at a chosen time.

Overtakes along a coastal route. The ship follows the reference run's route
(run.json road_map) at constant speed and leaves it at vertex --leave-at,
carrying straight on along the leg that ends there instead of turning with the
route. Its start is set so that the reference ego draws level with it --
reaches the same distance along the route -- at --level seconds. Level a little
before the vertex where the ship leaves puts the ego beside the ship, on the
outside of the turn, when the route turns across the ship's bow.

encounter_suite places a target by where it meets the reference ego at one
moment. Near a bend that sets the wrong situation: P5's "bend overtake" ships
crossed the ego's pre-bend leg and locked as give-way crossings 45 s before the
meeting (see results/baselines_ablations). A ship on the ego's own route, ahead
and slower, is overtaken from the start, on the same heading.

Each scene is checked against land along its whole track (--duration), against
a do-nothing ego (the reference run), and for the role the ship would get when
it first enters the COLREGs CPA gate along the reference track.

Run from the repo root:
    python PathPlanning/DWAT_v5_modular/adapters/route_follower.py \
        --reference-log runs/v21.13/baseline/v3_baseline_step__<ts>/log_details.json \
        --base-scenario PathPlanning/DWAT_v5_modular/configs/scenarios/baseline/v3_baseline_step.json \
        --leave-at 5 --speed 0.35 0.40 0.45 --level 155 163 171 --prefix v3follow2113

writes PREFIX_s040_l163.json, ... (speed in hundredths of u/s, level time in s)
and PREFIX_manifest.json into --out-dir.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

try:
    from . import do_nothing_check as dnc
    from . import encounter_suite as es
except ImportError:                     # run as a script
    import do_nothing_check as dnc
    import encounter_suite as es


def arclength(route, p):
    """Distance along the polyline `route` of the point nearest to p, and that distance to p."""
    seg = np.diff(route, axis=0)
    lens = np.hypot(seg[:, 0], seg[:, 1])
    start = np.concatenate([[0.0], np.cumsum(lens)[:-1]])
    t = np.clip(((p - route[:-1]) * seg).sum(axis=1) / lens ** 2, 0.0, 1.0)
    near = route[:-1] + t[:, None] * seg
    d = np.hypot(*(near - p).T)
    i = int(np.argmin(d))
    return float(start[i] + t[i] * lens[i]), float(d[i])


def follower(route, leave_at, s0, speed, duration):
    """Waypoints of a ship at distance s0 along `route` at t = 0 that sails the route to
    vertex `leave_at` and then straight on, far enough never to reach its last waypoint."""
    seg = np.diff(route, axis=0)
    lens = np.hypot(seg[:, 0], seg[:, 1])
    vert_s = np.concatenate([[0.0], np.cumsum(lens)])
    if not 0.0 <= s0 < vert_s[leave_at]:
        raise ValueError(f"the ship would start {s0:.1f} u along the route, outside "
                         f"0..{vert_s[leave_at]:.1f} (vertex {leave_at}): change --level or --speed")
    i = int(np.searchsorted(vert_s, s0, side="right") - 1)
    start = route[i] + seg[i] / lens[i] * (s0 - vert_s[i])
    u = seg[leave_at - 1] / lens[leave_at - 1]
    far = route[leave_at] + u * (speed * duration * 1.5 + 10.0)
    pts = [start] + [route[k] for k in range(i + 1, leave_at + 1)] + [far]
    return [[round(float(x), 4), round(float(y), 4)] for x, y in pts]


def gate_entry(traj, dt, obstacle, cfg):
    """First reference-run cycle at which the ship passes the CPA gate, and its role there."""
    from PathPlanning.DWAT_v4_st import colregs as C
    times = np.arange(traj.shape[0]) * dt
    q = dnc.target_track(obstacle, times)
    u = np.gradient(q, dt, axis=0)
    dcpa_gate = float(cfg["colregs_dcpa_gate"])
    tcpa_gate = cfg.get("colregs_tcpa_gate")
    tcpa_gate = float(cfg["predict_time_obstacle"] if tcpa_gate is None else tcpa_gate)
    head_on = math.radians(float(cfg["colregs_head_on_deg"]))
    for k, x in enumerate(traj):
        r = q[k] - x[:2]
        w = u[k] - x[3] * np.array([math.cos(x[2]), math.sin(x[2])])
        tc, dc = C._tcpa_dcpa(r, w)
        if 0.0 < tc <= tcpa_gate and dc <= dcpa_gate:
            beta = C._wrap(math.atan2(r[1], r[0]) - x[2])
            alpha = C._wrap(math.atan2(-r[1], -r[0]) - math.atan2(u[k, 1], u[k, 0]))
            return {"t_s": round(float(times[k]), 1), "role": C._role(beta, alpha, head_on),
                    "bearing_deg": round(math.degrees(beta), 1), "alpha_deg": round(math.degrees(alpha), 1),
                    "range": round(float(np.hypot(*r)), 2)}
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference-log", required=True, help="obstacle-free run; its run.json gives the route")
    ap.add_argument("--base-scenario", required=True)
    ap.add_argument("--leave-at", type=int, required=True, help="route vertex (0 = start) where the ship carries straight on")
    ap.add_argument("--speed", type=float, nargs="+", required=True)
    ap.add_argument("--level", type=float, nargs="+", required=True, help="times (s) at which the reference ego draws level")
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--radius", type=float, default=0.5)
    ap.add_argument("--duration", type=float, default=240.0, help="span of the land check (s)")
    ap.add_argument("--out-dir", default="PathPlanning/DWAT_v5_modular/configs/scenarios/encounters")
    a = ap.parse_args()

    ref = Path(a.reference_log)
    traj, dt, cfg = dnc.load_obstacle_free_log(ref)
    route = np.asarray(json.load(open(ref.parent / "run.json"))["road_map"], float)
    if not 0 < a.leave_at < len(route) - 1:
        raise SystemExit(f"--leave-at must be an inner vertex, 1..{len(route) - 2}")
    base_path = Path(a.base_scenario)
    base = json.load(open(base_path))

    class _Cfg:
        enable_map_boundary_obstacles = False
        obstacle_radius = 0.5
        robot_radius = 0.5
    land = np.asarray(es.load_static_obstacles(base, base_path, _Cfg()), float)

    out_dir = Path(a.out_dir)
    manifest = {"reference_log": str(ref), "base_scenario": str(base_path), "route": route.tolist(),
                "leave_at": a.leave_at, "scenes": {}}
    for speed in a.speed:
        for level in a.level:
            s_level, xte = arclength(route, np.asarray(es.ego_state_at(traj, dt, level)[0]))
            ship = {"motion_type": "waypoint",
                    "waypoints": follower(route, a.leave_at, s_level - speed * level, speed, a.duration),
                    "speed": speed, "radius": a.radius}
            blocked, first = es.check_navigable(ship, land, a.duration, a.radius, step=0.5)
            hit = dnc.check_targets([ship], traj, dt, cfg)[0]
            gate = gate_entry(traj, dt, ship, cfg)
            name = f"{a.prefix}_s{round(speed * 100):03d}_l{level:g}"
            scn = dict(base)
            scn["group"] = "encounters"
            scn["dynamic_obstacles"] = [ship]
            json.dump(scn, open(out_dir / f"{name}.json", "w"), indent=2)
            manifest["scenes"][name] = {"speed": speed, "level_s": level, "level_route_s": round(s_level, 2),
                                        "ego_off_route_at_level": round(xte, 2),
                                        "start": ship["waypoints"][0], "land_blocked_samples": blocked,
                                        "first_land_t": first, "do_nothing": hit, "gate_entry": gate}
            print(f"{name}: start {ship['waypoints'][0]}, "
                  f"{'ON LAND from %.0f s' % first if blocked else 'clear of land'}, "
                  f"do-nothing {hit['min_clearance']:+.2f} u at {hit['t_cpa_s']:.0f} s "
                  f"({'threat' if hit['threat'] else 'MISSES'}), "
                  + ("never enters the CPA gate" if gate is None else
                     f"gate at {gate['t_s']:.0f} s as {gate['role']} (bearing {gate['bearing_deg']:.0f}, "
                     f"alpha {gate['alpha_deg']:.0f}, range {gate['range']:.1f})"))
    json.dump(manifest, open(out_dir / f"{a.prefix}_manifest.json", "w"), indent=2)


if __name__ == "__main__":
    sys.exit(main())

"""Retime one ship of a scenario: slide it along its track, change its speed, or both.

Used to sweep when an encounter happens without changing where: the ship keeps
its track, and with --pivot its position at that time is kept for every speed
before the shift is applied. A shift of +s seconds puts the ship s seconds
further along, so it reaches every point of its track s seconds earlier.

Only `waypoint` ships: the first waypoint moves along the first leg (backwards
for a negative shift, extending the leg). Each new track is checked against
land and against the other ships, as encounter_suite does.

Run from the repo root:
    python PathPlanning/DWAT_v5_modular/adapters/ship_retime.py SCENARIO --ship 2 \
        --speed 0.35 0.40 --shift -2 0 2 --pivot 115 [--out-dir DIR]

writes SCENARIO_s035_dtm2.json, SCENARIO_s035_dt0.json, ... (speed in hundredths
of u/s; dtpN / dtmN for a shift of +N / -N s) and SCENARIO_retime_manifest.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np

try:
    from . import encounter_suite as es
except ImportError:                     # run as a script
    import encounter_suite as es


def retime(ship, speed, shift, pivot):
    w = np.asarray(ship["waypoints"], float)
    u = (w[1] - w[0]) / np.linalg.norm(w[1] - w[0])
    v0 = float(ship["speed"])
    start = w[0] + u * ((v0 - speed) * pivot + speed * shift)
    if (start - w[0]) @ u >= np.linalg.norm(w[1] - w[0]):
        raise SystemExit("the shift moves the ship past its second waypoint")
    out = dict(ship)
    out["waypoints"] = [[round(float(start[0]), 4), round(float(start[1]), 4)]] + ship["waypoints"][1:]
    out["speed"] = speed
    return out


def tag(speed, shift):
    s = f"dt{'p' if shift > 0 else 'm' if shift < 0 else ''}{abs(shift):g}"
    return f"s{round(speed * 100):03d}_{s}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario")
    ap.add_argument("--ship", type=int, required=True, help="ship number, 1-based")
    ap.add_argument("--speed", type=float, nargs="*", default=None, help="new speeds (default: unchanged)")
    ap.add_argument("--shift", type=float, nargs="*", default=[0.0], help="seconds along the track")
    ap.add_argument("--pivot", type=float, default=0.0,
                    help="time whose position is kept when the speed changes")
    ap.add_argument("--duration", type=float, default=230.0,
                    help="span for the land and separation checks (about one coastal run)")
    ap.add_argument("--out-dir")
    a = ap.parse_args()

    src = Path(a.scenario)
    scn = json.load(open(src))
    ships = scn["dynamic_obstacles"]
    j = a.ship - 1
    if not 0 <= j < len(ships) or ships[j].get("motion_type", "waypoint") != "waypoint":
        raise SystemExit(f"ship {a.ship} is not a waypoint ship of {src.name}")
    out_dir = Path(a.out_dir) if a.out_dir else src.parent

    class _Cfg:
        enable_map_boundary_obstacles = False
        obstacle_radius = 0.5
        robot_radius = 0.5
    land = es.load_static_obstacles(scn, src, _Cfg())
    radii = [float(s.get("radius", 0.5)) for s in ships]

    # Several calls (one per speed, say) add to the same manifest.
    man_path = out_dir / f"{src.stem}_retime_manifest.json"
    manifest = json.load(open(man_path)) if man_path.exists() else {"source": src.name, "scenes": {}}
    for speed in (a.speed or [float(ships[j]["speed"])]):
        for shift in a.shift:
            new = dict(scn)
            new["dynamic_obstacles"] = list(ships)
            new["dynamic_obstacles"][j] = retime(ships[j], speed, shift, a.pivot)
            try:
                blocked, first = es.check_navigable(new["dynamic_obstacles"][j], land, a.duration, radii[j], step=0.5)
            except ValueError as exc:       # the ship would reach its last waypoint and wrap
                raise SystemExit(f"{tag(speed, shift)}: {exc}")
            gaps = [g for g in es.check_target_separation(new["dynamic_obstacles"], radii, a.duration) if j in g[:2]]
            gap = min((g[2] for g in gaps), default=float("inf"))
            name = f"{src.stem}_{tag(speed, shift)}"
            json.dump(new, open(out_dir / f"{name}.json", "w"), indent=2)
            manifest["scenes"][name] = {"ship": f"T{a.ship}", "speed": speed, "shift": shift, "pivot": a.pivot,
                                        "start": new["dynamic_obstacles"][j]["waypoints"][0],
                                        "land_blocked_samples": blocked, "min_gap_to_other_ships": round(gap, 2)}
            print(f"{name}: start {new['dynamic_obstacles'][j]['waypoints'][0]} "
                  f"{'ON LAND from %.0f s' % first if blocked else 'clear of land'}, "
                  f"closest other ship {gap:.2f} u")
    json.dump(manifest, open(man_path, "w"), indent=2)


if __name__ == "__main__":
    main()

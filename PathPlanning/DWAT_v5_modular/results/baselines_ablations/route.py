"""Route-following behaviour on multi-leg routes: what the local goal does at bends.

Run from the repo root:
    python -m PathPlanning.DWAT_v5_modular.results.baselines_ablations.route LOG [LOG ...]

Each LOG is a run's log_details.json (its run.json, next to it, gives the
Theta* road map and the scenario). Open-water routes have one leg and say
nothing here; the coastal route (v3_baseline_step, v3crowd, v3enc) has five
bends of 8-73 deg, two on legs shorter than the 7 u switch radius.

Per run: cross-track error to the road map, local-goal jumps (the local goal
moving more than JUMP u in one step), yaw-rate peaks and reversals, and the
closest approach to land. Per bend: how far the ego cuts inside the corner and
the peak yaw rate within +-WIN s of passing it.
"""
import json
import sys
from pathlib import Path

import numpy as np

JUMP = 1.0          # [u] a local-goal move per step larger than this is a jump
WIN = 15.0          # [s] half-window around each bend
REV_ON = 0.5        # [deg/s] yaw rate below this counts as straight


def _seg_dist(p, a, b):
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-12), 0.0, 1.0)
    return np.hypot(*(p - (a + t[:, None] * ab)).T)


def cross_track(xy, road):
    return np.min([_seg_dist(xy, road[i], road[i + 1]) for i in range(len(road) - 1)], axis=0)


def land_points(run):
    """Static obstacle points and radii of the run's scenario map."""
    from PathPlanning.DWAT_v5_modular.adapters.encounter_suite import load_static_obstacles
    from PathPlanning.DWAT_v4_st.dwa_average_weighted_side import Config
    inputs = run["inputs"] if isinstance(run["inputs"], dict) else eval(run["inputs"])
    scen = Path(inputs["scenario"])
    pts = load_static_obstacles(json.load(open(scen)), scen.parent, Config())
    return np.asarray(pts, float)


_LAND = {}


def route_metrics(log):
    log = Path(log)
    d = json.load(open(log))
    run = json.load(open(log.parent / "run.json"))
    road = np.array(eval(run["road_map"]) if isinstance(run["road_map"], str) else run["road_map"], float)
    dt = float(d["config"].get("dt", 0.1))
    tr = np.asarray(d["trajectory"], float)
    xy = tr[:, :2]
    xte = cross_track(xy, road)
    lg = np.array([[e["local_goal_x"], e["local_goal_y"]] for e in d["log_entries"]])
    jumps = np.hypot(*np.diff(lg, axis=0).T)
    yaw = np.degrees(np.unwrap(tr[:, 2]))
    r = np.diff(yaw) / dt
    s = np.sign(r[np.abs(r) > REV_ON])
    rev = int(np.sum(s[1:] != s[:-1]))
    key = str(run["inputs"])
    if key not in _LAND:
        _LAND[key] = land_points(run)
    land = _LAND[key]
    dl = np.min(np.hypot(xy[:, None, 0] - land[None, :, 0], xy[:, None, 1] - land[None, :, 1]), axis=1)
    bends = []
    for k in range(1, len(road) - 1):
        dk = np.hypot(*(xy - road[k]).T)
        i = int(np.argmin(dk))
        w = slice(max(0, i - int(WIN / dt)), min(len(r), i + int(WIN / dt)))
        turn = np.degrees(np.arctan2(*(road[k + 1] - road[k])[::-1]) - np.arctan2(*(road[k] - road[k - 1])[::-1]))
        turn = (turn + 180) % 360 - 180
        bends.append(dict(k=k, turn=turn, cut=float(dk[i]), peak_rate=float(np.abs(r[w]).max())))
    return dict(goal=bool(d.get("reached_goal")), contact=bool(d.get("collision_event")),
                error=(d.get("error") or "")[:60], T=len(tr) * dt,
                xte_mean=float(xte.mean()), xte_max=float(xte.max()),
                jumps=int(np.sum(jumps > JUMP)), max_jump=float(jumps.max()),
                peak_rate=float(np.abs(r).max()), rev=rev,
                turn_total=float(np.sum(np.abs(np.diff(yaw)))), land_min=float(dl.min()), bends=bends)


def main(logs):
    for p in logs:
        m = route_metrics(p)
        name = "/".join(Path(p).parts[-4:-1])
        print(f"{name}\n  goal={m['goal']} contact={m['contact']} T={m['T']:.1f}s {m['error']}")
        print(f"  cross-track mean {m['xte_mean']:.2f} max {m['xte_max']:.2f} | local-goal jumps {m['jumps']} "
              f"(max {m['max_jump']:.2f} u) | peak yaw rate {m['peak_rate']:.2f} deg/s, reversals {m['rev']} | "
              f"total heading change {m['turn_total']:.0f} deg | closest land point {m['land_min']:.2f} u")
        print("  bends: " + "; ".join(f"#{b['k']} {b['turn']:+.0f}deg cut {b['cut']:.2f} peak {b['peak_rate']:.1f}"
                                      for b in m["bends"]))


if __name__ == "__main__":
    main(sys.argv[1:])

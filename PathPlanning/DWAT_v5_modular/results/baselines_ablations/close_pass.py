"""How the ego gets past a ship it comes abeam of: on which side, and ahead of it or behind.

Run from the repo root:
    python -m PathPlanning.DWAT_v5_modular.results.baselines_ablations.close_pass SCENE_GLOB [...]

over the arms of compare.py (latest run per scene). Built for the coastal bend
overtakes (P6-P8): a ship beside the ego when the route turns towards it blocks
the turn, and the ego has to get ahead of it or drop back behind it.

For every ship that comes abeam (range < ABEAM_R and bearing 45-135 deg on
either side) the row gives:
  role     the planner's role for the ship when the abeam phase starts
  side     where the ship is while abeam: port or stbd of the ego (median bearing)
  bend     the largest route turn (> 20 deg) from BACK u behind to LOOK u ahead of
           the ego when it comes abeam: "port 47" etc. The ego is on the OUTSIDE of that bend
           when the ship is on the side the route turns to.
  round    where the ego next crosses the ship's track after coming abeam: ahead
           of its bow ("bow", it got ahead first) or across its wake ("wake", it
           dropped back behind it); "same side" if it never crosses, "never" if
           the run stops while the ship is still abeam
  vmin     the ego's lowest speed from 10 s before the abeam phase to 10 s after
  off      the ego's largest distance from the route over the same span
  clear    surface clearance to that ship over the whole run (colregs_report)
"""
import ast
import json
import math
import sys
from pathlib import Path

import numpy as np

from PathPlanning.DWAT_v5_modular.adapters.colregs_report import score
from PathPlanning.DWAT_v5_modular.results.baselines_ablations.compare import ARMS, latest, outcome
from PathPlanning.DWAT_v5_modular.results.baselines_ablations.route import cross_track

ABEAM_R = 4.0       # [u] centre range below which a ship beside the ego counts as abeam
LOOK = 12.0         # [u] how far along the route to look for the bend the pass is at
BACK = 6.0          # [u] ... and behind, for an ego already in the bend
ARM_NAMES = ["v21.13", "A1 waypoint", "A2 binary", "A3 no pred"]


def _p(v):
    return ast.literal_eval(v) if isinstance(v, str) else v


def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def bend_ahead(route, p):
    """Largest route turn within LOOK u ahead of the point of the route nearest p (deg, + = port)."""
    seg = np.diff(route, axis=0)
    lens = np.hypot(seg[:, 0], seg[:, 1])
    vs = np.concatenate([[0.0], np.cumsum(lens)])
    t = np.clip(((p - route[:-1]) * seg).sum(axis=1) / lens ** 2, 0.0, 1.0)
    i = int(np.argmin(np.hypot(*(route[:-1] + t[:, None] * seg - p).T)))
    s = vs[i] + t[i] * lens[i]
    hd = np.degrees(np.arctan2(seg[:, 1], seg[:, 0]))
    turns = [(_wrap(hd[k] - hd[k - 1]), k) for k in range(1, len(seg)) if s - BACK <= vs[k] <= s + LOOK]
    turns = [x for x in turns if abs(x[0]) > 20.0]
    return max(turns, key=lambda x: abs(x[0]))[0] if turns else 0.0


def passes(log_path):
    d = json.load(open(log_path))
    dt = d["config"]["dt"]
    ents = d["log_entries"]
    n = len(ents)
    e = np.asarray(d["trajectory"], float)[1:n + 1]
    route = np.asarray(json.load(open(Path(log_path).parent / "run.json"))["road_map"], float)
    off = cross_track(e[:, :2], route)
    obs = np.array([_p(x["dynamic_obstacles_pos"]) for x in ents])
    roles = [(_p(x.get("colregs")) or {}).get("roles", []) for x in ents]
    _, sc = score(log_path)
    clear = {r["target"]: r["clear"] for r in sc}
    out = []
    for j in range(obs.shape[1]):
        rel = obs[:, j] - e[:, :2]
        rng = np.hypot(rel[:, 0], rel[:, 1])
        brg = _wrap(np.degrees(np.arctan2(rel[:, 1], rel[:, 0]) - e[:, 2]))
        abeam = (rng < ABEAM_R) & (np.abs(brg) >= 45.0) & (np.abs(brg) <= 135.0)
        if not abeam.any():
            continue
        k0 = int(np.argmax(abeam))
        # the abeam phase: from its first cycle to its last, gaps of up to 5 s bridged
        idx = np.where(abeam)[0]
        gaps = np.where(np.diff(idx) > int(5.0 / dt))[0]
        k1 = int(idx[gaps[0]] if len(gaps) else idx[-1])
        side = "port" if np.median(brg[k0:k1 + 1][abeam[k0:k1 + 1]]) > 0 else "stbd"
        # where the ego next crosses the ship's track: ahead of its bow or across its wake
        sv = np.gradient(obs[:, j], dt, axis=0)
        sd = sv / np.maximum(np.hypot(sv[:, 0], sv[:, 1]), 1e-9)[:, None]
        lat = sd[:, 0] * (-rel[:, 1]) - sd[:, 1] * (-rel[:, 0])     # ego left (+) / right of the ship's track
        lon = -(sd * rel).sum(axis=1)                               # ego ahead (+) / astern of the ship
        flip = np.where(np.sign(lat[k0 + 1:]) != np.sign(lat[k0:-1]))[0]
        if len(flip):
            how = "bow" if lon[k0 + flip[0]] > 0 else "wake"
        else:
            how = "never" if d.get("error") else "same side"
        bend = bend_ahead(route, e[k0, :2])
        a, b = max(0, k0 - int(10.0 / dt)), min(n, k1 + int(10.0 / dt) + 1)
        role = roles[k0][j] if len(roles[k0]) > j else "none"
        out.append({"ship": j, "t_abeam": (k0 + 1) * dt, "t_end": (k1 + 1) * dt, "role": role,
                    "side": side, "bend": bend,
                    "outside": bool(bend) and (side == "port") == (bend > 0),
                    "round": how, "vmin": float(e[a:b, 3].min()), "off": float(off[a:b].max()),
                    "clear": clear.get(j, float("nan")), "out": outcome(log_path)})
    return out


def main(globs):
    print(f"{'scene':<34}{'arm':<13}{'ship':>4}{'abeam':>7}{'end':>6}  {'role':<18}{'side':<5}{'bend':>9}"
          f"  {'round':<10}{'vmin':>6}{'off':>6}{'clear':>7}  out")
    for arm in ARM_NAMES:
        for g in globs:
            for scene, p in sorted(latest(ARMS[arm], g).items()):
                for r in passes(p):
                    bend = f"{'port' if r['bend'] > 0 else 'stbd'} {abs(r['bend']):.0f}" if r["bend"] else "-"
                    print(f"{scene:<34}{arm:<13}{'T%d' % (r['ship'] + 1):>4}{r['t_abeam']:7.0f}{r['t_end']:6.0f}  "
                          f"{r['role']:<18}{r['side']:<5}{bend:>9}{'*' if r['outside'] else ' '} {r['round']:<10}"
                          f"{r['vmin']:6.2f}{r['off']:6.2f}{r['clear']:7.2f}  {r['out']}")
    print("\n* the ego is on the outside of the bend: the ship is on the side the route turns to")


if __name__ == "__main__":
    main(sys.argv[1:])

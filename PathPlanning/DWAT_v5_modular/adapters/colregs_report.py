"""Per-target COLREGs verdict for DWAT runs.

encounter_report scores one target per run -- the closest. A combined scene has
up to seven, each with its own rule, so this scores every target:

  role       from the run's own colregs log if it has one, else replayed with
             colregs.classify on the logged states using the colregs_* keys of
             --colregs-config, so v21.12 and earlier are scored the same way.
             A role can change mid-run; the most frequent one is reported.
  clearance  closest approach, capsule axis to target surface (as encounter_report).
  side       which side of the ego the target was on at that instant.
  verdict    head_on and crossing_give_way pass only with the target to PORT;
             a wrong-side pass more than 2.0 clear is "far", not a violation.
             Other roles carry no side rule and print "-".

Usage
-----
    python -m PathPlanning.DWAT_v5_modular.adapters.colregs_report \\
        runs/v21.13/port_turn_release6/*/*/log_details.json
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from PathPlanning.DWAT_v4_st import colregs, dwa_average_weighted_side as base

_DEFAULT_COLREGS_CONFIG = (Path(__file__).resolve().parents[1]
                           / "configs" / "algorithm" / "v21.13.json")
_FAR = 2.0


def score(log_path, colregs_config=_DEFAULT_COLREGS_CONFIG, replay=False):
    """(reached_goal, [one dict per target]).

    ``replay`` ignores the run's own colregs log and re-classifies with
    ``colregs_config``, so runs that used different role logic (the COLREGs
    comparison encodings) are judged against the same roles.
    """
    d = json.load(open(log_path))
    cfg = d["config"]
    traj = np.asarray(d["trajectory"], float)
    ents = d.get("log_entries") or []
    rows = [e.get("dynamic_obstacles_pos") for e in ents if e.get("dynamic_obstacles_pos")]
    if traj.ndim != 2 or not rows:
        return d.get("reached_goal"), []
    n = min(len(r) for r in rows)
    # log_entries[k] holds the targets at (k+1)*dt: pair with trajectory[k+1],
    # as encounter_report.paired_states does.
    obs = np.array([r[:n] for r in rows])
    T = min(obs.shape[0], traj.shape[0] - 1)
    ego, obs = traj[1:T + 1], obs[:T]
    dt = float(cfg["dt"])

    logged = [e.get("colregs", {}).get("roles") for e in ents[:T]]
    if any(logged) and not replay:
        roles_t = [r or [] for r in logged]
        src = "log"
    else:
        c = base.Config()
        default = base.Config()
        for k, v in cfg.items():
            if hasattr(c, k) and k != "robot_type":
                setattr(c, k, v)
        for k in vars(default):
            if k.startswith("colregs_"):        # the run's own role logic does not carry over
                setattr(c, k, getattr(default, k))
        for k, v in json.load(open(colregs_config)).items():
            if k.startswith("colregs_"):
                setattr(c, k, v)
        vel = np.gradient(obs, dt, axis=0)
        roles_t = [colregs.classify(traj[k], obs[k], vel[k], c) for k in range(T)]
        src = "replay"

    half = max(0.5 * (cfg["robot_length"] - cfg["robot_width"]), 0.0)
    yaw = ego[:, 2]
    axis = np.stack([np.cos(yaw), np.sin(yaw)], -1)[:, None, :]
    to_ob = obs - ego[:, None, :2]
    proj = np.clip(np.sum(to_ob * axis, 2), -half, half)
    perp = to_ob - proj[:, :, None] * axis
    clear = (np.hypot(perp[..., 0], perp[..., 1])
             - cfg.get("obstacle_radius", 0.5) - cfg["robot_width"] / 2)

    out = []
    for j in range(n):
        k = int(np.argmin(clear[:, j]))
        lat = -math.sin(yaw[k]) * to_ob[k, j, 0] + math.cos(yaw[k]) * to_ob[k, j, 1]
        side = "port" if lat > 0 else "stbd"
        seen = Counter(r[j] for r in roles_t if len(r) > j and r[j] != colregs.NONE)
        role = seen.most_common(1)[0][0] if seen else colregs.NONE
        if role in colregs.SIDE_RULE_ROLES:
            verdict = "OK" if side == "port" else ("far" if clear[k, j] > _FAR else "VIOLATION")
        else:
            verdict = "-"
        out.append(dict(target=j, role=role, roles_seen=dict(seen),
                        clear=round(float(clear[k, j]), 3), t_cpa=round((k + 1) * dt, 1),
                        side=side, verdict=verdict, src=src))
    return d.get("reached_goal"), out


def main():
    ap = argparse.ArgumentParser(description="Per-target COLREGs verdict for DWAT runs.")
    ap.add_argument("logs", nargs="+", help="log_details.json files")
    ap.add_argument("--colregs-config", default=str(_DEFAULT_COLREGS_CONFIG),
                    help="config whose colregs_* keys replay roles for runs without a colregs log")
    ap.add_argument("--replay", action="store_true",
                    help="replay roles with --colregs-config even where the run logged its own")
    args = ap.parse_args()

    tally = Counter()
    for p in args.logs:
        goal, rows = score(p, args.colregs_config, replay=args.replay)
        name = Path(p).parent.name.split("__")[0]
        for r in rows:
            tally[r["verdict"]] += 1
            if r["clear"] > 3.0 and r["role"] == colregs.NONE:
                continue  # never came near
            flips = r["roles_seen"] if len(r["roles_seen"]) > 1 else ""
            print(f"{name:<28} goal={str(goal):<5} t{r['target']} {r['role']:<18} "
                  f"clear={r['clear']:>6.3f} t_cpa={r['t_cpa']:>6.1f} side={r['side']:<4} "
                  f"{r['verdict']:<9} [{r['src']}] {flips}")
    print(f"\nside-rule encounters: OK {tally['OK']}  VIOLATION {tally['VIOLATION']}  far {tally['far']}")


if __name__ == "__main__":
    main()

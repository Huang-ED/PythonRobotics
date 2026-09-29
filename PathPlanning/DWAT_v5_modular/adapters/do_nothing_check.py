"""Would each target hit an ego that does nothing?

An encounter suite is aimed at ONE obstacle-free reference run: each target is
placed so that it meets that run's ego at `t_meet` (see encounter_suite.py). If
the ego under test sails a different obstacle-free path -- another config,
another start heading, a retimed route -- the targets are aimed at water it is
no longer in, and "3/3 passed" measures traffic sailing past, not avoidance.
That has voided results three times (OPEN_PROBLEMS.md item 19).

The check: replay the ego's OBSTACLE-FREE log against each target's fixed
track. Targets are open-loop constant-velocity legs, so the result is exact --
it is what the obstacle run would show if the planner never reacted.

A target is a THREAT if the hulls would touch (surface clearance <= 0), using
the simulator's own contact test (check_collision_at_current_position_circle_
approximation): capsule or half-diagonal disc per `robot_collision_shape`, and
`trajectory[i]` paired with the target at `i * dt` -- the order the runner
steps them in (targets move, planner acts, ego moves, contact is checked).
Verified against the logged `dynamic_obstacles_pos` of a v3crowd6 run: the
predicted track matches to 0.0.

Usage
-----
    python PathPlanning/DWAT_v5_modular/adapters/do_nothing_check.py \
        --log runs/v21.11/baseline/v3_baseline_step__<ts>/log_details.json \
        --scenario PathPlanning/DWAT_v5_modular/configs/scenarios/combined/v3crowd6_v2111.json

Exits 1 if any target meant as a collision course (planned_cpa 0, or no
manifest) misses the do-nothing ego.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_obstacle_free_log(path: Path):
    """(trajectory (N, 5), dt, config dict) from a run with no dynamic obstacles."""
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    entries = data.get("log_entries") or []
    if any(e.get("dynamic_obstacles_pos") for e in entries):
        raise ValueError(
            f"{path} has dynamic obstacles in it: its trajectory already contains "
            "avoidance, so it is not what a do-nothing ego would sail")
    traj = np.asarray(data["trajectory"], dtype=float)
    if traj.ndim != 2 or traj.shape[0] < 2:
        raise ValueError(f"{path} recorded no trajectory ({data.get('error')!r})")
    return traj, float(data["config"]["dt"]), data["config"]


def target_track(obstacle: dict, times: np.ndarray) -> np.ndarray:
    """Position of a one-leg `waypoint` target at each time in `times`."""
    if obstacle.get("motion_type", "waypoint") != "waypoint":
        raise ValueError(f"only waypoint targets are open-loop straight legs; "
                         f"got {obstacle.get('motion_type')!r}")
    w = np.asarray(obstacle["waypoints"], dtype=float)
    leg = w[1] - w[0]
    length = float(np.hypot(*leg))
    run = obstacle["speed"] * times
    if run.max() > length:
        # encounter_suite makes the leg 1.5x the run, so this never fires for
        # generated targets; a hand-made multi-leg target is out of scope.
        raise ValueError("target reaches its second waypoint inside the run; "
                         "multi-leg targets are not supported")
    return w[0][None, :] + (leg / length)[None, :] * run[:, None]


def surface_clearance(traj: np.ndarray, target_xy: np.ndarray, target_r: float,
                      cfg: dict) -> np.ndarray:
    """Ego-hull-to-target surface gap per row; <= 0 is contact.

    Same hull as the simulator's contact test and encounter_report.py.
    """
    capsule = str(cfg.get("robot_collision_shape", "disc")) == "capsule"
    L, W = float(cfg["robot_length"]), float(cfg["robot_width"])
    to_ob = target_xy - traj[:, 0:2]
    if capsule:
        half = max(0.5 * (L - W), 0.0)
        axis = np.stack([np.cos(traj[:, 2]), np.sin(traj[:, 2])], axis=1)
        proj = np.clip(np.sum(to_ob * axis, axis=1), -half, half)
        perp = to_ob - proj[:, None] * axis
        d = np.hypot(perp[:, 0], perp[:, 1])
        ego_r = 0.5 * W
    else:
        d = np.hypot(to_ob[:, 0], to_ob[:, 1])
        ego_r = math.hypot(L / 2.0, W / 2.0)
    return d - target_r - ego_r


def check_targets(obstacles, traj, dt, cfg):
    """Closest approach of each target to a do-nothing ego.

    Returns one dict per target: min_clearance, t_cpa_s, threat (bool).
    """
    times = np.arange(traj.shape[0]) * dt
    out = []
    for ob in obstacles:
        r = float(ob.get("radius", cfg.get("obstacle_radius", 0.5)))
        clr = surface_clearance(traj, target_track(ob, times), r, cfg)
        k = int(np.argmin(clr))
        out.append({"min_clearance": round(float(clr[k]), 3),
                    "t_cpa_s": round(float(times[k]), 1),
                    "threat": bool(clr[k] <= 0.0)})
    return out


def expected_threat(diag) -> bool:
    """A target meant to collide: planned_cpa 0 (the null case sets it > 0)."""
    return diag is None or float(diag.get("planned_cpa", 0.0)) <= 0.0


def report(names, results, expected, label) -> int:
    """Print the table; return the number of intended threats that miss."""
    print(f"\ndo-nothing check against {label}")
    header = f"{'target':<22}{'clearance':>10}{'t_cpa':>8}  verdict"
    print(header)
    print("-" * len(header))
    misses = 0
    for name, res, exp in zip(names, results, expected):
        if res["threat"]:
            verdict = "threat" if exp else "THREAT (meant to pass clear)"
        elif exp:
            verdict = "MISSES -- not a collision course for this ego"
            misses += 1
        else:
            verdict = "passes clear (control)"
        print(f"{name:<22}{res['min_clearance']:>10.3f}{res['t_cpa_s']:>8.1f}  {verdict}")
    n_exp = sum(expected)
    print(f"{n_exp - misses}/{n_exp} intended threats would hit a do-nothing ego")
    return misses


def _manifest_for(scenario_path: Path):
    """The manifest entries describing this scenario's targets, in order.

    A --combine scene has its own `<stem>_manifest.json`, one entry per target.
    A --suite writes one file per case and a single `<prefix>_manifest.json`
    whose entries name their file; find the entry that names this one.
    """
    own = scenario_path.with_name(scenario_path.stem + "_manifest.json")
    if own.exists():
        with own.open("r", encoding="utf-8") as f:
            diags = json.load(f)
        return diags if isinstance(diags, list) else None
    for m in sorted(scenario_path.parent.glob("*_manifest.json")):
        try:
            with m.open("r", encoding="utf-8") as f:
                diags = json.load(f)
        except (OSError, ValueError):
            continue
        for d in diags if isinstance(diags, list) else []:
            if isinstance(d, dict) and Path(str(d.get("file", ""))).name == scenario_path.name:
                return [d]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", action="append", required=True,
                        help="obstacle-free log_details.json of the ego under test; repeatable")
    parser.add_argument("--scenario", nargs="+", required=True,
                        help="scenario JSON(s) whose dynamic_obstacles to check")
    args = parser.parse_args()
    root = _repo_root()

    total_misses = total_expected = 0
    for log in args.log:
        log_path = Path(log) if Path(log).is_absolute() else root / log
        traj, dt, cfg = load_obstacle_free_log(log_path)
        for sc in args.scenario:
            sc_path = Path(sc) if Path(sc).is_absolute() else root / sc
            with sc_path.open("r", encoding="utf-8") as f:
                scenario = json.load(f)
            if not isinstance(scenario, dict):
                continue            # a manifest caught by a glob, not a scenario
            obstacles = scenario.get("dynamic_obstacles") or []
            if not obstacles:
                print(f"\n{sc}: no dynamic obstacles")
                continue
            diags = _manifest_for(sc_path)
            if diags is not None and len(diags) == len(obstacles):
                names = [d.get("name", f"#{i}") for i, d in enumerate(diags)]
                expected = [expected_threat(d) for d in diags]
            else:
                # No manifest: every target is presumed a collision course.
                names = [f"{sc_path.stem}#{i}" for i in range(len(obstacles))]
                expected = [True] * len(obstacles)
            results = check_targets(obstacles, traj, dt, cfg)
            total_misses += report(names, results, expected,
                                   f"{log}\n  scenario {sc}")
            total_expected += sum(expected)
    if len(args.scenario) > 1 or len(args.log) > 1:
        print(f"\nTOTAL: {total_expected - total_misses}/{total_expected} intended "
              "threats would hit a do-nothing ego")
    return 1 if total_misses else 0


if __name__ == "__main__":
    raise SystemExit(main())

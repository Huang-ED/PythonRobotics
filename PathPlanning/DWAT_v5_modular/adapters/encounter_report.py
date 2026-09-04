"""Score encounter runs against an obstacle-free baseline.

Each run is reduced to the handful of numbers that decide whether the encounter
was handled well:

  clearance      Closest approach over the run, measured surface-to-surface
                 (centre distance minus the target's radius and the ego's
                 half-diagonal). Negative means the hulls overlapped. This is
                 the safety number.

  pass side      Which side of the ego the target was on at closest approach.
                 Not scored today -- the planner's dynamic-obstacle cost is
                 mirror-symmetric, so it has no side preference to score -- but
                 recorded because it is exactly the baseline needed to show what
                 changes once a COLREGs side rule goes in.

  deviation      How far the run departed from the obstacle-free baseline, and
                 from the global path. Distinguishes a proportionate manoeuvre
                 from an over-reaction.

  delay          Extra time to goal against the baseline.

The null case is read differently from the rest: there, deviation near zero is
the pass condition, because the target was never a threat.

Usage
-----
    python PathPlanning/DWAT_v5_modular/adapters/encounter_report.py \
        --baseline-log Logs/<obstacle-free run>/log_details.json \
        --runs Logs/openwater_head_on_*/log_details.json ...
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _display(path: Path, root: Path) -> str:
    """Repo-relative when it can be, absolute otherwise (outputs may live in /tmp)."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def load_run(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def obstacle_tracks(data):
    """(T, n_obs, 2) array of logged obstacle positions, or None if there were none."""
    entries = data.get("log_entries") or []
    rows = [e.get("dynamic_obstacles_pos") for e in entries]
    rows = [r for r in rows if r]
    if not rows:
        return None
    n = min(len(r) for r in rows)
    return np.array([r[:n] for r in rows], dtype=float)


def cross_track_error(points: np.ndarray, path: np.ndarray) -> np.ndarray:
    """Perpendicular distance from each point to the polyline `path`."""
    out = np.empty(points.shape[0])
    seg_a, seg_b = path[:-1], path[1:]
    seg = seg_b - seg_a
    seg_len2 = np.einsum("ij,ij->i", seg, seg)
    seg_len2 = np.where(seg_len2 < 1e-12, 1e-12, seg_len2)
    for i, p in enumerate(points):
        t = np.clip(np.einsum("ij,ij->i", p - seg_a, seg) / seg_len2, 0.0, 1.0)
        proj = seg_a + t[:, None] * seg
        out[i] = np.min(np.hypot(proj[:, 0] - p[0], proj[:, 1] - p[1]))
    return out


def analyse(run_path: Path, baseline_traj: np.ndarray, manifest: dict, path_xy=None):
    data = load_run(run_path)
    traj = np.asarray(data["trajectory"], dtype=float)
    dt = float(data["config"]["dt"])
    cfg = data["config"]

    # The ego is a rectangle; use its half-diagonal as the effective radius, the
    # same quantity the planner's own collision check uses.
    ego_r = math.hypot(cfg["robot_length"] / 2.0, cfg["robot_width"] / 2.0)

    result = {
        "run": run_path.parent.name,
        "reached_goal": bool(data.get("reached_goal")),
        "error": data.get("error"),
        "collision": data.get("collision_event") is not None,
        "duration_s": round((traj.shape[0] - 1) * dt, 1),
    }

    tracks = obstacle_tracks(data)
    if tracks is not None:
        T = min(tracks.shape[0], traj.shape[0])
        ego = traj[:T, 0:2]
        obs = tracks[:T]
        target_r = manifest.get("radius", cfg.get("obstacle_radius", 0.5))
        d = np.hypot(obs[:, :, 0] - ego[:, None, 0], obs[:, :, 1] - ego[:, None, 1])
        clearance = d - target_r - ego_r
        k = int(np.unravel_index(np.argmin(clearance), clearance.shape)[0])
        j = int(np.unravel_index(np.argmin(clearance), clearance.shape)[1])
        result["min_clearance"] = round(float(clearance[k, j]), 3)
        result["t_cpa_s"] = round(k * dt, 1)

        # Side of the ego the target sat on at closest approach: cross product of
        # the ego's heading with the bearing to the target.
        heading = traj[k, 2]
        to_t = obs[k, j] - ego[k]
        cross = math.cos(heading) * to_t[1] - math.sin(heading) * to_t[0]
        result["target_side_at_cpa"] = "port" if cross > 0 else "starboard"
        rel_b = (math.degrees(heading - math.atan2(to_t[1], to_t[0]))) % 360.0
        result["rel_bearing_at_cpa_deg"] = round(rel_b, 1)

    n = min(traj.shape[0], baseline_traj.shape[0])
    dev = np.hypot(traj[:n, 0] - baseline_traj[:n, 0], traj[:n, 1] - baseline_traj[:n, 1])
    result["max_dev_from_baseline"] = round(float(dev.max()), 3)
    result["delay_vs_baseline_s"] = round(
        (traj.shape[0] - baseline_traj.shape[0]) * dt, 1)
    # Minimum speed only means something inside the encounter: every run starts
    # from rest, so a whole-run minimum is always 0 and says nothing about
    # whether the ego slowed down to give way.
    if "t_cpa_s" in result:
        lo = max(0, int((result["t_cpa_s"] - 40.0) / dt))
        hi = min(traj.shape[0], int((result["t_cpa_s"] + 20.0) / dt))
    else:
        lo, hi = min(int(15.0 / dt), traj.shape[0] - 1), traj.shape[0]
    window = traj[lo:hi, 3]
    result["min_speed_in_encounter"] = round(float(window.min()), 3) if window.size else None

    if path_xy is not None:
        result["max_cross_track"] = round(float(cross_track_error(traj[:, 0:2], path_xy).max()), 3)

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline-log", required=True)
    parser.add_argument("--manifest", default=None,
                        help="manifest emitted by encounter_suite.py, for target radii "
                             "and the intended geometry of each case")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    root = _repo_root()
    base_path = Path(args.baseline_log)
    if not base_path.is_absolute():
        base_path = root / base_path
    baseline = load_run(base_path)
    baseline_traj = np.asarray(baseline["trajectory"], dtype=float)

    manifest_by_name = {}
    if args.manifest:
        mpath = Path(args.manifest)
        if not mpath.is_absolute():
            mpath = root / mpath
        with mpath.open("r", encoding="utf-8") as f:
            for entry in json.load(f):
                manifest_by_name[entry["name"]] = entry

    rows = []
    for run in args.runs:
        p = Path(run)
        if not p.is_absolute():
            p = root / p
        if not p.exists():
            print(f"  [skip] {run} (not found)")
            continue
        # Longest match wins, so a case name that is a prefix of another
        # ("overtake_ego" vs "overtaken_ego") cannot shadow it.
        hits = [n for n in manifest_by_name if n in p.parent.name]
        name = max(hits, key=len) if hits else None
        rows.append((name, analyse(p, baseline_traj, manifest_by_name.get(name, {}))))

    hdr = (f"{'case':<20}{'goal':>6}{'clear':>8}{'t_cpa':>7}{'side':>10}"
           f"{'dev':>8}{'delay':>7}{'vmin':>7}  notes")
    print()
    print(hdr)
    print("-" * len(hdr))
    for name, r in rows:
        note = []
        if r.get("collision"):
            note.append("COLLISION")
        if r.get("error"):
            note.append(str(r["error"]))
        intended = manifest_by_name.get(name, {}).get("sector", "")
        if intended:
            note.append(intended)
        print(f"{(name or r['run'])[:20]:<20}"
              f"{'yes' if r['reached_goal'] else 'NO':>6}"
              f"{r.get('min_clearance', float('nan')):>8.2f}"
              f"{r.get('t_cpa_s', float('nan')):>7.1f}"
              f"{r.get('target_side_at_cpa', '-'):>10}"
              f"{r['max_dev_from_baseline']:>8.2f}"
              f"{r['delay_vs_baseline_s']:>7.1f}"
              f"{(r.get('min_speed_in_encounter') if r.get('min_speed_in_encounter') is not None else float('nan')):>7.2f}"
              f"  {'; '.join(note)}")

    if args.json_out:
        out = Path(args.json_out)
        if not out.is_absolute():
            out = root / out
        with out.open("w", encoding="utf-8") as f:
            json.dump([r for _, r in rows], f, indent=2)
            f.write("\n")
        print(f"\nwrote {_display(out, root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

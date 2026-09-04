"""Contact-sheet plot of a set of encounter runs.

One panel per case: the ego's track against its obstacle-free baseline, the
target's track, and a marker at closest approach. The point is to see at a
glance which way the ego went and how hard it had to work -- a table of
clearances tells you whether a run was safe, not what it did.

Usage
-----
    python PathPlanning/DWAT_v5_modular/adapters/encounter_plot.py \
        --baseline-log Logs/<obstacle-free run>/log_details.json \
        --manifest PathPlanning/DWAT_v5_modular/configs/scenarios/owenc_manifest.json \
        --runs "Logs/*owenc_*/log_details.json" \
        --out scratch/openwater_contact_sheet.png
"""

import argparse
import glob
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def obstacle_tracks(data):
    rows = [e.get("dynamic_obstacles_pos") for e in (data.get("log_entries") or [])]
    rows = [r for r in rows if r]
    if not rows:
        return None
    n = min(len(r) for r in rows)
    return np.array([r[:n] for r in rows], dtype=float)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline-log", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="log paths or globs")
    ap.add_argument("--static-map", default=None,
                    help="optional .npy of static obstacle points to draw as land")
    ap.add_argument("--out", required=True)
    ap.add_argument("--extent", nargs=4, type=float, default=[0, 100, 0, 100],
                    metavar=("XMIN", "XMAX", "YMIN", "YMAX"))
    args = ap.parse_args()

    root = _repo_root()

    def resolve(p):
        q = Path(p)
        return q if q.is_absolute() else root / q

    baseline = np.asarray(load(resolve(args.baseline_log))["trajectory"], dtype=float)
    with resolve(args.manifest).open("r", encoding="utf-8") as f:
        manifest = {m["name"]: m for m in json.load(f)}

    paths = []
    for pattern in args.runs:
        hits = sorted(glob.glob(str(resolve(pattern))))
        paths.extend(hits if hits else [str(resolve(pattern))])

    land = None
    if args.static_map:
        land = np.load(str(resolve(args.static_map)))

    cases = []
    for p in paths:
        pp = Path(p)
        if not pp.exists():
            continue
        hits = [n for n in manifest if n in pp.parent.name]
        if not hits:
            continue
        cases.append((max(hits, key=len), load(pp)))
    # Keep the manifest's ordering so the sheet reads in the intended sequence.
    order = list(manifest)
    cases.sort(key=lambda kv: order.index(kv[0]))

    if not cases:
        print("no runs matched the manifest")
        return 1

    ncol = min(5, len(cases))
    nrow = math.ceil(len(cases) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 4.2 * nrow), squeeze=False)

    for ax in axes.flat:
        ax.set_visible(False)

    for idx, (name, data) in enumerate(cases):
        ax = axes[idx // ncol][idx % ncol]
        ax.set_visible(True)
        traj = np.asarray(data["trajectory"], dtype=float)
        m = manifest[name]

        if land is not None:
            ax.scatter(land[:, 0], land[:, 1], s=2, c="0.8", marker="s", linewidths=0)

        ax.plot(baseline[:, 0], baseline[:, 1], "--", color="0.55", lw=1.2,
                label="baseline (no target)")
        ax.plot(traj[:, 0], traj[:, 1], "-r", lw=1.8, label="ego")

        tracks = obstacle_tracks(data)
        if tracks is not None:
            ax.plot(tracks[:, 0, 0], tracks[:, 0, 1], "-", color="saddlebrown",
                    lw=1.5, label="target")
            ax.plot(tracks[0, 0, 0], tracks[0, 0, 1], "o", color="saddlebrown", ms=5)
            T = min(tracks.shape[0], traj.shape[0])
            d = np.hypot(tracks[:T, 0, 0] - traj[:T, 0], tracks[:T, 0, 1] - traj[:T, 1])
            k = int(d.argmin())
            ax.plot([traj[k, 0], tracks[k, 0, 0]], [traj[k, 1], tracks[k, 0, 1]],
                    "-", color="tab:green", lw=2.0)
            ax.plot(traj[k, 0], traj[k, 1], "x", color="tab:green", ms=9, mew=2)
            cpa = d[k] - m.get("radius", 0.5) - math.hypot(
                data["config"]["robot_length"] / 2.0, data["config"]["robot_width"] / 2.0)
            sub = f"CPA {cpa:.2f} @ t={k * data['config']['dt']:.0f}s"
        else:
            sub = "no target logged"

        ok = data.get("reached_goal")
        status = "" if ok else "  DID NOT REACH GOAL"
        ax.set_title(f"{name}\n{m['sector'].split('(')[0].strip()}  |  {sub}{status}",
                     fontsize=9, color="black" if ok else "crimson")
        ax.set_xlim(args.extent[0], args.extent[1])
        ax.set_ylim(args.extent[2], args.extent[3])
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(loc="lower left", fontsize=7)

    fig.tight_layout()
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=110, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

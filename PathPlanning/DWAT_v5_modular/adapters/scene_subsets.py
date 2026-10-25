"""Derive ship subsets of a multi-ship scenario: leave-one-out and keep-only.

Used to find which ships a crowded-scene failure needs. The kept ships are
copied unchanged and in their original order; the manifest records each new
scene's original ship numbers (T1 = first entry of dynamic_obstacles).

Run from the repo root:
    python PathPlanning/DWAT_v5_modular/adapters/scene_subsets.py SCENARIO \
        --drop 1 2 3 --keep 4,5 4,5,6 [--out-dir DIR]

writes SCENARIO_drop1.json, ..., SCENARIO_only45.json, ... and
SCENARIO_subsets_manifest.json next to SCENARIO (or in DIR).
"""
import argparse
import json
from pathlib import Path


def subset(scn, keep):
    out = dict(scn)
    out["dynamic_obstacles"] = [scn["dynamic_obstacles"][i - 1] for i in keep]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario")
    ap.add_argument("--drop", type=int, nargs="*", default=[], help="ship numbers (1-based) to leave out, one scene each")
    ap.add_argument("--keep", nargs="*", default=[], help="comma-separated ship numbers to keep, one scene each")
    ap.add_argument("--out-dir")
    a = ap.parse_args()

    src = Path(a.scenario)
    scn = json.load(open(src))
    n = len(scn["dynamic_obstacles"])
    out_dir = Path(a.out_dir) if a.out_dir else src.parent
    jobs = [(f"drop{d}", [i for i in range(1, n + 1) if i != d]) for d in a.drop]
    jobs += [("only" + "".join(k.split(",")), [int(i) for i in k.split(",")]) for k in a.keep]
    manifest = {"source": src.name, "n_ships": n, "scenes": {}}
    for tag, keep in jobs:
        if not keep or any(not 1 <= i <= n for i in keep):
            raise SystemExit(f"{tag}: ship numbers must be in 1..{n}")
        name = f"{src.stem}_{tag}"
        json.dump(subset(scn, keep), open(out_dir / f"{name}.json", "w"), indent=2)
        manifest["scenes"][name] = {"ships": [f"T{i}" for i in keep]}
        print(f"{name}: {', '.join(f'T{i}' for i in keep)}")
    json.dump(manifest, open(out_dir / f"{src.stem}_subsets_manifest.json", "w"), indent=2)


if __name__ == "__main__":
    main()

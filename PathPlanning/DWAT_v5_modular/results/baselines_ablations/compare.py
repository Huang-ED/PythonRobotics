"""Compare v21.13 with the baselines and ablations, scene by scene.

Run from the repo root:
    python -m PathPlanning.DWAT_v5_modular.results.baselines_ablations.compare
    python -m PathPlanning.DWAT_v5_modular.results.baselines_ablations.compare --v3crowd

Every arm differs from v21.13 only in the keys its config sets
(configs/algorithm/v21.13/{baseline,ablation}_*.json); all keep the vessel
model, Theta* path, Stage 1, the predicted-contact veto and the COLREGs layer.
Metrics are those of ../ablation_horizon_weighting/compare.py, plus the run's
outcome: goal, contact (collision_event) or abort (error, e.g. path lost).
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

from PathPlanning.DWAT_v5_modular.results.ablation_horizon_weighting.compare import (
    BASE, load, metrics)

ARMS = {
    "v21.13": "runs/v21.13/port_turn_release6",
    "B1 frozen g.05": "runs/v21.13/baseline_frozen_mincl_g005",
    "B1 frozen g.10": "runs/v21.13/baseline_frozen_mincl_g010",
    "B1 frozen g.20": "runs/v21.13/baseline_frozen_mincl_g020",
    "B2 veto only": "runs/v21.13/baseline_veto_only",
    "B3 CRI g.10": "runs/v21.13/baseline_cri_g010",
    "B3 CRI g.25": "runs/v21.13/baseline_cri_g025",
    "B3 CRI g.50": "runs/v21.13/baseline_cri_g050",
    "A1 waypoint": "runs/v21.13/ablation_waypoint",
    "A2 binary": "runs/v21.13/ablation_binary",
    "A3 no pred": "runs/v21.13/ablation_nopred",
}
OPEN_WATER = ["owenc215_*", "ow_sequence215", "ow_simultaneous215"]
V3CROWD = ["v3crowd5_v2111", "v3crowd6_v2111", "v3crowd7_v2111"]


def outcome(p):
    d = json.load(open(p))
    if d.get("collision_event"):
        return "CONTACT"
    if d.get("error"):
        return "abort"
    return "goal" if d.get("reached_goal") else "nogoal"


def latest(root, scene_glob):
    out = {}
    for p in sorted(glob.glob(f"{root}/*/{scene_glob}__*/log_details.json")):
        out[Path(p).parent.name.split("__")[0]] = p     # sorted: latest timestamp wins
    return out


def main(scenes):
    bases = {k: load(v) for k, v in BASE.items()}
    table = {}
    for arm, root in ARMS.items():
        for g in scenes:
            for scene, p in latest(root, g).items():
                m = metrics(p, bases["v3" if "v3crowd" in scene else "ow"])
                m["out"] = outcome(p)
                table.setdefault(scene, {})[arm] = m
    cols = ["clear", "delay", "turn", "maxdev", "ok", "bad"]
    print(f"{'scene':<28}{'arm':<16}{'out':>8}" + "".join(f"{c:>8}" for c in cols))
    for scene in sorted(table):
        for arm in ARMS:
            m = table[scene].get(arm)
            if m is None:
                continue
            print(f"{scene:<28}{arm:<16}{m['out']:>8}" + "".join(
                f"{m[c]:8.2f}" if isinstance(m[c], float) else f"{m[c]:>8}" for c in cols))
        print()
    print("SUMMARY")
    for arm in ARMS:
        ms = [table[s][arm] for s in table if arm in table[s]]
        if not ms:
            continue
        outs = [m["out"] for m in ms]
        print(f"  {arm:<16} n={len(ms):>2} goal={outs.count('goal'):>2} contact={outs.count('CONTACT')} "
              f"abort={outs.count('abort') + outs.count('nogoal')} "
              f"min clear={min(m['clear'] for m in ms):6.3f} mean clear={np.mean([m['clear'] for m in ms]):.3f} "
              f"mean delay={np.mean([m['delay'] for m in ms]):5.1f}s "
              f"mean extra turn={np.mean([m['turn'] for m in ms]):6.1f}deg "
              f"side OK/VIOL={sum(m['ok'] for m in ms)}/{sum(m['bad'] for m in ms)}")


if __name__ == "__main__":
    main(V3CROWD if "--v3crowd" in sys.argv else OPEN_WATER)

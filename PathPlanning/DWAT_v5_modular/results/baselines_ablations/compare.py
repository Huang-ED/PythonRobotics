"""Compare v21.13 with the baselines and ablations, scene by scene.

Run from the repo root:
    python -m PathPlanning.DWAT_v5_modular.results.baselines_ablations.compare
    python -m PathPlanning.DWAT_v5_modular.results.baselines_ablations.compare --v3crowd
    python -m PathPlanning.DWAT_v5_modular.results.baselines_ablations.compare --tight
    python -m PathPlanning.DWAT_v5_modular.results.baselines_ablations.compare --v3enc

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

# An arm may draw on several run roots (v21.13's scenes were run under two
# config names with identical contents); the latest run of a scene wins.
ARMS = {
    "v21.13": ["runs/v21.13/port_turn_release6", "runs/v21.13"],
    "v21.13 aligned": "runs/v21.13/time_aligned",
    "B1 frozen g.05": "runs/v21.13/baseline_frozen_mincl_g005",
    "B1 frozen g.10": "runs/v21.13/baseline_frozen_mincl_g010",
    "B1 frozen g.20": "runs/v21.13/baseline_frozen_mincl_g020",
    "B2 veto only": "runs/v21.13/baseline_veto_only",
    "B2 aligned": "runs/v21.13/baseline_veto_only_aligned",
    "B3 CRI g.10": "runs/v21.13/baseline_cri_g010",
    "B3 CRI g.25": "runs/v21.13/baseline_cri_g025",
    "B3 CRI g.50": "runs/v21.13/baseline_cri_g050",
    "A1 waypoint": "runs/v21.13/ablation_waypoint",
    "A2 binary": "runs/v21.13/ablation_binary",
    "A3 no pred": "runs/v21.13/ablation_nopred",
    # The same arms with the targets stepped back to the rollout's time base
    # (dynamic_ob_time_base "aligned"): their open-water contacts are all
    # shallower than the 0.05 u one-step offset.
    "B1 g.20 aligned": "runs/v21.13/baseline_frozen_mincl_g020_aligned",
    "B3 g.50 aligned": "runs/v21.13/baseline_cri_g050_aligned",
    "A3 aligned": "runs/v21.13/ablation_nopred_aligned",
    # v21.13 with catch_turning_point_dist = dist_localgoal = 9 u: no local-goal
    # jumps at the coastal route's short legs.
    "sr9": "runs/v21.13/switch_radius9",
}
OPEN_WATER = ["owenc215_*", "ow_sequence215", "ow_simultaneous215"]
V3CROWD = ["v3crowd5_v2111", "v3crowd6_v2111", "v3crowd7_v2111"]
# Encounters where v21.13's own clearance is near the binary zone (0.68 u):
# near-versus-far dilemma, course-altering targets, staggered slalom.
TIGHT = ["ow_dilemma_d??", "owturn215_*", "ow_slalom_*"]
# Single encounters along the coastal route, one per bend slot and geometry.
V3ENC = ["v3enc2113_*"]


def outcome(p):
    d = json.load(open(p))
    if d.get("collision_event"):
        return "CONTACT"
    if d.get("error"):
        return "abort"
    return "goal" if d.get("reached_goal") else "nogoal"


def latest(root, scene_glob):
    roots = [root] if isinstance(root, str) else root
    found = [p for r in roots for p in glob.glob(f"{r}/*/{scene_glob}__*/log_details.json")]
    out = {}
    for p in sorted(found, key=lambda q: Path(q).parent.name.split("__")[-1]):
        out[Path(p).parent.name.split("__")[0]] = p     # latest timestamp wins
    return out


def main(scenes):
    bases = {k: load(v) for k, v in BASE.items()}
    table = {}
    for arm, root in ARMS.items():
        for g in scenes:
            for scene, p in latest(root, g).items():
                m = metrics(p, bases["v3" if scene.startswith("v3") else "ow"])
                m["out"] = outcome(p)
                table.setdefault(scene, {})[arm] = m
    cols = ["clear", "delay", "turn", "maxdev", "ok", "bad"]
    print(f"{'scene':<34}{'arm':<16}{'out':>8}" + "".join(f"{c:>8}" for c in cols))
    for scene in sorted(table):
        for arm in ARMS:
            m = table[scene].get(arm)
            if m is None:
                continue
            print(f"{scene:<34}{arm:<16}{m['out']:>8}" + "".join(
                f"{m[c]:8.2f}" if isinstance(m[c], float) else f"{m[c]:>8}" for c in cols))
        print()
    print("SUMMARY")
    for arm in ARMS:
        ms = [table[s][arm] for s in table if arm in table[s]]
        if not ms:
            continue
        outs = [m["out"] for m in ms]
        # delay and turning only mean something for runs that reached the goal
        ok = [m for m in ms if m["out"] == "goal"] or [dict(delay=np.nan, turn=np.nan)]
        print(f"  {arm:<16} n={len(ms):>2} goal={outs.count('goal'):>2} contact={outs.count('CONTACT')} "
              f"abort={outs.count('abort') + outs.count('nogoal')} "
              f"min clear={min(m['clear'] for m in ms):6.3f} mean clear={np.mean([m['clear'] for m in ms]):.3f} "
              f"| goal runs: mean delay={np.mean([m['delay'] for m in ok]):5.1f}s "
              f"mean extra turn={np.mean([m['turn'] for m in ok]):6.1f}deg "
              f"| side OK/VIOL={sum(m['ok'] for m in ms)}/{sum(m['bad'] for m in ms)}")


if __name__ == "__main__":
    main(V3CROWD if "--v3crowd" in sys.argv else TIGHT if "--tight" in sys.argv
         else V3ENC if "--v3enc" in sys.argv else OPEN_WATER)

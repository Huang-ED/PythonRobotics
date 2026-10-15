"""Compare horizon-weighting arms scene by scene (README.md, same folder).

Run from the repo root:
    python -m PathPlanning.DWAT_v5_modular.results.ablation_horizon_weighting.compare
    python -m PathPlanning.DWAT_v5_modular.results.ablation_horizon_weighting.compare --turning
    python -m PathPlanning.DWAT_v5_modular.results.ablation_horizon_weighting.compare --frontier
    python -m PathPlanning.DWAT_v5_modular.results.ablation_horizon_weighting.compare --slalom


Per run: goal, min clearance (capsule, as encounter_report), delay vs the
obstacle-free baseline, COLREGs side verdicts, and the two things the paper
says discounting buys -- *earlier* and *smoother* evasion:

  lead_s     seconds between the ego first leaving its obstacle-free track
             (deviation > DEV_ON) and its closest approach to any target
  turn_deg   total |heading change| over the run, minus the baseline's
  reversals  sign changes of the commanded yaw rate (|omega| > OMEGA_ON).
             Counts path-following wobble after the pass as well; not used
             in the README.
"""
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

from PathPlanning.DWAT_v5_modular.adapters import colregs_report as cr  # noqa: E402

DEV_ON = 0.25
OMEGA_ON = 0.01
ARMS = {
    "uniform (lT=0)": "runs/v21.13/ablation_uniform",
    "uni x1.58": "runs/v21.13/ablation_uniform_gain1p58",
    "v21.13 (lT=1)": "runs/v21.13/port_turn_release6",
    "uni x3.15": "runs/v21.13/ablation_uniform_gain3p15",
    "strong (lT=3)": "runs/v21.13/ablation_discount_strong",
}
BASE = {"ow": "runs/v21.11/baseline/openwater_baseline__*/log_details.json",
        "v3": "runs/v21.11/baseline/v3_baseline__*/log_details.json"}


def load(p):
    return json.load(open(glob.glob(p)[0]))


def metrics(p, base):
    d = json.load(open(p))
    tr = np.asarray(d["trajectory"], float)
    bt = np.asarray(base["trajectory"], float)
    goal, rows = cr.score(p)
    near = [r for r in rows if r["role"] != "none" or r["clear"] <= 3.0]
    clear = min(r["clear"] for r in rows)
    t_cpa = min(rows, key=lambda r: r["clear"])["t_cpa"]
    n = min(len(tr), len(bt))
    # deviation from the baseline's PATH (nearest point), not its same-time point
    dev = np.array([np.min(np.hypot(bt[:, 0] - x, bt[:, 1] - y)) for x, y in tr[:n, :2]])
    on = np.argmax(dev > DEV_ON) if np.any(dev > DEV_ON) else None
    first_cpa = min((r["t_cpa"] for r in near), default=t_cpa)
    lead = (first_cpa - on * 0.1) if on is not None else float("nan")
    turn = float(np.sum(np.abs(np.diff(np.unwrap(tr[:, 2]))))) - float(np.sum(np.abs(np.diff(np.unwrap(bt[:, 2])))))
    om = np.array([e["chosen_omega"] for e in d["log_entries"]])
    s = np.sign(om[np.abs(om) > OMEGA_ON])
    rev = int(np.sum(s[1:] != s[:-1]))
    return dict(goal=bool(goal), clear=clear, delay=(len(tr) - len(bt)) * 0.1,
                ok=sum(r["verdict"] == "OK" for r in rows),
                bad=sum(r["verdict"] == "VIOLATION" for r in rows),
                lead=lead, turn=math.degrees(turn), rev=rev, maxdev=float(dev.max()))


def main():
    bases = {k: load(v) for k, v in BASE.items()}
    table = {}
    for arm, root in ARMS.items():
        for p in sorted(glob.glob(root + "/encounters/owenc215_*/log_details.json")
                        + glob.glob(root + "/combined/*/log_details.json")):
            scene = Path(p).parent.name.split("__")[0]
            table.setdefault(scene, {})[arm] = metrics(p, bases["v3" if "v3crowd" in scene else "ow"])

    cols = ["goal", "clear", "delay", "lead", "turn", "rev", "ok", "bad"]
    print(f"{'scene':<28}{'arm':<16}" + "".join(f"{c:>8}" for c in cols))
    for scene in sorted(table):
        for arm in ARMS:
            m = table[scene].get(arm)
            if m is None:
                print(f"{scene:<28}{arm:<16}   (missing)")
                continue
            print(f"{scene:<28}{arm:<16}" + "".join(
                f"{str(m[c]) if c == 'goal' else format(m[c], '8.2f') if isinstance(m[c], float) else m[c]:>8}"
                for c in cols))
        print()
    print("SUMMARY (complete scenes only)")
    full = [s for s in table if len(table[s]) == len(ARMS)]
    for arm in ARMS:
        ms = [table[s][arm] for s in full]
        print(f"  {arm:<16} n={len(ms):>2} goal={sum(m['goal'] for m in ms):>2} "
              f"min clear={min(m['clear'] for m in ms):.3f} mean clear={np.mean([m['clear'] for m in ms]):.3f} "
              f"mean delay={np.mean([m['delay'] for m in ms]):5.1f}s "
              f"median lead={np.nanmedian([m['lead'] for m in ms]):5.1f}s "
              f"mean extra turn={np.mean([m['turn'] for m in ms]):6.1f}deg "
              f"reversals={sum(m['rev'] for m in ms):>4} side OK/VIOL={sum(m['ok'] for m in ms)}/{sum(m['bad'] for m in ms)}")


if __name__ == "__main__" and not {"--turning", "--frontier", "--slalom"} & set(sys.argv):
    main()


ARMS_TURN = {
    "uniform (lT=0)": "runs/v21.13/ablation_uniform",
    "uni x1.58": "runs/v21.13/ablation_uniform_gain1p58",
    "v21.13 (lT=1)": "runs/v21.13",
    "uni x3.15": "runs/v21.13/ablation_uniform_gain3p15",
    "strong (lT=3)": "runs/v21.13/ablation_discount_strong",
}
T_TURN = 70.0


def turning():
    base = load(BASE["ow"])
    bt = np.asarray(base["trajectory"], float)
    print(f"\nTURNING SUITE (turn at t={T_TURN:.0f}s, meeting at 90s)")
    print(f"{'scene':<26}{'arm':<16}{'goal':>6}{'clear':>7}{'dev@turn':>9}{'maxdev':>8}"
          f"{'delay':>7}{'turn':>8}{'react':>7}")
    for scene in [f"owturn215_{b}_{m}" for m in ("away", "in")
                  for b in ("head_on", "cross_stbd_090", "cross_port_090")]:
        for arm, root in ARMS_TURN.items():
            ps = sorted(glob.glob(f"{root}/encounters/{scene}__*/log_details.json"))
            if not ps:
                print(f"{scene:<26}{arm:<16}   (missing)")
                continue
            m = metrics(ps[-1], base)
            tr = np.asarray(json.load(open(ps[-1]))["trajectory"], float)
            dev = np.array([np.min(np.hypot(bt[:, 0] - x, bt[:, 1] - y)) for x, y in tr[:, :2]])
            k = int(T_TURN / 0.1)
            after = np.nonzero(dev[k:] > DEV_ON)[0]
            react = after[0] * 0.1 if (scene.endswith("_in") and len(after) and dev[k] <= DEV_ON) else float("nan")
            print(f"{scene:<26}{arm:<16}{str(m['goal']):>6}{m['clear']:>7.2f}{dev[k]:>9.2f}"
                  f"{m['maxdev']:>8.2f}{m['delay']:>7.1f}{m['turn']:>8.1f}{react:>7.1f}")
        print()


if __name__ == "__main__" and "--turning" in sys.argv:
    turning()


UNIFORM = ["uniform (lT=0)", "uni x1.58", "uni x3.15"]
DISCOUNTED = ["v21.13 (lT=1)", "strong (lT=3)"]


def frontier():
    """Discounted arm vs the uniform curve, at the discounted arm's own cost.

    Per scene, the three uniform gains trace clearance against cost (turning,
    and separately delay); the uniform clearance at the discounted arm's cost
    is linearly interpolated on that curve. Scenes where the ego never
    manoeuvres (null_safe_pass, cross_port_090_away) carry no information
    and are left out, as is any point outside the uniform curve's cost range.
    """
    bases = {k: load(v) for k, v in BASE.items()}
    rows = []
    for suite, arms in (("straight", ARMS), ("turning", ARMS_TURN)):
        scenes = {}
        for arm, root in arms.items():
            if suite == "turning":
                ps = glob.glob(root + "/encounters/owturn215_*/log_details.json")
            else:
                ps = (glob.glob(root + "/encounters/owenc215_*/log_details.json")
                      + glob.glob(root + "/combined/*/log_details.json"))
            for p in ps:
                sc = Path(p).parent.name.split("__")[0]
                scenes.setdefault(sc, {})[arm] = metrics(p, bases["v3" if "v3crowd" in sc else "ow"])
        for sc, m in sorted(scenes.items()):
            if len(m) < len(arms) or "null" in sc or "port_090_away" in sc:
                continue
            for d in DISCOUNTED:
                gaps = {}
                for cost in ("turn", "delay"):
                    xs = np.array([m[u][cost] for u in UNIFORM])
                    ys = np.array([m[u]["clear"] for u in UNIFORM])
                    x = m[d][cost]
                    if xs.min() <= x <= xs.max():
                        o = np.argsort(xs)
                        gaps[cost] = m[d]["clear"] - float(np.interp(x, xs[o], ys[o]))
                rows.append((sc, d, gaps))
    print("discounted minus uniform clearance, at the discounted arm's own cost")
    for cost, name in (("turn", "turning"), ("delay", "delay")):
        for d in DISCOUNTED:
            v = [g[cost] for _, a, g in rows if a == d and cost in g]
            out = sum(1 for _, a, g in rows if a == d and cost not in g)
            print(f"  equal {name:<8} {d:<14} mean {np.mean(v):+.3f}  "
                  f"better {sum(x > 0.01 for x in v)}/{len(v)}  worse {sum(x < -0.01 for x in v)}/{len(v)}"
                  f"  (outside uniform range: {out})")


if __name__ == "__main__" and "--frontier" in sys.argv:
    frontier()


ARMS_SLALOM = {
    "uniform (lT=0)": "runs/v21.13/ablation_uniform",
    "uni x1.58": "runs/v21.13/ablation_uniform_gain1p58",
    "v21.13 (lT=1)": "runs/v21.13",
    "uni x3.15": "runs/v21.13/ablation_uniform_gain3p15",
    "strong (lT=3)": "runs/v21.13/ablation_discount_strong",
}


def slalom():
    """ow_slalom_*: clearance to A (met first) and to B (delta s later).

    The test of ranking between threats. Per delta, both mirrors pooled:
    the worse of the two clearances, its cost, and the discounted arms
    against the uniform curve at equal cost.
    """
    base = load(BASE["ow"])
    tab = {}
    for arm, root in ARMS_SLALOM.items():
        for p in glob.glob(f"{root}/combined/ow_slalom_*/log_details.json"):
            sc = Path(p).parent.name.split("__")[0]
            goal, rows = cr.score(p)
            m = metrics(p, base)
            ca, cb = rows[0]["clear"], rows[1]["clear"]
            tab.setdefault(sc, {})[arm] = dict(goal=goal, a=ca, b=cb, worst=min(ca, cb),
                                              delay=m["delay"], turn=m["turn"])
    print(f"{'scene':<22}{'arm':<16}{'goal':>6}{'clear A':>9}{'clear B':>9}{'worse':>7}{'delay':>7}{'turn':>8}")
    for sc in sorted(tab):
        for arm in ARMS_SLALOM:
            m = tab[sc].get(arm)
            if m is None:
                print(f"{sc:<22}{arm:<16}   (missing)")
                continue
            print(f"{sc:<22}{arm:<16}{str(m['goal']):>6}{m['a']:>9.2f}{m['b']:>9.2f}{m['worst']:>7.2f}"
                  f"{m['delay']:>7.1f}{m['turn']:>8.1f}")
        print()
    print("discounted minus uniform 'worse' clearance at the discounted arm's own cost, by delta")
    for d in sorted({sc[-3:] for sc in tab}):
        for cost in ("turn", "delay"):
            for darm in DISCOUNTED:
                v = []
                for sc in (s for s in tab if s.endswith(d)):
                    m = tab[sc]
                    if len(m) < len(ARMS_SLALOM):
                        continue
                    xs = np.array([m[u][cost] for u in UNIFORM])
                    ys = np.array([m[u]["worst"] for u in UNIFORM])
                    x = m[darm][cost]
                    if xs.min() <= x <= xs.max():
                        o = np.argsort(xs)
                        v.append(m[darm]["worst"] - float(np.interp(x, xs[o], ys[o])))
                txt = f"{np.mean(v):+.3f} (n={len(v)})" if v else "outside uniform range"
                print(f"  {d}  equal {cost:<6} {darm:<14} {txt}")


if __name__ == "__main__" and "--slalom" in sys.argv:
    slalom()

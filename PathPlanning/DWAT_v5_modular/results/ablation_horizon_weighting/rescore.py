"""Open-loop re-scoring: does the horizon weighting change the planner's choice?

At logged states of a run, rebuild the planner's exact inputs (state, local
goal, map, target positions and velocities), score every candidate under each
weighting arm, and report:

  reproduce   re-scoring under the run's own config picks the control it
              logged -- the check that the inputs are rebuilt faithfully
  choice      which candidate each arm picks; whether a discounted arm ever
              picks differently from its gain-matched uniform arm
  influence   for each time window of the horizon, how much that window's
              share of the dynamic cost varies ACROSS candidates (standard
              deviation over the finite-cost candidates). A window whose
              contribution is the same for every candidate cannot move the
              argmin, however it is weighted.
  veto        candidates scored inf because the rollout predicts contact
              somewhere in the horizon, and how far ahead that contact is.

    python -m PathPlanning.DWAT_v5_modular.results.ablation_horizon_weighting.rescore \\
        runs/v21.13/combined/ow_slalom_stbd_d20__<ts> [more run dirs] [--every 10]
"""
import argparse
import copy
import importlib
import json
from pathlib import Path

import numpy as np

from PathPlanning.DWAT_v4_st import dwa_ramped as dr
from PathPlanning.DWAT_v5_modular import main as runner
from PathPlanning.DWAT_v5_modular import map_creation

ARMS = {  # name: (time_decay_lambda, side_cost_gain multiplier)
    "uniform": (0.0, 1.0),
    "uni x1.58": (0.0, 1.580),
    "v21.13": (0.029958, 1.0),
    "uni x3.15": (0.0, 3.145),
    "strong": (0.089874, 1.0),
}
PAIRS = (("v21.13", "uni x1.58"), ("strong", "uni x3.15"))
WINDOWS = ((0.0, 3.0), (3.0, 8.0), (8.0, 15.0), (15.0, 1e9))
MIN_SPREAD = 1e-6


def load(run_dir: Path):
    d = json.load(open(run_dir / "log_details.json"))
    inputs = json.load(open(run_dir / "run.json"))["inputs"]
    # The algorithm config file, not the log's config dump: the log omits
    # dwa_module and carries alias keys the loader does not take.
    cfgd = json.load(open(inputs["algorithm_config"]))
    mod = importlib.import_module(cfgd.pop("dwa_module"))
    assert mod is dr, "rescore.py drives dwa_ramped only"
    config = runner.apply_algorithm_overrides(mod.Config(), cfgd)
    scenario = inputs["scenario"]
    ctx = map_creation.create_map_context(config, scenario)
    sc = json.load(open(scenario))
    radii = [float(o.get("radius", config.obstacle_radius)) for o in sc["dynamic_obstacles"]]
    return d, config, ctx, radii


def state(d, k, dt):
    """Inputs of decision k: entry k's targets are where the planner saw them."""
    e = d["log_entries"]
    pos = np.asarray(e[k]["dynamic_obstacles_pos"], float)
    j = k + 1 if k + 1 < len(e) else k - 1
    vel = (np.asarray(e[j]["dynamic_obstacles_pos"], float) - pos) / ((j - k) * dt)
    x = np.asarray(d["trajectory"][k], float).copy()
    goal = np.array([e[k]["local_goal_x"], e[k]["local_goal_y"]])
    return x, goal, pos, vel, (e[k]["chosen_v"], e[k]["chosen_omega"])


def arm_config(base_cfg, lam, gain_mult):
    c = copy.copy(base_cfg)
    c.time_decay_lambda = lam
    c.side_cost_gain = base_cfg.side_cost_gain * gain_mult
    # Fresh COLREGs state per call: roles are re-derived from this one state.
    # Exact for overtaking-only scenes (Rule 13 has no cost); see reproduce.
    return c


def severity_profiles(trajs, pos, vel, radii, x, cfg):
    """(n_cand, T) severity S(d(t)) against the targets the planner keeps."""
    p, r, v = dr.filter_obstacles_by_direction(x, pos, radii, vel, max_angle=cfg.obstacle_max_angle)
    if len(p) == 0:
        return None
    m = max(cfg.max_side_weight_dist, 1e-6)
    out = []
    for tr in trajs:
        d_arr, _ = dr.calc_trajectory_clearance_and_collision(tr, p, np.array(r), v, cfg)
        out.append(m * np.exp(-(cfg.side_clearance_gaussian_decay / m ** 2) * d_arr ** 2))
    return np.array(out), (p, r, v)


def first_contact_times(trajs, kept, cfg):
    p, r, v = kept
    ts = []
    for tr in trajs:
        d_arr, hit = dr.calc_trajectory_clearance_and_collision(tr, p, np.array(r), v, cfg)
        if hit:
            ts.append(float(np.argmax(d_arr <= 0)) * cfg.dt)
    return ts


def analyse(run_dir: Path, every: int):
    d, base_cfg, ctx, radii = load(run_dir)
    dt = base_cfg.dt
    static_ob = ctx.map_manager.get_static_obstacles()
    static_r = ctx.map_manager.get_static_obstacle_radii()
    rows = []
    for k in range(0, len(d["log_entries"]) - 1, every):
        if d["log_entries"][k]["dynamic_ob_cost_before"] <= 0.0:
            continue                       # no target in play at this step
        x, goal, pos, vel, logged = state(d, k, dt)
        rec = {"k": k, "t": k * dt}
        for name, (lam, g) in ARMS.items():
            cfg = arm_config(base_cfg, lam, g)
            cap = {}
            dr.CAPTURE = cap
            try:
                u = dr.dwa_control_merged(x.copy(), cfg, goal, static_ob, static_r, pos, radii, vel)[0]
            finally:
                dr.CAPTURE = None
            rec[name] = (float(u[0]), float(u[1]))
            if name == "uniform":
                cand = cap
        own = (base_cfg.time_decay_lambda, 1.0)
        own_arm = next(n for n, a in ARMS.items() if np.isclose(a[0], own[0]) and a[1] == 1.0)
        rec["reproduce"] = np.allclose(rec[own_arm], logged, atol=1e-9)

        # Which part of the horizon decides: per-window contribution of each
        # candidate under each weighting, and its spread across candidates.
        finite = np.isfinite(cand["dynamic"])
        prof = severity_profiles([cand["trajectories"][i] for i in np.nonzero(finite)[0]],
                                 pos, vel, radii, x, base_cfg)
        if prof is not None and finite.sum() > 1:
            S, kept = prof
            t = np.arange(S.shape[1]) * dt
            for name, (lam, g) in ARMS.items():
                w = np.exp(-lam * t)
                w = w / w.sum()
                gain = base_cfg.side_cost_gain * g
                rec[f"infl:{name}"] = [float(np.std(gain * (S[:, (t >= a) & (t < b)] * w[(t >= a) & (t < b)]).sum(axis=1)))
                                       for a, b in WINDOWS]
            vetoed = [cand["trajectories"][i] for i in np.nonzero(~finite)[0]]
            rec["veto_frac"] = float((~finite).mean())
            rec["veto_t"] = first_contact_times(vetoed, kept, base_cfg) if vetoed else []
        rows.append(rec)
    return rows


def report(run_dir, rows):
    n = len(rows)
    print(f"\n{run_dir.name}: {n} states with a target in play")
    if not n:
        return
    print(f"  reproduce logged control: {sum(r['reproduce'] for r in rows)}/{n}")
    names = list(ARMS)
    same_as_uniform = {a: sum(r[a] == r["uniform"] for r in rows) for a in names[1:]}
    print("  same choice as uniform:   " + "  ".join(f"{a} {v}/{n}" for a, v in same_as_uniform.items()))
    for disc, uni in PAIRS:
        diff = sum(r[disc] != r[uni] for r in rows)
        print(f"  {disc:<7} vs {uni:<9}: different choice at {diff}/{n} states")
    infl = [r for r in rows if "infl:uniform" in r]
    if infl:
        # States where the dynamic cost barely differs between candidates (a
        # target at the edge of range: totals down to 1e-54) carry no
        # information about which window decides; shares of ~0/~0 are noise.
        print(f"  influence by window (std across candidates, share of total), mean over "
              f"states with spread > {MIN_SPREAD:g}:")
        hdr = "".join(f"{f'{a:g}-{b:g}s' if b < 1e8 else f'{a:g}s+':>11}" for a, b in WINDOWS)
        print(f"    {'':<10}{hdr}{'states':>9}")
        for name in names:
            m = np.array([r[f"infl:{name}"] for r in infl])
            tot = m.sum(axis=1)
            share = m[tot > MIN_SPREAD] / tot[tot > MIN_SPREAD, None]
            print(f"    {name:<10}" + "".join(f"{v:>10.0%} " for v in share.mean(axis=0))
                  + f"{len(share):>8}")
        vf = np.array([r["veto_frac"] for r in infl])
        vt = np.concatenate([r["veto_t"] for r in infl if r["veto_t"]] or [np.array([])])
        print(f"  vetoed (inf) candidates: mean {vf.mean():.0%} of the window; "
              + (f"predicted contact {np.percentile(vt, 10):.1f}-{np.percentile(vt, 90):.1f} s ahead (10-90%)"
                 if len(vt) else "none"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--json-out")
    args = ap.parse_args()
    allrows = {}
    for r in args.runs:
        rd = Path(r)
        rows = analyse(rd, args.every)
        report(rd, rows)
        allrows[rd.name] = rows
    if args.json_out:
        json.dump(allrows, open(args.json_out, "w"), default=lambda o: list(o) if isinstance(o, tuple) else str(o))


if __name__ == "__main__":
    main()

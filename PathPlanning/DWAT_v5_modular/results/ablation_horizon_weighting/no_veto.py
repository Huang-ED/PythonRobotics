"""Open-loop: what would the planner choose if the inf collision veto were removed?

At logged states where the veto removes some (not all) candidates, give each
vetoed candidate the finite cost the formula would assign -- "naive" (the
symmetric barrier on negative clearance, so deeper overlap scores LOWER) or
"clamp" (overlap held at maximum severity) -- and report how often the
cheapest candidate becomes one whose rollout collides. README section 11.

    python PathPlanning/DWAT_v5_modular/results/ablation_horizon_weighting/no_veto.py \
        'runs/v21.13/port_turn_release6/combined/v3crowd6_v2111__*'
"""
import glob, sys, numpy as np
from pathlib import Path
from PathPlanning.DWAT_v4_st import dwa_ramped as dr
from PathPlanning.DWAT_v5_modular.results.ablation_horizon_weighting import rescore as R

def finite_dyn(tr, kept, cfg, clamp):
    p, r, v = kept
    d, hit = dr.calc_trajectory_clearance_and_collision(tr, p, np.array(r), v, cfg)
    m = cfg.max_side_weight_dist
    dd = np.maximum(d, 0.0) if clamp else d
    S = m * np.exp(-(cfg.side_clearance_gaussian_decay / m ** 2) * dd ** 2)
    t = np.arange(len(S)) * cfg.dt
    w = np.exp(-max(cfg.time_decay_lambda, 0.0) * t)
    c, _, _ = dr.aggregate_compound_costs(S, w, "mean")
    first = float(np.argmax(d <= 0)) * cfg.dt if hit else None
    return cfg.side_cost_gain * c, first

for run in sys.argv[1:]:
    rd = Path(glob.glob(run)[0]); d, cfg, ctx, radii = R.load(rd)
    so = ctx.map_manager.get_static_obstacles(); sr = ctx.map_manager.get_static_obstacle_radii()
    stats = {"naive": [], "clamp": []}; n = 0; anyveto = 0
    for k in range(0, len(d["log_entries"]) - 1, 10):
        if d["log_entries"][k]["dynamic_ob_cost_before"] <= 0: continue
        x, goal, pos, vel, _ = R.state(d, k, cfg.dt)
        c = R.arm_config(cfg, cfg.time_decay_lambda, 1.0)
        cap = {}; dr.CAPTURE = cap
        try: dr.dwa_control_merged(x.copy(), c, goal, so, sr, pos, radii, vel)
        finally: dr.CAPTURE = None
        dyn = cap["dynamic"]; veto = ~np.isfinite(dyn)
        n += 1
        if not veto.any() or veto.all(): continue
        anyveto += 1
        kept = dr.filter_obstacles_by_direction(x, pos, radii, vel, max_angle=c.obstacle_max_angle)
        base = cap["to_goal"] + cap["speed"] + cap["static"] + cap["colregs_side"] + cap["colregs_port"]
        for mode in stats:
            dd = dyn.copy(); first = {}
            for i in np.nonzero(veto)[0]:
                dd[i], first[i] = finite_dyn(cap["trajectories"][i], kept, c, mode == "clamp")
            j = int(np.argmin(base + dd))
            stats[mode].append(first.get(j))
    print(f"{rd.name}: {n} states, {anyveto} with some (not all) candidates vetoed")
    for mode, ch in stats.items():
        coll = [t for t in ch if t is not None]
        print(f"  {mode:<5}: cheapest candidate is a COLLIDING one at {len(coll)}/{anyveto} of those states"
              + (f"; its predicted contact {min(coll):.1f}-{max(coll):.1f} s ahead (median {np.median(coll):.1f})" if coll else ""))

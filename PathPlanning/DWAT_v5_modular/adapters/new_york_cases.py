"""Generate reasonable New York AIS run configurations, and read the results back.

This writes cases; it deliberately does not run them. Scheduling work is
something the shell (or an agent driving it) already does better than a
bespoke worker pool, and the runner has no iteration cap, so supervision
belongs with whatever is watching the jobs. What actually needs domain
knowledge -- and is therefore what lives here -- is deciding which cases are
worth running at all.

Two kinds of case, both against the same real traffic:

historical
    One recorded vessel is promoted to ego and removed from the obstacle set.
    Start, goal, heading and hull size all come from its own AIS record, so its
    recorded transit is a human baseline for exactly the problem the planner is
    given. Candidates are filtered to continuous, genuinely-under-way transits,
    because a baseline stitched across a ten-minute hole is not a transit
    anyone sailed.

synthetic
    An invented mission -- start and goal sampled from open, connected water --
    run through the same recorded traffic, with nothing removed. Hull and
    service speed are still drawn from a real vessel in the dataset, so the ego
    is plausible even though its voyage never happened. These test geometry the
    harbour did not happen to supply, and by construction have no human
    baseline.

Each case gets its own scenario JSON and its own algorithm config nested under
the base config, because hull and speed are properties of the vessel, not of
the tuning:

    configs/algorithm/ais_v1.1/<case>.json   -> runs/ais_v1.1/<case>/

Only the keys `ego_kinematics` derives differ between cases; every gain and
horizon is inherited unchanged, so a batch is a sweep over the ego and nothing
else.

    # write the cases, and print one run command per case
    python -m PathPlanning.DWAT_v5_modular.adapters.new_york_cases \
        --historical 4 --synthetic 4

    # afterwards, collect whatever has finished into one table
    python -m PathPlanning.DWAT_v5_modular.adapters.new_york_cases --summarise
"""

import argparse
import json
import math
import os
import sys

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from PathPlanning.DWAT_v5_modular.adapters.new_york_ais_adapter import (  # noqa: E402
    DEFAULT_ROI, DEFAULT_WINDOW, DERIVED, beam_for, convert, ego_kinematics,
    load_window, navigable_mask, rasterise_land, split_runs, trim_ego,
)

BASE_ALGORITHM = os.path.join('PathPlanning', 'DWAT_v5_modular', 'configs',
                              'algorithm', 'ais_v1.1.json')
ALGORITHM_DIR = os.path.join('PathPlanning', 'DWAT_v5_modular', 'configs',
                             'algorithm', 'ais_v1.1')
SCENARIO_DIR = os.path.join('PathPlanning', 'DWAT_v5_modular', 'configs',
                            'scenarios', 'imported')


# --------------------------------------------------------------- cases -----

def historical_candidates(pts, navigable, occupancy, resolution, ego_min_sog,
                          min_transit_m, min_samples):
    """Recorded vessels whose track is a usable transit, best first."""
    rows = []
    for mmsi, g in pts.groupby('mmsi'):
        g = g.sort_values('t')
        try:
            trimmed, _, _ = trim_ego(g, navigable, resolution, ego_min_sog)
        except ValueError:
            continue
        if len(trimmed) < min_samples:
            continue
        net = float(np.hypot(trimmed.x.iloc[-1] - trimmed.x.iloc[0],
                             trimmed.y.iloc[-1] - trimmed.y.iloc[0]))
        if net < min_transit_m:
            continue
        gaps = trimmed.t.diff().dt.total_seconds().iloc[1:]
        rows.append(dict(
            mmsi=int(mmsi), n=len(trimmed), net_m=net,
            path_m=float(np.hypot(np.diff(trimmed.x.values),
                                  np.diff(trimmed.y.values)).sum()),
            max_gap_s=float(gaps.max()) if len(gaps) else 0.0,
            length=float(trimmed.Length.median()) if trimmed.Length.notna().any() else np.nan,
            vtype=float(trimmed.VesselType.median()) if trimmed.VesselType.notna().any() else np.nan,
        ))
    if not rows:
        return pd.DataFrame(columns=['mmsi', 'n', 'net_m'])
    out = pd.DataFrame(rows)
    # Prefer unbroken, long transits: a baseline stitched across a 10-minute
    # hole is not a transit anyone actually sailed.
    out = out[out.max_gap_s <= 120.0]
    return out.sort_values(['net_m', 'n'], ascending=False).reset_index(drop=True)


def sample_missions(navigable, resolution, count, rng, min_m, max_m,
                    clearance_cells=3):
    """Start/goal pairs in open, connected water, far enough apart to be a run.

    Points are required to sit well clear of the shoreline, not merely in a
    navigable cell: a start wedged one cell from a pier is navigable on paper
    and unplannable in practice.
    """
    import cv2
    room = cv2.erode(navigable.astype(np.uint8),
                     np.ones((2 * clearance_cells + 1,) * 2, np.uint8))
    cells = np.argwhere(room == 1)
    if len(cells) < 2:
        raise ValueError('no water with enough room to start a vessel in')
    world = (cells + 0.5) * resolution

    out = []
    for _ in range(count * 400):
        if len(out) == count:
            break
        a, b = rng.integers(0, len(world), size=2)
        if a == b:
            continue
        p, q = world[a], world[b]
        d = float(np.hypot(*(q - p)))
        if min_m <= d <= max_m:
            out.append((p, q, d))
    if len(out) < count:
        raise ValueError(f'only found {len(out)} of {count} missions in range')
    return out


def sample_vessels(pts, count, rng):
    """Real (length, type, speed) triples, so an invented ego is still plausible."""
    per = pts.groupby('mmsi').agg(length=('Length', 'median'),
                                  vtype=('VesselType', 'median'),
                                  sog=('SOG', lambda s: np.nanpercentile(s, 90)))
    per = per[per.length.notna() & (per.length > 5) & per.sog.notna() & (per.sog > 1)]
    if per.empty:
        raise ValueError('no vessel in this window has both a length and a speed')
    picks = per.iloc[rng.integers(0, len(per), size=count)]
    return [dict(length=float(r.length), vtype=float(r.vtype) if np.isfinite(r.vtype) else None,
                 speed=float(r.sog) * 0.514444) for _, r in picks.iterrows()]


# ------------------------------------------------------------- plumbing ----

def derive_ais_v12(cfg, resolution):
    """The per-ego values ais_v1.2 derives from the map cell size.

    Land is a set of cell-centre points, each a circle standing for a square
    cell, so its radius is the cell's circumscribed circle, res/sqrt(2) --
    with res/2 (ais_v1.1) the circles leave gaps between diagonal neighbours and
    the ego sailed into the land through them (OPEN_PROBLEMS item 21).

    dt: at most half a cell of travel per step, capped at 10 s (item 3).
    global_planner_rr: Theta*'s buffer, so the whole global path -- segments,
    not just nodes -- stays outside the local contact distance: obstacle radius
    + half-beam + half a cell's diagonal (item 21).
    """
    cfg['obstacle_radius'] = round(resolution / math.sqrt(2), 2)
    cfg['dt'] = min(10.0, math.floor(0.5 * resolution / cfg['max_speed'] * 10) / 10)
    cfg['global_planner_rr'] = round(cfg['obstacle_radius'] + 0.5 * cfg['robot_width']
                                     + resolution / math.sqrt(2), 2)
    return cfg


def write_algorithm_config(case, kinematics, base_path=BASE_ALGORITHM,
                           out_dir=ALGORITHM_DIR, resolution=None, derive=None):
    with open(base_path) as f:
        cfg = json.load(f)
    cfg.update(kinematics)
    if derive is not None:
        cfg = derive(cfg, resolution)
    # obstacle_radius is tied to the map, not the vessel, so it is inherited.
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{case}.json')
    with open(path, 'w') as f:
        json.dump(cfg, f, indent=4)
    return path


def _hull_clearance(ego_xy, yaw, points, radii, config):
    """Distance from the ego hull to each point, using the planner's own geometry.

    Mirrors calc_trajectory_clearance_and_collision: a capsule when the config
    asks for one, otherwise the circumscribed disc. Measuring with a different
    hull than the planner used is how a run gets reported as "collision at 359 m
    clear".
    """
    L, W = float(config['robot_length']), float(config['robot_width'])
    if (config.get('robot_collision_shape', 'disc') == 'capsule'
            and config.get('robot_type', 'rectangle') == 'rectangle'):
        half = max(0.5 * (L - W), 0.0)
        axis = np.array([np.cos(yaw), np.sin(yaw)])
        to_pt = points - ego_xy
        proj = np.clip(to_pt @ axis, -half, half)
        perp = to_pt - proj[:, None] * axis
        return np.hypot(perp[:, 0], perp[:, 1]) - (radii + 0.5 * W)
    return np.hypot(*(points - ego_xy).T) - (radii + float(config['robot_radius']))


def summarise(run_dir, scenario_path):
    """Read one finished run back into a row of the results table.

    Clearance is re-measured here at 1 s rather than read from the log, because
    the planner only samples its own trajectory every `dt` and can step over a
    contact between samples -- which is how a run reaches its goal having
    passed through another vessel. Shore and ship are reported separately: a
    grounding and a collision are different failures with different fixes.
    """
    log_path = os.path.join(run_dir, 'log_details.json')
    if not os.path.exists(log_path):
        return dict(outcome='no log')
    with open(log_path) as f:
        log = json.load(f)
    with open(scenario_path) as f:
        scene = json.load(f)

    cfg = log['config']
    tr = np.array(log['trajectory'], dtype=float)
    dt = float(cfg['dt'])
    path = float(np.hypot(*np.diff(tr[:, :2], axis=0).T).sum()) if len(tr) > 1 else 0.0
    duration = (len(tr) - 1) * dt

    bundle = np.load(scene['dynamic_obstacles'][0]['track_file'])
    pos, lens, times, radii = (bundle['positions'], bundle['lengths'],
                               bundle['times'], bundle['radii'])
    # Before its first fix a vessel is wherever the run put it: parked at p[0]
    # ("hold", "hold_wrong_speed" -- the only behaviour before 2026-10-01, so
    # logs without the key are parked), or back along its first-leg velocity
    # ("extrapolate"). Score against the world the planner was in.
    extrapolate = cfg.get('track_on_start', 'hold_wrong_speed') == 'extrapolate'
    vel0 = (bundle['velocities'][:, 0] if 'velocities' in bundle.files
            else (pos[:, 1] - pos[:, 0]) / np.maximum(
                (times[:, 1] - times[:, 0])[:, None], 1e-12))

    # Shoreline: the rim cells the planner actually costs against.
    occ = np.load(scene['occupancy_path']).astype(np.uint8)
    res = float(scene['occupancy_resolution'])
    rim = cv2.subtract(occ, cv2.erode(occ, np.ones((3, 3), np.uint8)))
    idx = np.argwhere(rim == 1).astype(float)
    shore = (idx + 0.5) * res + np.asarray(scene['occupancy_origin'], dtype=float)
    shore_r = np.full(len(shore), float(cfg['obstacle_radius']))
    shore_tree = cKDTree(shore)

    worst_ship, worst_shore = float('inf'), float('inf')
    for t in np.arange(0.0, duration + 1e-9, 1.0):
        k = t / dt
        i = int(np.clip(np.floor(k), 0, max(len(tr) - 2, 0)))
        f = k - i
        j = min(i + 1, len(tr) - 1)
        ego = tr[i, :2] * (1 - f) + tr[j, :2] * f
        yaw = tr[i, 2]

        here = np.empty((pos.shape[0], 2))
        for o in range(pos.shape[0]):
            L = int(lens[o])
            tt = times[o, :L]
            if extrapolate and t < tt[0]:
                here[o] = pos[o, 0] + vel0[o] * (t - tt[0])
                continue
            tc = min(max(t, tt[0]), tt[-1])
            here[o] = (np.interp(tc, tt, pos[o, :L, 0]),
                       np.interp(tc, tt, pos[o, :L, 1]))
        worst_ship = min(worst_ship, float(_hull_clearance(ego, yaw, here, radii, cfg).min()))

        near = shore_tree.query_ball_point(ego, r=float(cfg['robot_radius']) + res * 4)
        if near:
            worst_shore = min(worst_shore, float(
                _hull_clearance(ego, yaw, shore[near], shore_r[near], cfg).min()))

    ce = log.get('collision_event') or {}
    if log.get('reached_goal'):
        outcome = 'goal'
    elif ce:
        # A static cell carries exactly obstacle_radius; a vessel carries its own.
        hit_shore = abs(float(ce.get('obstacle_radius', -1))
                        - float(cfg['obstacle_radius'])) < 1e-9
        outcome = 'grounding' if hit_shore else 'collision'
    elif log.get('error'):
        outcome = 'no-admissible' if 'No best trajectory' in str(log['error']) else 'error'
    else:
        outcome = 'stalled'

    src = scene['source']
    human = src.get('ego_recorded_duration_s')
    return dict(
        outcome=outcome, steps=len(tr), duration_s=duration, path_m=path,
        mean_speed=path / duration if duration else 0.0,
        ship_clear_m=worst_ship,
        shore_clear_m=None if not np.isfinite(worst_shore) else worst_shore,
        undetected=bool(outcome == 'goal' and min(worst_ship, worst_shore) < 0),
        human_duration_s=human if human and np.isfinite(human) else None,
    )


def collect(manifest_path, out_path):
    """Read every finished run named in a manifest into one table."""
    with open(manifest_path) as f:
        built = json.load(f)

    rows = []
    for b in built:
        version = os.path.splitext(os.path.relpath(
            b['algorithm'], os.path.join('PathPlanning', 'DWAT_v5_modular',
                                         'configs', 'algorithm')))[0]
        run_root = os.path.join('runs', version, 'imported')
        row = dict(b)
        if not os.path.isdir(run_root):
            row['outcome'] = 'not run'
            rows.append(row)
            continue
        # Newest run of this case wins; re-running a case is normal.
        runs = sorted((os.path.join(run_root, d) for d in os.listdir(run_root)),
                      key=os.path.getmtime)
        if not runs:
            row['outcome'] = 'not run'
        else:
            row['run_dir'] = runs[-1]
            row.update(summarise(runs[-1], b['scenario']))
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(['kind', 'case'])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    df.to_csv(out_path, index=False)
    cols = ['case', 'kind', 'outcome', 'steps', 'duration_s', 'path_m',
            'mean_speed', 'ship_clear_m', 'shore_clear_m', 'undetected',
            'human_duration_s']
    print(df[[c for c in cols if c in df]].to_string(index=False))
    print(f'\nwrote {out_path}')
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--roi', nargs=4, type=float, default=list(DEFAULT_ROI))
    p.add_argument('--window', nargs=2, default=list(DEFAULT_WINDOW))
    p.add_argument('--historical', type=int, default=4)
    p.add_argument('--synthetic', type=int, default=4)
    p.add_argument('--resolution', type=float, default=50.0)
    p.add_argument('--version', choices=('ais_v1.1', 'ais_v1.2'), default='ais_v1.1',
                   help="algorithm config version to write per-ego configs for; "
                        "ais_v1.2 derives obstacle_radius, dt and global_planner_rr "
                        "from --resolution and each ego (derive_ais_v12)")
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--min-transit', type=float, default=2000.0)
    p.add_argument('--mission-min', type=float, default=3000.0)
    p.add_argument('--mission-max', type=float, default=9000.0)
    p.add_argument('--ego-min-sog', type=float, default=0.5)
    p.add_argument('--manifest', default=os.path.join(DERIVED, 'cases.json'))
    p.add_argument('--out', default=os.path.join(DERIVED, 'batch_results.csv'))
    p.add_argument('--summarise', action='store_true',
                   help='skip generation; collect the runs named in --manifest')
    a = p.parse_args()

    if a.summarise:
        collect(a.manifest, a.out)
        return 0

    roi, window = tuple(a.roi), tuple(a.window)
    rng = np.random.default_rng(a.seed)

    print('preparing shared map and traffic ...')
    _, occupancy, _ = rasterise_land(roi, a.resolution)
    navigable = navigable_mask(occupancy)
    pts = load_window(roi, window, verbose=False)

    from pyproj import Transformer
    from PathPlanning.DWAT_v5_modular.adapters.new_york_ais_adapter import UTM18N
    tf = Transformer.from_crs('EPSG:4326', UTM18N, always_xy=True)
    xs, ys = tf.transform([roi[2], roi[3], roi[2], roi[3]],
                          [roi[0], roi[0], roi[1], roi[1]])
    px, py = tf.transform(pts.LON.values, pts.LAT.values)
    pts = pts.assign(x=px - min(xs), y=py - min(ys))

    cases = []
    if a.historical:
        cand = historical_candidates(pts, navigable, occupancy, a.resolution,
                                     a.ego_min_sog, a.min_transit, min_samples=10)
        print(f'  {len(cand)} usable historical egos; taking the {a.historical} longest')
        for _, r in cand.head(a.historical).iterrows():
            cases.append(dict(kind='historical', name=f'ny_hist_{int(r.mmsi)}',
                              kwargs=dict(ego_mmsi=int(r.mmsi))))
    if a.synthetic:
        missions = sample_missions(navigable, a.resolution, a.synthetic, rng,
                                   a.mission_min, a.mission_max)
        vessels = sample_vessels(pts, a.synthetic, rng)
        for i, ((start, goal, _), v) in enumerate(zip(missions, vessels)):
            cases.append(dict(kind='synthetic', name=f'ny_synth_{i:02d}',
                              kwargs=dict(ego_mmsi=None, ego_start=start,
                                          ego_goal=goal, ego_length=v['length'],
                                          ego_vessel_type=v['vtype'],
                                          ego_speed=v['speed'])))

    print(f'\nbuilding {len(cases)} cases ...')
    built = []
    for c in cases:
        print(f'--- {c["name"]} ({c["kind"]})')
        try:
            res = convert(roi=roi, window=window, resolution=a.resolution,
                          name=c['name'], ego_min_sog=a.ego_min_sog,
                          verbose=False, **c['kwargs'])
        except Exception as exc:
            print(f'    SKIPPED: {exc}')
            continue
        built.append(dict(case=c['name'], kind=c['kind'],
                          scenario=res['config_path'],
                          algorithm=write_algorithm_config(
                              c['name'], res['kinematics'],
                              base_path=os.path.join(os.path.dirname(BASE_ALGORITHM), f'{a.version}.json'),
                              out_dir=os.path.join(os.path.dirname(ALGORITHM_DIR), a.version),
                              resolution=a.resolution,
                              derive=derive_ais_v12 if a.version == 'ais_v1.2' else None),
                          **{f'ego_{k}': v for k, v in res['ego'].items()}))

    os.makedirs(os.path.dirname(os.path.abspath(a.manifest)), exist_ok=True)
    with open(a.manifest, 'w') as f:
        json.dump(built, f, indent=2, default=str)

    print(f'\nbuilt {len(built)} cases; manifest: {a.manifest}')
    print('\nRun them however you like -- there is no iteration cap in the runner,')
    print('so give each one a timeout and watch for the "stalled" case:\n')
    for b in built:
        print(f'  python -m PathPlanning.DWAT_v5_modular.main '
              f'--scenario {os.path.splitext(os.path.basename(b["scenario"]))[0]} '
              f'--algorithm-config ais_v1.1/{b["case"]} '
              f'--no-animation --no-save-frames')
    print(f'\nThen:  python -m PathPlanning.DWAT_v5_modular.adapters.new_york_cases '
          f'--summarise')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

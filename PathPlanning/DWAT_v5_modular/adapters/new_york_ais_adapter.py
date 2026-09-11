"""Convert EnvData/datasets/new_york (NOAA AIS, January 2021) into a scenario.

Unlike the scene_export adapter, nothing here is a repackaging step: the source
is a WGS84 land shapefile plus 10,162 per-vessel AIS track segments, and the
scenario is a real place at real scale. Four things therefore have to be
decided rather than copied, and each is done from the data:

Projection and scale
    Land and tracks are projected to UTM 18N (EPSG:32618) and shifted so the
    region of interest starts at (0, 0). The scenario is then in **metres and
    seconds** -- the planner tuning has to be in SI too, which is what
    configs/algorithm/ais_v1.json is for. Do not run this scenario
    against v20.7.8-2: that config is in map units on a 50-unit world and its
    obstacle/side gains multiply distances, so at kilometre scale the goal term
    is swamped and the ego stops steering.

Segment re-splitting
    Despite the dataset readme, track_segment_N.csv files are not contiguous:
    92.8% contain an internal time gap over 90 s, the median worst gap is 79
    minutes and the largest is 30 days. Most are benign -- 77.8% of gaps are
    under 5 minutes, and a further 18.6% are long but the vessel did not move
    (berth dwell, compressed out). Only 3.6% are true discontinuities. Tracks
    are therefore cut wherever a gap exceeds `gap_seconds` *and* the vessel
    moved more than `gap_metres`, which splits the real jumps without
    shattering every track at a pier.

Vessel size
    `Length` is missing from 24% of rows, and there is no beam field anywhere,
    so obstacle radius is taken as Length/2 with two fallbacks: the median
    length of the same VesselType within this dataset, then `default_length`.
    Per ship rather than per row the coverage is much better (78%), and the
    ships missing a length are overwhelmingly those also missing a VesselType,
    i.e. Class B small craft.

Ego vessel
    The source has no ego. One AIS vessel is promoted to ego and removed from
    the obstacle set, so its own recorded transit is a human baseline for the
    same start, goal and traffic. Its length/beam set the ego footprint, so
    pick a vessel the planner's kinematics can honestly represent -- see the
    note on max_accel in the algorithm config.

Run from the repository root:
    python -m PathPlanning.DWAT_v5_modular.adapters.new_york_ais_adapter

The first run indexes all 10,162 CSVs (a few minutes) and caches the result to
EnvData/derived/new_york/ais_index.parquet; later runs read only the segments
that overlap the requested window.
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from scene_format import write_scene_config, write_track_bundle  # noqa: E402

DATASET = os.path.join('EnvData', 'datasets', 'new_york')
LAND_SHP = os.path.join(DATASET, 'land_clipped', 'land_polygons_nyc_clip.shp')
TRACK_ROOT = os.path.join(DATASET, 'processed_interpolated_final')
DERIVED = os.path.join('EnvData', 'derived', 'new_york')
INDEX_PATH = os.path.join(DERIVED, 'ais_index.parquet')
CONFIG_DIR = os.path.join('PathPlanning', 'DWAT_v5_modular', 'configs',
                          'scenarios', 'imported')

UTM18N = 'EPSG:32618'

# The shapefile is a rectangular clip: four polygons are truncated flat at
# lat 40.70, so an ROI crossing that line silently loses its northern shore.
SHAPEFILE_CLIP_LAT = 40.70

# Narrows + Upper Bay + the mouth of Kill Van Kull: the densest water in the
# dataset (591 distinct vessels over the month) and safely below the clip.
DEFAULT_ROI = (40.56, 40.68, -74.12, -73.98)
DEFAULT_WINDOW = ('2021-01-28 18:20', '2021-01-28 19:20')
# 29 m towing vessel, 8.7 km of continuous 1-minute data across the window.
DEFAULT_EGO = 338862000


# ----------------------------------------------------------------- index ----

def build_index(track_root=TRACK_ROOT, index_path=INDEX_PATH, verbose=True):
    """Cache each segment's time span and bounding box so later runs can skip.

    Opening all 10,162 CSVs takes minutes; almost none of them overlap any one
    window, and the index is what makes that skippable.
    """
    files = sorted(glob.glob(os.path.join(track_root, '*', 'track_segment_*.csv')))
    if not files:
        raise FileNotFoundError(f'no track segments under {track_root}')
    rows = []
    for n, f in enumerate(files):
        if verbose and n % 2000 == 0:
            print(f'  indexing {n}/{len(files)}', flush=True)
        df = pd.read_csv(f, encoding='utf-8-sig', usecols=['BaseDateTime', 'LAT', 'LON'])
        if not len(df):
            continue
        t = pd.to_datetime(df['BaseDateTime'], utc=True, format='mixed')
        rows.append(dict(
            file=os.path.relpath(f, track_root),
            mmsi=int(os.path.basename(os.path.dirname(f))),
            t0=t.min(), t1=t.max(),
            lat_min=df['LAT'].min(), lat_max=df['LAT'].max(),
            lon_min=df['LON'].min(), lon_max=df['LON'].max(),
        ))
    index = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(index_path)), exist_ok=True)
    index.to_parquet(index_path)
    if verbose:
        print(f'  wrote {index_path} ({len(index)} segments)')
    return index


def load_index(track_root=TRACK_ROOT, index_path=INDEX_PATH, verbose=True):
    if os.path.exists(index_path):
        return pd.read_parquet(index_path)
    if verbose:
        print(f'no cached index at {index_path}; building it once')
    return build_index(track_root, index_path, verbose)


def load_window(roi, window, track_root=TRACK_ROOT, index_path=INDEX_PATH,
                verbose=True):
    """Read every AIS sample inside the ROI box and the time window."""
    la0, la1, lo0, lo1 = roi
    t0, t1 = (pd.Timestamp(w, tz='UTC') for w in window)
    index = load_index(track_root, index_path, verbose)

    hit = index[(index.t0 <= t1) & (index.t1 >= t0)
                & (index.lat_min <= la1) & (index.lat_max >= la0)
                & (index.lon_min <= lo1) & (index.lon_max >= lo0)]
    if verbose:
        print(f'  {len(hit)} of {len(index)} segments overlap the window and ROI')

    frames = []
    for f in hit.file:
        df = pd.read_csv(os.path.join(track_root, f), encoding='utf-8-sig',
                         usecols=['BaseDateTime', 'LAT', 'LON', 'MMSI', 'SOG',
                                  'COG', 'VesselType', 'Length'])
        df['t'] = pd.to_datetime(df['BaseDateTime'], utc=True, format='mixed')
        frames.append(df)
    if not frames:
        raise ValueError('no AIS data in the requested ROI and window')

    pts = pd.concat(frames, ignore_index=True)
    pts = pts[(pts.t.between(t0, t1))
              & (pts.LAT.between(la0, la1)) & (pts.LON.between(lo0, lo1))]
    pts['mmsi'] = pts['MMSI'].astype('int64')
    for c in ('SOG', 'COG', 'VesselType', 'Length'):
        pts[c] = pd.to_numeric(pts[c], errors='coerce')
    return pts.sort_values(['mmsi', 't']).reset_index(drop=True)


# ------------------------------------------------------------------ map ----

def rasterise_land(roi, resolution, out_dir=DERIVED):
    """Rasterise the land polygons over the ROI onto a metric occupancy grid.

    Returns (path, occupancy, extent) where occupancy[i, j] == 1 marks land,
    axis 0 is x (easting) and axis 1 is y (northing), matching MapManager's
    convention rather than image row/column order.
    """
    import cv2
    import geopandas as gpd
    from shapely.geometry import box
    from pyproj import Transformer

    la0, la1, lo0, lo1 = roi
    if la1 > SHAPEFILE_CLIP_LAT:
        raise ValueError(
            f'ROI reaches lat {la1}, above the shapefile clip at '
            f'{SHAPEFILE_CLIP_LAT}: the land north of that line is missing from '
            f'the source, so the map would show open water where there is shore'
        )

    land = gpd.clip(gpd.read_file(LAND_SHP), box(lo0, la0, lo1, la1)).to_crs(UTM18N)
    tf = Transformer.from_crs('EPSG:4326', UTM18N, always_xy=True)
    xs, ys = tf.transform([lo0, lo1, lo0, lo1], [la0, la0, la1, la1])
    x0, y0 = min(xs), min(ys)
    width, height = max(xs) - x0, max(ys) - y0
    nx, ny = int(width // resolution), int(height // resolution)

    # Fill polygon exteriors, then punch out interior rings, in image order
    # (row 0 = max northing) before flipping back to an x/y-major grid.
    img = np.zeros((ny, nx), np.uint8)

    def to_px(coords):
        a = np.asarray(coords)
        return np.round(np.c_[(a[:, 0] - x0) / resolution,
                              (ny - 1) - (a[:, 1] - y0) / resolution]).astype(np.int32)

    outer, holes = [], []
    for geom in land.geometry:
        for poly in (geom.geoms if geom.geom_type == 'MultiPolygon' else [geom]):
            outer.append(to_px(poly.exterior.coords))
            holes.extend(to_px(r.coords) for r in poly.interiors)
    if outer:
        cv2.fillPoly(img, outer, 1)
    if holes:
        cv2.fillPoly(img, holes, 0)

    occupancy = np.flipud(img).T.copy()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'occupancy_{int(resolution)}m.npy')
    np.save(path, occupancy)
    return path, occupancy, (x0, y0, nx, ny)


# AIS carries no beam field anywhere in this dataset, so beam is estimated from
# length by vessel class. These are length/beam ratios in the usual range for
# each type, not measurements.
BEAM_RATIO = {
    30: 4.0,    # fishing
    31: 3.0, 32: 3.0, 52: 3.0,   # towing / tug
    36: 4.0, 37: 3.5,            # sailing / pleasure
    60: 5.0, 61: 5.0, 62: 5.0, 63: 5.0, 64: 5.0, 65: 5.0, 66: 5.0, 69: 5.0,
    70: 7.0, 71: 7.0, 72: 7.0, 73: 7.0, 74: 7.0, 79: 7.0,   # cargo
    80: 6.5, 81: 6.5, 82: 6.5, 83: 6.5, 84: 6.5, 89: 6.5,   # tanker
}
DEFAULT_BEAM_RATIO = 5.0


def beam_for(length, vessel_type):
    ratio = DEFAULT_BEAM_RATIO
    if vessel_type is not None and np.isfinite(vessel_type):
        ratio = BEAM_RATIO.get(int(vessel_type), DEFAULT_BEAM_RATIO)
    return float(length) / ratio


def ego_kinematics(length, beam, service_speed):
    """Algorithm-config keys implied by one vessel's size and service speed.

    Every value here is a property of the ego, not of the tuning, so a batch
    over different vessels varies exactly these and nothing else. The cost gains
    and horizons stay in the base config.
    """
    length, beam = float(length), float(beam)
    v = float(service_speed)

    # Turn radius of 2.2 lengths -- a working figure for a manoeuvrable harbour
    # vessel, and what the hand-checked 29 m case used. Capped at 10 deg/s,
    # because for a short fast vessel (a 15 m ferry at 25 kn) that ratio implies
    # over 20 deg/s, which no hull actually delivers.
    yaw_rate = min(v / (2.2 * length), 0.175)
    # Braking authority, not hydrodynamics: see the note in README_ais_v1.md.
    # Scaled so a 30 m vessel gets ~0.3 m/s^2, then floored so that a large ship
    # does not demand more clear water than a harbour contains.
    accel = float(np.clip(9.0 / length, 0.05, 0.4))

    localgoal = float(np.clip(v * 90.0, 300.0, 2000.0))

    return {
        'robot_type': 'rectangle',
        'robot_length': round(length, 2),
        'robot_width': round(beam, 2),
        'robot_radius': round(float(np.hypot(length, beam) / 2.0), 2),
        'max_speed': round(v, 3),
        'max_accel': round(accel, 4),
        'max_yaw_rate': round(yaw_rate, 6),
        'max_delta_yaw_rate': round(yaw_rate / 20.0, 8),
        'v_resolution': round(2 * accel * 10.0 / 20, 4),
        'yaw_rate_resolution': round(2 * (yaw_rate / 20.0) * 10.0 / 40, 8),
        'speed_cost_gain': round(1.0 / v, 4),
        # Route-following distances scale with the ego as well. _choose_local_goal
        # raises outright once the ego is further than dist_localgoal from the
        # current route segment, so a distance tuned for a 5.5 m/s tug aborts a
        # 13 m/s ferry, which covers 130 m in a single control step. Expressed as
        # travel time along the route (~90 s) rather than a fixed distance.
        'dist_localgoal': round(localgoal, 1),
        'catch_turning_point_dist': round(localgoal, 1),
        # Three lengths to call it arrived, but never more than half the
        # lookahead: at parity the ego would count the goal as reached from
        # as far away as it plans, and stop short.
        'catch_goal_dist': round(min(float(np.clip(3.0 * length, 50.0, 300.0)),
                                     0.5 * localgoal), 1),
    }


def navigable_mask(occupancy):
    """Cells belonging to the largest connected body of water.

    The ROI contains genuine inland ponds, and a vessel alongside a pier sits in
    a slip the coastline polygon seals off. Both are water, and neither can be
    navigated to -- so "not land" is not a strong enough test for a start or a
    goal; being in the same connected component as the rest of the harbour is.
    """
    import cv2
    free = (occupancy == 0).astype(np.uint8)
    count, labels = cv2.connectedComponents(free)
    if count <= 1:
        raise ValueError('the occupancy grid contains no water at all')
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0                      # label 0 is the land background
    return labels == int(sizes.argmax())


def trim_ego(ego, navigable, resolution, min_sog):
    """Cut a berth-to-berth AIS track down to the part that is actually a transit.

    Leading and trailing samples are dropped while the vessel is either not
    under way or sitting in water disconnected from the harbour, so the scenario
    starts where the vessel does and the global planner has a route to plan.
    """
    nx, ny = navigable.shape
    i = np.clip((ego.x.values // resolution).astype(int), 0, nx - 1)
    j = np.clip((ego.y.values // resolution).astype(int), 0, ny - 1)
    usable = (ego.SOG.values > min_sog) & navigable[i, j]
    if not usable.any():
        raise ValueError(
            'the ego vessel is never both under way and in navigable water '
            'inside this ROI and window'
        )
    first, last = int(np.argmax(usable)), int(len(usable) - 1 - np.argmax(usable[::-1]))
    return ego.iloc[first:last + 1], first, len(ego) - 1 - last


# --------------------------------------------------------------- tracks ----

def vessel_radii(pts, default_length):
    """Collision radius per MMSI: Length/2, then VesselType median, then default."""
    per_ship = pts.groupby('mmsi').agg(length=('Length', 'median'),
                                       vtype=('VesselType', 'median'))
    by_type = per_ship.dropna(subset=['length']).groupby('vtype').length.median()
    filled = per_ship.length.fillna(per_ship.vtype.map(by_type))
    source = np.where(per_ship.length.notna(), 'ais',
                      np.where(filled.notna(), 'vessel_type', 'default'))
    return (filled.fillna(default_length) / 2.0), pd.Series(source, index=per_ship.index)


def split_runs(g, gap_seconds, gap_metres, max_leg_speed):
    """Split one vessel's samples wherever the recording genuinely jumps.

    Two kinds of break are cut, and one deliberately is not:

    * a long gap the vessel *moved* across is missing data, and interpolating
      it would drag the obstacle through whatever lies between;
    * a leg whose implied speed is impossible is a bad fix. Filtering on the
      reported SOG alone does not catch these -- SOG can look ordinary while
      the positions jump -- so the speed is recomputed from the positions;
    * a long gap the vessel did *not* move across is berth dwell that the
      upstream pipeline compressed out. Holding position across it is exactly
      what happened, so it is left joined.
    """
    dt = g.t.diff().dt.total_seconds().values
    step = np.hypot(g.x.diff().values, g.y.diff().values)
    with np.errstate(divide='ignore', invalid='ignore'):
        implied = np.where(dt > 0, step / dt, 0.0)
    cut = ((dt > gap_seconds) & (step > gap_metres)) | (implied > max_leg_speed)
    return np.cumsum(np.nan_to_num(cut, nan=0.0)).astype(int)


def convert(roi=DEFAULT_ROI,
            window=DEFAULT_WINDOW,
            ego_mmsi=DEFAULT_EGO,
            ego_start=None,
            ego_goal=None,
            ego_length=None,
            ego_vessel_type=None,
            ego_speed=None,
            resolution=50.0,
            global_planner_resolution=None,
            gap_seconds=300.0,
            gap_metres=100.0,
            ego_min_sog=0.5,
            default_length=15.0,
            min_samples=2,
            max_sog_knots=25.0,
            name=None,
            config_path=None,
            tracks_path=None,
            verbose=True):
    from pyproj import Transformer

    name = name or f'ny_narrows_{ego_mmsi}'
    config_path = config_path or os.path.join(CONFIG_DIR, f'{name}.json')
    tracks_path = tracks_path or os.path.join(DERIVED, f'{name}_tracks.npz')
    if global_planner_resolution is None:
        global_planner_resolution = resolution

    print('rasterising land ...')
    occ_path, occupancy, (x0, y0, nx, ny) = rasterise_land(roi, resolution)
    print(f'  occupancy      : {occupancy.shape} at {resolution:g} m/cell, '
          f'{int(occupancy.sum())} land cells ({occupancy.mean()*100:.1f}%)')

    max_leg_speed = max_sog_knots * 0.514444   # knots -> m/s

    print('reading AIS ...')
    pts = load_window(roi, window, verbose=verbose)
    tf = Transformer.from_crs('EPSG:4326', UTM18N, always_xy=True)
    px, py = tf.transform(pts.LON.values, pts.LAT.values)
    pts['x'], pts['y'] = px - x0, py - y0

    navigable = navigable_mask(occupancy)
    synthetic = ego_mmsi is None

    if synthetic:
        # An invented mission through real traffic: nothing is removed from the
        # obstacle set, so every recorded vessel stays an obstacle and the ego
        # is an extra vessel the harbour never contained.
        if ego_start is None or ego_goal is None:
            raise ValueError('a synthetic ego needs both ego_start and ego_goal')
        ego = None
        t_origin = pts.t.min()
        pts = pts.copy()
        pts['sec'] = (pts.t - t_origin).dt.total_seconds()
        print(f'  {len(pts)} samples, {pts.mmsi.nunique()} vessels, '
              f'{pts.sec.max()/60:.0f} min (synthetic ego, all vessels kept)')
    else:
        if ego_mmsi not in set(pts.mmsi):
            raise ValueError(f'ego {ego_mmsi} has no samples in this ROI and window')

        ego_full = pts[pts.mmsi == ego_mmsi].sort_values('t')
        ego, cut_head, cut_tail = trim_ego(ego_full, navigable, resolution, ego_min_sog)
        if cut_head or cut_tail:
            print(f'  ego trimmed    : dropped {cut_head} leading and {cut_tail} trailing '
                  f'samples where the ego was moored or in a berth slip')

        # t = 0 is when the EGO gets under way, not when the first of ~75
        # unrelated vessels enters. Anchoring on the earliest sample in the
        # window would offset every obstacle from the traffic the ego met.
        t_origin = ego.t.min()
        dropped = int((pts.t < t_origin).sum())
        pts = pts[pts.t >= t_origin].copy()
        pts['sec'] = (pts.t - t_origin).dt.total_seconds()
        ego = ego.assign(sec=(ego.t - t_origin).dt.total_seconds())
        print(f'  {len(pts)} samples, {pts.mmsi.nunique()} vessels, '
              f'{pts.sec.max()/60:.0f} min')
        if dropped:
            print(f'  time origin    : ego under way at {t_origin}, '
                  f'{(t_origin - pd.Timestamp(window[0], tz="UTC")).total_seconds()/60:.0f} min '
                  f'into the requested window; {dropped} earlier samples dropped')

    # A max SOG of 102 kn appears in this dataset; 2% of segments exceed 30 kn.
    # Those are decoding errors, and they produce obstacles that cross the map
    # in seconds, so drop the samples rather than the whole vessel.
    fast = pts.SOG > max_sog_knots
    if fast.any():
        print(f'  dropping {int(fast.sum())} samples over {max_sog_knots:g} kn')
        pts = pts[~fast]

    radii, radius_source = vessel_radii(pts, default_length)
    if verbose:
        print('  radius source  : ' + ', '.join(
            f'{k}={v}' for k, v in radius_source.value_counts().items()))

    others = pts if synthetic else pts[pts.mmsi != ego_mmsi]

    # A track whose first sample is after t=0 is clamped to that first position
    # until it starts, so the vessel is parked there beforehand. That is
    # harmless when the entry point is on the ROI boundary (the vessel really is
    # just outside), and a phantom when it is in open water well inside.
    edge_m = 2.0 * resolution
    span_x, span_y = nx * resolution, ny * resolution

    tracks, times, track_radii, ids = [], [], [], []
    late_edge = late_interior = 0
    for mmsi, g in others.groupby('mmsi', sort=True):
        g = g.sort_values('t')
        for run_id, run in g.groupby(
                split_runs(g, gap_seconds, gap_metres, max_leg_speed)):
            if len(run) < min_samples:
                continue
            tracks.append(np.c_[run.x.values, run.y.values])
            times.append(run.sec.values)
            track_radii.append(float(radii.loc[mmsi]))
            ids.append(f'{mmsi}_{run_id}')
            if run.sec.values[0] > 0:
                ex, ey = run.x.values[0], run.y.values[0]
                on_edge = (min(ex, span_x - ex) < edge_m
                           or min(ey, span_y - ey) < edge_m)
                # A vessel that never gets under way was moored there all along:
                # its samples are sparse because the upstream pipeline
                # compresses dwell, so holding position is what really happened.
                moored = float(np.nanmax(run.SOG.values)) <= 0.5
                if on_edge or moored:
                    late_edge += 1
                else:
                    late_interior += 1

    if not tracks:
        raise ValueError('no usable obstacle tracks in this ROI and window')

    bundle = write_track_bundle(tracks_path, tracks=tracks, times=times,
                                radii=track_radii, ids=ids)
    print(f'  dynamic tracks : {bundle["n_obs"]} from {others.mmsi.nunique()} vessels, '
          f'horizon {bundle["horizon"]} samples, {bundle["duration_sec"]:.0f} s')
    if late_edge or late_interior:
        print(f'  late entries   : {late_edge} benign (on the ROI boundary, or moored '
              f'throughout), {late_interior} park in open water before they start')
        if late_interior:
            print(f'                   those {late_interior} are stationary phantoms '
                  f'until their first sample; shorten the window, or pick an ego '
                  f'that enters earlier, if the route passes close to one')

    if synthetic:
        start = np.asarray(ego_start, dtype=float)
        goal = np.asarray(ego_goal, dtype=float)
        head = goal - start
        start_yaw = float(np.arctan2(head[1], head[0]))
        length = float(ego_length if ego_length is not None else default_length)
        recorded_path = float('nan')
        recorded_duration = float('nan')
    else:
        start = np.array([ego.x.iloc[0], ego.y.iloc[0]])
        goal = np.array([ego.x.iloc[-1], ego.y.iloc[-1]])
        # AIS COG is degrees clockwise from north; the planner's yaw is radians
        # counter-clockwise from +x (east).
        cog0 = ego.COG.iloc[0]
        if np.isfinite(cog0):
            start_yaw = np.deg2rad(90.0 - cog0)
        else:
            head = np.array([ego.x.iloc[1] - start[0], ego.y.iloc[1] - start[1]])
            start_yaw = np.arctan2(head[1], head[0])
        start_yaw = float(np.arctan2(np.sin(start_yaw), np.cos(start_yaw)))
        length = (float(ego.Length.median()) if ego.Length.notna().any()
                  else float(default_length))
        if ego_length is not None:
            length = float(ego_length)
        recorded_path = float(np.hypot(np.diff(ego.x.values), np.diff(ego.y.values)).sum())
        recorded_duration = float(ego.sec.max() - ego.sec.min())
        if ego_vessel_type is None and ego.VesselType.notna().any():
            ego_vessel_type = float(ego.VesselType.median())
        if ego_speed is None:
            ego_speed = float(np.nanpercentile(ego.SOG.values, 90)) * 0.514444

    ego_length = length
    beam = beam_for(length, ego_vessel_type)
    service_speed = float(ego_speed) if ego_speed else 5.0

    scene = write_scene_config(
        config_path,
        name=name,
        group='imported',
        occupancy_path=occ_path.replace(os.sep, '/'),
        occupancy_resolution=resolution,
        occupancy_origin=(0.0, 0.0),
        occupancy_cell_anchor='center',
        map_size=(occupancy.shape[0], occupancy.shape[1]),
        # The ROI edges are open water that traffic crosses; walling them would
        # trap every vessel that is only passing through.
        enable_map_boundary_obstacles=False,
        start_position=start,
        goal_position=goal,
        start_yaw=start_yaw,
        track_file=tracks_path.replace(os.sep, '/'),
        n_obs=bundle['n_obs'],
        radii=track_radii,
        on_finish='hold',
        source={
            'dataset': DATASET.replace(os.sep, '/'),
            'crs': UTM18N,
            'units': 'metres, seconds',
            'roi_latlon': list(roi),
            'window_utc': list(window),
            'time_origin_utc': str(t_origin),
            'utm_origin': [float(x0), float(y0)],
            'ego_mmsi': None if synthetic else int(ego_mmsi),
            'ego_kind': 'synthetic' if synthetic else 'historical',
            'ego_length_m': ego_length,
            'ego_beam_m': round(beam, 2),
            'ego_service_speed_mps': round(service_speed, 3),
            'ego_recorded_path_m': recorded_path,
            'ego_recorded_duration_s': recorded_duration,
            'gap_split_seconds': gap_seconds,
            'gap_split_metres': gap_metres,
            'max_sog_knots': max_sog_knots,
            'max_leg_speed_mps': max_leg_speed,
            'default_length_m': default_length,
        },
    )
    # Theta* grids the whole world at this spacing; in metres the default of
    # 1.0 would be a kilometre-scale grid of 1 m cells.
    scene['global_planner_resolution'] = float(global_planner_resolution)
    import json
    with open(config_path, 'w') as f:
        json.dump(scene, f, indent=4)

    _self_check(occupancy, navigable, resolution, start, goal, pts, ego_mmsi)

    net = np.hypot(*(goal - start))
    if synthetic:
        print(f'  ego            : synthetic, {ego_length:.0f} x {beam:.0f} m, '
              f'{service_speed:.1f} m/s')
    else:
        print(f'  ego            : MMSI {ego_mmsi}, {ego_length:.0f} x {beam:.0f} m, '
              f'{len(ego)} samples over {recorded_duration:.0f} s')
    print(f'                   start {tuple(np.round(start))} -> goal '
          f'{tuple(np.round(goal))}, net {net/1000:.2f} km, '
          f'yaw {np.degrees(start_yaw):.0f} deg')
    print(f'  wrote          : {occ_path}')
    print(f'  wrote          : {tracks_path}')
    print(f'  wrote          : {config_path}')
    return {
        'scene': scene,
        'config_path': config_path,
        'kinematics': ego_kinematics(ego_length, beam, service_speed),
        'ego': {
            'kind': 'synthetic' if synthetic else 'historical',
            'mmsi': None if synthetic else int(ego_mmsi),
            'length_m': ego_length, 'beam_m': beam,
            'service_speed_mps': service_speed,
            'net_distance_m': float(net),
            'recorded_path_m': recorded_path,
            'recorded_duration_s': recorded_duration,
        },
    }


def _self_check(occupancy, navigable, resolution, start, goal, pts, ego_mmsi):
    """Refuse to emit a scenario whose map contradicts its own tracks."""
    nx, ny = occupancy.shape

    def cell(p):
        return (int(np.clip(p[0] // resolution, 0, nx - 1)),
                int(np.clip(p[1] // resolution, 0, ny - 1)))

    for label, p in (('start', start), ('goal', goal)):
        i, j = cell(p)
        if occupancy[i, j] == 1:
            raise ValueError(
                f'ego {label} {tuple(np.round(p))} falls on a land cell -- the '
                f'coastline polygon covers it. Try a finer --resolution.'
            )
        if not navigable[i, j]:
            raise ValueError(
                f'ego {label} {tuple(np.round(p))} sits in water that is not '
                f'connected to the main harbour at {resolution:g} m/cell -- a '
                f'berth slip or an inland pond. The global planner cannot route '
                f'out of it. Use a finer --resolution, or raise --ego-min-sog so '
                f'the transit starts further from the berth.'
            )

    # Moored vessels legitimately sit inside the coastline polygon (piers are
    # drawn as land), so only moving samples say anything about map quality.
    moving = pts[(pts.SOG > 0.5) & (pts.mmsi != ego_mmsi)]
    if len(moving):
        i = np.clip((moving.x.values // resolution).astype(int), 0, nx - 1)
        j = np.clip((moving.y.values // resolution).astype(int), 0, ny - 1)
        on_land = float((occupancy[i, j] == 1).mean())
        print(f'  map check      : {on_land*100:.2f}% of moving AIS samples fall on '
              f'land cells (berth geometry; >10% would mean a projection error)')
        if on_land > 0.10:
            raise ValueError(
                f'{on_land*100:.1f}% of moving AIS samples are inside land cells -- '
                f'the map and the tracks disagree. Check the CRS and the x/y axis '
                f'order of the occupancy grid.'
            )


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--roi', nargs=4, type=float, metavar=('LAT0', 'LAT1', 'LON0', 'LON1'),
                   default=list(DEFAULT_ROI))
    p.add_argument('--window', nargs=2, metavar=('START_UTC', 'END_UTC'),
                   default=list(DEFAULT_WINDOW))
    p.add_argument('--ego', type=int, default=DEFAULT_EGO, help='ego MMSI')
    p.add_argument('--resolution', type=float, default=50.0,
                   help='occupancy cell size in metres (default 50). Each land '
                        'cell becomes a disc of radius resolution/2, so a coarse '
                        'grid fattens every shoreline by half a cell.')
    p.add_argument('--global-planner-resolution', type=float, default=None,
                   help='Theta* search grid in metres (default: --resolution)')
    p.add_argument('--gap-seconds', type=float, default=300.0)
    p.add_argument('--gap-metres', type=float, default=100.0)
    p.add_argument('--default-length', type=float, default=15.0)
    p.add_argument('--ego-min-sog', type=float, default=0.5,
                   help='knots; leading/trailing ego samples below this (or in a '
                        'berth slip) are trimmed so the scenario is a transit')
    p.add_argument('--max-sog', type=float, default=25.0)
    p.add_argument('--name', default=None)
    p.add_argument('--config-out', default=None)
    p.add_argument('--tracks-out', default=None)
    p.add_argument('--reindex', action='store_true',
                   help='rebuild the cached AIS index before converting')
    a = p.parse_args()

    if a.reindex:
        build_index()

    convert(roi=tuple(a.roi), window=tuple(a.window), ego_mmsi=a.ego,
            resolution=a.resolution,
            global_planner_resolution=a.global_planner_resolution,
            gap_seconds=a.gap_seconds, gap_metres=a.gap_metres,
            ego_min_sog=a.ego_min_sog,
            default_length=a.default_length, max_sog_knots=a.max_sog,
            name=a.name, config_path=a.config_out, tracks_path=a.tracks_out)


if __name__ == '__main__':
    main()

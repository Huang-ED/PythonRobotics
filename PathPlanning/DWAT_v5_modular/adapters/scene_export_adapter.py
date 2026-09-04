"""Convert EnvData/datasets/scene_export into the intermediary scene format.

The export is already a simulation snapshot on a 50x50 / 1 m grid, so this is a
pure repackaging step: no projection, no resampling, no unit conversion. The
one thing that matters is the cell convention.

    scene_export : cell (i, j) spans [i*res, (i+1)*res] x [j*res, (j+1)*res],
                   its centre is ((i + 0.5) * res, (j + 0.5) * res)
    MapManager   : each occupied cell becomes one obstacle point

so the grid is loaded with cell_anchor="center". Anchoring at the corner
instead would shift every static obstacle half a cell relative to the obstacle
trajectories, which are certified collision-free against the cell areas.

The occupancy grid is read from occupancy.npy rather than re-derived from the
bundled land_shapes_sf_crop.png: resizing that 1339x1339 image down to 50x50
reproduces only ~97% of the cells, and the trajectories are guaranteed against
the grid, not the image.

Algorithm tuning is deliberately not written here: DWAT_v5_modular keeps
scenario and algorithm configs separate, so this only produces the scenario.

Run from the repository root:
    python PathPlanning/DWAT_v5_modular/adapters/scene_export_adapter.py
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from scene_format import write_scene_config, write_track_bundle  # noqa: E402

DEFAULT_SCENE = os.path.join('EnvData', 'datasets', 'scene_export', 'scene.npz')
DEFAULT_OCCUPANCY = os.path.join('EnvData', 'datasets', 'scene_export', 'occupancy.npy')
DEFAULT_TRACKS = os.path.join('EnvData', 'derived', 'scene_export', 'dynamic_tracks.npz')
DEFAULT_IMAGE = os.path.join('EnvData', 'datasets', 'scene_export', 'land_shapes_sf_crop.png')
REFINED_DIR = os.path.join('EnvData', 'derived', 'scene_export')
DEFAULT_CONFIG = os.path.join('PathPlanning', 'DWAT_v5_modular', 'configs', 'scenarios',
                              'scene_export.json')


def refine_occupancy(image_path, cells, world_bounds, scene, out_dir=REFINED_DIR):
    """Re-rasterise the bundled map image onto a finer grid over the same world.

    The shipped occupancy.npy is 50x50 over a 50 m world -- 1 m cells, from a
    1339x1339 source image. That is coarser than the ego itself (1.2 m long), so
    corridor widths are quantised to 1 m and a 45-degree coastline becomes a
    staircase of 0.5 m disks.

    Refining is safe because the world extent is fixed at world_bounds whatever
    the grid size: the obstacle trajectories live in continuous world
    coordinates and do not move. Only the sampling of the land changes.

    Two things this must preserve from the shipped grid:
      * the deliberately blocked outer ring (the dataset relies on it instead of
        an explicit bounds check), and
      * the guarantee that every recorded trajectory point stays in free space --
        checked below, and refused if it ever fails.
    """
    x_min, x_max, y_min, y_max = world_bounds
    if not np.isclose(x_max - x_min, y_max - y_min):
        raise ValueError('refine_occupancy assumes a square world')

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f'map image not readable: {image_path}')

    resized = cv2.resize(img, (cells, cells), interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(resized, 128, 1, cv2.THRESH_BINARY)
    # Image row/col -> world (x, y): flip the rows, then transpose, so that
    # axis 0 is x and axis 1 is y. Black (0) is land.
    occ = (np.flipud(binary).T == 0).astype(np.uint8)

    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = 1

    resolution = (x_max - x_min) / cells

    # Refuse to emit a map that would invalidate the trajectories it ships with.
    traj, lens = scene['obs_traj'], scene['obs_traj_len']
    points = [traj[k, :int(lens[k])] for k in range(traj.shape[0])]
    points.append(scene['robot_dense_path'])
    bad = 0
    for arr in points:
        idx = np.clip(((arr - [x_min, y_min]) / resolution).astype(int), 0, cells - 1)
        bad += int(occ[idx[:, 0], idx[:, 1]].sum())
    if bad:
        raise ValueError(
            f'refining to {cells}x{cells} puts {bad} recorded trajectory points inside '
            f'blocked cells -- the finer map contradicts the tracks; use a coarser grid'
        )

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'occupancy_{cells}.npy')
    np.save(path, occ)
    return path, occ, resolution


def convert(scene_path=DEFAULT_SCENE,
            occupancy_path=DEFAULT_OCCUPANCY,
            tracks_path=DEFAULT_TRACKS,
            config_path=DEFAULT_CONFIG,
            name='scene_export',
            map_cells=None,
            image_path=DEFAULT_IMAGE):
    scene = np.load(scene_path)

    occupancy = scene['occupancy']
    resolution = float(scene['resolution'])
    time_step = float(scene['time_step'])
    obstacle_radius = float(scene['obstacle_radius'])
    x_min, x_max, y_min, y_max = [float(v) for v in scene['world_bounds']]

    if map_cells is not None:
        occupancy_path, occupancy, resolution = refine_occupancy(
            image_path, map_cells, (x_min, x_max, y_min, y_max), scene)

    obs_traj = scene['obs_traj']          # (n_obs, horizon, 2), padded
    obs_traj_len = scene['obs_traj_len']  # (n_obs,) real lengths
    obs_time = scene['obs_time']          # (horizon,) shared timeline

    n_obs = obs_traj.shape[0]

    # Trim each track back to its real samples; the padding is just the goal
    # repeated, and on_finish="hold" reproduces that behaviour exactly.
    tracks = [obs_traj[k, :int(obs_traj_len[k])] for k in range(n_obs)]

    bundle_info = write_track_bundle(
        tracks_path,
        tracks=tracks,
        times=obs_time,
        radii=[obstacle_radius] * n_obs,
        ids=[f'obs_{k}' for k in range(n_obs)],
    )

    # The scene ships the reference robot's mission; reuse it so the ego runs
    # the same problem the scene was authored around.
    dense_path = scene['robot_dense_path']
    start_position = dense_path[0]
    goal_position = dense_path[-1]

    # Face the ego down the first leg of its own route. The runner's historical
    # default of -pi/8 is arbitrary here and would burn several seconds -- and
    # several metres of traffic -- just turning around on the spot.
    heading = dense_path[min(5, len(dense_path) - 1)] - start_position
    start_yaw = float(np.arctan2(heading[1], heading[0]))

    scene_cfg = write_scene_config(
        config_path,
        name=name,
        occupancy_path=occupancy_path.replace(os.sep, '/'),
        occupancy_resolution=resolution,
        occupancy_origin=(x_min, y_min),
        occupancy_cell_anchor='center',
        map_size=(occupancy.shape[0], occupancy.shape[1]),
        # The export already marks the outer ring of cells as occupied, so the
        # planner must not add a second border on top of it.
        enable_map_boundary_obstacles=False,
        start_position=start_position,
        goal_position=goal_position,
        start_yaw=start_yaw,
        track_file=tracks_path.replace(os.sep, '/'),
        n_obs=n_obs,
        radii=[obstacle_radius] * n_obs,
        on_finish='hold',
        source={
            'dataset': 'EnvData/datasets/scene_export',
            'scene_npz': scene_path.replace(os.sep, '/'),
            'time_step': time_step,
            'world_bounds': [x_min, x_max, y_min, y_max],
            'reference_robot_radius': 0.3,
            'reference_robot_v_pref': 1.0,
        },
    )

    print(f'occupancy      : {occupancy.shape}, {resolution} m/cell, '
          f'{int((occupancy != 0).sum())} occupied of {occupancy.size}')
    if map_cells is not None:
        print(f'                 refined from {image_path}; all recorded trajectory '
              f'points verified in free space')
        print(f'  !! set obstacle_radius = {resolution / 2:g} in your algorithm config. '
              f'It is the disk radius used for EVERY static cell, and at the shipped '
              f'1 m grid it is 0.5 = half a cell. Leaving it at 0.5 here would inflate '
              f'every wall by {0.5 - resolution / 2:g} m per side.')
    print(f'dynamic tracks : {bundle_info["n_obs"]} obstacles, '
          f'horizon {bundle_info["horizon"]} samples, '
          f'{bundle_info["duration_sec"]:.2f} s, radius {obstacle_radius} m')
    print(f'ego            : start {tuple(start_position)} -> goal {tuple(goal_position)}, '
          f'yaw {np.degrees(start_yaw):.1f} deg')
    print(f'wrote          : {tracks_path}')
    print(f'wrote          : {config_path}')
    return scene_cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scene', default=DEFAULT_SCENE)
    parser.add_argument('--occupancy', default=DEFAULT_OCCUPANCY)
    parser.add_argument('--tracks-out', default=DEFAULT_TRACKS)
    parser.add_argument('--config-out', default=DEFAULT_CONFIG)
    parser.add_argument('--name', default='scene_export')
    parser.add_argument('--image', default=DEFAULT_IMAGE,
                        help='Map image to re-rasterise when --map-cells is given.')
    parser.add_argument('--map-cells', type=int, default=None,
                        help='Re-rasterise the bundled map image to this many cells per '
                             'side over the same 50 m world, instead of using the shipped '
                             '50x50 occupancy.npy. 100 gives 0.5 m cells. Remember to set '
                             'obstacle_radius = 25/map_cells in the algorithm config.')
    args = parser.parse_args()

    convert(args.scene, args.occupancy, args.tracks_out, args.config_out, args.name,
            map_cells=args.map_cells, image_path=args.image)


if __name__ == '__main__':
    main()

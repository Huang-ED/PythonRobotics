"""Shared writer for the intermediary scene format used by the DWAT runners.

One environment dataset -> one scenario JSON + one track bundle NPZ.

The point of the intermediary layer is that the planner never learns anything
about where the data came from. An adapter is responsible for resolving a
dataset (occupancy grid or shapefile, simulated tracks or AIS messages) into
these two artefacts; everything downstream reads only these.

Track bundle NPZ layout
-----------------------
    positions   (n_obs, K, 2) float64  padded track samples, world metres
    lengths     (n_obs,)      int64    real sample count of each track
    times       (K,)          float64  shared timeline in seconds
                or (n_obs, K)          per-track timeline in seconds
    velocities  (n_obs, K, 2) float64  optional; derived on load if absent
    radii       (n_obs,)      float64  collision radius per obstacle
    ids         (n_obs,)      <U        optional label per obstacle

Padding beyond `lengths[k]` is never read: the config's `on_finish` decides
what an obstacle does once its samples run out.
"""

import json
import os

import numpy as np


def derive_velocities(positions: np.ndarray, times: np.ndarray) -> np.ndarray:
    """v[k] = (p[k+1] - p[k]) / (t[k+1] - t[k]), v[-1] = 0.

    This is the piecewise-constant velocity that exactly reproduces linear
    interpolation of `positions` between samples, so a planner that predicts
    an obstacle forward at v[k] agrees with the recording at every sample.
    """
    positions = np.asarray(positions, dtype=float)
    times = np.asarray(times, dtype=float)
    velocities = np.zeros_like(positions)
    if positions.shape[0] < 2:
        return velocities
    dt = np.diff(times)
    safe_dt = np.where(dt > 1e-12, dt, np.inf)
    velocities[:-1] = np.diff(positions, axis=0) / safe_dt[:, None]
    return velocities


def write_track_bundle(path: str,
                       tracks,
                       times,
                       radii,
                       ids=None,
                       velocities=None) -> dict:
    """Write a track bundle NPZ from a list of (n_k, 2) arrays.

    Args:
        tracks:     list of per-obstacle (n_k, 2) position arrays.
        times:      shared (K,) timeline, or a list of per-obstacle (n_k,) timelines.
        radii:      per-obstacle collision radius.
        ids:        optional per-obstacle label.
        velocities: optional list of per-obstacle (n_k, 2) arrays; derived if None.

    Returns a small summary dict for logging.
    """
    n_obs = len(tracks)
    if n_obs == 0:
        raise ValueError("write_track_bundle needs at least one track")

    lengths = np.array([len(t) for t in tracks], dtype=np.int64)
    horizon = int(lengths.max())

    positions = np.zeros((n_obs, horizon, 2), dtype=float)
    vel_out = np.zeros((n_obs, horizon, 2), dtype=float)

    shared_times = not isinstance(times, (list, tuple)) or np.ndim(times[0]) == 0
    if shared_times:
        times_arr = np.asarray(times, dtype=float).reshape(-1)
        if times_arr.shape[0] < horizon:
            raise ValueError(
                f"shared times has {times_arr.shape[0]} entries but the longest "
                f"track has {horizon} samples"
            )
        times_arr = times_arr[:horizon]
    else:
        times_arr = np.zeros((n_obs, horizon), dtype=float)

    for k, track in enumerate(tracks):
        n = lengths[k]
        track = np.asarray(track, dtype=float).reshape(-1, 2)
        positions[k, :n] = track
        # Hold the last real sample through the padding so that any consumer
        # that ignores `lengths` still sees a parked obstacle, not the origin.
        positions[k, n:] = track[-1]

        t_k = times_arr[:n] if shared_times else np.asarray(times[k], dtype=float)
        if not shared_times:
            times_arr[k, :n] = t_k
            times_arr[k, n:] = t_k[-1]

        v_k = derive_velocities(track, t_k) if velocities is None \
            else np.asarray(velocities[k], dtype=float).reshape(-1, 2)
        vel_out[k, :n] = v_k

    payload = {
        'positions': positions,
        'lengths': lengths,
        'times': times_arr,
        'velocities': vel_out,
        'radii': np.asarray(radii, dtype=float).reshape(-1),
    }
    if ids is not None:
        payload['ids'] = np.asarray([str(i) for i in ids])

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **payload)

    return {
        'n_obs': n_obs,
        'horizon': horizon,
        'lengths': lengths.tolist(),
        'duration_sec': float(times_arr.max()),
    }


def write_scene_config(path: str,
                       name: str,
                       occupancy_path: str,
                       occupancy_resolution: float,
                       occupancy_origin,
                       map_size,
                       start_position,
                       goal_position,
                       track_file: str,
                       n_obs: int,
                       radii,
                       start_yaw: float = None,
                       occupancy_cell_anchor: str = "center",
                       enable_map_boundary_obstacles: bool = False,
                       on_finish: str = "hold",
                       group: str = None,
                       source: dict = None) -> dict:
    """Write the scenario JSON that DWAT_v5_modular's map_creation consumes."""
    radii = np.asarray(radii, dtype=float).reshape(-1)

    scene = {
        'format_version': 1,
        'name': name,
        'source': source or {},

        'occupancy_path': occupancy_path,
        'occupancy_resolution': float(occupancy_resolution),
        'occupancy_origin': [float(v) for v in occupancy_origin],
        'occupancy_cell_anchor': occupancy_cell_anchor,
        'map_size': [int(v) for v in map_size],
        'enable_map_boundary_obstacles': bool(enable_map_boundary_obstacles),

        'start_position': [float(v) for v in start_position],
        'goal_position': [float(v) for v in goal_position],

        'dynamic_obstacles': [
            {
                'motion_type': 'trajectory',
                'track_file': track_file,
                'track_index': k,
                'radius': float(radii[k]),
                'start_time': 0.0,
                'on_finish': on_finish,
            }
            for k in range(n_obs)
        ],
    }
    if start_yaw is not None:
        scene['start_yaw'] = float(start_yaw)

    if group is not None:
        # Every scenario carries a group: it selects the configs/scenarios/
        # subdirectory the file belongs to and decides where the run output
        # lands under runs/.
        scene = {'group': group, **scene}

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(scene, f, indent=4)

    return scene

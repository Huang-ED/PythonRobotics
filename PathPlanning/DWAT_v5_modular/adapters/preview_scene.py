"""Render a scenario JSON so an imported scene can be checked by eye.

This verifies the *data* independently of the planner: if the map, the tracks
and the start/goal all land where the source dataset says they should, the
adapter did its job, whatever the planner then does with it.

Run from the repository root:
    python PathPlanning/DWAT_v5_modular/adapters/preview_scene.py \
        --scenario PathPlanning/DWAT_v5_modular/configs/scenarios/scene_export.json
"""

import argparse
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from PathPlanning.DWAT_v4_st.dwa_consistent import Config  # noqa: E402
from PathPlanning.DWAT_v5_modular.map_creation import create_map_context  # noqa: E402


def preview(scenario_path: str, out_path: str, plot_global_path: bool = True):
    config = Config()
    ctx = create_map_context(config=config, scenario_path=scenario_path)
    mm = ctx.map_manager

    fig, ax = plt.subplots(figsize=(9, 9))

    if mm.static_obstacles.shape[0] > 0:
        ax.plot(mm.static_obstacles[:, 0], mm.static_obstacles[:, 1],
                's', color='0.82', markersize=3, label='static (all)')
    if mm.boundary_obstacles.shape[0] > 0:
        ax.plot(mm.boundary_obstacles[:, 0], mm.boundary_obstacles[:, 1],
                's', color='0.25', markersize=3, label='static (rim, costed)')

    cmap = plt.get_cmap('tab10')
    for i, obs in enumerate(mm.dynamic_obstacles):
        colour = cmap(i % 10)
        if obs.motion_type == 'trajectory':
            traj = obs.trajectory
            ax.plot(traj[:, 0], traj[:, 1], '-', color=colour, linewidth=1.4, alpha=0.9)
            ax.plot(traj[0, 0], traj[0, 1], 'o', color=colour, markersize=5)
            ax.plot(traj[-1, 0], traj[-1, 1], 'x', color=colour, markersize=7)
        ax.add_patch(plt.Circle(obs.current_position, obs.radius,
                                color=colour, alpha=0.85, zorder=6))
        label = obs.track_id if obs.track_id else str(i)
        ax.annotate(label, obs.current_position, fontsize=7,
                    xytext=(3, 3), textcoords='offset points', zorder=7)

    if plot_global_path and ctx.road_map is not None and len(ctx.road_map) > 0:
        ax.plot(ctx.road_map[:, 0], ctx.road_map[:, 1], '-b', linewidth=1.6,
                label='Theta* global path')

    ax.plot(ctx.start[0], ctx.start[1], 'or', markersize=9, zorder=10, label='start')
    ax.plot(ctx.goal[0], ctx.goal[1], 'sr', markersize=9, zorder=10, label='goal')
    ax.arrow(ctx.start[0], ctx.start[1],
             3.0 * math.cos(ctx.start_yaw), 3.0 * math.sin(ctx.start_yaw),
             head_width=0.8, color='r', zorder=10)

    ax.set_aspect('equal')
    ax.set_xlim(-1, mm.map_size[0])
    ax.set_ylim(-1, mm.map_size[1])
    ax.grid(True, alpha=0.3)
    ax.set_title(os.path.basename(scenario_path))
    ax.legend(loc='upper right', fontsize=8)

    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f'map          : {mm.map_size}, {len(mm.static_obstacles)} occupied cells, '
          f'{len(mm.boundary_obstacles)} rim cells')
    print(f'obstacles    : {len(mm.dynamic_obstacles)}')
    print(f'start/goal   : {tuple(ctx.start)} -> {tuple(ctx.goal)}, '
          f'yaw {math.degrees(ctx.start_yaw):.1f} deg')
    print(f'wrote        : {out_path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scenario', required=True)
    parser.add_argument('--out', default=None)
    parser.add_argument('--no-global-path', action='store_true',
                        help='Skip Theta* plotting (faster on large maps).')
    args = parser.parse_args()

    out = args.out or os.path.splitext(os.path.basename(args.scenario))[0] + '_preview.png'
    preview(args.scenario, out, plot_global_path=not args.no_global_path)


if __name__ == '__main__':
    main()

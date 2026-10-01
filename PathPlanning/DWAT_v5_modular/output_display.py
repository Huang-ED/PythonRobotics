import json
import sys
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np

from ..DWAT_v4_st import dwa_consistent as dwa

from .models import MapContext, OutputArtifacts, RuntimeOptions, SimulationResult


# One Circle per occupied cell is fine for a 100x100 hand-authored map, but a
# harbour rasterised in metres has tens of thousands of them, and every artist
# is re-rasterised on every savefig -- which is what rendering a video does a
# few hundred times. Above this many cells, draw the occupancy grid as a single
# image instead. The circles are the cells inflated by robot_radius, so the
# image is the same information without the inflation.
#
# Deliberately above the largest hand-authored scenario (scene_export_100, 5284
# cells) so every pre-existing scenario keeps rendering exactly as before; only
# the metric harbour maps, which are an order of magnitude larger, switch over.
MAX_OBSTACLE_PATCHES = 8000


def _draw_static_map(map_manager: Any, config: Any) -> None:
    static = map_manager.static_obstacles
    boundary = map_manager.boundary_obstacles
    occupancy = getattr(map_manager, "occupancy", None)

    if occupancy is not None and static.shape[0] > MAX_OBSTACLE_PATCHES:
        res = float(map_manager.occupancy_resolution)
        ox, oy = map_manager.occupancy_origin
        plt.imshow(
            np.asarray(occupancy).T,
            origin="lower",
            extent=[ox, ox + occupancy.shape[0] * res,
                    oy, oy + occupancy.shape[1] * res],
            cmap="gray_r",
            vmin=0,
            vmax=1,
            interpolation="nearest",
            zorder=0,
        )
        return

    for (x_s, y_s) in static:
        plt.gca().add_patch(plt.Circle((x_s, y_s), config.robot_radius, color="darkgrey"))
    for (x_b, y_b) in boundary:
        plt.gca().add_patch(plt.Circle((x_b, y_b), config.robot_radius, color="k"))


class SimulationRenderer:
    def __init__(self, runtime: RuntimeOptions):
        self.runtime = runtime
        self.fig_dir: Optional[Path] = None
        self.i_fig = 0
        self.plt_elements = []

    def _save_frame(self) -> None:
        if self.runtime.save_animation_to_figs and self.fig_dir is not None:
            plt.savefig(
                str(self.fig_dir / f"frame_{self.i_fig}.png"),
                bbox_inches="tight",
                pad_inches=0.1,
            )
            self.i_fig += 1

    def initialize_scene(
        self,
        map_ctx: MapContext,
        config: Any,
        base_dir: Path,
        reuse_existing_figure: bool = False,
    ) -> None:
        if not self.runtime.show_animation:
            return

        if not reuse_existing_figure:
            plt.figure(figsize=(10, 10))

        if self.runtime.save_animation_to_figs:
            # base_dir is the run directory; frames get their own subfolder so
            # run.json and log_details.json stay visible beside thousands of PNGs.
            #
            # With frames_root set they go to a scratch directory instead. The
            # run does NOT remove it: deleting scratch is the caller's business,
            # not the planner's. See RuntimeOptions and purge_scratch_frames.py.
            root = str(getattr(self.runtime, "frames_root", "") or "").strip()
            if root:
                # The run directory's own name is <scenario>__<timestamp>, so
                # two configs started on one scenario in the same second share
                # it -- and wrote into ONE frames folder, overwriting each other
                # (2026-10-01, six AIS runs). Claim the folder atomically; if it
                # is taken, add a suffix. RUN_DIR says whose frames these are.
                parent = Path(root).expanduser()
                parent.mkdir(parents=True, exist_ok=True)
                n = 1
                while True:
                    name = base_dir.name if n == 1 else f"{base_dir.name}__{n}"
                    try:
                        (parent / name).mkdir()
                        break
                    except FileExistsError:
                        n += 1
                self.fig_dir = parent / name
                (self.fig_dir / "RUN_DIR").write_text(str(base_dir) + "\n", encoding="utf-8")
            else:
                self.fig_dir = base_dir / "frames"
                self.fig_dir.mkdir(parents=True, exist_ok=False)

        if not reuse_existing_figure:
            _draw_static_map(map_ctx.map_manager, config)

            plt.plot(map_ctx.start[0], map_ctx.start[1], "or", zorder=10)
            plt.plot(map_ctx.goal[0], map_ctx.goal[1], "sr", zorder=10)

        plt.plot(map_ctx.rx, map_ctx.ry, "-b")

        for obstacle in map_ctx.map_manager.dynamic_obstacles:
            circle = plt.Circle(obstacle.current_position, obstacle.radius, color="brown", zorder=10)
            plt.gca().add_patch(circle)
            self.plt_elements.append(circle)

        plt.grid(True)
        plt.axis("equal")
        origin = getattr(map_ctx.map_manager, "occupancy_origin", None)
        x_lo, y_lo = (-1, -1) if origin is None else (origin[0], origin[1])
        plt.xlim(x_lo, map_ctx.map_manager.map_size[0])
        plt.ylim(y_lo, map_ctx.map_manager.map_size[1])

        plt.gcf().canvas.mpl_connect(
            "key_release_event",
            lambda event: exit(0) if event.key == "escape" else None,
        )

        plt.pause(0.001)
        self._save_frame()

    def render_step(
        self,
        x: np.ndarray,
        trajectory: np.ndarray,
        predicted_trajectory: np.ndarray,
        candidate_trajectories: list,
        dwagoal: np.ndarray,
        dynamic_ob_pos: np.ndarray,
        dynamic_ob_vel: np.ndarray,
        map_manager: Any,
        config: Any,
    ) -> None:
        if not self.runtime.show_animation:
            return

        for ele in self.plt_elements:
            ele.remove()
        self.plt_elements = []

        for obstacle in map_manager.dynamic_obstacles:
            circle = plt.Circle(obstacle.current_position, obstacle.radius, color="brown", zorder=10)
            plt.gca().add_patch(circle)
            self.plt_elements.append(circle)

        if self.runtime.plot_predicted_dynamic_obstacles and len(dynamic_ob_pos) > 0:
            pred_time = np.arange(0.0, config.predict_time_obstacle + config.dt, config.dt)
            for ob_i in range(len(dynamic_ob_pos)):
                ob_pred_x = dynamic_ob_pos[ob_i, 0] + dynamic_ob_vel[ob_i, 0] * pred_time
                ob_pred_y = dynamic_ob_pos[ob_i, 1] + dynamic_ob_vel[ob_i, 1] * pred_time
                self.plt_elements.append(
                    plt.plot(ob_pred_x, ob_pred_y, "--", color="peru", linewidth=1.0, alpha=0.8)[0]
                )

        if self.runtime.plot_candidate_trajectories:
            for cand_traj in candidate_trajectories:
                self.plt_elements.append(
                    plt.plot(cand_traj[:, 0], cand_traj[:, 1], "-", color="deepskyblue", linewidth=0.8, alpha=0.25)[0]
                )

        config_plot = config
        self.plt_elements.append(plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], "-g", linewidth=2.0)[0])
        self.plt_elements.append(plt.plot(x[0], x[1], "xr")[0])
        self.plt_elements.extend(dwa.plot_robot(x[0], x[1], x[2], config_plot))
        self.plt_elements.extend(dwa.plot_arrow(x[0], x[1], x[2]))
        self.plt_elements.append(plt.plot(trajectory[:, 0], trajectory[:, 1], "-r")[0])
        self.plt_elements.append(plt.plot(dwagoal[0], dwagoal[1], "Db")[0])

        plt.pause(0.001)
        self._save_frame()

    def show(self) -> None:
        """Hold the final figure open -- but only when someone is there to see it.

        ``plt.show()`` on an interactive backend BLOCKS until a human closes the
        window. In a background batch there is no human, so the run sits at 100%
        complete forever: measured 2026-09-21, six runs finished their 1,515
        frames in 38 min each and then hung here for a further 10.1 hours.

        stdout not being a tty is the signal that nobody is watching. Rendering
        itself is unaffected -- it costs ~15-25% against a headless run, and the
        frames are already written by the time this is reached.
        """
        if not self.runtime.show_animation:
            return
        if not sys.stdout.isatty():
            plt.close("all")
            return
        plt.show()


def save_log(
    config: Any,
    runtime: RuntimeOptions,
    result: SimulationResult,
    run_dir: Path,
) -> Optional[Path]:
    """Write log_details.json into the run directory created by run_paths.

    A run that aborts on iteration 0 -- as the two "ego is overtaken" cases do
    at predict_time_obstacle = 30 s -- produces no log entries at all. Skipping
    the write there meant the failure existed only on stdout, so the case left
    no durable record that it had even run. An empty-but-failed run is still a
    result worth keeping, so the log is written whenever there is anything to
    say: entries, an error, or a collision.
    """
    if not (result.log_entries or result.error or result.collision_event):
        return None

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    details_filename = run_dir / "log_details.json"
    with details_filename.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {k: v for k, v in config.__dict__.items() if not k.startswith("_")},
                "log_entries": result.log_entries,
                "trajectory": result.trajectory.tolist(),
                "reached_goal": result.reached_goal,
                "collision_event": None
                if result.collision_event is None
                else {
                    "iteration": result.collision_event.iteration,
                    "obstacle_index": result.collision_event.obstacle_index,
                    "distance_to_center": result.collision_event.distance_to_center,
                    "robot_shape_info": result.collision_event.robot_shape_info,
                    "obstacle_position": result.collision_event.obstacle_position,
                    "obstacle_radius": result.collision_event.obstacle_radius,
                    "collision_threshold": result.collision_event.collision_threshold,
                },
                "error": result.error,
                "traceback": result.traceback_text,
            },
            f,
            indent=2,
        )

    return details_filename


def build_output_artifacts(renderer: Optional[SimulationRenderer], log_file: Optional[Path]) -> OutputArtifacts:
    return OutputArtifacts(fig_dir=None if renderer is None else renderer.fig_dir, log_file=log_file)

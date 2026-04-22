import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np

from PathPlanning.DWAT_v4_st import dwa_average_weighted_side as dwa

from .models import MapContext, OutputArtifacts, RuntimeOptions, SimulationResult


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
            self.fig_dir = base_dir / self.runtime.output_tag
            self.fig_dir.mkdir(exist_ok=False)

        if not reuse_existing_figure:
            if map_ctx.map_manager.static_obstacles.shape[0] > 0:
                for (x_s, y_s) in map_ctx.map_manager.static_obstacles:
                    circle = plt.Circle((x_s, y_s), config.robot_radius, color="darkgrey")
                    plt.gca().add_patch(circle)

            if map_ctx.map_manager.boundary_obstacles.shape[0] > 0:
                for (x_b, y_b) in map_ctx.map_manager.boundary_obstacles:
                    circle = plt.Circle((x_b, y_b), config.robot_radius, color="k")
                    plt.gca().add_patch(circle)

            plt.plot(map_ctx.start[0], map_ctx.start[1], "or", zorder=10)
            plt.plot(map_ctx.goal[0], map_ctx.goal[1], "sr", zorder=10)

        plt.plot(map_ctx.rx, map_ctx.ry, "-b")

        for obstacle in map_ctx.map_manager.dynamic_obstacles:
            circle = plt.Circle(obstacle.current_position, obstacle.radius, color="brown", zorder=10)
            plt.gca().add_patch(circle)
            self.plt_elements.append(circle)

        plt.grid(True)
        plt.axis("equal")
        plt.xlim(-1, map_ctx.map_manager.map_size[0])
        plt.ylim(-1, map_ctx.map_manager.map_size[1])

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
        if self.runtime.show_animation:
            plt.show()


def save_log(
    config: Any,
    runtime: RuntimeOptions,
    result: SimulationResult,
    logs_root: Path,
) -> Optional[Path]:
    if not result.log_entries:
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = logs_root / f"{runtime.output_tag}_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=False)

    details_filename = log_dir / "log_details.json"
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

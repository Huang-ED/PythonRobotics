import math
import traceback
from typing import Any, Optional

import numpy as np

from PathPlanning.DWAT_v4_st import dwa_average_weighted_side as dwa
from PathPlanning.DWAT_v4_st.dwa_average_weighted_side import line_circle_intersection

from .models import CollisionEvent, MapContext, RuntimeOptions, SimulationResult


def _choose_local_goal(
    road_map: np.ndarray,
    i_turning_point: int,
    current_xy: np.ndarray,
    dist_localgoal: float,
) -> np.ndarray:
    turning_point = road_map[i_turning_point]
    intersection_points = line_circle_intersection(
        road_map[i_turning_point - 1],
        road_map[i_turning_point],
        (current_xy[0], current_xy[1]),
        dist_localgoal,
    )

    valid_intersection_points = []
    dist_to_turning_global = math.hypot(current_xy[0] - turning_point[0], current_xy[1] - turning_point[1])
    for point in intersection_points:
        dist_intersection_turning = math.hypot(point[0] - turning_point[0], point[1] - turning_point[1])
        if dist_intersection_turning <= dist_to_turning_global + 1e-6:
            valid_intersection_points.append(point)

    if len(valid_intersection_points) == 0:
        if dist_to_turning_global <= dist_localgoal:
            return turning_point
        raise ValueError("No intersection points found and robot not within local-goal radius")

    if len(valid_intersection_points) == 1:
        return np.array(valid_intersection_points[0])

    dists = [math.hypot(p[0] - turning_point[0], p[1] - turning_point[1]) for p in valid_intersection_points]
    return np.array(valid_intersection_points[int(np.argmin(dists))])


def _build_collision_event(config: Any, x: np.ndarray, obstacle_index: int, obstacle_pos: np.ndarray, obstacle_radius: float, collision_distance: float, iteration: int) -> CollisionEvent:
    if config.robot_type == dwa.RobotType.rectangle:
        effective_radius = math.sqrt((config.robot_length / 2) ** 2 + (config.robot_width / 2) ** 2)
        robot_shape_info = f"rectangle (effective radius: {effective_radius:.3f}m)"
    else:
        effective_radius = config.robot_radius
        robot_shape_info = f"circle (radius: {effective_radius:.3f}m)"

    return CollisionEvent(
        iteration=iteration,
        obstacle_index=obstacle_index,
        distance_to_center=collision_distance,
        robot_shape_info=robot_shape_info,
        obstacle_position=[float(obstacle_pos[0]), float(obstacle_pos[1])],
        obstacle_radius=float(obstacle_radius),
        collision_threshold=float(effective_radius + obstacle_radius),
    )


def run_simulation(
    config: Any,
    map_ctx: MapContext,
    runtime: RuntimeOptions,
    renderer: Optional[Any] = None,
) -> SimulationResult:
    x = np.array([map_ctx.start[0], map_ctx.start[1], -math.pi / 8.0, 0.0, 0.0])
    trajectory = np.array(x)
    log_data = []
    iteration = 0
    reached_goal = False
    collision_event = None

    try:
        for i_turning_point, turning_point in enumerate(map_ctx.road_map):
            if i_turning_point == 0:
                continue

            collision_detected = False
            while True:
                dwagoal = _choose_local_goal(
                    road_map=map_ctx.road_map,
                    i_turning_point=i_turning_point,
                    current_xy=x[:2],
                    dist_localgoal=config.dist_localgoal,
                )

                map_ctx.map_manager.update_dynamic_obstacles(config.dt)

                static_ob = map_ctx.map_manager.get_static_obstacles()
                static_ob_radii = map_ctx.map_manager.get_static_obstacle_radii()
                dynamic_ob_pos = map_ctx.map_manager.get_dynamic_obstacles_pos()
                dynamic_ob_radii = map_ctx.map_manager.get_dynamic_obstacle_radii()
                dynamic_ob_vel = map_ctx.map_manager.get_dynamic_obstacles_vel()

                if runtime.plot_candidate_trajectories:
                    (
                        u,
                        predicted_trajectory,
                        dw,
                        to_goal_before,
                        speed_before,
                        static_ob_before,
                        dynamic_ob_before,
                        dyn_side_val,
                        dyn_direct_val,
                        to_goal_after,
                        speed_after,
                        static_ob_after,
                        dynamic_ob_after,
                        final_cost,
                        candidate_trajectories,
                    ) = dwa.dwa_control_merged(
                        x,
                        config,
                        dwagoal,
                        static_ob,
                        static_ob_radii,
                        dynamic_ob_pos,
                        dynamic_ob_radii,
                        dynamic_ob_vel,
                        return_candidates=True,
                        candidate_stride=runtime.candidate_plot_stride,
                        candidate_max_count=runtime.candidate_plot_max_count,
                    )
                else:
                    (
                        u,
                        predicted_trajectory,
                        dw,
                        to_goal_before,
                        speed_before,
                        static_ob_before,
                        dynamic_ob_before,
                        dyn_side_val,
                        dyn_direct_val,
                        to_goal_after,
                        speed_after,
                        static_ob_after,
                        dynamic_ob_after,
                        final_cost,
                    ) = dwa.dwa_control_merged(
                        x,
                        config,
                        dwagoal,
                        static_ob,
                        static_ob_radii,
                        dynamic_ob_pos,
                        dynamic_ob_radii,
                        dynamic_ob_vel,
                    )
                    candidate_trajectories = []

                log_data.append(
                    {
                        "iteration": iteration,
                        "chosen_v": float(u[0]),
                        "chosen_omega": float(u[1]),
                        "local_goal_x": float(dwagoal[0]),
                        "local_goal_y": float(dwagoal[1]),
                        "final_cost": float(final_cost),
                        "to_goal_cost_before": float(to_goal_before),
                        "speed_cost_before": float(speed_before),
                        "static_ob_cost_before": float(static_ob_before),
                        "dynamic_ob_cost_before": float(dynamic_ob_before),
                        "dynamic_ob_side_cost": float(dyn_side_val),
                        "dynamic_ob_direct_cost": float(dyn_direct_val),
                        "to_goal_cost_after": float(to_goal_after),
                        "speed_cost_after": float(speed_after),
                        "static_ob_after": float(static_ob_after),
                        "dynamic_ob_after": float(dynamic_ob_after),
                        "dynamic_window": [float(dw[0]), float(dw[1]), float(dw[2]), float(dw[3])],
                        "dynamic_obstacles_pos": dynamic_ob_pos.tolist() if len(dynamic_ob_pos) > 0 else [],
                    }
                )
                iteration += 1

                x = dwa.motion(x, u, config.dt)
                trajectory = np.vstack((trajectory, x))

                all_current_ob = map_ctx.map_manager.get_current_obstacles()
                all_current_ob_radii = map_ctx.map_manager.get_obstacle_radii()

                collision_detected, obstacle_index, collision_distance = (
                    dwa.check_collision_at_current_position_circle_approximation(
                        x,
                        all_current_ob,
                        all_current_ob_radii,
                        config,
                    )
                )

                if collision_detected:
                    obstacle_pos = all_current_ob[obstacle_index]
                    obstacle_radius = all_current_ob_radii[obstacle_index]
                    collision_event = _build_collision_event(
                        config=config,
                        x=x,
                        obstacle_index=obstacle_index,
                        obstacle_pos=obstacle_pos,
                        obstacle_radius=obstacle_radius,
                        collision_distance=collision_distance,
                        iteration=iteration - 1,
                    )

                if renderer is not None:
                    renderer.render_step(
                        x=x,
                        trajectory=trajectory,
                        predicted_trajectory=predicted_trajectory,
                        candidate_trajectories=candidate_trajectories,
                        dwagoal=dwagoal,
                        dynamic_ob_pos=dynamic_ob_pos,
                        dynamic_ob_vel=dynamic_ob_vel,
                        map_manager=map_ctx.map_manager,
                        config=config,
                    )

                dist_to_turning = math.hypot(x[0] - turning_point[0], x[1] - turning_point[1])
                if i_turning_point == len(map_ctx.road_map) - 1:
                    if dist_to_turning <= config.catch_goal_dist:
                        reached_goal = True
                        break
                else:
                    if dist_to_turning <= config.catch_turning_point_dist:
                        break

                if collision_detected:
                    break

            if collision_detected:
                break

        return SimulationResult(
            trajectory=trajectory,
            log_entries=log_data,
            reached_goal=reached_goal,
            collision_event=collision_event,
        )

    except KeyboardInterrupt:
        return SimulationResult(
            trajectory=trajectory,
            log_entries=log_data,
            reached_goal=False,
            collision_event=collision_event,
            error="Simulation manually stopped.",
        )
    except Exception as exc:
        return SimulationResult(
            trajectory=trajectory,
            log_entries=log_data,
            reached_goal=False,
            collision_event=collision_event,
            error=str(exc),
            traceback_text=traceback.format_exc(),
        )

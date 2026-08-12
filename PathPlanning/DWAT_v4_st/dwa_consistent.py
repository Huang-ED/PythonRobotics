"""DWA variant that evaluates every obstacle with the same spatiotemporal cost.

Static obstacles are handled as zero-velocity dynamic obstacles so the objective
remains consistent across all obstacle sources.
"""

import math
import traceback
from typing import Any, Optional

import numpy as np

from PathPlanning.DWAT_v4_st import dwa_average_weighted_side as base

Config = base.Config
RobotType = base.RobotType
line_circle_intersection = base.line_circle_intersection
motion = base.motion
calc_dynamic_window = base.calc_dynamic_window
predict_trajectory_to_goal = base.predict_trajectory_to_goal
predict_trajectory_obstacle = base.predict_trajectory_obstacle
closest_obstacle_on_curve = base.closest_obstacle_on_curve
calc_trajectory_clearance_and_collision = base.calc_trajectory_clearance_and_collision
calc_to_goal_cost = base.calc_to_goal_cost
plot_arrow = base.plot_arrow
plot_robot = base.plot_robot
check_collision_at_current_position_circle_approximation = (
    base.check_collision_at_current_position_circle_approximation
)
filter_obstacles_by_direction = base.filter_obstacles_by_direction


def _combine_obstacles(
    static_ob: Optional[np.ndarray],
    static_ob_radii,
    dynamic_ob_pos: Optional[np.ndarray],
    dynamic_ob_radii,
    dynamic_ob_vel: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    combined_pos = []
    combined_radii = []
    combined_vel = []

    if static_ob is not None and static_ob.shape[0] > 0:
        combined_pos.append(np.asarray(static_ob, dtype=float))
        combined_radii.extend(list(static_ob_radii))
        combined_vel.append(np.zeros((static_ob.shape[0], 2), dtype=float))

    if dynamic_ob_pos is not None and dynamic_ob_pos.shape[0] > 0:
        combined_pos.append(np.asarray(dynamic_ob_pos, dtype=float))
        combined_radii.extend(list(dynamic_ob_radii))
        if dynamic_ob_vel is None:
            combined_vel.append(np.zeros((dynamic_ob_pos.shape[0], 2), dtype=float))
        else:
            combined_vel.append(np.asarray(dynamic_ob_vel, dtype=float))

    if not combined_pos:
        return np.empty((0, 2)), np.empty((0,), dtype=float), np.empty((0, 2), dtype=float)

    return (
        np.vstack(combined_pos),
        np.asarray(combined_radii, dtype=float),
        np.vstack(combined_vel),
    )


def dwa_control_merged(
    x,
    config,
    goal,
    static_ob,
    static_ob_radii,
    dynamic_ob_pos,
    dynamic_ob_radii,
    dynamic_ob_vel,
    return_candidates=False,
    candidate_stride=1,
    candidate_max_count=None,
):
    """DWA control wrapper that keeps static and dynamic obstacles on one model."""
    combined_ob_pos, combined_ob_radii, combined_ob_vel = _combine_obstacles(
        static_ob=static_ob,
        static_ob_radii=static_ob_radii,
        dynamic_ob_pos=dynamic_ob_pos,
        dynamic_ob_radii=dynamic_ob_radii,
        dynamic_ob_vel=dynamic_ob_vel,
    )

    filtered_ob_pos, filtered_ob_radii, filtered_ob_vel = filter_obstacles_by_direction(
        x,
        combined_ob_pos,
        combined_ob_radii,
        combined_ob_vel,
        max_angle=config.obstacle_max_angle,
    )

    return calc_control_and_trajectory_merged(
        x,
        calc_dynamic_window(x, config),
        config,
        goal,
        None,
        None,
        filtered_ob_pos,
        filtered_ob_radii,
        filtered_ob_vel,
        return_candidates=return_candidates,
        candidate_stride=candidate_stride,
        candidate_max_count=candidate_max_count,
    )


def calc_control_and_trajectory_merged(
    x,
    dw,
    config,
    goal,
    static_ob,
    static_ob_radii,
    dynamic_ob_pos,
    dynamic_ob_radii,
    dynamic_ob_vel,
    return_candidates=False,
    candidate_stride=1,
    candidate_max_count=None,
):
    """Evaluate all candidate controls using a single obstacle objective."""
    combined_ob_pos, combined_ob_radii, combined_ob_vel = _combine_obstacles(
        static_ob=static_ob,
        static_ob_radii=static_ob_radii,
        dynamic_ob_pos=dynamic_ob_pos,
        dynamic_ob_radii=dynamic_ob_radii,
        dynamic_ob_vel=dynamic_ob_vel,
    )

    has_all_ob = combined_ob_pos.shape[0] > 0
    all_ob_radii = combined_ob_radii if has_all_ob else np.empty((0,), dtype=float)

    min_cost = float("inf")
    best_u = [0.0, 0.0]
    best_trajectory = np.array([x])
    best_index = -1

    to_goal_costs = []
    speed_costs = []
    static_ob_costs = []
    dynamic_ob_costs = []
    side_cost_components = []
    direct_cost_components = []

    trajectories = []
    controls = []

    for v in np.arange(dw[0], dw[1] + 1e-6, config.v_resolution):
        for y in np.arange(dw[2], dw[3] + 1e-6, config.yaw_rate_resolution):
            dist_all = float("inf")
            if has_all_ob:
                dist_all, _ = closest_obstacle_on_curve(
                    x.copy(),
                    combined_ob_pos,
                    all_ob_radii,
                    v,
                    y,
                    config,
                )

            if v**2 + 2 * config.max_accel * v * config.dt > 2 * config.max_accel * dist_all:
                continue

            trajectory_to_goal = predict_trajectory_to_goal(x.copy(), v, y, config)
            trajectory_for_obstacle = predict_trajectory_obstacle(x.copy(), v, y, config)

            to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(trajectory_to_goal, goal)
            speed_cost = config.speed_cost_gain * (config.max_speed - trajectory_to_goal[-1, 3])

            obstacle_cost = 0.0
            current_side_comp = 0.0
            current_direct_comp = 0.0

            if has_all_ob:
                d_side_arr, is_collision = calc_trajectory_clearance_and_collision(
                    trajectory_for_obstacle,
                    combined_ob_pos,
                    combined_ob_radii,
                    combined_ob_vel,
                    config,
                )

                if is_collision:
                    obstacle_cost = float("inf")
                    current_side_comp = float("inf")
                    current_direct_comp = float("inf")
                else:
                    side_norm = np.maximum(config.max_side_weight_dist, 1e-6)

                    if config.side_clearance_barrier_type == "exponential":
                        decay_rate = config.side_clearance_gaussian_decay / (side_norm**2)
                        cost_side_vec = config.max_side_weight_dist * np.exp(-decay_rate * (d_side_arr**2))
                    else:
                        side_closeness = np.maximum(0.0, 1.0 - (d_side_arr / side_norm))
                        cost_side_vec = config.max_side_weight_dist * (side_closeness**2)

                    times = np.arange(trajectory_for_obstacle.shape[0]) * config.dt
                    cost_direct_vec = np.exp(-max(config.time_decay_lambda, 0.0) * times)
                    compound_costs = cost_side_vec * cost_direct_vec

                    if len(compound_costs) > 0:
                        weight_sum = np.sum(cost_direct_vec)
                        if weight_sum > 1e-12:
                            obstacle_cost = np.sum(compound_costs) / weight_sum
                            current_side_comp = np.sum(cost_side_vec * cost_direct_vec) / weight_sum
                        else:
                            obstacle_cost = np.mean(compound_costs)
                            current_side_comp = np.mean(cost_side_vec)

                        current_direct_comp = np.mean(cost_direct_vec)

                    obstacle_cost = config.side_cost_gain * obstacle_cost

            final_cost = to_goal_cost + speed_cost + obstacle_cost

            to_goal_costs.append(to_goal_cost)
            speed_costs.append(speed_cost)
            static_ob_costs.append(0.0)
            dynamic_ob_costs.append(obstacle_cost)
            side_cost_components.append(current_side_comp)
            direct_cost_components.append(current_direct_comp)

            trajectories.append(trajectory_for_obstacle)
            controls.append([v, y])

            if final_cost < min_cost:
                min_cost = final_cost
                best_index = len(controls) - 1
                best_u = controls[best_index]
                best_trajectory = trajectories[best_index]

    if len(to_goal_costs) == 0:
        raise ValueError("No admissible (v, ω) pairs found in dynamic window")

    if best_index == -1:
        raise ValueError("No best trajectory found")

    to_goal_before = to_goal_costs[best_index]
    speed_before = speed_costs[best_index]
    static_ob_before = static_ob_costs[best_index]
    dynamic_ob_before = dynamic_ob_costs[best_index]
    best_side_component = side_cost_components[best_index]
    best_direct_component = direct_cost_components[best_index]

    to_goal_after = to_goal_before
    speed_after = speed_before
    static_ob_after = static_ob_before
    dynamic_ob_after = dynamic_ob_before

    if abs(best_u[0]) < config.robot_stuck_flag_cons and abs(x[3]) < config.robot_stuck_flag_cons:
        best_u[1] = -config.max_delta_yaw_rate

    if return_candidates:
        stride = max(1, int(candidate_stride))
        candidate_trajectories = trajectories[::stride]

        if candidate_max_count is not None and candidate_max_count > 0 and len(candidate_trajectories) > candidate_max_count:
            idx = np.linspace(0, len(candidate_trajectories) - 1, candidate_max_count, dtype=int)
            candidate_trajectories = [candidate_trajectories[i] for i in idx]

        return (
            best_u,
            best_trajectory,
            dw,
            to_goal_before,
            speed_before,
            static_ob_before,
            dynamic_ob_before,
            best_side_component,
            best_direct_component,
            to_goal_after,
            speed_after,
            static_ob_after,
            dynamic_ob_after,
            min_cost,
            candidate_trajectories,
        )

    return (
        best_u,
        best_trajectory,
        dw,
        to_goal_before,
        speed_before,
        static_ob_before,
        dynamic_ob_before,
        best_side_component,
        best_direct_component,
        to_goal_after,
        speed_after,
        static_ob_after,
        dynamic_ob_after,
        min_cost,
    )


def calc_trajectory_clearance_and_collision(trajectory, ob_pos, ob_radii, ob_vel, config):
    """Proxy to the legacy clearance routine for direct compatibility."""
    return base.calc_trajectory_clearance_and_collision(trajectory, ob_pos, ob_radii, ob_vel, config)


def calc_to_goal_cost(trajectory, goal):
    """Proxy to the legacy goal objective for direct compatibility."""
    return base.calc_to_goal_cost(trajectory, goal)


def plot_arrow(x, y, yaw, length=0.5, width=0.1):  # pragma: no cover
    return base.plot_arrow(x, y, yaw, length=length, width=width)


def plot_robot(x, y, yaw, config):  # pragma: no cover
    return base.plot_robot(x, y, yaw, config)


def check_collision_at_current_position_circle_approximation(x, ob, ob_radii, config):
    """Proxy to the legacy collision checker for direct compatibility."""
    return base.check_collision_at_current_position_circle_approximation(x, ob, ob_radii, config)

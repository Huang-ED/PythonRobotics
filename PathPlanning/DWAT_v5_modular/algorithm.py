import importlib
import math
import traceback
from typing import Any, Optional

import time as _time

import numpy as np

_dwa_module = importlib.import_module("PathPlanning.DWAT_v4_st.dwa_average_weighted_side")
line_circle_intersection = _dwa_module.line_circle_intersection

from .models import CollisionEvent, MapContext, RuntimeOptions, SimulationResult


def _describe_stall(trajectory, window, loop_ratio=2.0):
    """Is the ego making headway, or going nowhere?

    Deliberately NOT distance-to-goal: with static obstacles the ego often has
    to move AWAY from the goal for a while (the v3crowd routes do exactly that),
    so a goal-relative metric cries wolf on correct behaviour.

    Instead compare, over a trailing window, how far the ego TRAVELLED against
    how large a REGION it stayed within. That is scale-free, needs no reference
    point, and does not care which way the goal lies:

        travelled / extent  ~1    running (straight or gently curving)
                            >2.0  going round -- covering ground inside a box
                                  it never leaves, i.e. circling

    Net displacement over the window is NOT the right measure. A closed loop
    only returns ``net ~ 0`` at a whole number of laps; at 1.5 laps the ego ends
    diametrically opposite and ``net/travelled`` reads 0.21, which passes any
    sane threshold. Extent has no such blind spot.

    This is what the real-time render was being watched for by eye.
    """
    if window <= 0 or trajectory.shape[0] < window + 1:
        return None

    recent = trajectory[-(window + 1):, :2]
    travelled = float(np.hypot(*np.diff(recent, axis=0).T).sum())
    # The LARGER SIDE of the bounding box, not its diagonal. The diagonal
    # inflates extent by sqrt(2) for anything roughly circular, which is
    # exactly the case being detected: a full lap reads 2.22 by diagonal
    # (passes) but 3.14 by larger side (fires).
    extent = float(max(np.ptp(recent[:, 0]), np.ptp(recent[:, 1])))

    if travelled < 1e-6:
        return f"STOPPED: no motion in {window} steps"
    if extent > 1e-9 and travelled / extent > loop_ratio:
        return (f"CIRCLING: travelled {travelled:.1f} u in {window} steps but never "
                f"left a {extent:.1f} u region (ratio {travelled / extent:.1f})")
    return None


def _rejoin_point(
    seg_start: np.ndarray,
    seg_end: np.ndarray,
    current_xy: np.ndarray,
    dist_localgoal: float,
) -> np.ndarray:
    """Aim back at a leg the ego has strayed off by more than the look-ahead.

    Project the ego onto the leg, then advance `dist_localgoal` along it from
    that projection, clamped to the leg's end. The result is always on the leg
    and always ahead of the ego, which is what the look-ahead circle would have
    given had it still reached.
    """
    seg = seg_end - seg_start
    length = float(np.hypot(seg[0], seg[1]))
    if length < 1e-9:
        return seg_end
    unit = seg / length
    # Arc-length of the ego's projection along the leg, clamped to it.
    s = float(np.dot(current_xy - seg_start, unit))
    s = min(max(s, 0.0), length)
    return seg_start + min(s + dist_localgoal, length) * unit


def _choose_local_goal(
    road_map: np.ndarray,
    i_turning_point: int,
    current_xy: np.ndarray,
    dist_localgoal: float,
    recovery: str = "abort",
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
        if recovery == "rejoin":
            # Open problem 15: the ego has evaded by more than dist_localgoal,
            # so the look-ahead circle no longer meets the leg. Steer back to
            # it instead of aborting a run that has not collided with anything.
            return _rejoin_point(
                road_map[i_turning_point - 1], turning_point, current_xy, dist_localgoal
            )
        raise ValueError("No intersection points found and robot not within local-goal radius")

    if len(valid_intersection_points) == 1:
        return np.array(valid_intersection_points[0])

    dists = [math.hypot(p[0] - turning_point[0], p[1] - turning_point[1]) for p in valid_intersection_points]
    return np.array(valid_intersection_points[int(np.argmin(dists))])


def _build_collision_event(
    config: Any,
    dwa_mod: Any,
    x: np.ndarray,
    obstacle_index: int,
    obstacle_pos: np.ndarray,
    obstacle_radius: float,
    collision_distance: float,
    iteration: int,
) -> CollisionEvent:
    if config.robot_type == dwa_mod.RobotType.rectangle:
        # Honour robot_collision_shape, like the check that fired to get here.
        # DETECTION is already capsule-aware (see
        # check_collision_at_current_position_circle_approximation); only the
        # threshold RECORDED in the event was still the disc, so a capsule run
        # logged a collision against a radius 3x larger than the one that
        # actually triggered it. Reporting only -- no decision depends on this.
        if str(getattr(config, "robot_collision_shape", "disc")) == "capsule":
            effective_radius = config.robot_width / 2.0
            robot_shape_info = f"capsule (half-beam: {effective_radius:.3f}m)"
        else:
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
    dwa_module: str = "PathPlanning.DWAT_v4_st.dwa_average_weighted_side",
) -> SimulationResult:
    dwa_mod = importlib.import_module(dwa_module)
    x = np.array([map_ctx.start[0], map_ctx.start[1], map_ctx.start_yaw, 0.0, 0.0])
    trajectory = np.array(x)
    log_data = []
    iteration = 0

    # --- progress reporting -------------------------------------------------
    # The control loop used to print nothing, so a run was opaque until it
    # ended: a livelock burned 11h49m before anyone could see the ego had not
    # moved in 4,000 iterations, and two processes died leaving no record of
    # where they stopped (log_details.json is written only at the end).
    # stdout is flushed as it goes, so the last line survives a killed process.
    _progress_every = float(getattr(runtime, "progress_every_s", 0.0) or 0.0)
    _stall_window = int(getattr(runtime, "stall_window_steps", 0) or 0)
    _t_start = _time.perf_counter()
    _t_next = _t_start + _progress_every
    _stall_warned = False
    reached_goal = False
    collision_event = None

    try:
        for i_turning_point, turning_point in enumerate(map_ctx.road_map):
            if i_turning_point == 0:
                continue

            collision_detected = False
            max_steps = int(getattr(runtime, "max_sim_steps", 0) or 0)
            while True:
                if max_steps and iteration >= max_steps:
                    raise ValueError(
                        f"Simulation did not converge within max_sim_steps={max_steps} "
                        f"(state x={x[0]:.3f} y={x[1]:.3f} v={x[3]:.4f} omega={x[4]:.5f}); "
                        "the ego reached neither the goal, a collision, nor a "
                        "local-goal divergence -- most likely stalled just outside "
                        "catch_goal_dist"
                    )
                dwagoal = _choose_local_goal(
                    road_map=map_ctx.road_map,
                    i_turning_point=i_turning_point,
                    current_xy=x[:2],
                    dist_localgoal=config.dist_localgoal,
                    recovery=str(getattr(config, "local_goal_recovery", "abort")),
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
                    ) = dwa_mod.dwa_control_merged(
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
                    ) = dwa_mod.dwa_control_merged(
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

                x = dwa_mod.motion(x, u, config.dt)
                trajectory = np.vstack((trajectory, x))

                if _progress_every > 0.0:
                    _now = _time.perf_counter()
                    if _now >= _t_next:
                        _t_next = _now + _progress_every
                        _el = _now - _t_start
                        _msg = (
                            f"[progress] step {iteration} t={iteration * config.dt:.1f}s "
                            f"pos=({x[0]:.2f}, {x[1]:.2f}) v={x[3]:.3f} omega={x[4]:+.4f} "
                            f"| leg {i_turning_point}/{len(map_ctx.road_map) - 1} "
                            f"{_el / 60.0:.1f} min, {iteration / max(_el, 1e-9):.2f} steps/s"
                        )
                        _stall = _describe_stall(trajectory, _stall_window)
                        if _stall:
                            _msg += f"  <-- {_stall}"
                            _stall_warned = True
                        print(_msg, flush=True)

                all_current_ob = map_ctx.map_manager.get_current_obstacles()
                all_current_ob_radii = map_ctx.map_manager.get_obstacle_radii()

                collision_detected, obstacle_index, collision_distance = (
                    dwa_mod.check_collision_at_current_position_circle_approximation(
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
                        dwa_mod=dwa_mod,
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

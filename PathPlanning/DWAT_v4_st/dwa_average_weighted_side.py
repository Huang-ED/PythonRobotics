"""

Mobile robot motion planning sample with Dynamic Window Approach

author: Atsushi Sakai (@Atsushi_twi), Göktuğ Karakaşlı

Modified by: Huang Erdong (@Huang-ED)
In this version of codes, 
    the methodology proposed in the original DWA paper is implemented.
Compared to dynamic_window_approach_paper.py,
    robot are considered in rectangle shape when indicated,
    and the obstacle is considered as a circle with radius.

This "merged" file combines two features:
1. Split obstacle cost functions (direct dist for static, side dist for dynamic)
2. Support for dynamic obstacles with individual, varying radii.

"""

import os
import math
from enum import Enum
from typing import List

import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings("error", category=RuntimeWarning)

show_animation = True
save_animation_to_figs = False
save_costs_fig = False

if save_costs_fig:
    to_goal_cost_list, speed_cost_list, ob_cost_list = [], [], []


class RobotType(Enum):
    circle = 0
    rectangle = 1


class Config:
    """
    simulation parameter class
    """

    def __init__(self):
        # robot parameter
        self.max_speed = 0.5  # [m/s]
        self.min_speed = 0.0  # [m/s]
        self.max_yaw_rate = 40.0 * math.pi / 180.0  # [rad/s]
        self.max_accel = 0.2  # [m/ss]
        self.max_delta_yaw_rate = 40.0 * math.pi / 180.0  # [rad/ss]
        self.v_resolution = 0.01  # [m/s]
        self.yaw_rate_resolution = 0.1 * math.pi / 180.0  # [rad/s]
        self.dt = 0.1  # [s] Time tick for motion prediction

        self.to_goal_cost_gain = 0.6
        self.speed_cost_gain = 1.0
        self.obstacle_cost_gain = 0.05  # Gain for static obstacles (direct dist)
        self.side_cost_gain = 0.6      # Gain for dynamic obstacles (side dist)

        self.max_obstacle_cost_dist = 8.0  # [m] max distance for static obstacle cost calculation
        self.max_side_weight_dist = 3.0      # [m] max distance for dynamic obstacle side cost calculation
        self.time_decay_lambda = 0.1  # [1/s] exponential time-discount for dynamic obstacle cost

        # Side clearance barrier function configuration
        self.side_clearance_barrier_type = "exponential"  # "exponential" or "quadratic"
        self.side_clearance_gaussian_decay = 3.0  # Decay parameter λ for exponential: S = A*exp(-λ*d²)

        # How the per-point compound costs (side_cost * time_cost) are reduced to
        # one scalar along the prediction horizon:
        #   "mean" -> time-weighted average (default, horizon-length invariant)
        #   "max"  -> worst single (discounted) point along the horizon
        # An unnormalized sum is deliberately not offered: it equals "mean" with
        # side_cost_gain scaled by sum(time_cost), i.e. a gain change, not a new cost.
        self.dynamic_ob_cost_aggregation = "mean"

        self.predict_time_to_goal = 1.0  # [s]
        self.predict_time_obstacle = 30.0  # [s]

        # --- v20.7.9 prototype knobs -------------------------------------
        # Defaults below reproduce the classic behaviour exactly; only the
        # dwa_ramped module reads them. See DWAT_v4_st/dwa_ramped.py.
        # Seconds of acceleration authority used to size the v window.
        # 0.0 -> classic one-dt dynamic window.
        self.v_window_ramp_time = 0.0  # [s]
        # Same idea for the yaw axis. 0.0 -> classic one-dt yaw window.
        # With a one-dt window you cannot match both a ship's angular
        # ACCELERATION and a usable window width at a fine dt: a realistic
        # max_delta_yaw_rate leaves a window a fraction of a percent wide.
        # Sizing the window from a ramp time decouples the two.
        self.omega_window_ramp_time = 0.0  # [s]
        # Treat a sampled (v, omega) as a setpoint approached at max_accel /
        # max_delta_yaw_rate, instead of a velocity applied instantly.
        self.ramped_rollout = False
        # What to do when every candidate that PASSED admissibility scores inf
        # -- i.e. Guard B, "No best trajectory found":
        # "raise" -> ValueError (classic), "least_bad" -> keep the safest one.
        #
        # Named for the guard it actually catches. It was once called
        # "no_admissible_fallback", which collided with Guard A's message
        # ("No admissible (v, omega) pairs") -- a guard it does NOT catch and
        # never did. Guard A fires when admissibility rejects everything, and
        # raises unconditionally regardless of this setting. The old name is
        # still accepted as an alias; see the property below.
        self.all_candidates_collide_fallback = "raise"

        self.obstacle_max_angle = np.pi / 180 * 180  # [rad] max angle to consider obstacles in front

        self.enable_map_boundary_obstacles = False  # add obstacles along the outer map border

        self.dist_localgoal = 7.0  # [m] distance to local goal
        self.catch_goal_dist = 0.5  # [m] goal radius
        self.catch_turning_point_dist = self.dist_localgoal  # [m] local goal radius

        self.robot_stuck_flag_cons = 0.001  # constant to prevent robot stucked
        self.robot_type = RobotType.rectangle
        self.robot_radius = 0.5  # [m] for collision check
        # if robot_type == RobotType.circle
        # Also used to check if goal is reached in both types
        self.obstacle_radius = 0.5  # [m] default radius for static obstacles

        # if robot_type == RobotType.rectangle
        self.robot_width = 0.5  # [m] for collision check
        self.robot_length = 1.2  # [m] for collision check

        # How a rectangular ego is modelled when checking dynamic obstacles:
        #   "disc"    -> circumscribed circle, radius sqrt(L^2 + W^2)/2 (historical)
        #   "capsule" -> oriented segment of length (L - W), radius W/2
        # "disc" is the default so every existing result stays reproducible; it
        # is only badly wrong for long thin hulls. See calc_trajectory_clearance
        # _and_collision.
        self.robot_collision_shape = "disc"


    @property
    def no_admissible_fallback(self):
        """Deprecated alias for ``all_candidates_collide_fallback``.

        Kept so that config JSONs written before the rename keep loading:
        ``main.apply_algorithm_overrides`` gates on ``hasattr`` and then
        ``setattr``, both of which this property satisfies.

        Being a property, it is absent from ``Config.__dict__`` and therefore
        from the config block that ``output_display`` writes into every
        ``run.json`` / ``log_details.json``. Logs written after the rename
        carry ``all_candidates_collide_fallback``; logs written before it carry
        ``no_admissible_fallback``. Both name the same setting.
        """
        return self.all_candidates_collide_fallback

    @no_admissible_fallback.setter
    def no_admissible_fallback(self, value):
        self.all_candidates_collide_fallback = value

    @property
    def robot_type(self):
        return self._robot_type

    @robot_type.setter
    def robot_type(self, value):
        if not isinstance(value, RobotType):
            raise TypeError("robot_type must be an instance of RobotType")
        self._robot_type = value


def line_circle_intersection(line_endpoint_1, line_endpoint_2, center, r):
    x1, y1 = line_endpoint_1
    x2, y2 = line_endpoint_2
    x0, y0 = center

    # Vector representation of the line segment
    dx, dy = x2 - x1, y2 - y1
    fx, fy = x1 - x0, y1 - y0
    
    # Quadratic coefficients
    a = dx**2 + dy**2
    b = 2 * (fx * dx + fy * dy)
    c = fx**2 + fy**2 - r**2

    # Discriminant
    discriminant = b**2 - 4 * a * c

    # Check if the discriminant is effectively zero
    if abs(discriminant) < 1e-9:
        discriminant = 0

    # No intersection if discriminant is negative
    if discriminant < 0:
        return []  # No intersections

    # If discriminant is zero or positive, calculate the solutions
    discriminant = np.sqrt(discriminant)
    t1 = (-b + discriminant) / (2 * a)
    t2 = (-b - discriminant) / (2 * a)

    # Collect the intersection points that lie within the line segment
    intersection_points = []
    for t in [t1, t2]:
        if 0 <= t <= 1:
            # Point of intersection
            ix = x1 + t * dx
            iy = y1 + t * dy
            intersection_points.append((ix, iy))

    return intersection_points


def dwa_control_merged(x, config, goal,
                       static_ob, static_ob_radii,
                       dynamic_ob_pos, dynamic_ob_radii,
                       dynamic_ob_vel,
                       return_candidates=False,
                       candidate_stride=1,
                       candidate_max_count=None): # <--- New Argument
    """
    DWA control function with spatiotemporal dynamic obstacle handling.
    """
    dw = calc_dynamic_window(x, config)
    
    # Filter positions, radii, AND velocities
    dyn_ob_pos_filt, dyn_ob_radii_filt, dyn_ob_vel_filt = filter_obstacles_by_direction(
        x, dynamic_ob_pos, dynamic_ob_radii, dynamic_ob_vel, max_angle=config.obstacle_max_angle
    )

    result = calc_control_and_trajectory_merged(
        x, dw, config, goal,
        static_ob, static_ob_radii,
        dyn_ob_pos_filt, dyn_ob_radii_filt,
        dyn_ob_vel_filt,  # <--- Pass filtered velocities
        return_candidates=return_candidates,
        candidate_stride=candidate_stride,
        candidate_max_count=candidate_max_count,
    )

    if return_candidates:
        (u, trajectory, dw,
         to_goal_before, speed_before, ob_before, dynamic_ob_before,
         dyn_side_component, dyn_direct_component,
         to_goal_after, speed_after, ob_after, dynamic_ob_after,
         final_cost, candidate_trajectories) = result

        return (u, trajectory, dw,
                to_goal_before, speed_before, ob_before, dynamic_ob_before,
                dyn_side_component, dyn_direct_component,
                to_goal_after, speed_after, ob_after, dynamic_ob_after,
                final_cost, candidate_trajectories)

    (u, trajectory, dw,
     to_goal_before, speed_before, ob_before, dynamic_ob_before,
     dyn_side_component, dyn_direct_component,
     to_goal_after, speed_after, ob_after, dynamic_ob_after,
     final_cost) = result

    return (u, trajectory, dw,
            to_goal_before, speed_before, ob_before, dynamic_ob_before,
            dyn_side_component, dyn_direct_component,
            to_goal_after, speed_after, ob_after, dynamic_ob_after,
            final_cost)


def filter_obstacles_by_direction(current_state, obstacles, obstacle_radii, obstacle_vel, max_angle=np.pi/2):
    """
    Filter obstacles to remove only those that are safely behind and moving away.
    """
    x, y, yaw = current_state[0], current_state[1], current_state[2]
    filtered_obstacles = []
    filtered_radii = []
    filtered_vel = [] 
    
    # Robot velocity vector (approximate direction)
    # Note: We don't strictly need robot speed here, just heading for the angle check,
    # but strictly speaking, we should know if we are faster than the guy behind us.
    # For simplicity, we keep the angle check but add a velocity check.
    
    for i, (obs_x, obs_y) in enumerate(obstacles):
        dx = obs_x - x
        dy = obs_y - y
        
        # 1. Calculate Relative Position
        # Transform obstacle position to robot frame
        local_x = dx * math.cos(yaw) + dy * math.sin(yaw)
        local_y = -dx * math.sin(yaw) + dy * math.cos(yaw)
        
        # 2. Check if strictly in front (local_x > 0)
        is_in_front = local_x > -1.0 # Allow a small buffer (1m) behind center
        
        if is_in_front:
            # If in front, always keep it
            filtered_obstacles.append([obs_x, obs_y])
            filtered_radii.append(obstacle_radii[i])
            filtered_vel.append(obstacle_vel[i])
        else:
            # 3. If BEHIND, check if it's a threat (Overtaking)
            # We project the relative velocity onto the line connecting them
            obs_vx, obs_vy = obstacle_vel[i]
            
            # Simple check: Is the obstacle moving in the same general direction 
            # as the robot but faster? Or simply moving TOWARDS the robot?
            
            # Vector from Obs to Robot
            vec_to_robot = np.array([-dx, -dy])
            dist = np.linalg.norm(vec_to_robot)
            if dist > 0:
                vec_to_robot /= dist
                
            # Obstacle velocity vector
            vec_obs_vel = np.array([obs_vx, obs_vy])
            
            # Speed towards robot = dot product
            speed_towards_robot = np.dot(vec_obs_vel, vec_to_robot)
            
            # If speed_towards_robot > 0, it is closing the distance (Threat!)
            # We also check if it's close enough to worry about (e.g. < 10m)
            if speed_towards_robot > 0 and dist < 15.0:
                filtered_obstacles.append([obs_x, obs_y])
                filtered_radii.append(obstacle_radii[i])
                filtered_vel.append(obstacle_vel[i])

    # Return empty arrays if no obstacles found
    if not filtered_obstacles:
        return np.empty((0, 2)), [], np.empty((0, 2))

    return np.array(filtered_obstacles), filtered_radii, np.array(filtered_vel)


def motion(x, u, dt):
    """
    motion model
    """
    x[2] += u[1] * dt
    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt
    x[3] = u[0]
    x[4] = u[1]

    return x


def calc_dynamic_window(x, config):
    """
    calculation dynamic window based on current state x
    """
    # Dynamic window from robot specification
    Vs = [config.min_speed, config.max_speed,
          -config.max_yaw_rate, config.max_yaw_rate]

    # Dynamic window from motion model
    Vd = [x[3] - config.max_accel * config.dt,
          x[3] + config.max_accel * config.dt,
          x[4] - config.max_delta_yaw_rate * config.dt,
          x[4] + config.max_delta_yaw_rate * config.dt]

    #  [v_min, v_max, yaw_rate_min, yaw_rate_max]
    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
          max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

    return dw


def predict_trajectory_to_goal(x_init, v, y, config):
    """
    predict trajectory with an input
    """
    x = np.array(x_init)
    trajectory = np.array(x)
    time = 0
    while time <= config.predict_time_to_goal:
        x = motion(x, [v, y], config.dt)
        trajectory = np.vstack((trajectory, x))
        time += config.dt
    return trajectory

def predict_trajectory_obstacle(x_init, v, y, config):
    """
    predict trajectory with an input for obstacle distance calculation
    """
    x = np.array(x_init)
    trajectory = np.array(x)
    time = 0
    while time <= config.predict_time_obstacle:
        x = motion(x, [v, y], config.dt)
        trajectory = np.vstack((trajectory, x))
        time += config.dt
    return trajectory


def aggregate_compound_costs(cost_side_vec, cost_direct_vec, mode="mean"):
    """Reduce the per-point compound costs (side * time) to a single scalar.

    Returns (obstacle_cost, side_component, direct_component), where the last two
    are only used for logging.
    """
    compound_costs = cost_side_vec * cost_direct_vec
    if len(compound_costs) == 0:
        return 0.0, 0.0, 0.0

    if mode == "max":
        i_max = int(np.argmax(compound_costs))
        return float(compound_costs[i_max]), float(cost_side_vec[i_max]), float(cost_direct_vec[i_max])

    if mode == "mean":
        # Time-weighted average of the side costs.
        weight_sum = np.sum(cost_direct_vec)
        if weight_sum > 1e-12:
            obstacle_cost = float(np.sum(compound_costs) / weight_sum)
            side_component = obstacle_cost
        else:
            obstacle_cost = float(np.mean(compound_costs))
            side_component = float(np.mean(cost_side_vec))
        return obstacle_cost, side_component, float(np.mean(cost_direct_vec))

    raise ValueError(
        f"Unknown dynamic_ob_cost_aggregation: {mode!r} (expected 'mean' or 'max')"
    )


def calc_control_and_trajectory_merged(x, dw, config, goal,
                                     static_ob, static_ob_radii,
                                     dynamic_ob_pos, dynamic_ob_radii,
                                     dynamic_ob_vel,
                                     return_candidates=False,
                                     candidate_stride=1,
                                     candidate_max_count=None): # <--- New Argument
    """
    Calculation final input with dynamic window, splitting cost functions
    for static and dynamic obstacles.
    """

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

    # --- Prepare combined obstacle list for admissibility check ---
    has_static = static_ob is not None and static_ob.shape[0] > 0
    has_dynamic = dynamic_ob_pos is not None and dynamic_ob_pos.shape[0] > 0

    all_ob = np.empty((0, 2))
    all_ob_radii = []

    if has_static:
        all_ob = np.vstack((all_ob, static_ob))
        all_ob_radii.extend(static_ob_radii)
    
    if has_dynamic:
        all_ob = np.vstack((all_ob, dynamic_ob_pos))
        all_ob_radii.extend(dynamic_ob_radii)
    
    has_all_ob = all_ob.shape[0] > 0
    if has_all_ob:
        all_ob_radii_np = np.array(all_ob_radii)
    
    # A window can be EMPTY rather than merely restrictive: if the state has
    # drifted outside the actuator envelope, calc_dynamic_window returns
    # lo > hi and np.arange yields nothing. Every candidate list stays empty and
    # the Guard A raise below fires -- reporting "the braking test rejected
    # everything" when in truth nothing was ever tested. Those are opposite
    # diagnoses, so say which one happened.
    if dw[0] > dw[1] or dw[2] > dw[3]:
        raise ValueError(
            "Dynamic window is empty -- state is outside the actuator envelope: "
            f"v window [{dw[0]:.6f}, {dw[1]:.6f}], "
            f"omega window [{dw[2]:.6f}, {dw[3]:.6f}], "
            f"state v={x[3]:.6f} omega={x[4]:.6f}, "
            f"limits max_speed={config.max_speed} max_yaw_rate={config.max_yaw_rate}"
        )

    # evaluate all trajectory with sampled input in dynamic window
    for v in np.arange(dw[0], dw[1] + 1e-6, config.v_resolution):
        for y in np.arange(dw[2], dw[3] + 1e-6, config.yaw_rate_resolution):
            
            # --- Admissible velocities check ---
            dist_all = float("inf")
            if has_all_ob:
                dist_all, _ = closest_obstacle_on_curve(
                    x.copy(), all_ob, all_ob_radii_np, v, y, config
                )
            
            if v**2 + 2 * config.max_accel * v * config.dt > 2 * config.max_accel * dist_all:
                continue
                
            trajectory_to_goal = predict_trajectory_to_goal(x.copy(), v, y, config)
            trajectory_for_obstacle = predict_trajectory_obstacle(x.copy(), v, y, config)
            
            # --- Calculate costs ---
            to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(trajectory_to_goal, goal)
            speed_cost = config.speed_cost_gain * (config.max_speed - trajectory_to_goal[-1, 3])
            
            # Static obstacle cost
            static_ob_cost = 0.0
            if has_static:
                dist_static, _ = closest_obstacle_on_curve(
                    x.copy(), static_ob, np.array(static_ob_radii), v, y, config
                )
                static_ob_cost = config.obstacle_cost_gain * max(0., config.max_obstacle_cost_dist - dist_static)

            # --- NEW DYNAMIC OBSTACLE COST LOGIC ---
            dynamic_ob_cost = 0.0
            current_side_comp = 0.0
            current_direct_comp = 0.0

            if has_dynamic:
                # CALL NEW SPATIOTEMPORAL CHECKER
                d_side_arr, is_collision = calc_trajectory_clearance_and_collision(
                    trajectory_for_obstacle, 
                    dynamic_ob_pos, 
                    np.array(dynamic_ob_radii), 
                    dynamic_ob_vel, # <--- Pass Velocity
                    config
                )

                if is_collision:
                    dynamic_ob_cost = float("inf")
                    current_side_comp = float("inf")
                    current_direct_comp = float("inf")
                else:
                    # 1. Calculate Side Cost Vector (one per trajectory point)
                    side_norm = np.maximum(config.max_side_weight_dist, 1e-6)
                    
                    if config.side_clearance_barrier_type == "exponential":
                        # Gaussian exponential barrier: S = A * exp(-λ*d²)
                        # Decay rate λ ≈ 3.0 / M² ensures cost ≈ 5% of max at distance = M
                        decay_rate = config.side_clearance_gaussian_decay / (side_norm ** 2)
                        cost_side_vec = config.max_side_weight_dist * np.exp(-decay_rate * (d_side_arr ** 2))
                    else:
                        # Quadratic barrier: S = A * (1 - d/M)²
                        # This emphasizes very small clearances more strongly than linear weighting.
                        side_closeness = np.maximum(0.0, 1.0 - (d_side_arr / side_norm))
                        cost_side_vec = config.max_side_weight_dist * (side_closeness ** 2)

                    # 2. Calculate Direct Cost Vector (one per trajectory point)
                    # Exponential time discount: near-future points matter more.
                    # w(t) = exp(-lambda * t)
                    times = np.arange(trajectory_for_obstacle.shape[0]) * config.dt
                    cost_direct_vec = np.exp(-max(config.time_decay_lambda, 0.0) * times)

                    # 3. Reduce the compound cost vector along the horizon
                    #    ("mean": time-weighted average, "max": worst point)
                    dynamic_ob_cost, current_side_comp, current_direct_comp = aggregate_compound_costs(
                        cost_side_vec,
                        cost_direct_vec,
                        getattr(config, "dynamic_ob_cost_aggregation", "mean"),
                    )

                    dynamic_ob_cost = config.side_cost_gain * dynamic_ob_cost

            final_cost = to_goal_cost + speed_cost + static_ob_cost + dynamic_ob_cost

            to_goal_costs.append(to_goal_cost)
            speed_costs.append(speed_cost)
            static_ob_costs.append(static_ob_cost)
            dynamic_ob_costs.append(dynamic_ob_cost)
            side_cost_components.append(current_side_comp)
            direct_cost_components.append(current_direct_comp)

            trajectories.append(trajectory_for_obstacle)
            controls.append([v, y])
            
            if final_cost < min_cost:
                min_cost = final_cost
                best_index = i = len(controls) - 1
                best_u = controls[i]
                best_trajectory = trajectories[i]
    
    if len(to_goal_costs) == 0:
        # Fallback if no admissible trajectory found
        # (Usually better to handle gracefully, but keeping original behavior)
        raise ValueError("No admissible (v, ω) pairs found in dynamic window")

    if best_index != -1:
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
    else:
        # Fallback logic if needed, or raise error
        raise ValueError("No best trajectory found")
        
    if abs(best_u[0]) < config.robot_stuck_flag_cons \
            and abs(x[3]) < config.robot_stuck_flag_cons:
        # A yaw RATE command, so it must respect max_yaw_rate. This historically
        # read -config.max_delta_yaw_rate, which is an angular ACCELERATION;
        # the two are equal in v20.7.8-2 and the acceleration is far smaller in
        # every AIS config, so min() reproduces all 34 committed configs
        # exactly. It only bites where max_delta_yaw_rate > max_yaw_rate.
        best_u[1] = -min(config.max_yaw_rate, config.max_delta_yaw_rate)

    if return_candidates:
        stride = max(1, int(candidate_stride))
        candidate_trajectories = trajectories[::stride]

        if candidate_max_count is not None and candidate_max_count > 0 and len(candidate_trajectories) > candidate_max_count:
            idx = np.linspace(0, len(candidate_trajectories) - 1, candidate_max_count, dtype=int)
            candidate_trajectories = [candidate_trajectories[i] for i in idx]

        return (best_u, best_trajectory, dw,
                to_goal_before, speed_before, static_ob_before, dynamic_ob_before,
                best_side_component, best_direct_component,
                to_goal_after, speed_after, static_ob_after, dynamic_ob_after,
                min_cost, candidate_trajectories)

    return (best_u, best_trajectory, dw,
            to_goal_before, speed_before, static_ob_before, dynamic_ob_before,
            best_side_component, best_direct_component,
            to_goal_after, speed_after, static_ob_after, dynamic_ob_after,
            min_cost)


def _swept_hull_extent(config, side_factor, arc_radius=None):
    """Rectangular ego hull, reduced to what the swept-path tests need.

    Returns ``(half, radius)``: the hull swept along the candidate path is
    contained in the band of ``radius`` about that path, with the bow leading
    the hull centre by ``half`` of path length.

    ``"disc"`` returns ``(0.0, sqrt((L/2)^2 + (W/2)^2) * side_factor)`` -- the
    circumscribed circle, which is what every branch below used unconditionally
    before the capsule existed, so disc configs are bit-identical.

    ``"capsule"`` returns ``half = (L - W)/2`` and ``radius = (W/2) *
    side_factor``. For a 19 x 6.33 m ego that is 3.165 m rather than 10.013 m --
    the disc treats a ship as being as wide as it is long, and in Stage 1 that
    is the difference between an admissible creep and an abort.

    ``arc_radius`` is ``|v / omega|`` for the circular-arc branch. A capsule
    rigidly swept along an arc, axis tangent, sweeps an annulus whose outer edge
    sits at ``hypot(R, half) + W/2`` while its inner edge stays at ``R - W/2``.
    Widening the band by ``hypot(R, half) - R`` makes the symmetric band contain
    that annulus, which keeps the existing circle-circle intersection exact and
    conservative. The term is 0.024 m at R = 836 m and grows only as the turn
    tightens; the bow lead ``half`` is handled by the caller and is the larger
    of the two effects by far.
    """
    shape = getattr(config, "robot_collision_shape", "disc")
    if shape == "capsule":
        half = max(0.5 * (config.robot_length - config.robot_width), 0.0)
        radius = 0.5 * config.robot_width * side_factor
        if arc_radius is not None and arc_radius > 0.0:
            radius += math.hypot(arc_radius, half) - arc_radius
        return half, radius

    diagonal = math.sqrt((config.robot_length / 2) ** 2 + (config.robot_width / 2) ** 2)
    return 0.0, diagonal * side_factor


def closest_obstacle_on_curve(x, ob, ob_radii, v, omega, config):
    """
    Calculate the distance to the closest obstacle that intersects with the curvature
    without time/span limitations - checks the entire trajectory
    
    Parameters:
    x: current state
        [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    ob: obstacle positions
        [[x(m), y(m)], ...]
    ob_radii: list/array of radii for each obstacle
    v: translational velocity (m/s)
    omega: angular velocity (rad/s)
    config: simulation configuration
    
    Returns:
    dist: distance to the closest obstacle
    t: time to reach the closest obstacle
    """
    # Handle empty obstacle list
    if ob is None or ob.shape[0] == 0:
        return float("inf"), float("inf")

    start_pos = (x[0], x[1])
    heading = x[2]
    min_dist = float("inf")
    min_time = float("inf")

    side_clearnce_factor = 0.1*v/config.max_speed + 1.0

    # Special case: when velocity is zero or very small
    if abs(v) < 1e-5:
        # Check obstacles along the facing direction (straight line)
        heading_vector = np.array([math.cos(heading), math.sin(heading)])
        
        for i in range(len(ob)):
            obstacle = np.array([ob[i, 0], ob[i, 1]])
            obstacle_radius = ob_radii[i]
            
            to_center = obstacle - np.array(start_pos)
            projection = np.dot(to_center, heading_vector)
            
            # Only consider obstacles in front of the robot
            if projection < 0:
                continue
                
            closest_approach = np.linalg.norm(to_center - projection * heading_vector)
            
            half_hull = 0.0
            if config.robot_type == RobotType.rectangle:
                half_hull, hull_radius = _swept_hull_extent(config, side_clearnce_factor)
                collision_threshold = obstacle_radius + hull_radius
            else:
                collision_threshold = obstacle_radius + config.robot_radius * side_clearnce_factor

            if closest_approach <= collision_threshold:
                # Distance to the edge of collision zone, travelled by the hull
                # CENTRE. The bow reaches the zone half_hull earlier; half_hull
                # is 0 for a disc, so this is unchanged there.
                dist_to_intersection = max(0, projection - half_hull - math.sqrt(
                    max(0, collision_threshold**2 - closest_approach**2)))
                
                if dist_to_intersection < min_dist:
                    min_dist = dist_to_intersection
                    min_time = float("inf")  # Time is infinite since v=0
                    
        return min_dist, min_time
    
    # Original logic for v > 0 cases
    elif abs(omega) < 1e-6:
        # Straight line motion - check entire line
        heading_vector = np.array([math.cos(heading), math.sin(heading)])
        
        for i in range(len(ob)):
            obstacle = np.array([ob[i, 0], ob[i, 1]])
            obstacle_radius = ob_radii[i]
            
            to_center = obstacle - np.array(start_pos)
            projection = np.dot(to_center, heading_vector)
            
            if projection < 0:
                continue
                
            closest_approach = np.linalg.norm(to_center - projection * heading_vector)
            
            half_hull = 0.0
            if config.robot_type == RobotType.rectangle:
                half_hull, hull_radius = _swept_hull_extent(config, side_clearnce_factor)
                collision_threshold = obstacle_radius + hull_radius
            else:
                collision_threshold = obstacle_radius + config.robot_radius * side_clearnce_factor

            if closest_approach > collision_threshold:
                continue

            dist_to_intersection = projection - half_hull - math.sqrt(
                collision_threshold**2 - closest_approach**2)
            if half_hull > 0.0:
                # A capsule's bow can already be inside the zone where the disc
                # centre had not yet arrived. Report contact now (0) rather than
                # letting the `0 <= ...` guard below drop the obstacle entirely
                # and return inf -- "no obstacle ahead" -- while overlapping it.
                dist_to_intersection = max(0.0, dist_to_intersection)

            if 0 <= dist_to_intersection < min_dist:
                min_dist = dist_to_intersection
                min_time = dist_to_intersection / v if v > 0 else float("inf")
                
        return min_dist, min_time
        
    else:
        # Circular arc motion - check entire circle
        radius = abs(v / omega)
        
        if omega > 0:
            center_x = x[0] - radius * math.sin(heading)
            center_y = x[1] + radius * math.cos(heading)
        else:
            center_x = x[0] + radius * math.sin(heading)
            center_y = x[1] - radius * math.cos(heading)
            
        arc_center = np.array([center_x, center_y])
        
        if omega > 0:
            start_angle = heading - math.pi/2
        else:
            start_angle = heading + math.pi/2
            
        for i in range(len(ob)):
            obstacle_center = np.array([ob[i, 0], ob[i, 1]])
            obstacle_radius = ob_radii[i]
            
            dist_between_centers = np.linalg.norm(arc_center - obstacle_center)
            
            half_hull = 0.0
            if config.robot_type == RobotType.rectangle:
                half_hull, hull_radius = _swept_hull_extent(
                    config, side_clearnce_factor, radius)
                collision_radius = obstacle_radius + hull_radius
            else:
                # NOTE: no side_clearnce_factor here, unlike every other branch
                # in this function. Pre-existing asymmetry, preserved so that
                # circle-robot runs stay reproducible; not part of this change.
                collision_radius = obstacle_radius + config.robot_radius
                
            if dist_between_centers > radius + collision_radius or dist_between_centers < radius - collision_radius:
                continue
                
            # Calculate intersection points without span checking
            a = radius
            b = dist_between_centers
            c = collision_radius
            
            if b == 0:
                continue
                
            d = (a*a + b*b - c*c) / (2*b)
            
            if abs(d) > radius:
                continue
                
            h_squared = a*a - d*d
            if h_squared < 0:
                continue
                
            h = math.sqrt(h_squared)
            
            unit_vector = (obstacle_center - arc_center) / dist_between_centers
            perp_vector = np.array([-unit_vector[1], unit_vector[0]])
            
            int_point1 = arc_center + d * unit_vector + h * perp_vector
            int_point2 = arc_center + d * unit_vector - h * perp_vector
            
            angle1 = math.atan2(int_point1[1] - arc_center[1], int_point1[0] - arc_center[0]) % (2 * math.pi)
            angle2 = math.atan2(int_point2[1] - arc_center[1], int_point2[0] - arc_center[0]) % (2 * math.pi)
            
            # Calculate angle differences without span limitation
            if omega > 0:
                angle_diff1 = (angle1 - start_angle) % (2 * math.pi)
                angle_diff2 = (angle2 - start_angle) % (2 * math.pi)
            else:
                angle_diff1 = (start_angle - angle1) % (2 * math.pi)
                angle_diff2 = (start_angle - angle2) % (2 * math.pi)
            
            # Check both intersection points (no span restriction)
            for angle_diff in [angle_diff1, angle_diff2]:
                # Arc length travelled by the hull centre, less the bow lead.
                dist = angle_diff * radius - half_hull
                if half_hull > 0.0:
                    dist = max(0.0, dist)
                time = dist / v if v > 0 else float("inf")
                
                if dist < min_dist:
                    min_dist = dist
                    min_time = time
                    
        return min_dist, min_time


def calc_trajectory_clearance_and_collision(trajectory, ob_pos, ob_radii, ob_vel, config):
    """
    Calculates the minimum clearance distance at each time step along a robot trajectory
    with respect to moving obstacles, and determines if a collision occurs.
    This function predicts obstacle positions over time based on their velocities and
    computes the distance from the robot to each obstacle at each trajectory point.
    Clearance is defined as the distance between the robot and obstacles minus their
    combined radii. A negative or zero clearance indicates a collision.
    Parameters
    ----------
    trajectory : np.ndarray
        Shape (T, >= 2), where T is the number of time steps.
        Each row contains at least [x, y] position of the robot at that time step.
        Additional columns (e.g., velocity, theta) are ignored.
    ob_pos : np.ndarray or None
        Shape (N, 2), where N is the number of obstacles.
        Each row contains [x, y] position of an obstacle at t=0.
        If None or empty, function returns no collision and infinite clearance.
    ob_radii : array-like
        Shape (N,), where N is the number of obstacles.
        Radius of each obstacle.
    ob_vel : np.ndarray
        Shape (N, 2), where N is the number of obstacles.
        Each row contains [vx, vy] velocity of an obstacle.
    config : Config
        Configuration object containing:
        - dt (float): Time step interval in seconds
        - robot_type (RobotType): Type of robot ('circle' or 'rectangle')
        - robot_radius (float): Radius of the robot (if circular)
        - robot_length (float): Length of the robot (if rectangular)
        - robot_width (float): Width of the robot (if rectangular)
    Returns
    -------
    min_clearance_per_step : np.ndarray
        Shape (T,), minimum clearance distance at each time step.
        Positive values indicate safe distance; zero or negative indicates collision.
        Returns full infinity array if no obstacles exist.
    is_collision : bool
        True if any point on the trajectory results in collision (clearance <= 0).
        False if the entire trajectory maintains safe distance or no obstacles exist.
    """

    if ob_pos is None or ob_pos.shape[0] == 0:
        return np.full(trajectory.shape[0], float("inf")), False

    num_steps = trajectory.shape[0]
    
    # 1. Time Vector: [0, dt, 2dt, ... T*dt]
    # Shape: (1, T, 1) for broadcasting
    times = np.arange(num_steps).reshape(1, -1, 1) * config.dt
    
    # 2. Predicted Obstacle Positions: Pos(t) = Pos(0) + Vel * t
    # Expand dims for broadcasting: (N, 1, 2)
    ob_pos_expanded = ob_pos[:, np.newaxis, :]
    ob_vel_expanded = ob_vel[:, np.newaxis, :]
    
    # Result Shape: (N, T, 2) -> (Num_Obs, Num_TimeSteps, XY)
    predicted_ob_pos = ob_pos_expanded + (ob_vel_expanded * times)
    
    # 3. Robot Trajectory Positions
    # Shape: (1, T, 2)
    robot_pos = trajectory[:, 0:2][np.newaxis, :, :]
    
    # 4. Calculate Distance Matrix (N, T)
    diff = robot_pos - predicted_ob_pos
    dist_matrix = np.hypot(diff[:, :, 0], diff[:, :, 1])
    
    # 5. Check Collision & Clearance
    ob_radii_expanded = np.array(ob_radii)[:, np.newaxis] # (N, 1)
    
    shape = getattr(config, "robot_collision_shape", "disc")
    if config.robot_type == RobotType.rectangle and shape == "capsule":
        # Oriented capsule: a segment of length (L - W) along the heading, with
        # radius W/2. Same overall length and width as the rectangle, only the
        # corners rounded off.
        #
        # The "disc" branch below bounds the rectangle by its circumscribed
        # circle, radius sqrt(L^2 + W^2)/2, which ignores the heading entirely
        # and so treats the hull as being as wide as it is long. The error grows
        # with aspect ratio: negligible for a 1.2 x 0.5 m robot, but it inflates
        # a 248 x 32 m ship into a 250 m-wide disc and leaves it no admissible
        # move in a harbour. Ships are the most elongated vehicles there are,
        # which is why this branch exists.
        half = max(0.5 * (config.robot_length - config.robot_width), 0.0)
        yaw = trajectory[:, 2]
        axis = np.stack([np.cos(yaw), np.sin(yaw)], axis=-1)[np.newaxis, :, :]

        # Offset of each obstacle from the hull centre, projected onto the
        # heading and clamped to the segment: the nearest point on the axis.
        to_ob = predicted_ob_pos - robot_pos                      # (N, T, 2)
        proj = np.clip(np.sum(to_ob * axis, axis=2), -half, half)  # (N, T)
        perp = to_ob - proj[:, :, np.newaxis] * axis
        axis_dist = np.hypot(perp[:, :, 0], perp[:, :, 1])
        clearance_matrix = axis_dist - (ob_radii_expanded + 0.5 * config.robot_width)
    elif config.robot_type == RobotType.rectangle:
        # Diagonal approximation
        robot_radius = math.sqrt(config.robot_length**2 + config.robot_width**2) / 2.0
        clearance_matrix = dist_matrix - (ob_radii_expanded + robot_radius)
    else:
        clearance_matrix = dist_matrix - (ob_radii_expanded + config.robot_radius)
        
    # Check if any point in time results in collision (clearance <= 0)
    is_collision = np.any(clearance_matrix <= 0)
    
    # Reduce over obstacles to find the closest danger at each time step
    # Shape: (T,)
    min_clearance_per_step = np.min(clearance_matrix, axis=0)

    return min_clearance_per_step, is_collision


def calc_to_goal_cost(trajectory, goal):
    """
    calc to goal cost with angle difference
    """
    dx = goal[0] - trajectory[-1, 0]
    dy = goal[1] - trajectory[-1, 1]
    error_angle = math.atan2(dy, dx)
    cost_angle = error_angle - trajectory[-1, 2]
    cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))

    return cost


def plot_arrow(x, y, yaw, length=0.5, width=0.1):  # pragma: no cover
    plt_elements = []
    arrow = plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
              head_length=width, head_width=width)
    point, = plt.plot(x, y)
    plt_elements.append(arrow)
    plt_elements.append(point)
    return plt_elements


def plot_robot(x, y, yaw, config):  # pragma: no cover
    plt_elements = []
    if config.robot_type == RobotType.rectangle:
        outline = np.array([[-config.robot_length / 2, config.robot_length / 2,
                             (config.robot_length / 2), -config.robot_length / 2,
                             -config.robot_length / 2],
                            [config.robot_width / 2, config.robot_width / 2,
                             - config.robot_width / 2, -config.robot_width / 2,
                             config.robot_width / 2]])
        Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                         [-math.sin(yaw), math.cos(yaw)]])
        outline = (outline.T.dot(Rot1)).T
        outline[0, :] += x
        outline[1, :] += y
        line, = plt.plot(
            np.array(outline[0, :]).flatten(),
            np.array(outline[1, :]).flatten(),
            "-k"
        )
        plt_elements.append(line)
    elif config.robot_type == RobotType.circle:
        circle = plt.Circle((x, y), config.robot_radius, color="b")
        plt_elements.append(circle)
        plt.gcf().gca().add_artist(circle)
        out_x, out_y = (np.array([x, y]) +
                        np.array([np.cos(yaw), np.sin(yaw)]) * config.robot_radius)
        line, = plt.plot([x, out_x], [y, out_y], "-k")
        plt_elements.append(line)
    return plt_elements


def check_collision_at_current_position_circle_approximation(x, ob, ob_radii, config):
    """
    Check if the robot at current position collides with any obstacles
    using circle approximation (same as DWA distance calculations)
    
    Parameters:
    x: current state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    ob: obstacle positions [[x(m), y(m)], ...]
    ob_radii: list/array of radii for each obstacle
    config: simulation configuration
    
    Returns:
    collision: boolean indicating if collision occurred
    obstacle_index: index of colliding obstacle (if any)
    distance: distance to colliding obstacle
    """
    # Handle empty obstacle list
    if ob is None or ob.shape[0] == 0:
        return False, None, float('inf')

    robot_pos = np.array([x[0], x[1]])
    ob = np.asarray(ob, dtype=float)
    ob_radii = np.asarray(ob_radii, dtype=float)

    # This must use the SAME hull as calc_trajectory_clearance_and_collision.
    # If the planner clears a gap with one hull and contact is declared with
    # another, the ego collides on a path its own planner certified -- which is
    # not a planning failure but a disagreement between two checks.
    #
    # The two disc forms below are algebraically identical
    # (sqrt(L^2 + W^2)/2 == sqrt((L/2)^2 + (W/2)^2)); they agreed before the
    # capsule was introduced and must keep agreeing after it.
    shape = getattr(config, "robot_collision_shape", "disc")
    if config.robot_type == RobotType.rectangle and shape == "capsule":
        half = max(0.5 * (config.robot_length - config.robot_width), 0.0)
        axis = np.array([math.cos(x[2]), math.sin(x[2])])
        to_ob = ob - robot_pos
        proj = np.clip(to_ob @ axis, -half, half)
        perp = to_ob - proj[:, None] * axis
        distances = np.hypot(perp[:, 0], perp[:, 1])
        effective_robot_radius = 0.5 * config.robot_width
    else:
        distances = np.linalg.norm(ob - robot_pos, axis=1)
        if config.robot_type == RobotType.rectangle:
            # Use diagonal approximation (same as in DWA distance calculation)
            effective_robot_radius = math.sqrt((config.robot_length/2)**2 + (config.robot_width/2)**2)
        else:
            # Use actual robot radius
            effective_robot_radius = config.robot_radius

    # Check collision for each obstacle with its specific radius
    for i in range(len(ob)):
        collision_threshold = ob_radii[i] + effective_robot_radius
        if distances[i] <= collision_threshold:
            return True, i, float(distances[i])

    return False, None, float('inf')
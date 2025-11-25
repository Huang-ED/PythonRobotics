"""

Mobile robot motion planning sample with Dynamic Window Approach

author: Atsushi Sakai (@Atsushi_twi), Göktuğ Karakaşlı

Modified by: Huang Erdong (@Huang-ED)
In this version of codes, 
    the methodology proposed in the original DWA paper is implemented.
Compared to dynamic_window_approach_paper.py,
    robot are considered in rectangle shape when indicated,
    and the obstacle is considered as a circle with radius.

This file (dwa_paper_split.py) is modified to accept separate
static and dynamic obstacle lists and apply different cost functions to them.
- Static Obstacles: Direct distance cost (from closest_obstacle_on_curve)
- Dynamic Obstacles: Side distance cost (from closest_obstacle_on_side)

"""

import os
import math
from enum import Enum

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
        self.predict_time = 1.0  # [s]
        # self.check_time = 100.0 # [s] Time to check for collision - a large number
        self.to_goal_cost_gain = 0.4
        self.speed_cost_gain = 1.0
        self.obstacle_cost_gain = 0.05  # Gain for static obstacles (direct dist)
        self.side_cost_gain = 4.0      # Gain for dynamic obstacles (side dist)
        self.robot_stuck_flag_cons = 0.001  # constant to prevent robot stucked
        self.robot_type = RobotType.rectangle
        self.catch_goal_dist = 0.5  # [m] goal radius
        self.catch_turning_point_dist = 5.0  # [m] local goal radius
        self.obstacle_radius = 0.5  # [m] for collision check

        self.max_obstacle_cost_dist = 5.0  # [m] max distance for static obstacle cost calculation

        # if robot_type == RobotType.circle
        # Also used to check if goal is reached in both types
        self.robot_radius = 0.5  # [m] for collision check

        # if robot_type == RobotType.rectangle
        self.robot_width = 0.5  # [m] for collision check
        self.robot_length = 1.2  # [m] for collision check

        self.dist_localgoal = 5.0  # [m] distance to local goal

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


def dwa_control(x, config, goal, ob):
    """
    Original DWA control function.
    """
    dw = calc_dynamic_window(x, config)
    (u, trajectory, dw, # admissible, inadmissible,
     to_goal_before, speed_before, ob_before,
     to_goal_after, speed_after, ob_after,
     final_cost) = calc_control_and_trajectory(x, dw, config, goal, ob)
    return (u, trajectory, dw, # admissible, inadmissible,
            to_goal_before, speed_before, ob_before,
            to_goal_after, speed_after, ob_after,
            final_cost)


def dwa_control_split_ob(x, config, goal, static_ob, dynamic_ob):
    """
    DWA control function that splits static and dynamic obstacles.
    """
    dw = calc_dynamic_window(x, config)
    (u, trajectory, dw,
     to_goal_before, speed_before, ob_before, dynamic_ob_before,
     to_goal_after, speed_after, ob_after, dynamic_ob_after,
     final_cost) = calc_control_and_trajectory_split_ob(x, dw, config, goal, static_ob, dynamic_ob)
    
    return (u, trajectory, dw,
            to_goal_before, speed_before, ob_before, dynamic_ob_before,
            to_goal_after, speed_after, ob_after, dynamic_ob_after,
            final_cost)


def motion(x, u, dt):
    """
    motion model
    Parameters:
        x: current state 
            [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
        u: translational and angular velocities 
            [v(m/s), omega(rad/s)]
        dt: time interval (s)
    Returns:
        x: updated state
            [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
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
    Parameters:
        x: current state
            [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
        config: simulation configuration
    Returns:
        dw: dynamic window
            [v_min, v_max, yaw_rate_min, yaw_rate_max]
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


def predict_trajectory(x_init, v, y, config):
    """
    predict trajectory with an input
    Parameters:
        x_init: initial state
            [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
        v: translational velocity (m/s)
        y: angular velocity (rad/s)
        config: simulation configuration
    Returns:
        trajectory: predicted trajectory
            [[x, y, yaw, v, omega], ...]
    """

    x = np.array(x_init)
    trajectory = np.array(x)
    time = 0
    while time <= config.predict_time:
        x = motion(x, [v, y], config.dt)
        trajectory = np.vstack((trajectory, x))
        time += config.dt

    return trajectory


def calc_control_and_trajectory(x, dw, config, goal, ob):
    """
    Original calculation function.
    """

    min_cost = float("inf")
    best_u = [0.0, 0.0]
    best_trajectory = np.array([x])
    best_index = -1
    
    to_goal_costs = []
    speed_costs = []
    ob_costs = []
    trajectories = []
    controls = []
    # admissible = []
    # inadmissible = []

    # evaluate all trajectory with sampled input in dynamic window
    for v in np.arange(dw[0], dw[1] + 1e-6, config.v_resolution):
        for y in np.arange(dw[2], dw[3] + 1e-6, config.yaw_rate_resolution):
            # Track all control inputs as inadmissible initially
            # inadmissible.append([float(v), float(y)])
            
            # admissible velocities check
            dist, _ = closest_obstacle_on_curve(x.copy(), ob, v, y, config)
            # if v > math.sqrt(2*config.max_accel*dist):
            # if v**2 + config.max_accel * v * config.dt > 2 * config.max_accel * dist:
            if v**2 + 2 * config.max_accel * v * config.dt > 2 * config.max_accel * dist:
                continue
                
            # If we reach here, the control is admissible
            # inadmissible.pop()  # Remove the last inadmissible entry
            # admissible.append([float(v), float(y)])
            
            trajectory = predict_trajectory(x.copy(), v, y, config)
            
            # calc costs
            to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(trajectory, goal)
            speed_cost = config.speed_cost_gain * (config.max_speed - trajectory[-1, 3])

            # ob_cost = float("inf") if dist == 0 else config.obstacle_cost_gain * (1 / dist)
            ob_cost = config.obstacle_cost_gain * max(0., config.max_obstacle_cost_dist-dist)

            # clearance = closest_obstacle_on_side(trajectory, ob, config)
            # # ob_cost = config.obstacle_cost_gain * 1 / clearance if clearance > 0 else float("inf")
            # side_cost = config.side_cost_gain * (- clearance) if clearance > 0 else 0

            final_cost = to_goal_cost + speed_cost + ob_cost # + side_cost

            # Store all costs and trajectories
            to_goal_costs.append(to_goal_cost)
            speed_costs.append(speed_cost)
            ob_costs.append(ob_cost)
            trajectories.append(trajectory)
            controls.append([v, y])
            
            # search minimum trajectory
            if final_cost < min_cost:
                min_cost = final_cost
                best_index = i = len(controls) - 1
                best_u = controls[i]
                best_trajectory = trajectories[i]
    
    if len(to_goal_costs) == 0:
        raise ValueError("No admissible (v, ω) pairs found in dynamic window")

    if best_index != -1:
        to_goal_before = to_goal_costs[best_index]
        speed_before = speed_costs[best_index]
        ob_before = ob_costs[best_index]
        # Since no normalization, after costs are the same as before
        to_goal_after = to_goal_before
        speed_after = speed_before
        ob_after = ob_before
    else:
        # This case should be covered by the len(to_goal_costs) == 0 check, but as a fallback:
        raise ValueError("No best trajectory found, though admissible (v, w) pairs existed.")
        
    # Handle the robot stuck scenario as in the original function
    if abs(best_u[0]) < config.robot_stuck_flag_cons \
            and abs(x[3]) < config.robot_stuck_flag_cons:
        # to ensure the robot does not get stuck
        best_u[1] = -config.max_delta_yaw_rate

    return (best_u, best_trajectory, dw, # admissible, inadmissible,
            to_goal_before, speed_before, ob_before,
            to_goal_after, speed_after, ob_after,
            min_cost)


def calc_control_and_trajectory_split_ob(x, dw, config, goal, static_ob, dynamic_ob):
    """
    Calculation final input with dynamic window, splitting cost functions
    for static and dynamic obstacles.
    
    Parameters:
        x: current state
        dw: dynamic window
        config: simulation configuration
        goal: goal position
        static_ob: static obstacle positions [[x(m), y(m)], ...]
        dynamic_ob: dynamic obstacle positions [[x(m), y(m)], ...]
    Returns:
        (best_u, best_trajectory, dw,
         to_goal_before, speed_before, static_ob_before, dynamic_ob_before,
         to_goal_after, speed_after, static_ob_after, dynamic_ob_after,
         min_cost)
    """

    min_cost = float("inf")
    best_u = [0.0, 0.0]
    best_trajectory = np.array([x])
    best_index = -1
    
    to_goal_costs = []
    speed_costs = []
    static_ob_costs = []   # Cost from static obstacles (direct)
    dynamic_ob_costs = []  # Cost from dynamic obstacles (side)
    trajectories = []
    controls = []

    # --- Prepare combined obstacle list for admissibility check ---
    has_static = static_ob is not None and static_ob.shape[0] > 0
    has_dynamic = dynamic_ob is not None and dynamic_ob.shape[0] > 0

    if has_static and has_dynamic:
        all_ob = np.vstack((static_ob, dynamic_ob))
    elif has_static:
        all_ob = static_ob
    elif has_dynamic:
        all_ob = dynamic_ob
    else:
        all_ob = np.empty((0, 2))
    
    has_all_ob = all_ob.shape[0] > 0

    # evaluate all trajectory with sampled input in dynamic window
    for v in np.arange(dw[0], dw[1] + 1e-6, config.v_resolution):
        for y in np.arange(dw[2], dw[3] + 1e-6, config.yaw_rate_resolution):
            
            # --- Admissible velocities check (using ALL obstacles) ---
            dist_all = float("inf")
            if has_all_ob:
                dist_all, _ = closest_obstacle_on_curve(x.copy(), all_ob, v, y, config)
            
            if v**2 + 2 * config.max_accel * v * config.dt > 2 * config.max_accel * dist_all:
                continue
                
            trajectory = predict_trajectory(x.copy(), v, y, config)
            
            # --- Calculate costs ---
            to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(trajectory, goal)
            speed_cost = config.speed_cost_gain * (config.max_speed - trajectory[-1, 3])

            # Static obstacle cost (direct distance)
            static_ob_cost = 0.0
            if has_static:
                # We can reuse dist_all if only static obs exist, but for clarity, let's recalculate
                dist_static, _ = closest_obstacle_on_curve(x.copy(), static_ob, v, y, config)
                static_ob_cost = config.obstacle_cost_gain * max(0., config.max_obstacle_cost_dist - dist_static)

            # Dynamic obstacle cost (side distance/clearance)
            dynamic_ob_cost = 0.0
            if has_dynamic:
                clearance_dynamic = closest_obstacle_on_side(trajectory, dynamic_ob, config)
                if clearance_dynamic <= 0:  # Collision or on boundary
                    dynamic_ob_cost = float("inf")
                else:
                    # Use inverse of clearance, so smaller clearance = higher cost
                    dynamic_ob_cost = config.side_cost_gain * (1.0 / clearance_dynamic)

            final_cost = to_goal_cost + speed_cost + static_ob_cost + dynamic_ob_cost

            # Store all costs and trajectories
            to_goal_costs.append(to_goal_cost)
            speed_costs.append(speed_cost)
            static_ob_costs.append(static_ob_cost)
            dynamic_ob_costs.append(dynamic_ob_cost)
            trajectories.append(trajectory)
            controls.append([v, y])
            
            # search minimum trajectory
            if final_cost < min_cost:
                min_cost = final_cost
                best_index = i = len(controls) - 1
                best_u = controls[i]
                best_trajectory = trajectories[i]
    
    if len(to_goal_costs) == 0:
        raise ValueError("No admissible (v, ω) pairs found in dynamic window")

    if best_index != -1:
        to_goal_before = to_goal_costs[best_index]
        speed_before = speed_costs[best_index]
        static_ob_before = static_ob_costs[best_index]
        dynamic_ob_before = dynamic_ob_costs[best_index]
        # Since no normalization, after costs are the same as before
        to_goal_after = to_goal_before
        speed_after = speed_before
        static_ob_after = static_ob_before
        dynamic_ob_after = dynamic_ob_before
    else:
        # This case should be covered by the len(to_goal_costs) == 0 check, but as a fallback:
        raise ValueError("No best trajectory found, though admissible (v, w) pairs existed.")
        
    # Handle the robot stuck scenario
    if abs(best_u[0]) < config.robot_stuck_flag_cons \
            and abs(x[3]) < config.robot_stuck_flag_cons:
        # to ensure the robot does not get stuck
        best_u[1] = -config.max_delta_yaw_rate

    return (best_u, best_trajectory, dw,
            to_goal_before, speed_before, static_ob_before, dynamic_ob_before,
            to_goal_after, speed_after, static_ob_after, dynamic_ob_after,
            min_cost)


def calc_control_and_trajectory_side_norm(x, dw, config, goal, ob):
    """
    calculation final input with dynamic window
    Parameters:
        x: current state
            [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
        dw: dynamic window
            [v_min, v_max, yaw_rate_min, yaw_rate_max]
        config: simulation configuration
        goal: goal position
            [x(m), y(m)]
        ob: obstacle positions 
            [[x(m), y(m)], ...]
    Returns:
        best_u: selected control input
            [v(m/s), omega(rad/s)]
        best_trajectory: predicted trajectory with selected input
            [[x, y, yaw, v, omega], ...]
        dw: dynamic window
        to_goal_before: raw to-goal cost
        speed_before: raw speed cost
        ob_before: raw obstacle cost
        to_goal_after: to-goal cost (same as before since no normalization)
        speed_after: speed cost (same as before since no normalization)
        ob_after: obstacle cost (same as before since no normalization)
        min_cost: minimum cost value
    """
    min_cost = float("inf")
    best_u = [0.0, 0.0]
    best_trajectory = np.array([x])
    best_index = -1

    # --- First Pass: Collection ---
    # Lists to store raw data for all admissible controls
    all_to_goal_costs = []
    all_speed_costs = []
    all_ob_costs = []
    all_side_costs = []  # Store raw side costs for normalization
    all_trajectories = []
    all_controls = []    # Store [v, y] for each entry
    all_vs = []          # Store v for each entry (for grouping)

    # Dictionary to group side costs by v for later normalization
    # Key: v (rounded to handle floating point), Value: list of side_cost indices or values
    side_costs_by_v = {}

    # evaluate all trajectory with sampled input in dynamic window
    for v in np.arange(dw[0], dw[1] + 1e-6, config.v_resolution):
        for y in np.arange(dw[2], dw[3] + 1e-6, config.yaw_rate_resolution):
            # admissible velocities check
            dist, _ = closest_obstacle_on_curve(x.copy(), ob, v, y, config)
            if v**2 + 2 * config.max_accel * v * config.dt > 2 * config.max_accel * dist:
                continue

            trajectory = predict_trajectory(x.copy(), v, y, config)

            # calc raw costs (side_cost is calculated here but not used in final_cost yet)
            to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(trajectory, goal)
            speed_cost = config.speed_cost_gain * (config.max_speed - trajectory[-1, 3])
            ob_cost = config.obstacle_cost_gain * max(0., config.max_obstacle_cost_dist - dist)
            side_cost = 0  # Initialize
            clearance = closest_obstacle_on_side(trajectory, ob, config)
            if clearance > 0:
                side_cost = config.side_cost_gain * (- clearance) # Note: Negative for cost
            else:
                side_cost = 0 # or potentially a large penalty if collision is detected, but function returns 0 on collision

            # Store all raw costs and data
            current_index = len(all_controls) # Index of this entry in the lists
            all_to_goal_costs.append(to_goal_cost)
            all_speed_costs.append(speed_cost)
            all_ob_costs.append(ob_cost)
            all_side_costs.append(side_cost) # Store RAW side cost
            all_trajectories.append(trajectory)
            all_controls.append([v, y])
            all_vs.append(v)

            # Group side_cost by v for normalization
            # Use rounding to handle floating point precision issues when using v as a key
            v_key = round(v, int(-math.log10(config.v_resolution)) + 2) # Adjust precision based on resolution
            if v_key not in side_costs_by_v:
                side_costs_by_v[v_key] = []
            side_costs_by_v[v_key].append(current_index) # Store the index

    if len(all_controls) == 0:
        raise ValueError("No admissible (v, ω) pairs found in dynamic window")

    # --- Normalization: Normalize side_cost within each v group ---
    normalized_side_costs = [0.0] * len(all_side_costs) # Initialize list for normalized values

    for v_key, indices in side_costs_by_v.items():
        if len(indices) == 0:
            continue
        # Extract the raw side costs for this v group
        group_side_costs = [all_side_costs[i] for i in indices]
        min_side = min(group_side_costs)
        max_side = max(group_side_costs)

        # Avoid division by zero if all side costs in the group are the same
        if abs(max_side - min_side) < 1e-9:
            # If all costs are identical, normalized cost can be 0.0, 0.5, or 1.0.
            # Setting to 0.0 might be reasonable if they are all "good" (less negative),
            # but since side_cost is usually negative (penalty for being close),
            # setting normalized value to 0.0 implies "best" in the group.
            # Alternatively, you could set it to the raw value if you don't want to alter it.
            # Here, we set normalized cost to 0.0 for identical values.
            norm_value = 0.0
            for idx in indices:
                normalized_side_costs[idx] = norm_value
        else:
            # Apply Min-Max normalization: (value - min) / (max - min)
            # This scales to [0, 1], where 0 is the least penalty (best) and 1 is the highest penalty (worst) in the group.
            for idx in indices:
                raw_sc = all_side_costs[idx]
                norm_sc = (raw_sc - min_side) / (max_side - min_side)
                normalized_side_costs[idx] = norm_sc

    # --- Second Pass: Calculate Final Cost and Find Minimum ---
    for i in range(len(all_controls)):
        to_goal_cost = all_to_goal_costs[i]
        speed_cost = all_speed_costs[i]
        ob_cost = all_ob_costs[i]
        norm_side_cost = config.side_cost_gain * normalized_side_costs[i] # Use the normalized side cost

        # Calculate final cost using normalized side cost
        final_cost = to_goal_cost + speed_cost + ob_cost + norm_side_cost

        # search minimum trajectory
        if final_cost < min_cost:
            min_cost = final_cost
            best_index = i
            best_u = all_controls[i]
            best_trajectory = all_trajectories[i]

    # --- Prepare Return Values ---
    if best_index != -1:
        to_goal_before = all_to_goal_costs[best_index]
        speed_before = all_speed_costs[best_index]
        ob_before = all_ob_costs[best_index]
        # Since no normalization for these, after costs are the same as before
        to_goal_after = to_goal_before
        speed_after = speed_before
        ob_after = ob_before
    else:
        # This should not happen if we have admissible controls, but keep for safety
        raise ValueError("No admissible (v, ω) pairs found in dynamic window")
        to_goal_before = speed_before = ob_before = 0.0
        to_goal_after = speed_after = ob_after = 0.0

    # Handle the robot stuck scenario
    if abs(best_u[0]) < config.robot_stuck_flag_cons \
            and abs(x[3]) < config.robot_stuck_flag_cons:
        best_u[1] = -config.max_delta_yaw_rate

    return (best_u, best_trajectory, dw,
            to_goal_before, speed_before, ob_before,
            to_goal_after, speed_after, ob_after,
            min_cost)


def closest_obstacle_on_curve(x, ob, v, omega, config):
    """
    Calculate the distance to the closest obstacle that intersects with the curvature
    without time/span limitations - checks the entire trajectory
    
    Parameters:
    x: current state
        [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    ob: obstacle positions
        [[x(m), y(m)], ...]
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
        
        # # Define a reasonable search distance for stationary obstacle detection
        # search_distance = config.obstacle_radius * 10  # or another reasonable value
        
        for i in range(len(ob)):
            obstacle = np.array([ob[i, 0], ob[i, 1]])
            obstacle_radius = config.obstacle_radius
            
            to_center = obstacle - np.array(start_pos)
            projection = np.dot(to_center, heading_vector)
            
            # Only consider obstacles in front of the robot within search distance
            if projection < 0:    #  or projection > search_distance:
                continue
                
            closest_approach = np.linalg.norm(to_center - projection * heading_vector)
            
            collision_threshold = 0
            if config.robot_type == RobotType.rectangle:
                robot_diagonal = math.sqrt((config.robot_length/2)**2 + (config.robot_width/2)**2)
                collision_threshold = obstacle_radius + robot_diagonal * side_clearnce_factor
            else:
                collision_threshold = obstacle_radius + config.robot_radius * side_clearnce_factor
                
            if closest_approach <= collision_threshold:
                # Distance to the edge of collision zone
                dist_to_intersection = max(0, projection - math.sqrt(
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
            obstacle_radius = config.obstacle_radius
            
            to_center = obstacle - np.array(start_pos)
            projection = np.dot(to_center, heading_vector)
            
            if projection < 0:
                continue
                
            closest_approach = np.linalg.norm(to_center - projection * heading_vector)
            
            collision_threshold = 0
            if config.robot_type == RobotType.rectangle:
                robot_diagonal = math.sqrt((config.robot_length/2)**2 + (config.robot_width/2)**2)
                collision_threshold = obstacle_radius + robot_diagonal * side_clearnce_factor
            else:
                collision_threshold = obstacle_radius + config.robot_radius * side_clearnce_factor
                
            if closest_approach > collision_threshold:
                continue
                
            dist_to_intersection = projection - math.sqrt(collision_threshold**2 - closest_approach**2)
            
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
            obstacle_radius = config.obstacle_radius
            
            dist_between_centers = np.linalg.norm(arc_center - obstacle_center)
            
            if config.robot_type == RobotType.rectangle:
                robot_diagonal = math.sqrt((config.robot_length/2)**2 + (config.robot_width/2)**2)
                collision_radius = obstacle_radius + robot_diagonal * side_clearnce_factor
            else:
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
                dist = angle_diff * radius
                time = dist / v if v > 0 else float("inf")
                
                if dist < min_dist:
                    min_dist = dist
                    min_time = time
                    
        return min_dist, min_time


def closest_obstacle_on_side(trajectory, ob, config):
    """
    Find the closest obstacle on the left or right side of the trajectory.
    Parameters:
        trajectory: predicted trajectory
            [[x, y, yaw, v, omega], ...]
        ob: obstacle positions 
            [[x(m), y(m)], ...]
        config: simulation configuration
    Returns:
        min_clearance: Minimum clearance to an obstacle. 
                       Returns <= 0 if collision is detected.
    """
    # Handle empty obstacle list
    if ob is None or ob.shape[0] == 0:
        return float("inf")

    ox = ob[:, 0]
    oy = ob[:, 1]
    # ob_radii = np.array(ob_radii)
    dx = trajectory[:, 0] - ox[:, None]
    dy = trajectory[:, 1] - oy[:, None]
    clearance = np.hypot(dx, dy)

    if config.robot_type == RobotType.rectangle:
        yaw = trajectory[:, 2]
        rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        rot = np.transpose(rot, [2, 0, 1])
        local_ob = ob[:, None] - trajectory[:, 0:2]
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        local_ob = np.array([local_ob @ x for x in rot])
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        upper_check = local_ob[:, 0] <= config.robot_length / 2
        right_check = local_ob[:, 1] <= config.robot_width / 2
        bottom_check = local_ob[:, 0] >= -config.robot_length / 2
        left_check = local_ob[:, 1] >= -config.robot_width / 2
        if (np.logical_and(np.logical_and(upper_check, right_check),
                           np.logical_and(bottom_check, left_check))).any():
            return 0  # Collision
    elif config.robot_type == RobotType.circle:
        if np.array(clearance <= (config.robot_radius + config.obstacle_radius)).any():
            return 0 # Collision
    
    # If no immediate collision, calculate minimum clearance
    if config.robot_type == RobotType.circle:
        # For circle, clearance is distance from robot edge to obstacle edge
        clearance = clearance - (config.robot_radius + config.obstacle_radius)
    else:
        # For rectangle, clearance is distance from robot center to obstacle edge
        # (Note: This is an approximation, original logic)
        clearance = clearance - config.obstacle_radius

    return np.min(clearance)


def calc_to_goal_cost(trajectory, goal):
    """
    calc to goal cost with angle difference
    Parameters:
        trajectory: predicted trajectory
            [[x, y, yaw, v, omega], ...]
        goal: goal position
            [x(m), y(m)]
    Returns:
        to goal cost
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


def check_collision_at_current_position_circle_approximation(x, ob, config):
    """
    Check if the robot at current position collides with any obstacles
    using circle approximation (same as DWA distance calculations)
    
    Parameters:
    x: current state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    ob: obstacle positions [[x(m), y(m)], ...]
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
    
    # Always use circle-circle collision detection (same as DWA)
    distances = np.linalg.norm(ob - robot_pos, axis=1)
    
    # Determine effective robot radius based on robot type
    if config.robot_type == RobotType.rectangle:
        # Use diagonal approximation (same as in DWA distance calculation)
        effective_robot_radius = math.sqrt((config.robot_length/2)**2 + (config.robot_width/2)**2)
    else:
        # Use actual robot radius
        effective_robot_radius = config.robot_radius
    
    collision_threshold = effective_robot_radius + config.obstacle_radius
    
    # Check for collisions
    collision_mask = distances <= collision_threshold
    if np.any(collision_mask):
        collision_index = np.argmin(distances)
        collision_distance = np.min(distances)
        return True, collision_index, collision_distance
    
    return False, None, float('inf')
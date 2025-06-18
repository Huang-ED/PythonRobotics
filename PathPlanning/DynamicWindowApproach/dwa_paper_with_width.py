"""

Mobile robot motion planning sample with Dynamic Window Approach

author: Atsushi Sakai (@Atsushi_twi), Göktuğ Karakaşlı

Modified by: Huang Erdong (@Huang-ED)
In this version of codes, 
    the methodology proposed in the original DWA paper is implemented.
Compared to dynamic_window_approach_paper.py,
    robot are considered in rectangle shape when indicated,
    and the obstacle is considered as a circle with radius.

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


def dwa_control(x, config, goal, ob):
    dw = calc_dynamic_window(x, config)
    (u, trajectory, dw, admissible, inadmissible,
     to_goal_before, speed_before, ob_before,
     to_goal_after, speed_after, ob_after,
     final_cost) = calc_control_and_trajectory(x, dw, config, goal, ob)
    return (u, trajectory, dw, admissible, inadmissible,
            to_goal_before, speed_before, ob_before,
            to_goal_after, speed_after, ob_after,
            final_cost)

def dwa_control_norm(x, config, goal, ob):
    dw = calc_dynamic_window(x, config)
    (u, trajectory, dw, admissible, inadmissible,
     to_goal_before, speed_before, ob_before,
     to_goal_after, speed_after, ob_after,
     final_cost) = calc_control_and_trajectory_norm(x, dw, config, goal, ob)
    return (u, trajectory, dw, admissible, inadmissible,
            to_goal_before, speed_before, ob_before,
            to_goal_after, speed_after, ob_after,
            final_cost)


class RobotType(Enum):
    circle = 0
    rectangle = 1


class Config:
    """
    simulation parameter class
    """

    def __init__(self):
        # robot parameter
        self.max_speed = 1.0  # [m/s]
        self.min_speed = 0.0  # [m/s]
        self.max_yaw_rate = 40.0 * math.pi / 180.0  # [rad/s]
        self.max_accel = 0.2  # [m/ss]
        self.max_delta_yaw_rate = 40.0 * math.pi / 180.0  # [rad/ss]
        self.v_resolution = 0.01  # [m/s]
        self.yaw_rate_resolution = 0.1 * math.pi / 180.0  # [rad/s]
        self.dt = 0.1  # [s] Time tick for motion prediction
        self.predict_time = 1.0  # [s]
        self.check_time = 100.0 # [s] Time to check for collision - a large number
        self.to_goal_cost_gain = 0.2
        self.speed_cost_gain = 1
        self.obstacle_cost_gain = 0.1
        self.robot_stuck_flag_cons = 0.001  # constant to prevent robot stucked
        self.robot_type = RobotType.rectangle
        self.catch_goal_dist = 0.5 # [m] goal radius
        self.obstacle_radius = 0.5  # [m] for collision check

        # if robot_type == RobotType.circle
        # Also used to check if goal is reached in both types
        self.robot_radius = 0.5  # [m] for collision check

        # if robot_type == RobotType.rectangle
        self.robot_width = 0.5  # [m] for collision check
        self.robot_length = 1.2  # [m] for collision check

    @property
    def robot_type(self):
        return self._robot_type

    @robot_type.setter
    def robot_type(self, value):
        if not isinstance(value, RobotType):
            raise TypeError("robot_type must be an instance of RobotType")
        self._robot_type = value


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
        admissible: list of admissible control inputs
        inadmissible: list of inadmissible control inputs
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
    
    to_goal_costs = []
    speed_costs = []
    ob_costs = []
    trajectories = []
    controls = []
    admissible = []
    inadmissible = []

    # evaluate all trajectory with sampled input in dynamic window
    for v in np.arange(dw[0], dw[1] + 1e-6, config.v_resolution):
        for y in np.arange(dw[2], dw[3] + 1e-6, config.yaw_rate_resolution):
            # Track all control inputs as inadmissible initially
            inadmissible.append([float(v), float(y)])
            
            # admissible velocities check
            dist, _ = closest_obstacle_on_curve(x.copy(), ob, v, y, config)
            # if v > math.sqrt(2*config.max_accel*dist):
            # if v**2 + config.max_accel * v * config.dt > 2 * config.max_accel * dist:
            if v**2 + 2 * config.max_accel * v * config.dt > 2 * config.max_accel * dist:
                continue
                
            # If we reach here, the control is admissible
            inadmissible.pop()  # Remove the last inadmissible entry
            admissible.append([float(v), float(y)])
            
            trajectory = predict_trajectory(x.copy(), v, y, config)
            
            # calc costs
            to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(trajectory, goal)
            speed_cost = config.speed_cost_gain * (config.max_speed - trajectory[-1, 3])
            ob_cost = float("inf") if dist == 0 else config.obstacle_cost_gain * (1 / dist)
            
            final_cost = to_goal_cost + speed_cost + ob_cost
            
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
        raise ValueError("No admissible (v, ω) pairs found in dynamic window")
        to_goal_before = speed_before = ob_before = 0.0
        to_goal_after = speed_after = ob_after = 0.0
        
    # Handle the robot stuck scenario as in the original function
    if abs(best_u[0]) < config.robot_stuck_flag_cons \
            and abs(x[3]) < config.robot_stuck_flag_cons:
        # to ensure the robot does not get stuck
        best_u[1] = -config.max_delta_yaw_rate

    return (best_u, best_trajectory, dw, admissible, inadmissible,
            to_goal_before, speed_before, ob_before,
            to_goal_after, speed_after, ob_after,
            min_cost)


def calc_control_and_trajectory_norm(x, dw, config, goal, ob):
    min_cost = float("inf")
    best_u = [0.0, 0.0]
    best_trajectory = np.array([x])
    best_index = -1

    to_goal_costs = []
    speed_costs = []
    ob_costs = []
    trajectories = []
    controls = []
    admissible = []
    inadmissible = []

    for v in np.arange(dw[0], dw[1], config.v_resolution):
        for y in np.arange(dw[2], dw[3], config.yaw_rate_resolution):
            dist, _ = closest_obstacle_on_curve(x.copy(), ob, v, y, config)
            inadmissible.append([float(v), float(y)])
            if v > math.sqrt(2 * config.max_accel * dist):
                continue
            admissible.append([float(v), float(y)])

            trajectory = predict_trajectory(x.copy(), v, y, config)
            to_goal_cost = calc_to_goal_cost(trajectory, goal)
            speed_cost = config.max_speed - trajectory[-1, 3]
            ob_cost = float("inf") if dist == 0 else 1 / dist

            to_goal_costs.append(to_goal_cost)
            speed_costs.append(speed_cost)
            ob_costs.append(ob_cost)
            trajectories.append(trajectory)
            controls.append([v, y])

    def normalize_costs(costs):
        total = sum(costs)
        # return [c / total if total != 0 else 0 for c in costs]
        if total == 0:  # Avoid division by zero
            return [1.0 / len(costs)] * len(costs)  # Equal weight
        return [c / total for c in costs]

    if to_goal_costs and speed_costs and ob_costs:
        norm_to_goal = normalize_costs(to_goal_costs)
        norm_speed = normalize_costs(speed_costs)
        norm_ob = normalize_costs(ob_costs)

        for i in range(len(controls)):
            final_cost = (config.to_goal_cost_gain * norm_to_goal[i] +
                          config.speed_cost_gain * norm_speed[i] +
                          config.obstacle_cost_gain * norm_ob[i])
            if final_cost < min_cost:
                min_cost = final_cost
                best_index = i
                best_u = controls[i]
                best_trajectory = trajectories[i]
    # print(f"len(to_goal_costs): {len(to_goal_costs)}\nlen(speed_costs): {len(speed_costs)}\nlen(ob_costs): {len(ob_costs)}")
    if len(to_goal_costs) == 0:
        raise ValueError("No admissible (v, ω) pairs found in dynamic window")

    if best_index != -1:
        to_goal_before = to_goal_costs[best_index]
        speed_before = speed_costs[best_index]
        ob_before = ob_costs[best_index]
        to_goal_after = norm_to_goal[best_index]
        speed_after = norm_speed[best_index]
        ob_after = norm_ob[best_index]
    else:
        raise ValueError("No admissible (v, ω) pairs found in dynamic window")
        to_goal_before = speed_before = ob_before = 0.0
        to_goal_after = speed_after = ob_after = 0.0

    return (best_u, best_trajectory, dw, admissible, inadmissible,
            to_goal_before, speed_before, ob_before,
            to_goal_after, speed_after, ob_after,
            min_cost)



def any_circle_overlap_with_box(circles, center, length, width, rot):
    """
    Check whether any of the given circles overlap with a rotated rectangular box.

    Parameters:
    - circles: 2D numpy array, shape (N, 3), where each row contains the 2D coordinate of the point and the radius of the circle (x, y, radius)
    - center: tuple (cx, cy), the 2D coordinate of the center of the box
    - length: float, length of the box
    - width: float, width of the box
    - rot: float, rotational angle of the box in radians

    Returns:
    - Boolean: True if any circle overlaps with the box, False otherwise
    """
    # Translate circle centers so that the center of the box is at the origin
    translated_circles = circles[:, :2] - np.array(center)
    
    # Rotation matrix for the negative of the box's rotation angle
    cos_rot = np.cos(-rot)
    sin_rot = np.sin(-rot)
    rotation_matrix = np.array([[cos_rot, -sin_rot], [sin_rot, cos_rot]])

    # Rotate all circle centers using matrix multiplication
    rotated_centers = translated_circles @ rotation_matrix.T

    # Half dimensions of the box
    half_length = length / 2
    half_width = width / 2

    # Calculate the distances of each circle center to the closest box boundary
    clamped_x = np.clip(rotated_centers[:, 0], -half_length, half_length)
    clamped_y = np.clip(rotated_centers[:, 1], -half_width, half_width)
    closest_points = np.vstack((clamped_x, clamped_y)).T
    
    # Calculate distances from circle centers to closest points on the box boundary
    distances = np.linalg.norm(rotated_centers - closest_points, axis=1)
    
    # Check if any distance is less than or equal to the circle radius
    overlap = distances <= circles[:, 2]

    return np.any(overlap)


def closest_obstacle_on_curve_old(x, ob, v, omega, config):
    """
    Calculate the distance to the closest obstacle that intersects with the curvature
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
    t = 0
    dist = 0
    while t < config.check_time:
        x = motion(x, [v, omega], config.dt / 10)
        # print(x, v, omega)
        if config.robot_type == RobotType.rectangle:
            ob_with_radius = np.concatenate([ob, np.full((len(ob), 1), config.obstacle_radius)], axis=1)
            if any_circle_overlap_with_box(ob_with_radius, x[:2], config.robot_length, config.robot_width, x[2]):
                return dist, t
        elif config.robot_type == RobotType.circle:
            distances = np.linalg.norm(ob - x[:2], axis=1)
            if np.any(distances <= (config.robot_radius + config.obstacle_radius)):
                return dist, t
        else:
            raise ValueError("Invalid robot type")
        t += config.dt / 10
        dist += v * config.dt / 10
    return float("Inf"), float("Inf")


def closest_obstacle_on_curve_with_check_time(x, ob, v, omega, config):
    """
    Was called "closest_obstacle_on_curve_math" after implementation
    Calculate the distance to the closest obstacle that intersects with the curvature
    in a mathematical way.
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
    start_pos = (x[0], x[1])
    heading = x[2]
    min_dist = float("inf")
    min_time = float("inf")

    # if abs(v) < 1e-6:
    #     heading = x[2] + omega * config.dt
    
    if abs(omega) < 1e-6:
    # if abs(omega) < 1e-6 or abs(v) < 1e-6:
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
                collision_threshold = obstacle_radius + robot_diagonal
            else:
                collision_threshold = obstacle_radius + config.robot_radius
            
            if closest_approach > collision_threshold:
                continue
                
            dist_to_intersection = projection - math.sqrt(collision_threshold**2 - closest_approach**2)
            
            if 0 <= dist_to_intersection < min_dist:
                min_dist = dist_to_intersection
                min_time = dist_to_intersection / v if v > 0 else float("inf")
        
        return min_dist, min_time
    
    else:
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
                collision_radius = obstacle_radius + robot_diagonal
            else:
                collision_radius = obstacle_radius + config.robot_radius
            
            if dist_between_centers > radius + collision_radius or dist_between_centers < radius - collision_radius:
                continue
            
            obstacle_angle = math.atan2(obstacle_center[1] - arc_center[1], 
                                      obstacle_center[0] - arc_center[0])
            
            obstacle_angle = obstacle_angle % (2 * math.pi)
            start_angle_norm = start_angle % (2 * math.pi)
            
            check_span = abs(omega) * config.check_time
            
            in_span = False
            if check_span > 2 * math.pi:
                in_span = True
            else:
                if omega > 0:
                    end_angle = (start_angle_norm + check_span) % (2 * math.pi)
                    if start_angle_norm <= end_angle:
                        in_span = start_angle_norm <= obstacle_angle <= end_angle
                    else:
                        in_span = obstacle_angle >= start_angle_norm or obstacle_angle <= end_angle
                else:
                    end_angle = (start_angle_norm - check_span) % (2 * math.pi)
                    if start_angle_norm >= end_angle:
                        in_span = end_angle <= obstacle_angle <= start_angle_norm
                    else:
                        in_span = obstacle_angle <= start_angle_norm or obstacle_angle >= end_angle

            if not in_span:
                continue
            
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
            
            in_arc1 = False
            in_arc2 = False
            
            if omega > 0:
                angle_diff1 = (angle1 - start_angle_norm) % (2 * math.pi)
                angle_diff2 = (angle2 - start_angle_norm) % (2 * math.pi)
            else:
                angle_diff1 = (start_angle_norm - angle1) % (2 * math.pi)
                angle_diff2 = (start_angle_norm - angle2) % (2 * math.pi)
            
            in_arc1 = angle_diff1 <= check_span
            in_arc2 = angle_diff2 <= check_span
            
            if in_arc1:
                dist1 = angle_diff1 * radius
                time1 = dist1 / v if v > 0 else float("inf")
                if dist1 < min_dist:
                    min_dist = dist1
                    min_time = time1
            
            if in_arc2:
                dist2 = angle_diff2 * radius
                time2 = dist2 / v if v > 0 else float("inf")
                if dist2 < min_dist:
                    min_dist = dist2
                    min_time = time2
    
    return min_dist, min_time


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
    start_pos = (x[0], x[1])
    heading = x[2]
    min_dist = float("inf")
    min_time = float("inf")

    # Special case: when velocity is zero or very small
    if abs(v) < 1e-5:
        # Check obstacles along the facing direction (straight line)
        heading_vector = np.array([math.cos(heading), math.sin(heading)])
        
        # Define a reasonable search distance for stationary obstacle detection
        search_distance = config.obstacle_radius * 10  # or another reasonable value
        
        for i in range(len(ob)):
            obstacle = np.array([ob[i, 0], ob[i, 1]])
            obstacle_radius = config.obstacle_radius
            
            to_center = obstacle - np.array(start_pos)
            projection = np.dot(to_center, heading_vector)
            
            # Only consider obstacles in front of the robot within search distance
            if projection < 0 or projection > search_distance:
                continue
                
            closest_approach = np.linalg.norm(to_center - projection * heading_vector)
            
            collision_threshold = 0
            if config.robot_type == RobotType.rectangle:
                robot_diagonal = math.sqrt((config.robot_length/2)**2 + (config.robot_width/2)**2)
                collision_threshold = obstacle_radius + robot_diagonal
            else:
                collision_threshold = obstacle_radius + config.robot_radius
                
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
                collision_threshold = obstacle_radius + robot_diagonal
            else:
                collision_threshold = obstacle_radius + config.robot_radius
                
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
                collision_radius = obstacle_radius + robot_diagonal
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


def dwa(x, goal, ob, config):
    '''
    Main function for the dynamic window approach.
    Parameters:
        gx: X-coordinate of the goal position.
        gy: Y-coordinate of the goal position.
        robot_type (RobotType): 
            Type of the robot. Default is RobotType.circle.
    Returns:
        None
    '''
    print(__file__ + " start!!")
    if save_animation_to_figs:
        cur_dir = os.path.dirname(__file__)
        fig_dir = os.path.join(cur_dir, 'figs')
        os.makedirs(fig_dir, exist_ok=False)
        i_fig = 0
        fig_path = os.path.join(fig_dir, 'frame_{}.png'.format(i_fig))

    trajectory = np.array(x)
    while True:
        u, predicted_trajectory = dwa_control(x, config, goal, ob)
        x = motion(x, u, config.dt)  # simulate robot
        trajectory = np.vstack((trajectory, x))  # store state history

        if show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], "-g")
            plt.plot(x[0], x[1], "xr")
            plt.plot(goal[0], goal[1], "xb")
            plt.plot(ob[:, 0], ob[:, 1], "ok")
            plot_robot(x[0], x[1], x[2], config)
            plot_arrow(x[0], x[1], x[2])
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.0001)

            if save_animation_to_figs:
                plt.savefig(fig_path)
                i_fig += 1
                fig_path = os.path.join(fig_dir, 'frame_{}.png'.format(i_fig))

        # check reaching goal
        dist_to_goal = math.hypot(x[0] - goal[0], x[1] - goal[1])
        if dist_to_goal <= config.catch_goal_dist:
            print("Goal!!")
            break

    print("Done")
    if show_animation:
        plt.plot(trajectory[:, 0], trajectory[:, 1], "-r")
        plt.pause(0.0001)

        if save_animation_to_figs:
            plt.savefig(fig_path)
            i_fig += 1
            fig_path = os.path.join(fig_dir, 'frame_{}.png'.format(i_fig))

        plt.show()


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


if __name__ == '__main__':
    # initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    x = np.array([0.0, 0.0, math.pi / 8.0, 0.0, 0.0])
    # goal position [x(m), y(m)]
    goal = np.array([10.0, 10.0])
    # obstacles [x(m) y(m), ....]
    ob = np.array([
        [-1, -1],
        [0, 2],
        [4.0, 2.0],
        [5.0, 4.0],
        [5.0, 5.0],
        [5.0, 6.0],
        [5.0, 9.0],
        [8.0, 9.0],
        [7.0, 9.0],
        [8.0, 10.0],
        [9.0, 11.0],
        [12.0, 13.0],
        [12.0, 12.0],
        [15.0, 15.0],
        [13.0, 13.0]
    ])
    config = Config()

    dwa(x, goal, ob, config)

    if save_costs_fig:
        time_list = [i * config.dt for i in range(len(to_goal_cost_list))]
        plt.figure(figsize=(32, 18))
        plt.plot(time_list, to_goal_cost_list, label='To goal cost')
        plt.plot(time_list, speed_cost_list, label='Speed cost')
        plt.plot(time_list, ob_cost_list, label='Obstacle cost')
        plt.xlabel('Time(s)')
        plt.ylabel('Cost')
        plt.legend()

        cur_dir = os.path.dirname(__file__)
        fig_path = os.path.join(cur_dir, 'costs.png')
        while os.path.exists(fig_path):
            try:
                i_fig += 1
            except NameError: # if i_fig is not defined, define it
                i_fig = 1
            fig_path = os.path.join(cur_dir, 'costs({}).png'.format(i_fig))
        plt.savefig(fig_path)
        print('Costs figure saved as costs.png')

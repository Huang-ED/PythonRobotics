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


# def calc_control_and_trajectory(x, dw, config, goal, ob):
#     """
#     calculation final input with dynamic window
#     Parameters:
#         x: current state
#             [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
#         dw: dynamic window
#             [v_min, v_max, yaw_rate_min, yaw_rate_max]
#         config: simulation configuration
#         goal: goal position
#             [x(m), y(m)]
#         ob: obstacle positions 
#             [[x(m), y(m)], ...]
#     Returns:
#         best_u: selected control input
#             [v(m/s), omega(rad/s)]
#         best_trajectory: predicted trajectory with selected input
#             [[x, y, yaw, v, omega], ...]
#     """

#     min_cost = float("inf")
#     best_u = [0.0, 0.0]
#     best_trajectory = np.array([x])

#     # evaluate all trajectory with sampled input in dynamic window
#     for v in np.arange(dw[0], dw[1], config.v_resolution):
#         for y in np.arange(dw[2], dw[3], config.yaw_rate_resolution):

#             # admissible velocities check
#             dist, _ = closest_obstacle_on_curve(x.copy(), ob, v, y, config)
#             if v > math.sqrt(2*config.max_accel*dist):
#                 continue
#             # if y > math.sqrt(2*config.max_delta_yaw_rate*dist):
#             #     continue

#             trajectory = predict_trajectory(x.copy(), v, y, config)
#             # calc cost
#             to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(trajectory, goal)
#             speed_cost = config.speed_cost_gain * (config.max_speed - trajectory[-1, 3])
#             if dist == 0:
#                 ob_cost = float("Inf")
#             else:
#                 ob_cost = config.obstacle_cost_gain * (1 / dist)

#             final_cost = to_goal_cost + speed_cost + ob_cost

#             # search minimum trajectory
#             if min_cost >= final_cost:
#                 min_cost = final_cost
#                 best_u = [v, y]
#                 best_trajectory = trajectory

#                 if save_costs_fig:
#                     to_goal_cost_list.append(to_goal_cost)
#                     speed_cost_list.append(speed_cost)
#                     ob_cost_list.append(ob_cost)

#                 if abs(best_u[0]) < config.robot_stuck_flag_cons \
#                         and abs(x[3]) < config.robot_stuck_flag_cons:
#                     # to ensure the robot do not get stuck in
#                     # best v=0 m/s (in front of an obstacle) and
#                     # best omega=0 rad/s (heading to the goal with
#                     # angle difference of 0)
#                     best_u[1] = -config.max_delta_yaw_rate
#     return best_u, best_trajectory


def calc_control_and_trajectory(x, dw, config, goal, ob):
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
    print(f"len(to_goal_costs): {len(to_goal_costs)}\nlen(speed_costs): {len(speed_costs)}\nlen(ob_costs): {len(ob_costs)}")
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
    rotated_centers = translated_circles @ rotation_matrix

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


def closest_obstacle_on_curve(x, ob, v, omega, config):
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
        x = motion(x, [v, omega], config.dt)
        if config.robot_type == RobotType.rectangle:
            ob_with_radius = np.concatenate([ob, np.full((len(ob), 1), config.obstacle_radius)], axis=1)
            if any_circle_overlap_with_box(ob_with_radius, x[:2], config.robot_length, config.robot_width, x[2]):
                return dist, t
        elif config.robot_type == RobotType.circle:
            distances = np.linalg.norm(ob - x[:2], axis=1)
            if np.any(distances <= config.robot_radius):
                return dist, t
        else:
            raise ValueError("Invalid robot type")
        t += config.dt
        dist += v * config.dt
    return float("Inf"), float("Inf")


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

import os, sys
rpath = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.append(rpath)

from PathPlanning.AStar import theta_star
from PathPlanning.DynamicWindowApproach import dwa_paper_with_width as dwa

import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
import warnings

# plt.switch_backend('Agg')
show_animation = True
save_animation_to_figs = True

"""
v3: Obstacles are put onto the A* path, to simulate dynamic obstacles and test the performance of DWA.
v4: The robot will head to the next local goal once it is "close enough" the current local goal. 
v5: For collision check,
    the robot is a rectangle with width and length when specified, rather than always a circle.
    Obstacles are circles with radius. 
v5_video: Local goals are selected waypoints (1 in every 10 A* waypoints) on the A* path. 
v6: Fix a radius around the robot to determine the local goal. 
v7: Replace A* with Theta*. 
"""

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
        self.check_time = 100.0 # [s] Time to check for collision - a large number
        self.to_goal_cost_gain = 0.4
        self.speed_cost_gain = 1
        self.obstacle_cost_gain = 0.08
        self.robot_stuck_flag_cons = 0.001  # constant to prevent robot stucked
        self.robot_type = dwa.RobotType.rectangle
        self.catch_goal_dist = 0.5  # [m] goal radius
        self.catch_turning_point_dist = 1.0  # [m] local goal radius
        self.obstacle_radius = 0.5  # [m] for collision check

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
        if not isinstance(value, dwa.RobotType):
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



config = Config()
config_plot = Config()
config_plot.robot_width *= 2
config_plot.robot_length *= 2

# ----- Set up the map -----
ox, oy = [], []
for i in range(60):
    ox.append(i)
    oy.append(0.0)
for i in range(60):
    ox.append(60.0)
    oy.append(i)
for i in range(61):
    ox.append(i)
    oy.append(60.0)
for i in range(61):
    ox.append(0.0)
    oy.append(i)
for i in range(40):
    ox.append(20.0)
    oy.append(i)
for i in range(40):
    ox.append(40.0)
    oy.append(60.0 - i)
ob = np.array([ox, oy]).transpose()

ob_astar = np.concatenate([
    ob,
    ob+np.array([1, 0]),
    ob+np.array([0, 1]),
    ob+np.array([-1, 0]),
    ob+np.array([0, -1]),
    ob+np.array([2, 0]),
    ob+np.array([0, 2]),
    ob+np.array([-2, 0]),
    ob+np.array([0, -2])
], axis=0)
ob_astar = np.unique(ob_astar, axis=0)

# Map for DWA
'''
to be implemented: boundary extraction + local map
'''

# ----- Set up the start and goal positions -----
# Set the start and goal positions
sx, sy = 10.0, 10.0
gx, gy = 50.0, 50.0

# Plot the map
if show_animation:  # pragma: no cover
    plt.figure(figsize=(10, 10))
    if save_animation_to_figs:
        cur_dir = os.path.dirname(__file__)
        fig_dir = os.path.join(cur_dir, 'figs_v7.2')
        os.makedirs(fig_dir, exist_ok=False)
        i_fig = 0

    else:
        fig_dir = None
        i_fig = 0
    # plt.plot(ox, oy, ".k")
    for (x, y) in ob:
        circle = plt.Circle((x, y), config.robot_radius, color="k")
        plt.gca().add_patch(circle)
    plt.plot(sx, sy, "or", zorder=10)
    plt.plot(gx, gy, "sr", zorder=10)
    plt.grid(True)
    plt.axis("equal")

# ----- Run A* path planning -----
theta_star_planner = theta_star.ThetaStarPlanner(
    ob=ob_astar, resolution=1.0, 
    rr=max(config.robot_width, config.robot_length),
    min_x=min(*ox, sx-2, gx-2), min_y=min(*oy, sy-2, gy-2),
    max_x=max(*ox, sx+2, gx+2), max_y=max(*oy, sy+2, gy+2),
    # save_animation_to_figs=save_animation_to_figs,
    save_animation_to_figs=False,
    fig_dir=fig_dir
)
rx, ry = theta_star_planner.planning(sx, sy, gx, gy, curr_i_fig=i_fig)  # full A* path (reversed)
road_map = np.array([rx, ry]).transpose()[::-1]  # full A* path
# print(road_map)

# Plot the A* path
if show_animation:  # pragma: no cover
    plt.plot(rx, ry, "-b")[0]
    plt.pause(0.001)
    if save_animation_to_figs:
        i_fig = theta_star_planner.i_fig # update i_fig
        plt.savefig(os.path.join(fig_dir, 'frame_{}.png'.format(i_fig)))
        i_fig += 1


# ----- Put new obstacles on the A* path -----
new_ob = np.array([
    [14, 14.5],
    [16, 19.5],
    [16, 24.5],
    [16, 29.5],
    [16, 34.5],
    # [17.5, 39.5],
    [21.5, 40.5],
    [26.5, 36.5],
    [31.5, 31.5],
    [34.5, 26.5],
    [36.5, 21.5],
    [40.5, 19.5],
    [43.5, 22.5],
    [44, 27.5],
    [44, 32.5],
    [44.5, 37.5],
    [46.5, 42.5],
    [49, 47.5]
])
new_ob1 = new_ob + np.array([0.5, 0.5])
new_ob2 = new_ob + np.array([-0.5, -0.5])
new_ob3 = new_ob + np.array([0.5, -0.5])
new_ob4 = new_ob + np.array([-0.5, 0.5])
new_ob = np.concatenate((new_ob1, new_ob2, new_ob3, new_ob4), axis=0)
ob = np.append(ob, new_ob, axis=0)
if show_animation:  # pragma: no cover
    # plt.plot(new_ob[:,0], new_ob[:,1], ".k")
    for (x, y) in new_ob:
        circle = plt.Circle((x, y), config.robot_radius, color="k", zorder=10)
        plt.gca().add_patch(circle)


# ----- Run DWA path planning -----
x = np.array([sx, sy, - math.pi / 8.0, 1.0, 0.0])
# config = Config()

print(__file__ + " start!!")
trajectory = np.array(x)

if show_animation:  # pragma: no cover
    # for stopping simulation with the esc key.
    plt.gcf().canvas.mpl_connect(
        'key_release_event',
        lambda event: [exit(0) if event.key == 'escape' else None])
    plt_elements = []


for i_turning_point, turning_point in enumerate(road_map):
    if i_turning_point == 0:  # skip the start point
        continue

    while True:

        ## Determine the local goal
        intersection_points = line_circle_intersection(
            road_map[i_turning_point-1], road_map[i_turning_point], (x[0], x[1]), config.dist_localgoal
        )
        # remove intersection points that are behind my current position
        for intersection_point in intersection_points:
            dist_intersection_turning = math.hypot(intersection_point[0] - turning_point[0], intersection_point[1] - turning_point[1])
            dist_to_turning_point = math.hypot(x[0] - turning_point[0], x[1] - turning_point[1])
            if dist_intersection_turning > dist_to_turning_point:
                intersection_points.remove(intersection_point)

        if len(intersection_points) == 0:
            dist_to_turning_point = math.hypot(x[0] - turning_point[0], x[1] - turning_point[1])
            if dist_to_turning_point <= config.dist_localgoal:
                dwagoal = turning_point
            else:
                """ To be implemented: Re-plan the path"""
                raise ValueError("No intersection points found")
        elif len(intersection_points) == 1:
            dwagoal = intersection_points[0]
        elif len(intersection_points) == 2:
            # Choose the intersection point that is closer to the turning point
            dist0 = math.hypot(intersection_points[0][0] - turning_point[0], intersection_points[0][1] - turning_point[1])
            dist1 = math.hypot(intersection_points[1][0] - turning_point[0], intersection_points[1][1] - turning_point[1])
            if dist0 < dist1:
                dwagoal = intersection_points[0]
            else:
                dwagoal = intersection_points[1]
            warnings.warn(f"""\
2 intersection points found: {intersection_points}.
This means the robot is too far away from the turning point ({turning_point}),
that the distance is longer than both intersection points. By right, this shall not happen.\
""", UserWarning)
        else:
            raise ValueError(f"More than 2 intersection points found - {intersection_points}")


        ## Execute DWA
        u, predicted_trajectory = dwa.dwa_control(x, config, dwagoal, ob)
        x = dwa.motion(x, u, config.dt)  # simulate robot
        trajectory = np.vstack((trajectory, x))  # store state history

        if show_animation:  # pragma: no cover
            for ele in plt_elements:
                ele.remove()
            plt_elements = []
            plt_elements.append(plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], "-g")[0])
            plt_elements.append(plt.plot(x[0], x[1], "xr")[0])
            plt_elements.extend(dwa.plot_robot(x[0], x[1], x[2], config_plot))
            plt_elements.extend(dwa.plot_arrow(x[0], x[1], x[2]))
            plt_elements.append(plt.plot(trajectory[:, 0], trajectory[:, 1], "-r")[0])

            plt_elements.append(plt.plot(dwagoal[0], dwagoal[1], "Db")[0])
            plt.pause(0.001)

            if save_animation_to_figs:
                plt.savefig(os.path.join(fig_dir, 'frame_{}.png'.format(i_fig)))
                i_fig += 1

        
        ## Check reaching turning point
        dist_to_turning_point = math.hypot(x[0] - turning_point[0], x[1] - turning_point[1])
        if i_turning_point == len(road_map) - 1:
            if dist_to_turning_point <= config.catch_goal_dist:
                print("Goal!!")
                break
        else:
            if dist_to_turning_point <= config.catch_turning_point_dist:
                print("Local goal!!")
                break

print("Done")
if show_animation:  # pragma: no cover
    plt.show()
        


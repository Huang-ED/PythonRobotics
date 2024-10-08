import os, sys
rpath = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.append(rpath)

from PathPlanning.AStar import a_star
from PathPlanning.DynamicWindowApproach import dynamic_window_approach_paper as dwa

import math
import numpy as np
import matplotlib.pyplot as plt

show_animation = True
save_animation_to_figs = False

"""
In this version of codes, a lower resolution A* path is used to guide the DWA path.
Obstacles are then put onto the A* path, to test the performance of DWA.
The robot will head to the next local goal once it is close enough the current local goal.
This version introduces a catch_localgoal_dist parameter that is larger than the catch_goal_dist. 
"""

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
        self.robot_type = dwa.RobotType.rectangle
        self.catch_goal_dist = 0.5  # [m] goal radius
        self.catch_localgoal_dist = 1.0  # [m] local goal radius

        # if robot_type == RobotType.circle
        # Also used to check if goal is reached in both types
        self.robot_radius = 0.5  # [m] for collision check

        # if robot_type == RobotType.rectangle
        self.robot_width = 1.0  # [m] for collision check
        self.robot_length = 2.0  # [m] for collision check

    @property
    def robot_type(self):
        return self._robot_type

    @robot_type.setter
    def robot_type(self, value):
        if not isinstance(value, dwa.RobotType):
            raise TypeError("robot_type must be an instance of RobotType")
        self._robot_type = value



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

# ----- Set up the start and goal positions -----
# Set the start and goal positions
sx, sy = 10.0, 10.0
gx, gy = 50.0, 50.0

# Plot the map
if show_animation:  # pragma: no cover
    if save_animation_to_figs:
        cur_dir = os.path.dirname(__file__)
        fig_dir = os.path.join(cur_dir, 'figs')
        os.makedirs(fig_dir, exist_ok=False)
        i_fig = 0
        fig_path = os.path.join(fig_dir, 'frame_{}.png'.format(i_fig))
    plt.plot(ox, oy, ".k")
    plt.plot(sx, sy, "og")
    plt.plot(gx, gy, "*b")
    plt.grid(True)
    plt.axis("equal")

# ----- Run A* path planning -----
a_star_planner = a_star.AStarPlanner(
    ob, resolution=5.0, rr=1.0,
    min_x=min(*ox, sx-2, gx-2), min_y=min(*oy, sy-2, gy-2),
    max_x=max(*ox, sx+2, gx+2), max_y=max(*oy, sy+2, gy+2)
)
rx, ry = a_star_planner.planning(sx, sy, gx, gy)

road_map = np.array([rx, ry]).transpose()[::-1]
# print(road_map)

# Plot the path
if show_animation:  # pragma: no cover
    # plt.plot(rx, ry, "-r")
    plt.plot(rx, ry, "xb")
    plt.pause(0.001)

    if save_animation_to_figs:
        plt.savefig(fig_path)
        i_fig += 1
        fig_path = os.path.join(fig_dir, 'frame_{}.png'.format(i_fig))

# ----- Put new obstacles on the A* path -----
new_ob = np.array([
    [12.5, 12.5], 
    [15.0, 17.5], 
    [15.0, 22.5], 
    [15.0, 27.5], 
    [15.0, 32.5], 
    [15.0, 37.5], 
    [17.5, 42.5], 
    [22.5, 42.5], 
    [27.5, 37.5], 
    [32.5, 32.5], 
    [35.0, 27.5], 
    [35.0, 22.5], 
    [37.5, 17.5], 
    [42.5, 17.5], 
    [45.0, 22.5], 
    [45.0, 27.5], 
    [45.0, 32.5], 
    [45.0, 37.5], 
    [45.0, 42.5], 
    [47.5, 47.5]
])
# new_ob1 = new_ob + np.array([0.5, 0.5])
# new_ob2 = new_ob + np.array([-0.5, -0.5])
# new_ob3 = new_ob + np.array([0.5, -0.5])
# new_ob4 = new_ob + np.array([-0.5, 0.5])
# new_ob = np.concatenate((new_ob1, new_ob2, new_ob3, new_ob4), axis=0)
ob = np.append(ob, new_ob, axis=0)
if show_animation:  # pragma: no cover
    plt.plot(new_ob[:,0], new_ob[:,1], ".k")


# ----- Run DWA path planning -----
x = np.array([sx, sy, math.pi / 8.0, 0.0, 0.0])
config = Config()

print(__file__ + " start!!")
trajectory = np.array(x)

if show_animation:  # pragma: no cover
    # for stopping simulation with the esc key.
    plt.gcf().canvas.mpl_connect(
        'key_release_event',
        lambda event: [exit(0) if event.key == 'escape' else None])
    plt_elements = []

for i_goal, dwagoal in enumerate(road_map):
    if i_goal == 0:  # Skip the start point
        continue

    while True:
        u, predicted_trajectory = dwa.dwa_control(x, config, dwagoal, ob)
        x = dwa.motion(x, u, config.dt)  # simulate robot
        trajectory = np.vstack((trajectory, x))  # store state history

        if show_animation:  # pragma: no cover
            for ele in plt_elements:
                ele.remove()
            plt_elements = []
            plt_elements.append(plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], "-g")[0])
            plt_elements.append(plt.plot(x[0], x[1], "xr")[0])
            plt_elements.extend(dwa.plot_robot(x[0], x[1], x[2], config))
            plt_elements.extend(dwa.plot_arrow(x[0], x[1], x[2]))
            plt_elements.append(plt.plot(trajectory[:, 0], trajectory[:, 1], "-r")[0])
            plt.pause(0.001)

            if save_animation_to_figs:
                plt.savefig(fig_path)
                i_fig += 1
                fig_path = os.path.join(fig_dir, 'frame_{}.png'.format(i_fig))

        # check reaching goal
        dist_to_goal = math.hypot(x[0] - dwagoal[0], x[1] - dwagoal[1])
        if i_goal == len(road_map) - 1:
            if dist_to_goal <= config.catch_goal_dist:
                print("Goal!!")
                break
        else:
            if dist_to_goal <= config.catch_localgoal_dist:
                print("Local goal!!")
                break
    
print("Done")
if show_animation:  # pragma: no cover
    plt.show()
        


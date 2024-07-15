import os, sys
rpath = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.append(rpath)

from PathPlanning.AStar import a_star
from PathPlanning.DynamicWindowApproach import dynamic_window_approach as dwa

import math
import numpy as np
import matplotlib.pyplot as plt

show_animation = True

# Set up the map
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

def main():
    # Set the start and goal positions
    sx, sy = 10.0, 10.0
    gx, gy = 50.0, 50.0

    # Plot the map
    if show_animation:  # pragma: no cover
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "xb")
        plt.grid(True)
        plt.axis("equal")

    # Run A* path planning
    a_star_planner = a_star.AStarPlanner(
        ob, resolution=2.0, rr=1.0,
        min_x=min(*ox, sx-2, gx-2), min_y=min(*oy, sy-2, gy-2),
        max_x=max(*ox, sx+2, gx+2), max_y=max(*oy, sy+2, gy+2)
    )
    rx, ry = a_star_planner.planning(sx, sy, gx, gy)
    # Plot the path
    if show_animation:  # pragma: no cover
        plt.plot(rx, ry, "-r")
        plt.pause(0.001)
        # plt.show()

    # # Run DWA path planning
    x = np.array([sx, sy, math.pi / 8.0, 0.0, 0.0])
    config = dwa.Config()
    config.robot_type = dwa.RobotType.rectangle
    config.robot_width = 1.0
    config.robot_length = 2.0
    goal = np.array([gx, gy])
    # dwa.dwa(x, goal, ob, config)

    print(__file__ + " start!!")

    if show_animation:  # pragma: no cover
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])

    trajectory = np.array(x)
    while True:
        plt_elements = []
        u, predicted_trajectory = dwa.dwa_control(x, config, goal, ob)
        x = dwa.motion(x, u, config.dt)  # simulate robot
        trajectory = np.vstack((trajectory, x))  # store state history

        if show_animation:
            # plt.cla()
            # # for stopping simulation with the esc key.
            # plt.gcf().canvas.mpl_connect(
            #     'key_release_event',
            #     lambda event: [exit(0) if event.key == 'escape' else None])
            plt_elements.append(plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], "-g")[0])
            plt_elements.append(plt.plot(x[0], x[1], "xr")[0])
            plt_elements.extend(dwa.plot_robot(x[0], x[1], x[2], config))
            plt_elements.extend(dwa.plot_arrow(x[0], x[1], x[2]))
            plt.pause(0.0001)
            for ele in plt_elements:
                ele.remove()

        # check reaching goal
        dist_to_goal = math.hypot(x[0] - goal[0], x[1] - goal[1])
        if dist_to_goal <= config.catch_goal_dist:
            print("Goal!!")
            break

    print("Done")
    if show_animation:
        plt.plot(trajectory[:, 0], trajectory[:, 1], "-r")
        plt.pause(0.0001)
        plt.show()


if __name__ == '__main__':
    main()

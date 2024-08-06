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
save_animation_to_figs = False

"""
In this version of codes, a lower resolution A* path is used to guide the DWA path.
"""

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
    if save_animation_to_figs:
        cur_dir = os.path.dirname(__file__)
        fig_dir = os.path.join(cur_dir, 'figs')
        os.makedirs(fig_dir, exist_ok=False)
        i_fig = 0
        fig_path = os.path.join(fig_dir, 'frame_{}.png'.format(i_fig))

    # Set the start and goal positions
    sx, sy = 10.0, 10.0
    gx, gy = 50.0, 50.0

    # Plot the map
    if show_animation:  # pragma: no cover
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "*b")
        plt.grid(True)
        plt.axis("equal")

    # Run A* path planning
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
        plt.plot(rx, ry, "-r")
        plt.pause(0.001)

        if save_animation_to_figs:
            plt.savefig(fig_path)
            i_fig += 1
            fig_path = os.path.join(fig_dir, 'frame_{}.png'.format(i_fig))

    # # Run DWA path planning
    x = np.array([sx, sy, math.pi / 8.0, 0.0, 0.0])
    config = dwa.Config()
    config.robot_type = dwa.RobotType.rectangle
    config.robot_width = 1.0
    config.robot_length = 2.0

    print(__file__ + " start!!")
    trajectory = np.array(x)

    if show_animation:  # pragma: no cover
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        plt_elements = []

    for dwagoal in road_map[1:]:
        while True:
            u, predicted_trajectory = dwa.dwa_control(x, config, dwagoal, ob)
            x = dwa.motion(x, u, config.dt)  # simulate robot
            trajectory = np.vstack((trajectory, x))  # store state history

            if show_animation:
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
            if dist_to_goal <= config.catch_goal_dist:
                print("Goal!!")
                break
        
    print("Done")
    if show_animation: # pragma: no cover
        plt.show()
        


if __name__ == '__main__':
    main()

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

import json
from datetime import datetime
import traceback

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
    Add boundary extraction for DWA map.
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
        # self.check_time = 100.0 # [s] Time to check for collision - a large number
        self.to_goal_cost_gain = 0.4
        self.speed_cost_gain = 1
        self.obstacle_cost_gain = 0.10
        self.robot_stuck_flag_cons = 0.001  # constant to prevent robot stucked
        self.robot_type = dwa.RobotType.rectangle
        self.catch_goal_dist = 0.5  # [m] goal radius
        self.catch_turning_point_dist = 1.0  # [m] local goal radius
        self.obstacle_radius = 0.5  # [m] for collision check

        self.max_obstacle_cost_dist = 5.0  # [m] max distance to obstacles for cost calculation

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


if __name__ == '__main__':

    config = Config()
    config_plot = Config()
    # config_plot.robot_width *= 2
    # config_plot.robot_length *= 2

    # ----- Set up the map -----
    ## Load the map from image
    image_path = "EnvData/AISData_20240827/land_shapes_ht_crop.png"
    arr = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    arr = cv2.resize(arr, (100, 100))
    _, arr = cv2.threshold(arr, 128, 1, cv2.THRESH_BINARY)

    ## add boundary obstacles
    arr[0, :] = 0  # Top edge
    arr[-1, :] = 0  # Bottom edge
    arr[:, 0] = 0  # Left edge
    arr[:, -1] = 0  # Right edge
    # arr = cv2.erode(arr, kernel=np.ones((5, 5), np.uint8), iterations=1)
    ob = np.argwhere(arr == 0)

    ## imread direction and plot direction are different
    ob[:, [0, 1]] = ob[:, [1, 0]]  # Swap columns to match (x, y)
    ob[:, 1] = arr.shape[0] - ob[:, 1] - 1  # Flip y-axis
    ox, oy = ob[:, 0], ob[:, 1]

    # Map for A*
    arr_astar = cv2.erode(arr, kernel=np.ones((3, 3), np.uint8), iterations=1)
    # arr_astar = arr
    ob_astar = np.argwhere(arr_astar == 0)
    ob_astar[:, [0, 1]] = ob_astar[:, [1, 0]]  # Swap columns to match (x, y)
    ob_astar[:, 1] = arr_astar.shape[0] - ob_astar[:, 1] - 1  # Flip y-axis
    ox_astar, oy_astar = ob_astar[:, 0], ob_astar[:, 1]

    # Map for DWA
    arr = 1 - arr
    eroded_arr = cv2.erode(arr, kernel=np.ones((3, 3), np.uint8), iterations=1)
    arr_dwa = cv2.subtract(arr, eroded_arr)
    arr_dwa = 1 - arr_dwa

    ob_dwa = np.argwhere(arr_dwa == 0)
    ob_dwa[:, [0, 1]] = ob_dwa[:, [1, 0]]  # Swap columns to match (x, y)
    ob_dwa[:, 1] = arr_dwa.shape[0] - ob_dwa[:, 1] - 1  # Flip y-axis
    # ox_dwa, oy_dwa = ob_dwa[:, 0], ob_dwa[:, 1]


    # ----- Set up the start and goal positions -----
    # Set the start and goal positions
    sx, sy = 70, 90
    gx, gy = 60, 10

    # Plot the map
    if show_animation:  # pragma: no cover
        plt.figure(figsize=(10, 10))
        if save_animation_to_figs:
            cur_dir = os.path.dirname(__file__)
            fig_dir = os.path.join(cur_dir, 'figs_v7.3.9-test2_vid1')
            os.makedirs(fig_dir, exist_ok=False)
            i_fig = 0

        else:
            fig_dir = None
            i_fig = 0
        # plt.plot(ox, oy, ".k")
        for (x, y) in ob:
            circle = plt.Circle((x, y), config.robot_radius, color="darkgrey")
            plt.gca().add_patch(circle)
        for (x, y) in ob_dwa:
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
        [67., 75.], [67., 74.], [66., 75.], [66., 74.],
        [57., 65.], [57., 66.], [58., 65.], [58., 66.],
        [43., 44.], [42., 43.], [43., 43.], [42., 44.],
        [67., 23.], [67., 24.], [68., 23.], [68., 24.]
    ])
    new_ob = np.array([
        [80., 63.], [80., 64.], [81., 63.], [81., 64.],
        [55., 68.], [55., 69.], [56., 68.], [56., 69.],
        [43., 44.], [42., 43.], [43., 43.], [42., 44.],
        [67., 23.], [67., 24.], [68., 23.], [68., 24.],
    ])
    ob_dwa = np.append(ob_dwa, new_ob, axis=0)
    if show_animation:  # pragma: no cover
        # plt.plot(new_ob[:,0], new_ob[:,1], ".k")
        for (x, y) in new_ob:
            circle = plt.Circle((x, y), config.robot_radius, color="k", zorder=10)
            plt.gca().add_patch(circle)


    # ----- Run DWA path planning -----
    x = np.array([sx, sy, - math.pi / 8.0, 0.0, 0.0])
    x = np.array([47, 56, - math.pi*3/4, 0.5, 0.])
    road_map = road_map[1:]    # roadmap remove the first few points

    print(__file__ + " start!!")
    trajectory = np.array(x)

    if show_animation:  # pragma: no cover
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        plt_elements = []

    # Initialize data logging
    log_data = []
    iteration = 0

    try:
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
                (u, predicted_trajectory, dw, admissible, inadmissible, 
                 to_goal_before, speed_before, ob_before, to_goal_after, 
                 speed_after, ob_after, final_cost) = dwa.dwa_control(
                    x, config, dwagoal, ob_dwa
                )

                # Record data for this iteration
                log_entry = {
                    "iteration": iteration,
                    "chosen_v": float(u[0]),
                    "chosen_omega": float(u[1]),
                    "local_goal_x": float(dwagoal[0]),
                    "local_goal_y": float(dwagoal[1]),
                    "final_cost": float(final_cost),
                    "to_goal_cost_before": float(to_goal_before),
                    "speed_cost_before": float(speed_before),
                    "ob_cost_before": float(ob_before),
                    "to_goal_cost_after": float(to_goal_after),
                    "speed_cost_after": float(speed_after),
                    "ob_cost_after": float(ob_after),
                    "dynamic_window": [float(dw[0]), float(dw[1]), float(dw[2]), float(dw[3])],
                    "admissible": admissible,
                    "inadmissible": inadmissible
                }
                log_data.append(log_entry)
                iteration += 1

                x = dwa.motion(x, u, config.dt)  # simulate robot
                trajectory = np.vstack((trajectory, x))  # store state history
                
                # ADD COLLISION DETECTION HERE (after motion, before animation)
                collision_detected, obstacle_index, collision_distance = dwa.check_collision_at_current_position_circle_approximation(x, ob_dwa, config)

                if collision_detected:
                    # Calculate effective radius for error message
                    if config.robot_type == dwa.RobotType.rectangle:
                        effective_radius = math.sqrt((config.robot_length/2)**2 + (config.robot_width/2)**2)
                        robot_shape_info = f"rectangle (effective radius: {effective_radius:.3f}m)"
                    else:
                        effective_radius = config.robot_radius
                        robot_shape_info = f"circle (radius: {effective_radius:.3f}m)"
                    
                    # Print collision details and terminate
                    error_message = f"""
COLLISION DETECTED!
- Iteration: {iteration-1}
- Robot position: ({x[0]:.3f}, {x[1]:.3f})
- Robot heading: {x[2]:.3f} rad ({math.degrees(x[2]):.1f}°)
- Robot shape: {robot_shape_info}
- Colliding obstacle index: {obstacle_index}
- Colliding obstacle position: ({ob_dwa[obstacle_index][0]:.3f}, {ob_dwa[obstacle_index][1]:.3f})
- Distance to obstacle center: {collision_distance:.3f}m
- Collision threshold: {effective_radius + config.obstacle_radius:.3f}m
- Simulation terminated due to collision.
                """
                    
                    # Terminate program with error
                    # sys.exit(error_message)
                    warnings.warn(error_message, UserWarning)

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

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        traceback.print_exc()  # Print stack trace for debugging

    except KeyboardInterrupt:
        print("Simulation manually stopped.")

    finally:
        # Always save data before exiting
        if log_data:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # details_filename = f"dwa_log_details_{timestamp}.json"
            details_filename = os.path.join("Logs", f"dwa_log_details_{timestamp}", "log_details.json")
            os.makedirs(os.path.dirname(details_filename), exist_ok=False)
            with open(details_filename, 'w') as f:
                json.dump({
                    "log_entries": log_data,
                    "trajectory": trajectory.tolist()  # Add trajectory data
                }, f, indent=2)
            print(f"Data saved to {details_filename}")

    print("Done")
    if show_animation:  # pragma: no cover
        plt.show()
            


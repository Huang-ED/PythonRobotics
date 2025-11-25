import os, sys
rpath = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.append(rpath)

import theta_star
# Import the new DWA logic file
import dwa_paper_split as dwa
from dwa_paper_split import Config, line_circle_intersection
from map_manager import MapManager

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
fig_folder = 'figs_v8.2.3.4-video1-split' # Changed fig folder
map_config_file = os.path.join("PathPlanning", "DWAT_v2_split", "map_config", "map_config_video1.json")
# fig_folder = 'figs_v8.2.3-vid2'
# map_config_file = os.path.join("PathPlanning", "DWAT_v2", "map_config", "map_config_video2.json")
# fig_folder = 'figs_v8.2.3-simple2'
# map_config_file = os.path.join("PathPlanning", "DWAT_v2", "map_config", "map_config_simple2.json")


if __name__ == '__main__':

    config = Config()
    config_plot = Config()
    # config_plot.robot_width *= 2
    # config_plot.robot_length *= 2

    # ----- Set up the map -----
    # Load map configuration
    with open(map_config_file, 'r') as f:
        map_config = json.load(f)
    
    # Initialize map manager
    map_manager = MapManager(config)
    map_manager.load_map_from_image(map_config['image_path'], map_size=map_config['map_size'])
    map_manager.add_dynamic_obstacles(map_config['dynamic_obstacles'])
    
    # Get start and goal positions
    sx, sy = map_config['start_position']
    gx, gy = map_config['goal_position']

    # Plot the map
    if show_animation:  # pragma: no cover
        plt.figure(figsize=(10, 10))
        if save_animation_to_figs:
            cur_dir = os.path.dirname(__file__)
            fig_dir = os.path.join(cur_dir, fig_folder)
            os.makedirs(fig_dir, exist_ok=True) # Use exist_ok=True for easier reruns
            i_fig = 0

        else:
            fig_dir = None
            i_fig = 0
        # plt.plot(ox, oy, ".k")
        for (x, y) in map_manager.static_obstacles:
            circle = plt.Circle((x, y), config.robot_radius, color="darkgrey")
            plt.gca().add_patch(circle)
        for (x, y) in map_manager.boundary_obstacles:
            circle = plt.Circle((x, y), config.robot_radius, color="k")
            plt.gca().add_patch(circle)
        plt.plot(sx, sy, "or", zorder=10)
        plt.plot(gx, gy, "sr", zorder=10)
        plt.grid(True)
        plt.axis("equal")

    # ----- Run A* path planning -----
    theta_star_planner = theta_star.ThetaStarPlanner(
        ob=map_manager.astar_obstacles,  # Use A* obstacles from MapManager
        resolution=1.0,
        rr=max(config.robot_width, config.robot_length),
        min_x=min(map_manager.astar_obstacles[:, 0].min(), sx-2, gx-2),
        min_y=min(map_manager.astar_obstacles[:, 1].min(), sy-2, gy-2),
        max_x=max(map_manager.astar_obstacles[:, 0].max(), sx+2, gx+2),
        max_y=max(map_manager.astar_obstacles[:, 1].max(), sy+2, gy+2),
        save_animation_to_figs=False,
        fig_dir=fig_dir
    )
    rx, ry = theta_star_planner.planning(sx, sy, gx, gy, curr_i_fig=i_fig)  # full A* path (reversed)
    road_map = np.array([rx, ry]).transpose()[::-1]  # full A* path
    # print(road_map)
    map_manager.set_road_map(road_map)  # Set the road map in MapManager

    # Plot the A* path
    if show_animation:  # pragma: no cover
        plt.plot(rx, ry, "-b")[0]
        plt.pause(0.001)
        if save_animation_to_figs:
            i_fig = theta_star_planner.i_fig # update i_fig
            plt.savefig(
                os.path.join(fig_dir, 'frame_{}.png'.format(i_fig)), 
                bbox_inches='tight', pad_inches=0.1
            )
            i_fig += 1


    # Get the obstacles for DWA
    if show_animation:  # pragma: no cover
        for (x, y) in map_manager.dynamic_obstacles:
            circle = plt.Circle((x, y), config.robot_radius, color="brown", zorder=10)
            plt.gca().add_patch(circle)


    # ----- Run DWA path planning -----
    # x = np.array([sx, sy, - math.pi / 8.0, 0.0, 0.0])
    # # x = np.array([47, 56, - math.pi*3/4, 0.5, 0.])
    # # road_map = road_map[1:]    # roadmap remove the first few points

    x = np.array([sx, sy, - math.pi / 8.0, 0.0, 0.0])
    # x = np.array([39, 60, - math.pi*3/4, 0.5, -0.])
    # road_map = road_map[4:]    # roadmap remove the first few points
    # x = np.array([3.39083423,8.02084971,-1.90310702,0.50000000,0.13089969])
    # road_map = road_map[8:]    # roadmap remove the first few points


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
                valid_intersection_points = []
                for intersection_point in intersection_points:
                    dist_intersection_turning = math.hypot(intersection_point[0] - turning_point[0], intersection_point[1] - turning_point[1])
                    dist_to_turning_point = math.hypot(x[0] - turning_point[0], x[1] - turning_point[1])
                    if dist_intersection_turning <= dist_to_turning_point:
                         valid_intersection_points.append(intersection_point)
                intersection_points = valid_intersection_points


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
                else:
                    # This case handles > 2 intersections, often when robot is on the line segment
                    # Choose the one closest to the robot
                    dists_to_robot = [math.hypot(p[0] - x[0], p[1] - x[1]) for p in intersection_points]
                    dwagoal = intersection_points[np.argmin(dists_to_robot)]
                    warnings.warn(f"""\
        {len(intersection_points)} intersection points found: {intersection_points}.
        Choosing closest to robot: {dwagoal}\
        """, UserWarning)



                ## Execute DWA
                # Get static and dynamic obstacles separately
                static_ob_all = map_manager.boundary_obstacles
                dynamic_ob_all = map_manager.dynamic_obstacles

                # Filter obstacles based on distance to robot
                # (Using 8m as in original script)
                temp_static_ob = [o for o in static_ob_all if np.linalg.norm(o - x[:2]) < 8]
                temp_static_ob = np.array(temp_static_ob) if temp_static_ob else np.empty((0, 2))

                temp_dynamic_ob = [o for o in dynamic_ob_all if np.linalg.norm(o - x[:2]) < 8]
                temp_dynamic_ob = np.array(temp_dynamic_ob) if temp_dynamic_ob else np.empty((0, 2))

                # Fallback for static obstacles (from original logic)
                if temp_static_ob.shape[0] == 0 and static_ob_all.shape[0] > 0:
                    temp_static_ob = static_ob_all[
                        np.argmin(np.linalg.norm(static_ob_all - x[:2], axis=1))
                    ].reshape(-1, 2)
                
                # Ensure dynamic obstacles are 2D array even if empty
                if temp_dynamic_ob.shape[0] == 0:
                    temp_dynamic_ob = np.empty((0, 2))

                # Call the new DWA control function
                (u, predicted_trajectory, dw, 
                 to_goal_before, speed_before, static_ob_before, dynamic_ob_before,
                 to_goal_after, speed_after, static_ob_after, dynamic_ob_after, 
                 final_cost) = dwa.dwa_control_split_ob(
                    x, config, dwagoal, temp_static_ob, temp_dynamic_ob
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
                    "static_ob_cost_before": float(static_ob_before),
                    "dynamic_ob_cost_before": float(dynamic_ob_before),
                    "to_goal_cost_after": float(to_goal_after),
                    "speed_cost_after": float(speed_after),
                    "static_ob_cost_after": float(static_ob_after),
                    "dynamic_ob_cost_after": float(dynamic_ob_after),
                    "dynamic_window": [float(dw[0]), float(dw[1]), float(dw[2]), float(dw[3])],
                    # "admissible": admissible,
                    # "inadmissible": inadmissible
                }
                log_data.append(log_entry)
                iteration += 1

                x = dwa.motion(x, u, config.dt)  # simulate robot
                trajectory = np.vstack((trajectory, x))  # store state history
                
                # ADD COLLISION DETECTION HERE (after motion, before animation)
                # Check against ALL obstacles
                all_current_obstacles = map_manager.get_current_obstacles()
                collision_detected, obstacle_index, collision_distance = \
                    dwa.check_collision_at_current_position_circle_approximation(
                        x, all_current_obstacles, config
                    )

                if collision_detected:
                    # Calculate effective radius for error message
                    if config.robot_type == dwa.RobotType.rectangle:
                        effective_radius = math.sqrt((config.robot_length/2)**2 + (config.robot_width/2)**2)
                        robot_shape_info = f"rectangle (effective radius: {effective_radius:.3f}m)"
                    else:
                        effective_radius = config.robot_radius
                        robot_shape_info = f"circle (radius: {effective_radius:.3f}m)"
                    
                    colliding_obstacle_pos = all_current_obstacles[obstacle_index]
                    
                    # Print collision details and terminate
                    error_message = f"""
COLLISION DETECTED!
- Iteration: {iteration-1}
- Robot position: ({x[0]:.3f}, {x[1]:.3f})
- Robot heading: {x[2]:.3f} rad ({math.degrees(x[2]):.1f}°)
- Robot shape: {robot_shape_info}
- Colliding obstacle index: {obstacle_index} (in combined list)
- Colliding obstacle position: ({colliding_obstacle_pos[0]:.3f}, {colliding_obstacle_pos[1]:.3f})
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
                        plt.savefig(
                            os.path.join(fig_dir, 'frame_{}.png'.format(i_fig)),
                            bbox_inches='tight', pad_inches=0.1
                        )
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
            log_dir = os.path.join("Logs", f"{fig_folder}_{timestamp}")
            os.makedirs(log_dir, exist_ok=True)
            details_filename = os.path.join(log_dir, "log_details.json")
            
            with open(details_filename, 'w') as f:
                json.dump({
                    "log_entries": log_data,
                    "trajectory": trajectory.tolist()  # Add trajectory data
                }, f, indent=2)
            print(f"Data saved to {details_filename}")

    print("Done")
    if show_animation:  # pragma: no cover
        plt.show()
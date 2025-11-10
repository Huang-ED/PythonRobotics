import os, sys
rpath = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
# Make sure the path to theta_star is correct
sys.path.append(rpath) 
# Add current directory to path to find the new modules
sys.path.append(os.path.dirname(__file__))

try:
    import theta_star
except ImportError:
    print("Error: 'theta_star' module not found.")
    print(f"Please ensure 'theta_star.py' is in the search path: {rpath}")
    sys.exit(1)
    
# Import the new MERGED logic files
import PathPlanning.DWAT_v3_split.dwa_paper as dwa
from PathPlanning.DWAT_v3_split.dwa_paper import Config, line_circle_intersection
from PathPlanning.DWAT_v3_split.map_manager import MapManager

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
fig_folder = 'figs_v9.2.3.2-video1-split' # New folder for merged results
map_config_file = os.path.join("PathPlanning", "DWAT_v3_split", "map_config", "map_config_video1.json")


if __name__ == '__main__':

    # Ensure map file exists before proceeding
    if not os.path.exists(map_config_file):
        print(f"Error: Map config file not found at '{map_config_file}'")
        print("Please update the 'map_config_file' variable in 'dwa_astar_merged.py'")
        sys.exit(1)

    config = Config()
    config_plot = Config()
    # config_plot.robot_width *= 2
    # config_plot.robot_length *= 2

    # ----- Set up the map -----
    # Initialize map manager
    map_manager = MapManager(config)
    
    # Load map configuration including dynamic obstacles
    map_manager.load_map_config(map_config_file)
    
    # Get start and goal positions
    start = map_manager.start_position
    goal = map_manager.goal_position

    # Plot the map
    if show_animation:  # pragma: no cover
        plt.figure(figsize=(10, 10))
        if save_animation_to_figs:
            cur_dir = os.path.dirname(__file__)
            fig_dir = os.path.join(cur_dir, fig_folder)
            os.makedirs(fig_dir, exist_ok=True) # Use exist_ok=True
            i_fig = 0

        else:
            fig_dir = None
            i_fig = 0
        
        # Plot static map features
        if map_manager.static_obstacles.shape[0] > 0:
            for (x_s, y_s) in map_manager.static_obstacles:
                circle = plt.Circle((x_s, y_s), config.robot_radius, color="darkgrey")
                plt.gca().add_patch(circle)
        if map_manager.boundary_obstacles.shape[0] > 0:
            for (x_b, y_b) in map_manager.boundary_obstacles:
                circle = plt.Circle((x_b, y_b), config.robot_radius, color="k")
                plt.gca().add_patch(circle)
        
        plt.plot(start[0], start[1], "or", zorder=10)
        plt.plot(goal[0], goal[1], "sr", zorder=10)
        plt.grid(True)
        plt.axis("equal")

    # ----- Run A* path planning -----
    theta_star_planner = theta_star.ThetaStarPlanner(
        ob=map_manager.astar_obstacles,  # Use A* obstacles from MapManager
        resolution=1.0,
        rr=max(config.robot_width, config.robot_length),
        min_x = min(map_manager.astar_obstacles[:, 0].min(), start[0]-2, goal[0]-2),
        min_y = min(map_manager.astar_obstacles[:, 1].min(), start[1]-2, goal[1]-2),
        max_x = max(map_manager.astar_obstacles[:, 0].max(), start[0]+2, goal[0]+2),
        max_y = max(map_manager.astar_obstacles[:, 1].max(), start[1]+2, goal[1]+2),
        save_animation_to_figs=False,
        fig_dir=fig_dir
    )
    rx, ry = theta_star_planner.planning(
        start[0], start[1], goal[0], goal[1], curr_i_fig=i_fig
    )  # full A* path (reversed)
    road_map = np.array([rx, ry]).transpose()[::-1]  # full A* path
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


    # Plot initial dynamic obstacles
    if show_animation:  # pragma: no cover
        plt_elements = [] # This will hold moving elements
        for obstacle in map_manager.dynamic_obstacles:
            circle = plt.Circle(obstacle.current_position, obstacle.radius, color="brown", zorder=10)
            plt.gca().add_patch(circle)
            plt_elements.append(circle)


    # ----- Run DWA path planning -----
    x = np.array([start[0], start[1], - math.pi / 8.0, 0.0, 0.0])  # Initial state
    print(__file__ + " start!!")
    trajectory = np.array(x)

    if show_animation:  # pragma: no cover
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])

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
                
                # remove intersection points that are "behind" the turning point
                valid_intersection_points = []
                dist_to_turning_point_global = math.hypot(x[0] - turning_point[0], x[1] - turning_point[1])
                for intersection_point in intersection_points:
                    dist_intersection_turning = math.hypot(intersection_point[0] - turning_point[0], intersection_point[1] - turning_point[1])
                    if dist_intersection_turning <= dist_to_turning_point_global + 1e-6: # Add tolerance
                         valid_intersection_points.append(intersection_point)
                intersection_points = valid_intersection_points


                if len(intersection_points) == 0:
                    dist_to_turning_point = math.hypot(x[0] - turning_point[0], x[1] - turning_point[1])
                    if dist_to_turning_point <= config.dist_localgoal:
                        dwagoal = turning_point
                    else:
                        """ To be implemented: Re-plan the path"""
                        raise ValueError("No intersection points found and not at local goal")
                elif len(intersection_points) == 1:
                    dwagoal = intersection_points[0]
                else: # 2 or more points
                    # Choose the intersection point that is closer to the turning point
                    dists_to_turning = [math.hypot(p[0] - turning_point[0], p[1] - turning_point[1]) for p in intersection_points]
                    dwagoal = intersection_points[np.argmin(dists_to_turning)]


                ## Execute DWA
                # Update dynamic obstacles
                map_manager.update_dynamic_obstacles(config.dt)
                
                # Get separate obstacle lists and their radii
                static_ob = map_manager.get_static_obstacles()
                static_ob_radii = map_manager.get_static_obstacle_radii()
                dynamic_ob_pos = map_manager.get_dynamic_obstacles_pos()
                dynamic_ob_radii = map_manager.get_dynamic_obstacle_radii()

                
                # Call the new MERGED DWA control function
                (u, predicted_trajectory, dw, 
                 to_goal_before, speed_before, static_ob_before, dynamic_ob_before,
                 to_goal_after, speed_after, static_ob_after, dynamic_ob_after, 
                 final_cost) = dwa.dwa_control_merged(
                    x, config, dwagoal, 
                    static_ob, static_ob_radii,
                    dynamic_ob_pos, dynamic_ob_radii
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
                    "static_ob_after": float(static_ob_after),
                    "dynamic_ob_after": float(dynamic_ob_after),
                    "dynamic_window": [float(dw[0]), float(dw[1]), float(dw[2]), float(dw[3])],
                }
                log_data.append(log_entry)
                iteration += 1

                x = dwa.motion(x, u, config.dt)  # simulate robot
                trajectory = np.vstack((trajectory, x))  # store state history
                
                # --- COLLISION DETECTION (after motion) ---
                # Get *all* current obstacles and radii for the check
                all_current_ob = map_manager.get_current_obstacles()
                all_current_ob_radii = map_manager.get_obstacle_radii()

                collision_detected, obstacle_index, collision_distance = \
                    dwa.check_collision_at_current_position_circle_approximation(
                        x, all_current_ob, all_current_ob_radii, config
                    )

                if collision_detected:
                    if config.robot_type == dwa.RobotType.rectangle:
                        effective_radius = math.sqrt((config.robot_length/2)**2 + (config.robot_width/2)**2)
                        robot_shape_info = f"rectangle (effective radius: {effective_radius:.3f}m)"
                    else:
                        effective_radius = config.robot_radius
                        robot_shape_info = f"circle (radius: {effective_radius:.3f}m)"
                    
                    colliding_ob_pos = all_current_ob[obstacle_index]
                    colliding_ob_radius = all_current_ob_radii[obstacle_index]
                    
                    error_message = f"""
COLLISION DETECTED!
- Iteration: {iteration-1}
- Robot position: ({x[0]:.3f}, {x[1]:.3f})
- Robot heading: {x[2]:.3f} rad ({math.degrees(x[2]):.1f}°)
- Robot shape: {robot_shape_info}
- Colliding obstacle index: {obstacle_index} (in combined list)
- Colliding obstacle position: ({colliding_ob_pos[0]:.3f}, {colliding_ob_pos[1]:.3f})
- Colliding obstacle radius: {colliding_ob_radius:.3f}m
- Distance to obstacle center: {collision_distance:.3f}m
- Collision threshold: {effective_radius + colliding_ob_radius:.3f}m
- Simulation terminated due to collision.
                """
                    warnings.warn(error_message, UserWarning)
                    # You might want to break or exit here
                    # break # Break from inner 'while' loop
                    # sys.exit(error_message) # Or exit entirely

                if show_animation:  # pragma: no cover
                    # Remove all old moving elements
                    for ele in plt_elements:
                        ele.remove()
                    plt_elements = []
                    
                    # Add new dynamic obstacle positions
                    for obstacle in map_manager.dynamic_obstacles:
                        circle = plt.Circle(obstacle.current_position, obstacle.radius, color="brown", zorder=10)
                        plt.gca().add_patch(circle)
                        plt_elements.append(circle)
                        
                    # Add robot elements
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
                        break # Break from inner 'while' loop
                else:
                    if dist_to_turning_point <= config.catch_turning_point_dist:
                        print("Local goal!!")
                        break # Break from inner 'while' loop
            
            # This 'if' is for the collision break
            if collision_detected:
                print("Exiting due to collision.")
                break # Break from outer 'for' loop


    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        traceback.print_exc()  # Print stack trace for debugging

    except KeyboardInterrupt:
        print("Simulation manually stopped.")

    finally:
        # Always save data before exiting
        if log_data:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = os.path.join("Logs", f"{fig_folder}_{timestamp}")
            os.makedirs(log_dir, exist_ok=True)
            details_filename = os.path.join(log_dir, "log_details.json")
            
            with open(details_filename, 'w') as f:
                json.dump({
                    "config": {k: v for k, v in config.__dict__.items() if not k.startswith('_')},
                    "log_entries": log_data,
                    "trajectory": trajectory.tolist()  # Add trajectory data
                }, f, indent=2)
            print(f"Data saved to {details_filename}")

    print("Done")
    if show_animation:  # pragma: no cover
        plt.show()
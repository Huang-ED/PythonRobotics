import os
import sys
rpath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(rpath)

import numpy as np
import pandas as pd
import math
import cv2
import matplotlib.pyplot as plt
from PathPlanning.DynamicWindowApproach.dwa_paper_with_width import (
    calc_dynamic_window, closest_obstacle_on_curve, predict_trajectory, calc_to_goal_cost, 
    motion as dwa_motion, any_circle_overlap_with_box, RobotType
)
from PathPlanning.LocalGlobal.dwa_astar_v7_video2 import Config

def get_robot_bounding_box(center_x, center_y, yaw, length, width):
    """Compute the bounding box vertices of the robot."""
    outline = np.array([
        [-length / 2, length / 2, length / 2, -length / 2, -length / 2],
        [width / 2, width / 2, -width / 2, -width / 2, width / 2]
    ])
    rotation_matrix = np.array([
        [np.cos(yaw), np.sin(yaw)],
        [-np.sin(yaw), np.cos(yaw)]
    ])
    rotated_outline = rotation_matrix @ outline
    translated_outline = rotated_outline + np.array([[center_x], [center_y]])
    return translated_outline.T

def find_colliding_obstacles(robot_center, robot_length, robot_width, robot_yaw, obstacles, obstacle_radius):
    """Find obstacles that collide with the robot."""
    colliding_obstacles = []
    for ob_point in obstacles:
        ob_with_radius = np.array([[ob_point[0], ob_point[1], obstacle_radius]])
        if any_circle_overlap_with_box(ob_with_radius, robot_center, robot_length, robot_width, robot_yaw):
            colliding_obstacles.append(ob_point)
    return np.array(colliding_obstacles)

def plot_collision_details(original_pose, collision_pose, colliding_obstacles, all_obstacles, config):
    """Plot detailed information when a collision occurs."""
    plt.figure(figsize=(10, 10))
    
    # Plot all obstacles
    plt.scatter(all_obstacles[:, 0], all_obstacles[:, 1], c='gray', s=50, label='Obstacles')
    
    # Highlight colliding obstacles
    if len(colliding_obstacles) > 0:
        plt.scatter(colliding_obstacles[:, 0], colliding_obstacles[:, 1], c='red', s=100, marker='X', label='Colliding Obstacle')
    
    # Original position and bounding box
    plt.scatter(original_pose[0], original_pose[1], c='blue', s=100, marker='o', label='Original Center')
    original_box = get_robot_bounding_box(original_pose[0], original_pose[1], original_pose[2], config.robot_length, config.robot_width)
    plt.plot(original_box[:, 0], original_box[:, 1], 'b--', linewidth=2, label='Original Bounding Box')
    
    # Collision position and bounding box
    plt.scatter(collision_pose[0], collision_pose[1], c='red', s=100, marker='x', label='Collision Center')
    collision_box = get_robot_bounding_box(collision_pose[0], collision_pose[1], collision_pose[2], config.robot_length, config.robot_width)
    plt.plot(collision_box[:, 0], collision_box[:, 1], 'r-', linewidth=2, label='Collision Bounding Box')
    
    # Set view limits
    plt.xlim([collision_pose[0] - 2*config.robot_length, collision_pose[0] + 2*config.robot_length])
    plt.ylim([collision_pose[1] - 2*config.robot_width, collision_pose[1] + 2*config.robot_width])
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title('Zoom-in View at Collision')
    plt.show()

def calculate_all_costs_debug(x, config, goal, ob):
    """
    Generate a cost matrix with debug information and print detailed calculations for specific velocity combinations.
    """
    dw = calc_dynamic_window(x, config)
    v_samples = np.arange(dw[0], dw[1] + 1e-6, config.v_resolution)
    omega_samples = np.arange(dw[2], dw[3] + 1e-6, config.yaw_rate_resolution)
    
    V, Omega = np.meshgrid(v_samples, omega_samples, indexing='ij')
    to_goal_cost = np.full_like(V, -1.0)
    speed_cost = np.full_like(V, -1.0)
    ob_cost = np.full_like(V, -1.0)
    
    # Debug target velocities
    debug_targets = [
        (0.36, 0.060),
        (0.36, 0.065),
        (0.36, 0.070)
    ]
    
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            v = V[i, j]
            omega = Omega[i, j]
            dist, _ = closest_obstacle_on_curve(x.copy(), ob, v, omega, config)
            
            # Dynamic window admission check
            if v > math.sqrt(2 * config.max_accel * dist):
                continue
                
            # Calculate base costs
            trajectory = predict_trajectory(x.copy(), v, omega, config)
            to_goal_cost[i, j] = calc_to_goal_cost(trajectory, goal)
            speed_cost[i, j] = config.max_speed - trajectory[-1, 3]
            ob_cost[i, j] = 1.0 / dist if dist != 0 and not np.isinf(dist) else np.inf
            
            # Debug logic: Capture target velocity combinations
            target_found = any(
                abs(v - tv) < 0.005 and abs(omega - tw) < 0.0005
                for tv, tw in debug_targets
            )
            
            if target_found:
                print(f"\n----- Detailed Calculation [v={v:.2f}, ω={omega:.5f}] -----")
                print(f"Current State: x={x[0]:.2f}, y={x[1]:.2f}, yaw={x[2]:.5f} rad")
                
                # Independent trajectory simulation
                x_sim = x.copy()
                collision_dist = float('inf')
                total_dist = 0.0
                
                for _ in range(int(config.predict_time / config.dt) + 1):
                    # Compute robot's current position and collision status
                    if config.robot_type == RobotType.rectangle:
                        ob_with_radius = np.c_[ob, np.full(len(ob), config.obstacle_radius)]
                        collision = any_circle_overlap_with_box(
                            ob_with_radius, x_sim[:2], 
                            config.robot_length, config.robot_width, x_sim[2]
                        )
                    else:
                        distances = np.linalg.norm(ob - x_sim[:2], axis=1)
                        collision = any(d <= config.robot_radius + config.obstacle_radius for d in distances)
                    
                    # Distance calculation logic
                    closest_distance = min(
                        np.linalg.norm(ob - x_sim[:2], axis=1) 
                        - (config.robot_radius if config.robot_type == RobotType.circle else 0)
                    )
                    
                    print(
                        f"Time: {_ * config.dt:.1f}s | "
                        f"Position: ({x_sim[0]:.2f}, {x_sim[1]:.2f}) | "
                        f"Closest Obstacle Distance: {closest_distance:.2f}m | "
                        f"Collision: {collision}"
                    )
                    
                    # Collision detection
                    if collision:
                        collision_dist = total_dist
                        print(f">>> Collision Detected! Total Distance Moved: {collision_dist:.2f}m <<<")
                        # Collect data and plot
                        collision_pose = x_sim.copy()
                        original_pose = x.copy()  # Original position for the current frame
                        colliding_obstacles = find_colliding_obstacles(
                            collision_pose[:2], config.robot_length, config.robot_width, collision_pose[2],
                            ob, config.obstacle_radius
                        )
                        plot_collision_details(original_pose, collision_pose, colliding_obstacles, ob, config)
                        break
                        
                    # Update state
                    total_dist += v * config.dt
                    x_sim = dwa_motion(x_sim, [v, omega], config.dt)
                
                # Output final calculation results
                final_cost_value = 1.0 / collision_dist if collision_dist != 0 else np.inf
                print(
                    f"----- Calculation Result [v={v:.2f}, ω={omega:.5f}] -----\n"
                    f"Distance to First Collision: {collision_dist:.2f}m | "
                    f"Obstacle Cost: {final_cost_value:.2f}\n"
                    f"------------------------------------------\n"
                )
    
    return to_goal_cost, speed_cost, ob_cost, v_samples, omega_samples

def main_debug():
    """Main debug function."""
    log_file_path = "Logs/dwa_log_details_20250306_155027/log_details.csv"
    config = Config()
    
    # Load example iteration (using iteration 227)
    df = pd.read_csv(log_file_path, index_col="iteration")
    iter_num = 226
    x = np.array(df.loc[iter_num, ['x_traj', 'y_traj', 'yaw_traj', 'v_traj', 'omega_traj']])
    goal = np.array(df.loc[iter_num, ['local_goal_x', 'local_goal_y']])
    
    # Create DWA-specific obstacle map (same as original process)
    image_path = "EnvData/AISData_20240827/land_shapes_sf_crop.png"
    arr = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    arr = cv2.resize(arr, (100, 100))
    _, arr = cv2.threshold(arr, 128, 1, cv2.THRESH_BINARY)
    arr[0, :] = 0    # Top boundary
    arr[-1, :] = 0   # Bottom boundary
    arr[:, 0] = 0    # Left boundary
    arr[:, -1] = 0   # Right boundary
    arr = 1 - arr    # Invert DWA map
    eroded_arr = cv2.erode(arr, kernel=np.ones((3, 3), np.uint8), iterations=1)
    arr_dwa = cv2.subtract(arr, eroded_arr)
    arr_dwa = 1 - arr_dwa
    ob_dwa = np.argwhere(arr_dwa == 0)
    ob_dwa[:, [0, 1]] = ob_dwa[:, [1, 0]]  # Swap xy coordinates
    ob_dwa[:, 1] = arr_dwa.shape[0] - ob_dwa[:, 1] - 1
    new_ob = np.array([  # Add extra obstacles
        [25., 79.], [25., 80.], [26., 79.], [26., 80.],
        [35., 55.], [36., 56], [28., 46.], [27., 47.],
        [10., 19.], [10., 20.], [11., 19.], [11., 20.]
    ])
    ob_dwa = np.append(ob_dwa, new_ob, axis=0)
    
    # Execute cost calculation with debugging
    calculate_all_costs_debug(x, config, goal, ob_dwa)

if __name__ == "__main__":
    main_debug()

import os
import sys
rpath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(rpath)

import numpy as np
import pandas as pd
import math
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

from PathPlanning.DynamicWindowApproach.dwa_paper_with_width import (
    calc_dynamic_window, closest_obstacle_on_curve, predict_trajectory, 
    calc_to_goal_cost, motion as dwa_motion, any_circle_overlap_with_box, RobotType
)
from PathPlanning.LocalGlobal.dwa_astar_v7_video2 import Config

def calculate_bounding_box(center, length, width, yaw):
    """Calculate vertices of rotated rectangle with proper length/width orientation"""
    half_length = length / 2  # Forward/backward dimension
    half_width = width / 2    # Side-to-side dimension
    
    # Create rectangle aligned with heading direction
    corners = np.array([
        [-half_length, -half_width],  # Rear-left
        [half_length, -half_width],   # Front-left 
        [half_length, half_width],    # Front-right
        [-half_length, half_width],   # Rear-right
        [-half_length, -half_width]   # Close polygon
    ])
    
    # Apply rotation matrix
    rot = np.array([[math.cos(yaw), -math.sin(yaw)],
                    [math.sin(yaw), math.cos(yaw)]])
    rotated = corners @ rot.T + center
    
    return rotated


def plot_collision_details(collision_info, config):
    """
    Plot the zoom-in view of a collision event
    
    Parameters:
        collision_info: dictionary containing collision details
        config: configuration object with robot parameters
    """
    # Extract collision data
    cc = np.array(collision_info['collision_center'])
    cy = collision_info['collision_yaw']
    oc = np.array(collision_info['original_center'])
    oy = collision_info['original_yaw']
    cos = np.array(collision_info['collided_obstacles'])
    
    # Calculate bounding boxes
    collision_bbox = calculate_bounding_box(
        cc, 
        config.robot_length,  # Forward dimension
        config.robot_width,   # Side dimension
        cy
    )
    original_bbox = calculate_bounding_box(oc, config.robot_length, config.robot_width, oy)
    
    # Create figure
    plt.figure(figsize=(10, 10))
    
    # Plot original position and bounding box
    plt.plot(oc[0], oc[1], 'bo', markersize=8, label='Original Position')
    plt.plot(original_bbox[:, 0], original_bbox[:, 1], 'b--', linewidth=1.5, label='Original BBox')
    
    # Plot collision position and bounding box
    plt.plot(cc[0], cc[1], 'ro', markersize=8, label='Collision Center')
    plt.plot(collision_bbox[:, 0], collision_bbox[:, 1], 'r-', linewidth=1.5, label='Collision BBox')
    
    # Plot collided obstacles
    if len(cos) > 0:
        for obstacle in cos:
            circle = Circle((obstacle[0], obstacle[1]), config.obstacle_radius, 
                           color='red', alpha=0.3, label='Collided Obstacle')
            plt.gca().add_patch(circle)
            plt.plot(obstacle[0], obstacle[1], 'rx', markersize=10, markeredgewidth=2)
    
    # Configure plot
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title('Collision Zoom-In View')
    
    # Set appropriate view limits
    all_x = np.concatenate([collision_bbox[:, 0], original_bbox[:, 0]])
    if len(cos) > 0:
        all_x = np.concatenate([all_x, cos[:, 0]])
    all_y = np.concatenate([collision_bbox[:, 1], original_bbox[:, 1]])
    if len(cos) > 0:
        all_y = np.concatenate([all_y, cos[:, 1]])
    
    x_margin = max(2, (np.max(all_x) - np.min(all_x)) * 0.3)
    y_margin = max(2, (np.max(all_y) - np.min(all_y)) * 0.3)
    
    plt.xlim(np.min(all_x) - x_margin, np.max(all_x) + x_margin)
    plt.ylim(np.min(all_y) - y_margin, np.max(all_y) + y_margin)
    
    plt.show()

def calculate_all_costs_debug(x, config, goal, ob):
    """
    Generate cost matrix with debugging information and collision visualization
    
    Parameters:
        x: current state [x, y, yaw, v, omega]
        config: configuration object
        goal: goal position [x, y]
        ob: obstacle positions array
    
    Returns:
        tuple: (to_goal_cost, speed_cost, ob_cost, v_samples, omega_samples, collisions)
    """
    dw = calc_dynamic_window(x, config)
    v_samples = np.arange(dw[0], dw[1] + 1e-6, config.v_resolution)
    omega_samples = np.arange(dw[2], dw[3] + 1e-6, config.yaw_rate_resolution)
    
    V, Omega = np.meshgrid(v_samples, omega_samples, indexing='ij')
    to_goal_cost = np.full_like(V, -1.0)
    speed_cost = np.full_like(V, -1.0)
    ob_cost = np.full_like(V, -1.0)
    
    debug_targets = [
        (0.36, 0.060),
        (0.36, 0.065),
        (0.36, 0.070)
    ]
    
    collisions = []  # Store collision information
    
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            v = V[i, j]
            omega = Omega[i, j]
            dist, _ = closest_obstacle_on_curve(x.copy(), ob, v, omega, config)
            
            # Dynamic window admission check
            if v > math.sqrt(2 * config.max_accel * dist):
                continue
                
            # Calculate basic costs
            trajectory = predict_trajectory(x.copy(), v, omega, config)
            to_goal_cost[i, j] = calc_to_goal_cost(trajectory, goal)
            speed_cost[i, j] = config.max_speed - trajectory[-1, 3]
            ob_cost[i, j] = 1.0 / dist if dist != 0 and not np.isinf(dist) else np.inf
            
            # Debug logic for target velocity combinations
            target_found = any(
                abs(v - tv) < 0.005 and abs(omega - tw) < 0.0005
                for tv, tw in debug_targets
            )
            
            if target_found:
                print(f"\n----- Detailed calculation [v={v:.2f}, ω={omega:.5f}] -----")
                print(f"Current state: x={x[0]:.2f}, y={x[1]:.2f}, yaw={x[2]:.5f} rad")
                
                # Independent trajectory simulation
                x_sim = x.copy()
                collision_dist = float('inf')
                total_dist = 0.0
                
                for _ in range(int(config.predict_time / config.dt) + 1):
                    # Calculate robot position and collision status
                    if config.robot_type == RobotType.rectangle:
                        ob_with_radius = np.c_[ob, np.full(len(ob), config.obstacle_radius)]
                        collision = any_circle_overlap_with_box(
                            ob_with_radius,
                            x_sim[:2],
                            config.robot_length,  # Forward dimension
                            config.robot_width,   # Side dimension
                            x_sim[2]
                        )
                    else:
                        distances = np.linalg.norm(ob - x_sim[:2], axis=1)
                        collision = any(d <= config.robot_radius + config.obstacle_radius for d in distances)
                    
                    # Distance calculation
                    closest_distance = min(
                        np.linalg.norm(ob - x_sim[:2], axis=1) 
                        - (config.robot_radius if config.robot_type == RobotType.circle else 0)
                    )
                    
                    print(
                        f"Time: {_ * config.dt:.1f}s | "
                        f"Position: ({x_sim[0]:.2f}, {x_sim[1]:.2f}) | "
                        f"Closest obstacle: {closest_distance:.2f}m | "
                        f"Collision: {collision}"
                    )
                    
                    # Collision detection
                    if collision:
                        collision_dist = total_dist
                        print(f">>> COLLISION DETECTED! Distance traveled: {collision_dist:.2f}m <<<")
                        
                        # Record collision information
                        collided_obstacles = []
                        robot_center = x_sim[:2]
                        robot_yaw = x_sim[2]
                        
                        # Identify which obstacles caused the collision
                        for ob_point in ob:
                            circle = np.array([[ob_point[0], ob_point[1], config.obstacle_radius]])
                            if any_circle_overlap_with_box(circle, robot_center, 
                                                          config.robot_length, config.robot_width, robot_yaw):
                                collided_obstacles.append(ob_point)
                        
                        collision_info = {
                            'collision_center': robot_center.tolist(),
                            'collision_yaw': float(robot_yaw),
                            'original_center': x[:2].tolist(),
                            'original_yaw': float(x[2]),
                            'collided_obstacles': collided_obstacles,
                            'velocity': (float(v), float(omega)),
                            'distance_traveled': float(collision_dist)
                        }
                        collisions.append(collision_info)
                        break
                            
                    # Update state
                    total_dist += v * config.dt
                    x_sim = dwa_motion(x_sim, [v, omega], config.dt)
                
                # Output final calculation results
                final_cost_value = 1.0 / collision_dist if collision_dist != 0 else np.inf
                print(
                    f"----- Results [v={v:.2f}, ω={omega:.5f}] -----\n"
                    f"Distance to first collision: {collision_dist:.2f}m | "
                    f"Obstacle cost: {final_cost_value:.2f}\n"
                    f"------------------------------------------\n"
                )
    
    return to_goal_cost, speed_cost, ob_cost, v_samples, omega_samples, collisions

def main_debug():
    """Main debugging function with collision visualization"""
    log_file_path = "Logs/dwa_log_details_20250306_155027/log_details.csv"
    config = Config()
    
    # Load example iteration (using iteration 227 from user's data)
    df = pd.read_csv(log_file_path, index_col="iteration")
    iter_num = 226
    x = np.array(df.loc[iter_num, ['x_traj', 'y_traj', 'yaw_traj', 'v_traj', 'omega_traj']])
    goal = np.array(df.loc[iter_num, ['local_goal_x', 'local_goal_y']])
    
    # Create DWA obstacle map (same as original process)
    image_path = "EnvData/AISData_20240827/land_shapes_sf_crop.png"
    arr = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    arr = cv2.resize(arr, (100, 100))
    _, arr = cv2.threshold(arr, 128, 1, cv2.THRESH_BINARY)
    arr[0, :] = 0    # Top boundary
    arr[-1, :] = 0   # Bottom boundary
    arr[:, 0] = 0    # Left boundary
    arr[:, -1] = 0   # Right boundary
    arr = 1 - arr    # Invert for DWA map
    eroded_arr = cv2.erode(arr, kernel=np.ones((3, 3), np.uint8), iterations=1)
    arr_dwa = cv2.subtract(arr, eroded_arr)
    arr_dwa = 1 - arr_dwa
    ob_dwa = np.argwhere(arr_dwa == 0)
    ob_dwa[:, [0, 1]] = ob_dwa[:, [1, 0]]  # Swap x,y coordinates
    ob_dwa[:, 1] = arr_dwa.shape[0] - ob_dwa[:, 1] - 1
    new_ob = np.array([  # Add extra obstacles
        [25., 79.], [25., 80.], [26., 79.], [26., 80.],
        [35., 55.], [36., 56], [28., 46.], [27., 47.],
        [10., 19.], [10., 20.], [11., 19.], [11., 20.]
    ])
    ob_dwa = np.append(ob_dwa, new_ob, axis=0)
    
    # Execute cost calculation with debugging
    to_goal_cost, speed_cost, ob_cost, v_samples, omega_samples, collisions = calculate_all_costs_debug(x, config, goal, ob_dwa)
    
    # Visualize each collision
    for i, collision in enumerate(collisions):
        print(f"\nVisualizing collision {i+1}/{len(collisions)}")
        print(f"Velocity: v={collision['velocity'][0]:.2f}, ω={collision['velocity'][1]:.5f}")
        print(f"Distance traveled before collision: {collision['distance_traveled']:.2f}m")
        plot_collision_details(collision, config)

if __name__ == "__main__":
    main_debug()

import os, sys
rpath = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.append(rpath)
# Add current directory to path to find the new modules
sys.path.append(os.path.dirname(__file__))

import numpy as np
import pandas as pd
import math
import cv2
import matplotlib.pyplot as plt
import json

# Import from the new "merged" files
try:
    from dwa import (
        Config, calc_dynamic_window, predict_trajectory, 
        calc_to_goal_cost, closest_obstacle_on_curve, 
        closest_obstacle_on_side
    )
    from map_manager import MapManager
except ImportError as e:
    print(f"Error: Could not import necessary modules. {e}")
    print("Please ensure 'dwa_paper_merged.py' and 'map_manager_merged.py' are in the same directory.")
    sys.exit(1)


def calculate_all_costs_merged(x, config, goal, 
                             static_ob, static_ob_radii, 
                             dynamic_ob_pos, dynamic_ob_radii):
    """
    Calculate cost matrices for all (v, ω) pairs in the dynamic window,
    with separate costs for static and dynamic obstacles.
    
    Parameters:
        x (np.array): Current state [x, y, yaw, v, ω]
        config (Config): Configuration parameters
        goal (np.array): Goal position [x, y]
        static_ob (np.array): Static obstacle positions
        static_ob_radii (np.array): Static obstacle radii
        dynamic_ob_pos (np.array): Dynamic obstacle positions
        dynamic_ob_radii (np.array): Dynamic obstacle radii
    
    Returns:
        tuple: (to_goal_cost, speed_cost, static_ob_cost, dynamic_ob_cost, v_samples, omega_samples)
    """
    # Calculate dynamic window
    dw = calc_dynamic_window(x, config)
    
    # Generate velocity and yaw rate samples
    v_samples = np.arange(dw[0], dw[1] + 1e-6, config.v_resolution)
    omega_samples = np.arange(dw[2], dw[3] + 1e-6, config.yaw_rate_resolution)
    
    # Create meshgrid for all (v, ω) pairs
    V, Omega = np.meshgrid(v_samples, omega_samples, indexing='ij')
    
    # Initialize cost matrices with nan (inadmissible)
    to_goal_cost = np.full(V.shape, np.nan)
    speed_cost = np.full(V.shape, np.nan)
    static_ob_cost = np.full(V.shape, np.nan)
    dynamic_ob_cost = np.full(V.shape, np.nan)
    
    # --- Prepare combined obstacle list for admissibility check ---
    has_static = static_ob is not None and static_ob.shape[0] > 0
    has_dynamic = dynamic_ob_pos is not None and dynamic_ob_pos.shape[0] > 0

    all_ob = np.empty((0, 2))
    all_ob_radii = []

    if has_static:
        all_ob = np.vstack((all_ob, static_ob))
        all_ob_radii.extend(static_ob_radii)
    
    if has_dynamic:
        all_ob = np.vstack((all_ob, dynamic_ob_pos))
        all_ob_radii.extend(dynamic_ob_radii)
    
    has_all_ob = all_ob.shape[0] > 0
    if has_all_ob:
        all_ob_radii_np = np.array(all_ob_radii)
    
    # --- Numpy array versions for cost functions ---
    if has_static:
        static_ob_radii_np = np.array(static_ob_radii)
    if has_dynamic:
        dynamic_ob_radii_np = np.array(dynamic_ob_radii)

    # --- Iterate over all (v, w) pairs ---
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            v = V[i, j]
            omega = Omega[i, j]
            
            # --- Admissible velocities check (using ALL obstacles) ---
            dist_all = float("inf")
            if has_all_ob:
                dist_all, _ = closest_obstacle_on_curve(
                    x.copy(), all_ob, all_ob_radii_np, v, omega, config
                )
            
            if v**2 + 2 * config.max_accel * v * config.dt > 2 * config.max_accel * dist_all:
                continue  # Skip inadmissible pairs
            
            # Calculate trajectory and costs for admissible pairs
            trajectory = predict_trajectory(x.copy(), v, omega, config)
            
            # To_goal cost
            to_goal_cost[i, j] = calc_to_goal_cost(trajectory, goal)
            
            # Speed cost
            speed_cost[i, j] = config.max_speed - trajectory[-1, 3]
            
            # Static obstacle cost (direct distance)
            static_cost_val = 0.0
            if has_static:
                dist_static, _ = closest_obstacle_on_curve(
                    x.copy(), static_ob, static_ob_radii_np, v, omega, config
                )
                static_cost_val = max(0., config.max_obstacle_cost_dist - dist_static)
            static_ob_cost[i, j] = static_cost_val

            # Dynamic obstacle cost (side distance/clearance)
            dynamic_cost_val = 0.0
            if has_dynamic:
                clearance_dynamic = closest_obstacle_on_side(
                    trajectory, dynamic_ob_pos, dynamic_ob_radii_np, config
                )
                if clearance_dynamic <= 0:  # Collision or on boundary
                    dynamic_cost_val = np.inf
                else:
                    dynamic_cost_val = 1.0 / clearance_dynamic
            dynamic_ob_cost[i, j] = dynamic_cost_val
    
    return to_goal_cost, speed_cost, static_ob_cost, dynamic_ob_cost, v_samples, omega_samples



def main():
    # --- Configuration ---
    # !! IMPORTANT: Update this log_file_path to point to your CSV log !!
    log_file_path = "Logs/figs_v9.2.4.7-video1_20251113_171124/log_details.csv" # Example path
    
    # !! IMPORTANT: Update this map_config_file to point to the map JSON !!
    map_config_file = "PathPlanning/DWAT_v3_split/map_config/map_config_video1.json" # Example path
    
    iter_nums = list(range(2855, 2865))  # Specify the iterations you want to process
    # ---------------------

    if not os.path.exists(log_file_path):
        print(f"Error: Log CSV file not found at '{log_file_path}'")
        print("Please update 'log_file_path' in this script.")
        return

    json_log_path = log_file_path.replace('.csv', '.json')
    if not os.path.exists(json_log_path):
        json_log_path = os.path.join(os.path.dirname(log_file_path), "log_details.json")
        if not os.path.exists(json_log_path):
            print(f"Error: Log JSON file not found at '{json_log_path}' or original path.")
            print("The JSON log is required to get the full trajectory.")
            return

    if not os.path.exists(map_config_file):
        print(f"Error: Map config file not found at '{map_config_file}'")
        print("Please update 'map_config_file' in this script.")
        return

    ## Config
    config = Config()

    ## Load log data
    df = pd.read_csv(log_file_path, index_col="iteration")
    with open(json_log_path, 'r') as f:
        log_data_json = json.load(f)
    trajectory_full = np.array(log_data_json['trajectory'])

    # Check if iter_nums are valid
    max_iter = df.index.max()
    valid_iter_nums = [i for i in iter_nums if i <= max_iter]
    if not valid_iter_nums:
        print(f"Error: None of the specified 'iter_nums' are in the log file (max iter: {max_iter}).")
        return
    
    print(f"Processing iterations: {valid_iter_nums}")

    for iter_num in valid_iter_nums:
        print(f"--- Processing iteration {iter_num} ---")
        
        ## Re-initialize map manager to reset dynamic obstacles
        map_manager = MapManager(config)
        map_manager.load_map_config(map_config_file)
        static_ob = map_manager.get_static_obstacles()
        static_ob_radii = map_manager.get_static_obstacle_radii()

        ## Simulate dynamic obstacles up to the current iteration
        # update_dynamic_obstacles is called *before* cost calculation in the main loop
        for _ in range(iter_num + 1):
            map_manager.update_dynamic_obstacles(config.dt)

        dynamic_ob_pos = map_manager.get_dynamic_obstacles_pos()
        dynamic_ob_radii = map_manager.get_dynamic_obstacle_radii()

        ## Get the state 'x' *at the start* of this iteration
        # trajectory_full[iter_num] corresponds to the state used for cost calculation
        x = trajectory_full[iter_num] 
        goal = np.array(df.loc[iter_num, ['local_goal_x', 'local_goal_y']])

        ## Calculate costs
        (tg_cost, sp_cost, 
         static_ob_cost, dynamic_ob_cost, 
         v_samples, omega_samples) = calculate_all_costs_merged(
            x, config, goal, 
            static_ob, static_ob_radii, 
            dynamic_ob_pos, dynamic_ob_radii
         )
        
        # Apply gains to get weighted costs
        tg_cost_weighted = config.to_goal_cost_gain * tg_cost
        sp_cost_weighted = config.speed_cost_gain * sp_cost
        static_ob_cost_weighted = config.obstacle_cost_gain * static_ob_cost
        dynamic_ob_cost_weighted = config.side_cost_gain * dynamic_ob_cost
        
        # Handle infinities before summing
        final_cost = (
            np.nan_to_num(tg_cost_weighted, nan=np.nan, posinf=np.inf, neginf=-np.inf) +
            np.nan_to_num(sp_cost_weighted, nan=np.nan, posinf=np.inf, neginf=-np.inf) +
            np.nan_to_num(static_ob_cost_weighted, nan=np.nan, posinf=np.inf, neginf=-np.inf) +
            np.nan_to_num(dynamic_ob_cost_weighted, nan=np.nan, posinf=np.inf, neginf=-np.inf)
        )
        
        # Get the chosen (v, omega) pair (which was for the *next* state)
        if iter_num + 1 in df.index:
            # Note: v_traj/omega_traj in the CSV are the *chosen* v/omega
            chosen_v = df.loc[iter_num, 'v_chosen']
            chosen_omega = df.loc[iter_num, 'omega_chosen']
        else:
            chosen_v, chosen_omega = None, None
        
        # Save results as text
        log_dir = os.path.dirname(log_file_path)
        curr_cost_dir = os.path.join(log_dir, f"cost_matrices_{iter_num}")
        os.makedirs(curr_cost_dir, exist_ok=True)
        
        np.savetxt(os.path.join(curr_cost_dir, "v_samples.txt"), v_samples, fmt='%.3f', delimiter='\t')
        np.savetxt(os.path.join(curr_cost_dir, "omega_samples.txt"), omega_samples, fmt='%.3f', delimiter='\t')
        
        np.savetxt(os.path.join(curr_cost_dir, "to_goal_cost.txt"), tg_cost, fmt='%.3f', delimiter='\t')
        np.savetxt(os.path.join(curr_cost_dir, "speed_cost.txt"), sp_cost, fmt='%.3f', delimiter='\t')
        np.savetxt(os.path.join(curr_cost_dir, "static_ob_cost.txt"), static_ob_cost, fmt='%.3f', delimiter='\t')
        np.savetxt(os.path.join(curr_cost_dir, "dynamic_ob_cost.txt"), dynamic_ob_cost, fmt='%.3f', delimiter='\t')
        
        np.savetxt(os.path.join(curr_cost_dir, "final_cost.txt"), final_cost, fmt='%.3f', delimiter='\t')

        # Display as image with custom ticks
        step = max(1, len(omega_samples) // 20) # Adjust step size for ticks
        x_ticks_indices = np.arange(0, len(omega_samples), step)
        omega_samples_rounded = np.round(omega_samples, 2)
        
        v_step = max(1, len(v_samples) // 20) # Adjust step size for ticks
        y_ticks_indices = np.arange(0, len(v_samples), v_step)
        v_samples_rounded = np.round(v_samples, 2)


        # Create combined figure with subplots
        fig, axs = plt.subplots(1, 5, figsize=(30, 6)) # 5 subplots now
        plt.subplots_adjust(wspace=0.3, hspace=0.3)

        def plot_cost(ax, cost, title, is_final=False):
            cmap = plt.get_cmap('jet')
            cmap.set_bad(color='black')  # Set NaN values to black
            
            # For final cost, infinities are bad, but for components, they can be normal
            if not is_final:
                cost[cost == np.inf] = 1e6 # Replace inf with a large number for plotting
            
            im = ax.imshow(cost, origin='lower', cmap=cmap, aspect='auto')
            plt.colorbar(im, ax=ax, label='Cost')
            ax.set_xticks(x_ticks_indices)
            ax.set_xticklabels(omega_samples_rounded[::step], rotation=45, ha="right")
            ax.set_yticks(y_ticks_indices)
            ax.set_yticklabels(v_samples_rounded[::v_step])
            ax.set_xlabel('Omega (rad/s)')
            ax.set_ylabel('V (m/s)')
            ax.set_title(title)
            if chosen_v is not None and chosen_omega is not None:
                # Find closest indices in samples
                chosen_v_idx = np.argmin(np.abs(v_samples - chosen_v))
                chosen_omega_idx = np.argmin(np.abs(omega_samples - chosen_omega))
                
                ax.plot(chosen_omega_idx, chosen_v_idx, 'ro', mfc='none', markersize=10, label='Chosen (v, ω)')
                ax.legend()
        
        plot_cost(axs[0], tg_cost_weighted, f"To Goal Cost (Gain: {config.to_goal_cost_gain})")
        plot_cost(axs[1], sp_cost_weighted, f"Speed Cost (Gain: {config.speed_cost_gain})")
        plot_cost(axs[2], static_ob_cost_weighted, f"Static Ob Cost (Gain: {config.obstacle_cost_gain})")
        plot_cost(axs[3], dynamic_ob_cost_weighted, f"Dynamic Ob Cost (Gain: {config.side_cost_gain})")
        plot_cost(axs[4], final_cost, "Final Cost", is_final=True)

        # Save combined figure
        os.makedirs(os.path.join(log_dir, "cost_images"), exist_ok=True)
        fig_path = os.path.join(log_dir, "cost_images", f"cost_matrices_{iter_num}.png")
        plt.savefig(fig_path)
        plt.close()
        print(f"Saved cost image to {fig_path}")


    print("Cost matrices saved to disk")

if __name__ == "__main__":
    main()
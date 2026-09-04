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
    from dwa_average_weighted_side import (
        Config, calc_dynamic_window, 
        predict_trajectory_to_goal,
        predict_trajectory_obstacle,
        calc_to_goal_cost, 
        closest_obstacle_on_curve,
        calc_trajectory_clearance_and_collision,
        # --- FIX: Import the filter function ---
        filter_obstacles_by_direction 
    )
    from map_manager import MapManager
except ImportError as e:
    print(f"Error: {e}")
    sys.exit(1)


def calculate_all_costs_merged(x, config, goal, 
                             static_ob, static_ob_radii, 
                             dynamic_ob_pos, dynamic_ob_radii):
    """
    Calculate cost matrices AND physical distance matrices for all (v, ω) pairs.
    """
    
    # --- FIX: Apply Obstacle Filtering to match Runtime DWA ---
    # The runtime planner filters obstacles based on direction (FOV).
    # We must do the same here, or costs will mismatch.
    if dynamic_ob_pos is not None and len(dynamic_ob_pos) > 0:
        dynamic_ob_pos, dynamic_ob_radii = filter_obstacles_by_direction(
            x, dynamic_ob_pos, dynamic_ob_radii, max_angle=config.obstacle_max_angle
        )
    
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
    
    # Dynamic obstacle COST matrices
    dynamic_ob_cost_total = np.full(V.shape, np.nan)
    dynamic_ob_cost_side = np.full(V.shape, np.nan)
    dynamic_ob_cost_direct = np.full(V.shape, np.nan)

    # NEW: Dynamic obstacle DISTANCE matrices (Physical meters)
    dist_side_matrix = np.full(V.shape, np.nan)
    dist_direct_matrix = np.full(V.shape, np.nan)
    
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
            
            # Check braking distance
            if v**2 + 2 * config.max_accel * v * config.dt > 2 * config.max_accel * dist_all:
                continue  # Skip inadmissible pairs
            
            # --- Generate Trajectories ---
            traj_goal = predict_trajectory_to_goal(x.copy(), v, omega, config)
            traj_obs = predict_trajectory_obstacle(x.copy(), v, omega, config)
            
            # --- 1. To Goal Cost ---
            to_goal_cost[i, j] = calc_to_goal_cost(traj_goal, goal)
            
            # --- 2. Speed Cost ---
            speed_cost[i, j] = config.max_speed - traj_goal[-1, 3]
            
            # --- 3. Static Obstacle Cost (Direct Distance) ---
            static_val = 0.0
            if has_static:
                dist_static, _ = closest_obstacle_on_curve(
                    x.copy(), static_ob, static_ob_radii_np, v, omega, config
                )
                static_val = max(0., config.max_obstacle_cost_dist - dist_static)
            static_ob_cost[i, j] = static_val

            # --- 4. Dynamic Obstacle Cost (Split Logic) ---
            dyn_total = 0.0
            dyn_side_val = 0.0
            dyn_direct_val = 0.0
            
            # Placeholders for physical distances
            dist_side_val = np.nan
            dist_direct_val = np.nan

            if has_dynamic:
                # 2. USE CORRECT FUNCTION CALL
                # Note: calc_trajectory_clearance_and_collision returns (d_side_arr, is_collision)
                d_side_arr, is_collision = calc_trajectory_clearance_and_collision(
                    traj_obs, dynamic_ob_pos, dynamic_ob_radii_np, config
                )

                if is_collision:
                    dyn_total = np.inf
                    dyn_side_val = np.inf
                    dyn_direct_val = np.inf
                    dist_side_val = 0.0 
                    dist_direct_val = 0.0
                else:
                    # Calculate D_side Cost Term
                    cost_side = config.max_side_weight_dist - d_side_arr
                    cost_side = np.maximum(0.0, cost_side)

                    # Calculate D_direct Cost Term (Length on Curve)
                    traj_points = traj_obs[:, 0:2]
                    segment_diffs = traj_points[1:] - traj_points[:-1]
                    segment_dists = np.linalg.norm(segment_diffs, axis=1)
                    # Cumulative distance for EVERY point: [0, d1, d1+d2, ...]
                    d_direct_arr = np.concatenate(([0], np.cumsum(segment_dists)))
                    
                    cost_direct = config.max_obstacle_cost_dist - d_direct_arr
                    cost_direct = np.maximum(0.0, cost_direct)

                    # Compound Cost = Side * Direct
                    compound_costs = cost_side * cost_direct
                    
                    if len(compound_costs) > 0:
                        # 3. CORRECT AGGREGATION LOGIC (MEAN vs MAX)
                        # Match logic in dwa_average_weighted_side.py line 348
                        dyn_total = np.mean(compound_costs)
                        
                        # For logging components, we also take the mean to match
                        dyn_side_val = np.mean(cost_side)
                        dyn_direct_val = np.mean(cost_direct)
                        
                        # For physical distances, mean is a fair representation for the "average" path logic
                        dist_side_val = np.mean(d_side_arr)
                        dist_direct_val = np.mean(d_direct_arr)

            dynamic_ob_cost_total[i, j] = dyn_total
            dynamic_ob_cost_side[i, j] = dyn_side_val
            dynamic_ob_cost_direct[i, j] = dyn_direct_val
            
            # Store physical distances
            dist_side_matrix[i, j] = dist_side_val
            dist_direct_matrix[i, j] = dist_direct_val
    
    return (to_goal_cost, speed_cost, static_ob_cost, 
            dynamic_ob_cost_total, dynamic_ob_cost_side, dynamic_ob_cost_direct,
            dist_side_matrix, dist_direct_matrix, # NEW RETURNS
            v_samples, omega_samples)



def main():
    # --- Configuration ---
    # !! IMPORTANT: Update these paths !!
    log_file_path = "Logs/figs_v11.7-video1-corrected_20251210_182350/log_details.csv" 
    map_config_file = "PathPlanning/DWAT_v3_split/map_config/map_config_video1.json"
    
    iter_nums = list(range(1155, 1165))  # Specify iterations
    # ---------------------

    if not os.path.exists(log_file_path):
        print(f"Error: Log CSV file not found at '{log_file_path}'")
        return

    json_log_path = log_file_path.replace('.csv', '.json')
    if not os.path.exists(json_log_path):
        json_log_path = os.path.join(os.path.dirname(log_file_path), "log_details.json")
        if not os.path.exists(json_log_path):
            print("Error: Log JSON file required for full trajectory.")
            return

    if not os.path.exists(map_config_file):
        print(f"Error: Map config file not found at '{map_config_file}'")
        return

    ## Config
    config = Config()

    ## Load log data
    df = pd.read_csv(log_file_path, index_col="iteration")
    with open(json_log_path, 'r') as f:
        log_data_json = json.load(f)
    trajectory_full = np.array(log_data_json['trajectory'])

    # Check validity
    max_iter = df.index.max()
    valid_iter_nums = [i for i in iter_nums if i <= max_iter]
    
    print(f"Processing iterations: {valid_iter_nums}")

    for iter_num in valid_iter_nums:
        print(f"--- Processing iteration {iter_num} ---")
        
        ## 1. Reset Map & Obstacles
        map_manager = MapManager(config)
        map_manager.load_map_config(map_config_file)
        
        static_ob = map_manager.get_static_obstacles()
        static_ob_radii = map_manager.get_static_obstacle_radii()

        ## 2. Sync Dynamic Obstacles
        for _ in range(iter_num + 1):
            map_manager.update_dynamic_obstacles(config.dt)

        dynamic_ob_pos = map_manager.get_dynamic_obstacles_pos()
        dynamic_ob_radii = map_manager.get_dynamic_obstacle_radii()

        ## 3. Get Robot State
        x = trajectory_full[iter_num] 
        goal = np.array(df.loc[iter_num, ['local_goal_x', 'local_goal_y']])

        ## 4. Calculate Costs (Unpacking new return values)
        (tg_cost, sp_cost, static_cost, 
         dyn_total, dyn_side, dyn_direct, 
         dist_side, dist_direct, # NEW UNPACKING
         v_samples, omega_samples) = calculate_all_costs_merged(
            x, config, goal, 
            static_ob, static_ob_radii, 
            dynamic_ob_pos, dynamic_ob_radii
         )
        
        ## 5. Apply Gains (Weighted Costs)
        tg_weighted = config.to_goal_cost_gain * tg_cost
        sp_weighted = config.speed_cost_gain * sp_cost
        static_weighted = config.obstacle_cost_gain * static_cost
        dyn_weighted = config.side_cost_gain * dyn_total
        
        # Final Sum
        final_cost = (
            np.nan_to_num(tg_weighted, nan=np.nan, posinf=np.inf) +
            np.nan_to_num(sp_weighted, nan=np.nan, posinf=np.inf) +
            np.nan_to_num(static_weighted, nan=np.nan, posinf=np.inf) +
            np.nan_to_num(dyn_weighted, nan=np.nan, posinf=np.inf)
        )
        
        # Get chosen control
        chosen_v = df.loc[iter_num, 'v_chosen']
        chosen_omega = df.loc[iter_num, 'omega_chosen']
        
        # --- Save Data ---
        log_dir = os.path.dirname(log_file_path)
        curr_cost_dir = os.path.join(log_dir, f"cost_matrices_{iter_num}")
        os.makedirs(curr_cost_dir, exist_ok=True)
        
        # Save Costs
        np.savetxt(os.path.join(curr_cost_dir, "to_goal_cost.txt"), tg_cost, fmt='%.3f')
        np.savetxt(os.path.join(curr_cost_dir, "speed_cost.txt"), sp_cost, fmt='%.3f')
        np.savetxt(os.path.join(curr_cost_dir, "static_cost.txt"), static_cost, fmt='%.3f')
        np.savetxt(os.path.join(curr_cost_dir, "dyn_total.txt"), dyn_total, fmt='%.3f')
        np.savetxt(os.path.join(curr_cost_dir, "dyn_side.txt"), dyn_side, fmt='%.3f')
        np.savetxt(os.path.join(curr_cost_dir, "dyn_direct.txt"), dyn_direct, fmt='%.3f')
        
        # NEW: Save Physical Distances
        np.savetxt(os.path.join(curr_cost_dir, "dist_side.txt"), dist_side, fmt='%.3f')
        np.savetxt(os.path.join(curr_cost_dir, "dist_direct.txt"), dist_direct, fmt='%.3f')
        
        np.savetxt(os.path.join(curr_cost_dir, "final_cost.txt"), final_cost, fmt='%.3f')

        # --- Plotting ---
        step = max(1, len(omega_samples) // 10)
        x_ticks = np.arange(0, len(omega_samples), step)
        v_step = max(1, len(v_samples) // 10)
        y_ticks = np.arange(0, len(v_samples), v_step)

        # Updated to 9 Subplots
        fig, axs = plt.subplots(1, 9, figsize=(45, 5))
        plt.subplots_adjust(wspace=0.3)

        def plot_cost(ax, cost, title, is_final=False, is_component=False, is_distance=False):
            cmap = plt.get_cmap('jet')
            cmap.set_bad(color='black')
            
            display_cost = cost.copy()
            if not is_final and not is_distance:
                # Cap infinities for better color range
                valid_vals = display_cost[display_cost != np.inf]
                if len(valid_vals) > 0:
                     cap_val = np.nanmax(valid_vals) * 1.2
                     display_cost[display_cost == np.inf] = cap_val
                else:
                    display_cost[:] = 10.0 # arbitrary fallback if all inf
            
            im = ax.imshow(display_cost, origin='lower', cmap=cmap, aspect='auto')
            
            # Label
            label_text = 'Distance (m)' if is_distance else 'Cost'
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label_text)
            
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(np.round(omega_samples[x_ticks], 2), rotation=90)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(np.round(v_samples[y_ticks], 2))
            
            ax.set_title(title, fontsize=10)
            
            # Plot chosen point
            if chosen_v is not None:
                v_idx = np.argmin(np.abs(v_samples - chosen_v))
                w_idx = np.argmin(np.abs(omega_samples - chosen_omega))
                ax.plot(w_idx, v_idx, 'ro', mfc='none', markersize=8, markeredgewidth=2)

        # 1-3. Basic Costs
        plot_cost(axs[0], tg_weighted, f"To Goal (Weighted)\nGain: {config.to_goal_cost_gain}")
        plot_cost(axs[1], sp_weighted, f"Speed (Weighted)\nGain: {config.speed_cost_gain}")
        plot_cost(axs[2], static_weighted, f"Static (Weighted)\nGain: {config.obstacle_cost_gain}")
        
        # 4-5. Dyn Component Costs (Calculated from Distances)
        plot_cost(axs[3], dyn_side, "Dyn Side COST\n(Derived from Clearance)", is_component=True)
        plot_cost(axs[4], dyn_direct, "Dyn Direct COST\n(Derived from Long. Dist)", is_component=True)
        
        # 6-7. NEW: Physical Distances
        plot_cost(axs[5], dist_side, "Side Distance (m)\n(Actual Lateral Clearance)", is_distance=True)
        plot_cost(axs[6], dist_direct, "Direct Distance (m)\n(Actual Long. Dist)", is_distance=True)

        # 8-9. Totals
        plot_cost(axs[7], dyn_weighted, f"Dyn Total (Weighted)\nGain: {config.side_cost_gain}")
        plot_cost(axs[8], final_cost, "Final Cost", is_final=True)

        fig_path = os.path.join(log_dir, "cost_images", f"cost_matrices_{iter_num}.png")
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close()
        print(f"Saved visualization to {fig_path}")

    print("Processing complete.")

if __name__ == "__main__":
    main()
import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Combined Log Processor
Usage:
    python process_log.py Logs/dwa_log_details_xxx/log_details.json
"""

def load_and_clean_data(log_file_path):
    """
    Loads JSON, extracts lists, and handles length mismatches.
    Returns a dictionary of aligned data arrays.
    """
    print(f"Loading log data from: {log_file_path}")
    
    with open(log_file_path, 'r') as f:
        log_data = json.load(f)
        
    log_entries = log_data['log_entries']
    trajectory = np.array(log_data['trajectory'])

    # --- 1. Extract Data from Log Entries ---
    data = {
        'iteration': [entry['iteration'] for entry in log_entries],
        'v_chosen': [entry['chosen_v'] for entry in log_entries],
        'omega_chosen': [entry['chosen_omega'] for entry in log_entries],
        'local_goal_x': [entry['local_goal_x'] for entry in log_entries],
        'local_goal_y': [entry['local_goal_y'] for entry in log_entries],
        'final_cost': [entry['final_cost'] for entry in log_entries],
        
        # Raw Costs
        'to_goal_before': [entry['to_goal_cost_before'] for entry in log_entries],
        'speed_before': [entry['speed_cost_before'] for entry in log_entries],
        'static_ob_before': [entry.get('static_ob_cost_before', 0) for entry in log_entries],
        'dynamic_ob_before': [entry.get('dynamic_ob_cost_before', 0) for entry in log_entries],
        
        # New Split Components (using .get for backward compatibility)
        'dyn_side_cost': [entry.get('dynamic_ob_side_cost', 0) for entry in log_entries],
        'dyn_direct_cost': [entry.get('dynamic_ob_direct_cost', 0) for entry in log_entries],
        
        # Weighted Costs
        'to_goal_after': [entry['to_goal_cost_after'] for entry in log_entries],
        'speed_after': [entry['speed_cost_after'] for entry in log_entries],
        'static_ob_after': [entry.get('static_ob_after', 0) for entry in log_entries],
        'dynamic_ob_after': [entry.get('dynamic_ob_after', 0) for entry in log_entries],
    }

    # --- 2. Extract Trajectory Data ---
    # Trajectory usually has initial state (index 0), so we slice from 1 to match iterations
    traj_data = {
        'x_traj': trajectory[1:, 0],
        'y_traj': trajectory[1:, 1],
        'yaw_traj': trajectory[1:, 2],
        'v_traj': trajectory[1:, 3],
        'omega_traj': trajectory[1:, 4]
    }

    # --- 3. Handle Length Mismatches ---
    # Determine the minimum length common to all arrays
    len_logs = len(data['iteration'])
    len_traj = len(traj_data['x_traj'])
    
    if len_traj != len_logs:
        print(f"Warning: Mismatch in trajectory length ({len_traj}) and log entries ({len_logs}).")
        min_len = min(len_logs, len_traj)
        print(f"Truncating all data to length {min_len}...")
    else:
        min_len = len_logs

    # Combine and truncate
    combined_data = {}
    
    # Process dictionary lists
    for key, lst in data.items():
        combined_data[key] = lst[:min_len]
        
    # Process numpy arrays
    for key, arr in traj_data.items():
        combined_data[key] = arr[:min_len]

    return combined_data

def save_csv(data_dict, original_log_path):
    """Saves the data dictionary to a CSV file."""
    df = pd.DataFrame(data_dict)
    csv_path = os.path.splitext(original_log_path)[0] + '.csv'
    df.to_csv(csv_path, index=False, float_format='%.8f')
    print(f"CSV saved to: {csv_path}")

def generate_plots(data, output_dir):
    """Generates and saves analysis plots."""
    print(f"Generating plots in: {output_dir}")
    
    iterations = data['iteration']
    
    # Helper for saving plots
    def save_plot(x, y, xlabel, ylabel, title, filename, labels=None):
        plt.figure(figsize=(10, 6))
        if labels and isinstance(y, list):
            for y_data, label in zip(y, labels):
                plt.plot(x, y_data, label=label, linewidth=1.5)
            plt.legend()
        else:
            plt.plot(x, y, linewidth=1.5)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, filename), dpi=100)
        plt.close()

    # 1. Translational Speed
    save_plot(iterations, data['v_chosen'], 
             'Iteration', 'Speed (m/s)', 'Chosen Translational Speed',
             '1. translational_speed.png')

    # 2. Rotational Speed
    save_plot(iterations, data['omega_chosen'], 
             'Iteration', 'Speed (rad/s)', 'Chosen Rotational Speed',
             '2. rotational_speed.png')

    # 3. Final Cost
    save_plot(iterations, data['final_cost'],
             'Iteration', 'Cost', 'Final Cost (Weighted Sum)',
             '3. final_cost.png')

    # 4. To Goal Cost (Raw)
    save_plot(iterations, data['to_goal_before'],
             'Iteration', 'Cost', 'Raw To-Goal Cost',
             '4. to_goal_cost_before.png')

    # 5. Speed Cost (Raw)
    save_plot(iterations, data['speed_before'],
             'Iteration', 'Cost', 'Raw Speed Cost',
             '5. speed_cost_before.png')

    # 6. Obstacle Costs (Raw)
    save_plot(iterations, [data['static_ob_before'], data['dynamic_ob_before']],
             'Iteration', 'Cost', 'Raw Obstacle Costs',
             '6. obstacle_costs_before.png',
             labels=['Static Obstacle', 'Dynamic Obstacle'])

    # 7. Normalized Costs Stacked
    save_plot(iterations, 
             [data['to_goal_after'], data['speed_after'], data['static_ob_after'], data['dynamic_ob_after']],
             'Iteration', 'Weighted Cost', 'Weighted Cost Components',
             '7. costs_after_normalization.png',
             labels=['To Goal', 'Speed', 'Static Obs', 'Dynamic Obs'])

    # 8. Position Components
    save_plot(iterations, [data['x_traj'], data['y_traj']],
             'Iteration', 'Position (m)', 'Position vs Iteration',
             '8. position_components.png',
             labels=['X', 'Y'])

    # 9. 2D Trajectory Map
    plt.figure(figsize=(8, 8))
    plt.plot(data['x_traj'], data['y_traj'], '-r', label='Path', linewidth=2)
    plt.plot(data['x_traj'][0], data['y_traj'][0], 'go', label='Start', markersize=8)
    plt.plot(data['x_traj'][-1], data['y_traj'][-1], 'bo', label='End', markersize=8)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Robot Trajectory 2D')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(os.path.join(output_dir, '9. trajectory_2d.png'), dpi=100)
    plt.close()

    # 10. Dynamic Cost Decomposition (New!)
    save_plot(iterations, [data['dyn_side_cost'], data['dyn_direct_cost']],
             'Iteration', 'Component Value', 
             'Dynamic Obstacle Cost Decomposition',
             '10. dynamic_cost_decomposition.png',
             labels=['Side Cost (Lateral Clearance)', 'Direct Cost (Longitudinal Dist)'])
             
    print("All plots generated successfully.")

def main():
    parser = argparse.ArgumentParser(description='Process DWA logs: Convert to CSV and Plot Figures')
    parser.add_argument('log_file', help='Path to DWA log JSON file')
    args = parser.parse_args()

    if not os.path.exists(args.log_file):
        print(f"Error: File not found {args.log_file}")
        return

    # 1. Load and Clean
    clean_data = load_and_clean_data(args.log_file)
    
    # 2. Save CSV
    save_csv(clean_data, args.log_file)
    
    # 3. Generate Plots
    output_dir = os.path.dirname(args.log_file)
    generate_plots(clean_data, output_dir)

if __name__ == "__main__":
    main()
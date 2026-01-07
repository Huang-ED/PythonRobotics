import os
import json
import numpy as np
import pandas as pd

"""
In the command line, run:
python log_to_df.py Logs/dwa_log_details_xxx/log_details.json
"""

def main(args):
    log_file_path = args.log_file

    with open(log_file_path, 'r') as f:
        log_data = json.load(f)
    log_entries = log_data['log_entries']
    trajectory = np.array(log_data['trajectory'])

    # Extract data from logs
    iterations = [entry['iteration'] for entry in log_entries]
    v_values = [entry['chosen_v'] for entry in log_entries]
    omega_values = [entry['chosen_omega'] for entry in log_entries]
    final_costs = [entry['final_cost'] for entry in log_entries]

    local_goals_x = [entry['local_goal_x'] for entry in log_entries]
    local_goals_y = [entry['local_goal_y'] for entry in log_entries]

    # Before normalization - UPDATED KEYS
    to_goal_before = [entry['to_goal_cost_before'] for entry in log_entries]
    speed_before = [entry['speed_cost_before'] for entry in log_entries]
    static_ob_before = [entry['static_ob_cost_before'] for entry in log_entries]
    dynamic_ob_before = [entry['dynamic_ob_cost_before'] for entry in log_entries]

    # After normalization - UPDATED KEYS
    to_goal_after = [entry['to_goal_cost_after'] for entry in log_entries]
    speed_after = [entry['speed_cost_after'] for entry in log_entries]
    static_ob_after = [entry['static_ob_after'] for entry in log_entries]
    dynamic_ob_after = [entry['dynamic_ob_after'] for entry in log_entries]

    # Position data
    # Trajectory includes initial state (index 0), so we slice from 1
    x = trajectory[1:, 0]
    y = trajectory[1:, 1]
    yaw = trajectory[1:, 2]
    v = trajectory[1:, 3]
    omega = trajectory[1:, 4]

    # Ensure all data lists have the same length as 'iterations'
    if len(x) != len(iterations):
        print(f"Warning: Mismatch in trajectory length ({len(x)}) and log entries ({len(iterations)}). Adjusting trajectory data.")
        # This can happen if logging stops before the last state is recorded
        # or if the initial state is logged differently.
        # We'll truncate the longer list to match the shorter one.
        min_len = min(len(iterations), len(x))
        iterations = iterations[:min_len]
        v_values = v_values[:min_len]
        omega_values = omega_values[:min_len]
        local_goals_x = local_goals_x[:min_len]
        local_goals_y = local_goals_y[:min_len]
        final_costs = final_costs[:min_len]
        to_goal_before = to_goal_before[:min_len]
        speed_before = speed_before[:min_len]
        static_ob_before = static_ob_before[:min_len]
        dynamic_ob_before = dynamic_ob_before[:min_len]
        to_goal_after = to_goal_after[:min_len]
        speed_after = speed_after[:min_len]
        static_ob_after = static_ob_after[:min_len]
        dynamic_ob_after = dynamic_ob_after[:min_len]
        x = x[:min_len]
        y = y[:min_len]
        yaw = yaw[:min_len]
        v = v[:min_len]
        omega = omega[:min_len]


    # Save every data above into a csv file
    data = {
        'iteration': iterations,
        'v_chosen': v_values,
        'omega_chosen': omega_values,
        'local_goal_x': local_goals_x,
        'local_goal_y': local_goals_y,
        'final_cost': final_costs,
        'to_goal_before': to_goal_before,
        'speed_before': speed_before,
        'static_ob_before': static_ob_before,   # UPDATED
        'dynamic_ob_before': dynamic_ob_before, # UPDATED
        'to_goal_after': to_goal_after,
        'speed_after': speed_after,
        'static_ob_after': static_ob_after,     # UPDATED
        'dynamic_ob_after': dynamic_ob_after,   # UPDATED
        'x_traj': x,
        'y_traj': y,
        'yaw_traj': yaw,
        'v_traj': v,
        'omega_traj': omega
    }

    df = pd.DataFrame(data)

    csv_file_path = os.path.splitext(log_file_path)[0] + '.csv'
    df.to_csv(csv_file_path, index=False, float_format='%.8f')
    print(f"DataFrame saved to {csv_file_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Convert DWA log JSON to CSV')
    parser.add_argument('log_file', help='Path to DWA log JSON file')
    args = parser.parse_args()
    main(args)
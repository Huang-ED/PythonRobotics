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

    # Before normalization
    to_goal_before = [entry['to_goal_cost_before'] for entry in log_entries]
    speed_before = [entry['speed_cost_before'] for entry in log_entries]
    ob_before = [entry['ob_cost_before'] for entry in log_entries]

    # After normalization
    to_goal_after = [entry['to_goal_cost_after'] for entry in log_entries]
    speed_after = [entry['speed_cost_after'] for entry in log_entries]
    ob_after = [entry['ob_cost_after'] for entry in log_entries]

    # Position data
    x = trajectory[1:, 0]
    y = trajectory[1:, 1]
    yaw = trajectory[1:, 2]
    v = trajectory[1:, 3]
    omega = trajectory[1:, 4]

    # Custom - Find the maximum value of to_goal_after
    to_goal_after = np.array(to_goal_after)
    # max_idx = np.argmax(to_goal_after[200:]) + 200
    # print(max_idx)
    # print(to_goal_after[max_idx-1])
    # print(to_goal_after[max_idx])
    # print(to_goal_after[max_idx+1])

    # Save every data above into a csv file
    data = {
        'iteration': iterations,
        'v': v_values,
        'omega': omega_values,
        'local_goal_x': local_goals_x,
        'local_goal_y': local_goals_y,
        'final_cost': final_costs,
        'to_goal_before': to_goal_before,
        'speed_before': speed_before,
        'ob_before': ob_before,
        'to_goal_after': to_goal_after,
        'speed_after': speed_after,
        'ob_after': ob_after,
        'x_traj': x,
        'y_traj': y,
        'yaw_traj': yaw,
        'v_traj': v,
        'omega_traj': omega
    }
    # # print the length of each data
    # for key in data:
    #     print(f"{key}: {len(data[key])}")
    df = pd.DataFrame(data)

    # # Convert omega from rad/s to deg/s
    # df['omega'] = np.rad2deg(df['omega'])
    # df['omega_traj'] = np.rad2deg(df['omega_traj'])

    csv_file_path = os.path.splitext(log_file_path)[0] + '.csv'
    df.to_csv(csv_file_path, index=False, float_format='%.8f')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Plot DWA log data')
    parser.add_argument('log_file', help='Path to DWA log JSON file')
    args = parser.parse_args()
    main(args)
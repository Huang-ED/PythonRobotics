import os
import json
import numpy as np
import pandas as pd

def main(args):
    log_file_path = args.log_file

    with open(log_file_path, 'r') as f:
        log_data = json.load(f)

    # Extract data from logs
    iterations = [entry['iteration'] for entry in log_data]
    v_values = [entry['chosen_v'] for entry in log_data]
    omega_values = [entry['chosen_omega'] for entry in log_data]
    final_costs = [entry['final_cost'] for entry in log_data]

    # Before normalization
    to_goal_before = [entry['to_goal_cost_before'] for entry in log_data]
    speed_before = [entry['speed_cost_before'] for entry in log_data]
    ob_before = [entry['ob_cost_before'] for entry in log_data]

    # After normalization
    to_goal_after = [entry['to_goal_cost_after'] for entry in log_data]
    speed_after = [entry['speed_cost_after'] for entry in log_data]
    ob_after = [entry['ob_cost_after'] for entry in log_data]

    to_goal_after = np.array(to_goal_after)
    max_idx = np.argmax(to_goal_after[200:]) + 200
    print(max_idx)
    print(to_goal_after[max_idx-1])
    print(to_goal_after[max_idx])
    print(to_goal_after[max_idx+1])

    # Save every data above into a csv file
    data = {
        'iteration': iterations,
        'v': v_values,
        'omega': omega_values,
        'final_cost': final_costs,
        'to_goal_before': to_goal_before,
        'speed_before': speed_before,
        'ob_before': ob_before,
        'to_goal_after': to_goal_after,
        'speed_after': speed_after,
        'ob_after': ob_after
    }
    df = pd.DataFrame(data)

    # Convert omega from rad/s to deg/s
    df['omega'] = np.rad2deg(df['omega'])

    csv_file_path = os.path.splitext(log_file_path)[0] + '.csv'
    df.to_csv(csv_file_path, index=False, float_format='%.3f')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Plot DWA log data')
    parser.add_argument('log_file', help='Path to DWA log JSON file')
    args = parser.parse_args()
    main(args)
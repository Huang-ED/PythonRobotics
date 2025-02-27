import json
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

def plot_dwa_logs(log_file_path):
    # Load log data
    with open(log_file_path, 'r') as f:
        data = json.load(f)
    
    log_entries = data['log_entries']
    trajectory = np.array(data['trajectory'])
    
    # Extract basic data
    iterations = [entry['iteration'] for entry in log_entries]
    v_values = [entry['chosen_v'] for entry in log_entries]
    omega_values = [entry['chosen_omega'] for entry in log_entries]
    final_costs = [entry['final_cost'] for entry in log_entries]
    
    # Cost data extraction
    to_goal_before = [entry['to_goal_cost_before'] for entry in log_entries]
    speed_before = [entry['speed_cost_before'] for entry in log_entries]
    ob_before = [entry['ob_cost_before'] for entry in log_entries]
    to_goal_after = [entry['to_goal_cost_after'] for entry in log_entries]
    speed_after = [entry['speed_cost_after'] for entry in log_entries]
    ob_after = [entry['ob_cost_after'] for entry in log_entries]

    # Position data
    x_pos = trajectory[:, 0]
    y_pos = trajectory[:, 1]
    pos_iterations = list(range(len(x_pos)))

    # Create output directory
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # output_dir = f"dwa_plots_{timestamp}"
    # os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.dirname(log_file_path)

    # Plotting helper function
    def save_plot(x, y, xlabel, ylabel, title, filename, labels=None):
        plt.figure()
        if labels and isinstance(y, list):
            for y_data, label in zip(y, labels):
                plt.plot(x, y_data, label=label)
            plt.legend()
        else:
            plt.plot(x, y)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

    # 1. Translational Speed
    save_plot(iterations, v_values, 
             'Iteration Number', 'Translational Speed (m/s)', 
             'Chosen Translational Speed vs Iteration',
             '1. translational_speed.png')

    # 2. Rotational Speed
    save_plot(iterations, omega_values, 
             'Iteration Number', 'Rotational Speed (rad/s)', 
             'Chosen Rotational Speed vs Iteration',
             '2. rotational_speed.png')

    # 3. Final Cost
    save_plot(iterations, final_costs,
             'Iteration Number', 'Final Cost',
             'Final Cost vs Iteration',
             '3. final_cost.png')

    # 4. To Goal Cost (Before Normalization)
    save_plot(iterations, to_goal_before,
             'Iteration Number', 'Cost',
             'To Goal Cost Before Normalization',
             '4. to_goal_cost_before.png')

    # 5. Speed Cost (Before Normalization)
    save_plot(iterations, speed_before,
             'Iteration Number', 'Cost',
             'Speed Cost Before Normalization',
             '5. speed_cost_before.png')

    # 6. Obstacle Cost (Before Normalization)
    save_plot(iterations, ob_before,
             'Iteration Number', 'Cost',
             'Obstacle Cost Before Normalization',
             '6. obstacle_cost_before.png')

    # 7. Normalized Costs (Combined)
    save_plot(iterations, [to_goal_after, speed_after, ob_after],
             'Iteration Number', 'Normalized Cost',
             'Normalized Cost Components',
             '7. costs_after_normalization.png',
             labels=['To Goal', 'Speed', 'Obstacle'])

    # 8. Position Components
    save_plot(pos_iterations, [x_pos, y_pos],
             'Iteration Number', 'Position (m)',
             'Position Components vs Iteration',
             '8. position_components.png',
             labels=['X Position', 'Y Position'])

    # 9. 2D Trajectory
    plt.figure(figsize=(8, 8))
    plt.plot(x_pos, y_pos, '-r', label='Path')
    plt.plot(x_pos[0], y_pos[0], 'go', label='Start')
    plt.plot(x_pos[-1], y_pos[-1], 'bo', label='End')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Ship Trajectory')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(os.path.join(output_dir, '9. trajectory_2d.png'))
    plt.close()

    print(f"Plots saved to directory: {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Plot DWA log data')
    parser.add_argument('log_file', help='Path to DWA log JSON file')
    args = parser.parse_args()
    
    plot_dwa_logs(args.log_file)

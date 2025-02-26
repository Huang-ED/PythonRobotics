import json
import matplotlib.pyplot as plt
import os
from datetime import datetime

def plot_dwa_logs(log_file_path):
    # Load log data
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

    # Create output directory
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # output_dir = f"dwa_plots_{timestamp}"
    # os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.dirname(log_file_path)

    # Plotting functions
    def save_plot(x, y, xlabel, ylabel, title, filename):
        plt.figure()
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

    # 7. Normalized Costs
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, to_goal_after, label='To Goal Cost (Normalized)')
    plt.plot(iterations, speed_after, label='Speed Cost (Normalized)')
    plt.plot(iterations, ob_after, label='Obstacle Cost (Normalized)')
    plt.xlabel('Iteration Number')
    plt.ylabel('Normalized Cost')
    plt.title('Normalized Cost Components')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, '7. costs_after_normalization.png'))
    plt.close()

    print(f"Plots saved to directory: {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Plot DWA log data')
    parser.add_argument('log_file', help='Path to DWA log JSON file')
    args = parser.parse_args()
    
    plot_dwa_logs(args.log_file)

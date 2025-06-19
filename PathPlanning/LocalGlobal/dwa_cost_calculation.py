import os, sys
rpath = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.append(rpath)

import numpy as np
import pandas as pd
import math
import cv2
import matplotlib.pyplot as plt

from PathPlanning.DynamicWindowApproach.dwa_paper_with_width \
    import calc_dynamic_window, closest_obstacle_on_curve, predict_trajectory, calc_to_goal_cost
from PathPlanning.LocalGlobal.dwa_astar_v7_video2 import Config


def calculate_all_costs(x, config, goal, ob):
    """
    Calculate cost matrices for all (v, ω) pairs in the dynamic window.
    
    Parameters:
        x (np.array): Current state [x, y, yaw, v, ω]
        config (Config): Configuration parameters
        goal (np.array): Goal position [x, y]
        ob (np.array): Obstacle positions (Nx2 array)
    
    Returns:
        tuple: (to_goal_cost, speed_cost, ob_cost, v_samples, omega_samples)
    """
    # Calculate dynamic window
    dw = calc_dynamic_window(x, config)
    
    # Generate velocity and yaw rate samples
    v_samples = np.arange(dw[0], dw[1] + 1e-6, config.v_resolution)
    omega_samples = np.arange(dw[2], dw[3] + 1e-6, config.yaw_rate_resolution)
    
    # Create meshgrid for all (v, ω) pairs
    V, Omega = np.meshgrid(v_samples, omega_samples, indexing='ij')
    
    # Initialize cost matrices with nan (inadmissible)
    # to_goal_cost = np.full_like(V, -1.0)
    # speed_cost = np.full_like(V, -1.0)
    # ob_cost = np.full_like(V, -1.0)
    to_goal_cost = np.full(V.shape, np.nan)
    speed_cost = np.full(V.shape, np.nan)
    ob_cost = np.full(V.shape, np.nan)
    
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            v = V[i, j]
            omega = Omega[i, j]
            
            # Check admissibility
            dist, _ = closest_obstacle_on_curve(x.copy(), ob, v, omega, config)
            # if v > math.sqrt(2*config.max_accel*dist):
            # if v**2 + config.max_accel * v * config.dt > 2 * config.max_accel * dist:
            if v**2 + 2 * config.max_accel * v * config.dt > 2 * config.max_accel * dist:
                continue  # Skip inadmissible pairs
            
            # Calculate trajectory and costs for admissible pairs
            trajectory = predict_trajectory(x.copy(), v, omega, config)
            
            # To_goal cost
            to_goal_cost[i, j] = calc_to_goal_cost(trajectory, goal)
            
            # Speed cost
            speed_cost[i, j] = config.max_speed - trajectory[-1, 3]
            
            # Obstacle cost
            ob_cost[i, j] = 1.0 / dist if dist != 0 and not np.isinf(dist) else (0.0 if np.isinf(dist) else np.inf)
    
    return to_goal_cost, speed_cost, ob_cost, v_samples, omega_samples



def main():
    log_file_path = "Logs/dwa_log_details_20250619_125137_v7.3.8-test3_vid1/log_details.csv"

    ## Config
    config = Config()

    ## Define the map
    # image_path = "EnvData/AISData_20240827/land_shapes_sf_crop.png"
    image_path = "EnvData/AISData_20240827/land_shapes_ht_crop.png"
    arr = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    arr = cv2.resize(arr, (100, 100))
    _, arr = cv2.threshold(arr, 128, 1, cv2.THRESH_BINARY)

    # Add boundary obstacles
    arr[0, :] = 0    # Top edge
    arr[-1, :] = 0   # Bottom edge
    arr[:, 0] = 0    # Left edge
    arr[:, -1] = 0   # Right edge

    ### DWA专用地图处理 ###
    arr = 1 - arr  # 反转值 (0=障碍,1=可行区域)
    eroded_arr = cv2.erode(arr, kernel=np.ones((3, 3), np.uint8), iterations=1)
    arr_dwa = cv2.subtract(arr, eroded_arr)  # 提取边界
    arr_dwa = 1 - arr_dwa  # 再次反转，获得原始障碍表示格式

    # 提取障碍坐标
    ob_dwa = np.argwhere(arr_dwa == 0)
    ob_dwa[:, [0, 1]] = ob_dwa[:, [1, 0]]  # 交换x,y坐标
    ob_dwa[:, 1] = arr_dwa.shape[0] - ob_dwa[:, 1] - 1  # 翻转y轴坐标系

    # 在A*路径上添加的额外障碍物
    # new_ob = np.array([
    #     [25., 79.], [25., 80.], [26., 79.], [26., 80.],
    #     [35., 55.], [36., 56],
    #     [28., 46.], [27., 47.], [29., 45.],
    #     [12., 19.], [12., 20.], [11., 19.], [11., 20.]
    # ])
    new_ob = np.array([
        [67., 75.], [67., 74.], [66., 75.], [66., 74.],
        [57., 65.], [57., 66.], [58., 65.], [58., 66.],
        [43., 44.], [42., 43.], [43., 43.], [42., 44.],
        [67., 23.], [67., 24.], [68., 23.], [68., 24.]
    ])
    ob_dwa = np.append(ob_dwa, new_ob, axis=0)  # 合并障碍物

    ## Obtain x and goal from log_datails.csv
    # x = np.array([0.0, 0.0, np.pi/4, 0.0, 0.0])  # [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    # goal = np.array([10.0, 10.0])                 # Target position
    # iter_num = 227
    iter_nums = list(range(1080, 1100))  # Specify the iterations you want to process
    for iter_num in iter_nums:
        df = pd.read_csv(log_file_path, index_col="iteration")
        # print(df.loc[iter_num])
        x = np.array(df.loc[iter_num, ['x_traj', 'y_traj', 'yaw_traj', 'v_traj', 'omega_traj']])
        goal = np.array(df.loc[iter_num, ['local_goal_x', 'local_goal_y']])

        ## Calculate costs
        tg_cost, sp_cost, ob_cost, v_samples, omega_samples = calculate_all_costs(x, config, goal, ob_dwa)
        
        # Get the chosen (v, omega) pair for the next time-step
        if iter_num + 1 in df.index:
            chosen_v = df.loc[iter_num + 1, 'v_traj']
            chosen_omega = df.loc[iter_num + 1, 'omega_traj']
        else:
            chosen_v, chosen_omega = None, None
        
        # Save results as text
        log_dir = os.path.dirname(log_file_path)
        curr_cost_dir = os.path.join(log_dir, f"cost_matrices_{iter_num}")
        os.makedirs(curr_cost_dir, exist_ok=True)
        np.savetxt(os.path.join(curr_cost_dir, "to_goal_cost.txt"), tg_cost, fmt='%.3f', delimiter='\t')
        np.savetxt(os.path.join(curr_cost_dir, "speed_cost.txt"), sp_cost, fmt='%.3f', delimiter='\t')
        np.savetxt(os.path.join(curr_cost_dir, "obstacle_cost.txt"), ob_cost, fmt='%.3f', delimiter='\t')
        np.savetxt(os.path.join(curr_cost_dir, "v_samples.txt"), v_samples, fmt='%.3f', delimiter='\t')
        np.savetxt(os.path.join(curr_cost_dir, "omega_samples.txt"), omega_samples, fmt='%.3f', delimiter='\t')

        # Display as image with custom ticks
        # Extract unique omega labels (show every 2nd or 3rd sample to reduce density)
        step = 4  # Adjust this value as needed
        x_ticks_indices = np.arange(0, len(omega_samples), step)
        omega_samples_rounded = np.round(omega_samples, 2)

        """
        # To Goal Cost
        plt.figure()
        plt.imshow(tg_cost, origin='lower', cmap='jet', aspect='auto')
        plt.colorbar(label='Cost')
        plt.xticks(x_ticks_indices, omega_samples_rounded[::step], rotation=45)  # Selective ticks
        plt.yticks(np.arange(len(v_samples)), np.round(v_samples, 2))
        plt.xlabel('Omega (rad/s)')
        plt.ylabel('V (m/s)')
        plt.title("To Goal Cost")
        plt.savefig(os.path.join(curr_cost_dir, "to_goal_cost.png"))
        plt.close()

        # Speed Cost
        plt.figure()
        plt.imshow(sp_cost, origin='lower', cmap='jet', aspect='auto')
        plt.colorbar(label='Cost')
        plt.xticks(x_ticks_indices, omega_samples_rounded[::step], rotation=45)
        plt.yticks(np.arange(len(v_samples)), np.round(v_samples, 2))
        plt.xlabel('Omega (rad/s)')
        plt.ylabel('V (m/s)')
        plt.title("Speed Cost")
        plt.savefig(os.path.join(curr_cost_dir, "speed_cost.png"))
        plt.close()

        # Obstacle Cost
        plt.figure()
        plt.imshow(ob_cost, origin='lower', cmap='jet', aspect='auto')
        plt.colorbar(label='Cost')
        plt.xticks(x_ticks_indices, omega_samples_rounded[::step], rotation=45)
        plt.yticks(np.arange(len(v_samples)), np.round(v_samples, 2))
        plt.xlabel('Omega (rad/s)')
        plt.ylabel('V (m/s)')
        plt.title("Obstacle Cost")
        plt.savefig(os.path.join(curr_cost_dir, "obstacle_cost.png"))
        plt.close()
        """

        # Create combined figure with subplots
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        plt.subplots_adjust(wspace=0.3, hspace=0.3)

        def plot_cost(ax, cost, title):
            cmap = plt.get_cmap('jet')
            cmap.set_bad(color='black')  # Set NaN values to black
            im = ax.imshow(cost, origin='lower', cmap=cmap, aspect='auto')
            plt.colorbar(im, ax=ax, label='Cost')
            ax.set_xticks(x_ticks_indices)
            ax.set_xticklabels(omega_samples_rounded[::step], rotation=45)
            ax.set_yticks(np.arange(len(v_samples)))
            ax.set_yticklabels(np.round(v_samples, 2))
            ax.set_xlabel('Omega (rad/s)')
            ax.set_ylabel('V (m/s)')
            ax.set_title(title)
            if chosen_v is not None and chosen_omega is not None:
                ax.plot(np.argmin(np.abs(omega_samples - chosen_omega)),
                            np.argmin(np.abs(v_samples - chosen_v)), 'ro', markersize=10, label='Chosen (v, ω)')
                ax.legend()
        
        plot_cost(axs[0], tg_cost, "To Goal Cost")
        plot_cost(axs[1], sp_cost, "Speed Cost")
        plot_cost(axs[2], ob_cost, "Obstacle Cost")

        # Save combined figure
        os.makedirs(os.path.join(log_dir, "cost_images"), exist_ok=True)
        plt.savefig(os.path.join(log_dir, "cost_images", f"cost_matrices_{iter_num}.png"))
        plt.close()


    print("Cost matrices saved to disk")

if __name__ == "__main__":
    main()

import os
import sys
rpath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(rpath)

import numpy as np
import pandas as pd
import math
import cv2
import matplotlib.pyplot as plt

from PathPlanning.DynamicWindowApproach.dwa_paper_with_width import (
    calc_dynamic_window, closest_obstacle_on_curve, predict_trajectory, calc_to_goal_cost, 
    motion as dwa_motion, any_circle_overlap_with_box, RobotType
)
from PathPlanning.LocalGlobal.dwa_astar_v7_video2 import Config

def calculate_all_costs_debug(x, config, goal, ob):
    """
    生成带调试信息的成本矩阵，并输出特定速度组合的详细计算过程
    
    Parameters:
        x, config, goal, ob: 同原始函数
    """
    # 核心动态窗口计算逻辑保持不变
    dw = calc_dynamic_window(x, config)
    v_samples = np.arange(dw[0], dw[1] + 1e-6, config.v_resolution)
    omega_samples = np.arange(dw[2], dw[3] + 1e-6, config.yaw_rate_resolution)
    
    V, Omega = np.meshgrid(v_samples, omega_samples, indexing='ij')
    to_goal_cost = np.full_like(V, -1.0)
    speed_cost = np.full_like(V, -1.0)
    ob_cost = np.full_like(V, -1.0)
    
    # 调试目标速度集合 
    debug_targets = [
        # (0.36, 0.060),
        # (0.36, 0.065),
        # (0.36, 0.070)
        (0.36, 0)
    ]
    
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            v = V[i, j]
            omega = Omega[i, j]
            dist, _ = closest_obstacle_on_curve(x.copy(), ob, v, omega, config)
            
            # 动态窗口准入检查
            if v > math.sqrt(2 * config.max_accel * dist):
                continue
                
            # 计算基础成本
            trajectory = predict_trajectory(x.copy(), v, omega, config)
            to_goal_cost[i, j] = calc_to_goal_cost(trajectory, goal)
            speed_cost[i, j] = config.max_speed - trajectory[-1, 3]
            ob_cost[i, j] = 1.0 / dist if dist != 0 and not np.isinf(dist) else np.inf
            
            # --- 调试逻辑：捕获目标速度组合 ---
            target_found = any(
                abs(v - tv) < 0.005 and abs(omega - tw) < 0.005
                for tv, tw in debug_targets
            )
            # print(f"v={v}, ω={omega} | 目标速度组合: {target_found}")
            
            if target_found:
                print(f"\n----- 开始详细计算 [v={v:.2f}, ω={omega:.5f}] -----")
                print(f"当前状态: x={x[0]:.2f}, y={x[1]:.2f}, yaw={x[2]:.5f} rad")
                
                # 独立轨迹模拟
                x_sim = x.copy()
                collision_dist = float('inf')
                total_dist = 0.0
                
                for _ in range(int(config.predict_time / config.dt) + 1):
                    # 计算机器人当前位置和碰撞状态
                    if config.robot_type == RobotType.rectangle:
                        ob_with_radius = np.c_[ob, np.full(len(ob), config.obstacle_radius)]
                        collision = any_circle_overlap_with_box(
                            ob_with_radius, x_sim[:2], 
                            config.robot_length, config.robot_width, x_sim[2]
                        )
                    else:
                        distances = np.linalg.norm(ob - x_sim[:2], axis=1)
                        collision = any(d <= config.robot_radius + config.obstacle_radius for d in distances)
                    
                    # 距离计算逻辑
                    closest_distance = min(
                        np.linalg.norm(ob - x_sim[:2], axis=1) 
                        - (config.robot_radius if config.robot_type == RobotType.circle else 0)
                    )
                    
                    print(
                        f"时间: {_ * config.dt:.1f}s | "
                        f"位置: ({x_sim[0]:.2f}, {x_sim[1]:.2f}) | "
                        f"最近障碍物距离: {closest_distance:.2f}m | "
                        f"碰撞状态: {collision}"
                    )
                    
                    # 碰撞检测
                    if collision:
                        collision_dist = total_dist
                        print(f">>> 碰撞触发! 累积运动距离: {collision_dist:.2f}m <<<")
                        break
                        
                    # 更新状态
                    total_dist += v * config.dt
                    x_sim = dwa_motion(x_sim, [v, omega], config.dt)
                
                # 输出最终计算结果
                final_cost_value = 1.0 / collision_dist if collision_dist != 0 else np.inf
                print(
                    f"----- 计算结果 [v={v:.2f}, ω={omega:.5f}] -----\n"
                    f"到达首次碰撞距离: {collision_dist:.2f}m | "
                    f"障碍物成本: {final_cost_value:.2f}\n"
                    f"------------------------------------------\n"
                )
    
    return to_goal_cost, speed_cost, ob_cost, v_samples, omega_samples

def main_debug():
    """主调试函数"""
    log_file_path = "Logs/dwa_log_details_20250306_155027/log_details.csv"
    config = Config()
    
    # 载入示例迭代（使用用户提供的迭代227）
    df = pd.read_csv(log_file_path, index_col="iteration")
    iter_num = 226
    x = np.array(df.loc[iter_num, ['x_traj', 'y_traj', 'yaw_traj', 'v_traj', 'omega_traj']])
    goal = np.array(df.loc[iter_num, ['local_goal_x', 'local_goal_y']])
    
    # 创建DWA专用障碍物地图（同原流程）
    image_path = "EnvData/AISData_20240827/land_shapes_sf_crop.png"
    arr = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    arr = cv2.resize(arr, (100, 100))
    _, arr = cv2.threshold(arr, 128, 1, cv2.THRESH_BINARY)
    arr[0, :] = 0    # 上边界
    arr[-1, :] = 0   # 下边界
    arr[:, 0] = 0    # 左边界
    arr[:, -1] = 0   # 右边界
    arr = 1 - arr    # DWA地图反色
    eroded_arr = cv2.erode(arr, kernel=np.ones((3, 3), np.uint8), iterations=1)
    arr_dwa = cv2.subtract(arr, eroded_arr)
    arr_dwa = 1 - arr_dwa
    ob_dwa = np.argwhere(arr_dwa == 0)
    ob_dwa[:, [0, 1]] = ob_dwa[:, [1, 0]]  # 交换xy坐标
    ob_dwa[:, 1] = arr_dwa.shape[0] - ob_dwa[:, 1] - 1
    new_ob = np.array([  # 添加额外障碍
        [25., 79.], [25., 80.], [26., 79.], [26., 80.],
        [35., 55.], [36., 56], [28., 46.], [27., 47.],
        [10., 19.], [10., 20.], [11., 19.], [11., 20.]
    ])
    ob_dwa = np.append(ob_dwa, new_ob, axis=0)
    
    # 执行带调试的成本计算
    calculate_all_costs_debug(x, config, goal, ob_dwa)

if __name__ == "__main__":
    main_debug()

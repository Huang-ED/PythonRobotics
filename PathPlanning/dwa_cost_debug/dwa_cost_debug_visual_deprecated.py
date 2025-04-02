import os
import sys
import numpy as np
import pandas as pd
import math
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

rpath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(rpath)

from PathPlanning.DynamicWindowApproach.dwa_paper_with_width import (
    calc_dynamic_window, closest_obstacle_on_curve, predict_trajectory, 
    calc_to_goal_cost, motion as dwa_motion, any_circle_overlap_with_box, 
    RobotType, plot_robot
)
from PathPlanning.LocalGlobal.dwa_astar_v7_video2 import Config

def visualize_debug_frame(x, config, goal, ob_dwa, curr_trajectory, output_dir, collision_point=None, debug_params=None):
    """
    可视化调试帧
    
    Parameters:
        x: 当前状态
        config: 配置参数
        goal: 目标位置
        ob_dwa: 障碍物坐标
        curr_trajectory: 当前预测轨迹 
        collision_point: 发生碰撞的位置
        debug_params: 调试参数字典 (v, omega)
    """
    plt.figure(figsize=(12, 12))
    plt.axis('equal')

    view_size = 10  # 10米见方的视图
    center_x, center_y = x[0], x[1]
    plt.xlim(center_x - view_size/2, center_x + view_size/2)
    plt.ylim(center_y - view_size/2, center_y + view_size/2)
    plt.gca().set_aspect('equal', adjustable='box')  # 锁定宽高比

    
    # 绘制障碍物
    for (ox, oy) in ob_dwa:
        circle = plt.Circle((ox, oy), config.obstacle_radius, color='darkgrey', zorder=5)
        plt.gca().add_patch(circle)
    
    # 绘制预测轨迹
    if curr_trajectory is not None:
        plt.plot(curr_trajectory[:,0], curr_trajectory[:,1], 
                linestyle='--', color='lime', linewidth=2, 
                label='Predicted Path', zorder=10)
    
    # 绘制碰撞点
    if collision_point is not None:
        plt.scatter(collision_point[0], collision_point[1], 
                   s=200, marker='x', color='red', linewidths=3, 
                   label='Collision Point', zorder=15)
    
    # 绘制机器人当前状态
    plot_robot(x[0], x[1], x[2], config)
    plt.plot(x[0], x[1], 'D', color='darkorange', markersize=12, label='Current Position')
    
    # 绘制目标点
    plt.plot(goal[0], goal[1], 's', color='gold', markersize=15, 
            markeredgecolor='black', label='Local Goal', zorder=20)
    
    # 图例和标注
    title_info = ""
    if debug_params:
        title_info = f"Debug: v={debug_params['v']:.2f} m/s, ω={debug_params['omega']:.2f} rad/s"
        plt.title(title_info, fontsize=14, pad=20)
        
    plt.grid(True)
    plt.legend(loc='upper right')
    
    # 保存图像
    output_dir.mkdir(parents=True, exist_ok=True)
    if debug_params:
        filename = output_dir / f"debug_v_{debug_params['v']:.2f}_w_{debug_params['omega']:.2f}.png"
    else:
        filename = output_dir / "debug_frame.png"
    
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def enhanced_calculate_costs(x, config, goal, ob_dwa, vis_params, output_dir):
    """
    增强版成本计算函数，集成可视化功能
    
    Parameters:
        x, config, goal, ob_dwa: 同常规参数
        vis_params: 可视化控制参数
    """
    # 动态窗口参数保持不变
    dw = calc_dynamic_window(x, config)
    v_samples = np.arange(dw[0], dw[1] + 1e-6, config.v_resolution)
    omega_samples = np.arange(dw[2], dw[3] + 1e-6, config.yaw_rate_resolution)
    
    # 准备调试目标速度组合
    debug_targets = [
        (0.36, 0.060, 'navy'),
        (0.36, 0.065, 'royalblue'),
        (0.36, 0.070, 'deepskyblue')
    ]
    
    # 主计算循环
    for v in v_samples:
        for omega in omega_samples:
            # 精度匹配检测
            matched = None
            for t_v, t_o, color in debug_targets:
                if abs(v - t_v) < 0.005 and abs(omega - t_o) < 0.005: 
                    matched = (t_v, t_o, color)
                    print(matched)
                    break
            if not matched:
                continue
                
            print(f"\n{'='*40}")
            print(f" 开始调试 v={matched[0]:.2f}, ω={matched[1]:.2f} ")
            print(f"{'='*40}")
            
            # 轨迹预测与碰撞检测
            trajectory = []
            collision_info = None
            x_sim = x.copy()
            for t in np.arange(0, config.predict_time + 1e-6, config.dt):
                # 记录轨迹点
                trajectory.append(x_sim.copy())
                
                # 碰撞检测逻辑
                if config.robot_type == RobotType.rectangle:
                    ob_radius = np.c_[ob_dwa, np.full(len(ob_dwa), config.obstacle_radius)]
                    if any_circle_overlap_with_box(
                        ob_radius, x_sim[:2], 
                        config.robot_length, config.robot_width, x_sim[2]
                    ):
                        collision_info = {
                            'pos': x_sim[:2].copy(),
                            'time': t,
                            'distance': np.linalg.norm(trajectory[0][:2] - x_sim[:2])
                        }
                        break
                else:
                    dists = np.linalg.norm(ob_dwa - x_sim[:2], axis=1)
                    if np.any(dists <= config.robot_radius + config.obstacle_radius):
                        collision_info = {
                            'pos': x_sim[:2].copy(),
                            'time': t,
                            'distance': np.linalg.norm(trajectory[0][:2] - x_sim[:2])
                        }
                        break
                
                # 更新状态
                x_sim = dwa_motion(x_sim, [v, omega], config.dt)
            
            trajectory = np.array(trajectory)
            
            # 执行可视化
            if vis_params.get('enable_visualization', False):
                debug_params = {
                    'v': matched[0],
                    'omega': matched[1],
                    'color': matched[2]
                }
                visualize_debug_frame(
                    x, config, goal, ob_dwa,
                    curr_trajectory=trajectory[:, :2],
                    collision_point=collision_info['pos'] if collision_info else None,
                    debug_params=debug_params,
                    output_dir=output_dir
                )
                
            # 打印碰撞信息
            if collision_info:
                print(f"碰撞发生于: t={collision_info['time']:.1f}s，距离起点{collision_info['distance']:.2f}m")
                print(f"障碍物成本计算值: {1/collision_info['distance'] if collision_info['distance']>0 else np.inf:.2f}")
            else:
                print("未检测到碰撞，障碍物成本采用最大安全距离")

def main_enhanced_debug():
    """增强版调试主函数"""
    log_file_path = "Logs/dwa_log_details_20250306_155027/log_details.csv"
    config = Config()
    
    # 加载数据并遍历迭代
    df = pd.read_csv(log_file_path, index_col="iteration")
    iter_nums = [225, 226, 227, 228, 229, 230, 231, 232, 233, 234]  # 与原脚本一致
    
    for iter_num in iter_nums:
        # 创建迭代专用目录
        log_dir = Path(os.path.dirname(log_file_path))
        iter_dir = log_dir / f"cost_matrices_{iter_num}"
        debug_dir = iter_dir / "debug_visualizations"
        debug_dir.mkdir(parents=True, exist_ok=True)

        # 加载状态数据
        x = np.array(df.loc[iter_num, ['x_traj', 'y_traj', 'yaw_traj', 'v_traj', 'omega_traj']])
        goal = np.array(df.loc[iter_num, ['local_goal_x', 'local_goal_y']])


        # 构建DWA障碍物地图
        image = cv2.imread("EnvData/AISData_20240827/land_shapes_sf_crop.png", cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (100, 100))
        _, bin_map = cv2.threshold(image, 128, 1, cv2.THRESH_BINARY)
        bin_map[0, :] = 0; bin_map[-1, :] = 0; bin_map[:, 0] = 0; bin_map[:, -1] = 0  # 边界障碍
        ob_coords = np.column_stack(np.where(1 - bin_map == 1))[:, [1, 0]]  # 转换坐标系
        ob_coords[:, 1] = bin_map.shape[0] - ob_coords[:, 1] - 1  # 翻转Y轴

        # 添加动态障碍物
        dynamic_obstacles = np.array([
            [25.,79], [25.,80], [26.,79], [26.,80],
            [35.,55], [36.,56], [28.,46], [27.,47],
            [10.,19], [10.,20], [11.,19], [11.,20]
        ])
        ob_dwa = np.vstack([ob_coords, dynamic_obstacles])


        # 执行增强版计算（指定输出目录）
        enhanced_calculate_costs(
            x, config, goal, ob_dwa,
            vis_params={'enable_visualization': True},
            output_dir=debug_dir
        )
        
        # 创建cost_images目录（与原脚本一致）
        cost_images_dir = log_dir / "cost_images"
        cost_images_dir.mkdir(exist_ok=True)
        
        # 此处可添加组合图的生成逻辑（如果需要）
        # plt.savefig(cost_images_dir / f"cost_matrices_{iter_num}.png")


if __name__ == "__main__":
    main_enhanced_debug()

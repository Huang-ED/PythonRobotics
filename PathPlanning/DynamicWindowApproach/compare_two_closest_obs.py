import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from enum import Enum
# from types import SimpleNamespace
from dwa_paper_with_width import Config, RobotType, closest_obstacle_on_curve, closest_obstacle_on_curve_math

config = Config()

# 在 closest_obstacle_on_curve_test.py 中添加以下测试函数

def test_cases():
    # 初始配置
    config.obstacle_radius = 0.5
    
    # Test Case 1：直线路径前方有障碍物（圆形机器人）
    print("Test Case 1: Straight path with front obstacle (circle)")
    config.robot_type = RobotType.rectangle
    config.robot_length = 1.2
    config.robot_width = 0.5
    x1 = np.array([0., 0., 0., 0., 0.])  # 朝向正东
    ob1 = np.array([[4.5, 0]])
    v1 = 1.0
    omega1 = 0.0
    res1 = closest_obstacle_on_curve(x1.copy(), ob1, v1, omega1, config)
    print(f"Closest obstacle distance: {res1}")
    res1_math = closest_obstacle_on_curve_math(x1, ob1, v1, omega1, config)
    print(f"Closest obstacle distance (math): {res1_math}")
    # visualize_test_case(x1, ob1, v1, omega1, config, res1)

    # Test Case 2：圆弧左转路径内有障碍物
    print("Test Case 2: Left turn arc with obstacle")
    config.robot_type = RobotType.circle
    x2 = np.array([0., 0., 0., 0., 0.])  # 圆心在(0,2)
    ob2 = np.array([[2, 2.5]])
    v2 = 2.0
    omega2 = 1.0  # 左转，半径2
    res2 = closest_obstacle_on_curve(x2.copy(), ob2, v2, omega2, config)
    print(f"Closest obstacle distance: {res2}")
    res2_math = closest_obstacle_on_curve_math(x2, ob2, v2, omega2, config)
    print(f"Closest obstacle distance (math): {res2_math}")
    # visualize_test_case(x2, ob2, v2, omega2, config, res2)

    # Test Case 3：多障碍物检测最近
    print("Test Case 3: Multiple obstacles")
    x3 = np.array([0., 0., np.pi/4, 0., 0.])  # 东北方向
    ob3 = np.array([[3, -3], [3.5, 0], [2, 2]])
    v3 = 1.0
    omega3 = 0.0
    res3 = closest_obstacle_on_curve(x3.copy(), ob3, v3, omega3, config)
    print(f"Closest obstacle distance: {res3}")
    res3_math = closest_obstacle_on_curve_math(x3, ob3, v3, omega3, config)
    print(f"Closest obstacle distance (math): {res3_math}")
    # visualize_test_case(x3, ob3, v3, omega3, config, res3)

    # Test Case 4：矩形机器人右转检测
    print("Test Case 4: Rectangle robot right turn")
    config.robot_type = RobotType.rectangle
    config.robot_length = 1.2
    config.robot_width = 0.5
    x4 = np.array([0., 0., 0., 0., 0.])  # 朝向正东
    ob4 = np.array([[2, -1]])
    v4 = 1.0
    omega4 = -0.5  # 右转，半径2
    res4 = closest_obstacle_on_curve(x4.copy(), ob4, v4, omega4, config)
    print(f"Closest obstacle distance: {res4}")
    res4_math = closest_obstacle_on_curve_math(x4, ob4, v4, omega4, config)
    print(f"Closest obstacle distance (math): {res4_math}")
    # visualize_test_case(x4, ob4, v4, omega4, config, res4)

    # Test Case 5：无碰撞情况
    print("Test Case 5: No collision")
    config.robot_type = RobotType.circle
    x5 = np.array([0., 0., 0., 0., 0.])  # 朝向正东
    ob5 = np.array([[5, 3]])
    v5 = 1.0
    omega5 = 0.0
    res5 = closest_obstacle_on_curve(x5.copy(), ob5, v5, omega5, config)
    print(f"Closest obstacle distance: {res5}")
    res5_math = closest_obstacle_on_curve_math(x5, ob5, v5, omega5, config)
    print(f"Closest obstacle distance (math): {res5_math}")
    # visualize_test_case(x5, ob5, v5, omega5, config, res5)

if __name__ == '__main__':
    test_cases()



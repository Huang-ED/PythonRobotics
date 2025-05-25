import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from enum import Enum
# from types import SimpleNamespace
from dwa_paper_with_width import Config

config = Config()

class RobotType(Enum):
    circle = 1
    rectangle = 2

def closest_obstacle_on_curve(x, ob, v, omega, config):
    """Original function implementation"""
    start_pos = (x[0], x[1])
    heading = x[2]
    min_dist = float("inf")
    min_time = float("inf")
    
    if abs(omega) < 1e-6:
        heading_vector = np.array([math.cos(heading), math.sin(heading)])
        
        for i in range(len(ob)):
            obstacle = np.array([ob[i, 0], ob[i, 1]])
            obstacle_radius = config.obstacle_radius
            to_center = obstacle - np.array(start_pos)
            projection = np.dot(to_center, heading_vector)
            
            if projection < 0:
                continue
                
            closest_approach = np.linalg.norm(to_center - projection * heading_vector)
            
            collision_threshold = 0
            if config.robot_type == RobotType.rectangle:
                robot_diagonal = math.sqrt((config.robot_length/2)**2 + (config.robot_width/2)**2)
                collision_threshold = obstacle_radius + robot_diagonal
            else:
                collision_threshold = obstacle_radius + config.robot_radius
            
            if closest_approach > collision_threshold:
                continue
                
            dist_to_intersection = projection - math.sqrt(collision_threshold**2 - closest_approach**2)
            
            if 0 <= dist_to_intersection < min_dist:
                min_dist = dist_to_intersection
                min_time = dist_to_intersection / v if v > 0 else float("inf")
        
        return min_dist, min_time
    
    else:
        radius = abs(v / omega)
        
        if omega > 0:
            center_x = x[0] - radius * math.sin(heading)
            center_y = x[1] + radius * math.cos(heading)
        else:
            center_x = x[0] + radius * math.sin(heading)
            center_y = x[1] - radius * math.cos(heading)
        
        arc_center = np.array([center_x, center_y])
        
        if omega > 0:
            start_angle = heading - math.pi/2
        else:
            start_angle = heading + math.pi/2
        
        for i in range(len(ob)):
            obstacle_center = np.array([ob[i, 0], ob[i, 1]])
            obstacle_radius = config.obstacle_radius
            
            dist_between_centers = np.linalg.norm(arc_center - obstacle_center)
            
            if config.robot_type == RobotType.rectangle:
                robot_diagonal = math.sqrt((config.robot_length/2)**2 + (config.robot_width/2)**2)
                collision_radius = obstacle_radius + robot_diagonal
            else:
                collision_radius = obstacle_radius + config.robot_radius
            
            if dist_between_centers > radius + collision_radius or dist_between_centers < radius - collision_radius:
                continue
            
            obstacle_angle = math.atan2(obstacle_center[1] - arc_center[1], 
                                      obstacle_center[0] - arc_center[0])
            
            obstacle_angle = obstacle_angle % (2 * math.pi)
            start_angle_norm = start_angle % (2 * math.pi)
            
            check_span = (v / radius) * config.check_time
            
            in_span = False
            if omega > 0:
                end_angle = (start_angle_norm + check_span) % (2 * math.pi)
                if start_angle_norm <= end_angle:
                    in_span = start_angle_norm <= obstacle_angle <= end_angle
                else:
                    in_span = obstacle_angle >= start_angle_norm or obstacle_angle <= end_angle
            else:
                end_angle = (start_angle_norm - check_span) % (2 * math.pi)
                if start_angle_norm >= end_angle:
                    in_span = end_angle <= obstacle_angle <= start_angle_norm
                else:
                    in_span = obstacle_angle <= start_angle_norm or obstacle_angle >= end_angle
            
            if not in_span:
                continue
            
            a = radius
            b = dist_between_centers
            c = collision_radius
            
            if b == 0:
                continue
                
            d = (a*a + b*b - c*c) / (2*b)
            
            if abs(d) > radius:
                continue
                
            h_squared = a*a - d*d
            if h_squared < 0:
                continue
            h = math.sqrt(h_squared)
            
            unit_vector = (obstacle_center - arc_center) / dist_between_centers
            perp_vector = np.array([-unit_vector[1], unit_vector[0]])
            
            int_point1 = arc_center + d * unit_vector + h * perp_vector
            int_point2 = arc_center + d * unit_vector - h * perp_vector
            
            angle1 = math.atan2(int_point1[1] - arc_center[1], int_point1[0] - arc_center[0]) % (2 * math.pi)
            angle2 = math.atan2(int_point2[1] - arc_center[1], int_point2[0] - arc_center[0]) % (2 * math.pi)
            
            in_arc1 = False
            in_arc2 = False
            
            if omega > 0:
                angle_diff1 = (angle1 - start_angle_norm) % (2 * math.pi)
                angle_diff2 = (angle2 - start_angle_norm) % (2 * math.pi)
            else:
                angle_diff1 = (start_angle_norm - angle1) % (2 * math.pi)
                angle_diff2 = (start_angle_norm - angle2) % (2 * math.pi)
            
            in_arc1 = angle_diff1 <= check_span
            in_arc2 = angle_diff2 <= check_span
            
            if in_arc1:
                dist1 = angle_diff1 * radius
                time1 = dist1 / v if v > 0 else float("inf")
                if dist1 < min_dist:
                    min_dist = dist1
                    min_time = time1
            
            if in_arc2:
                dist2 = angle_diff2 * radius
                time2 = dist2 / v if v > 0 else float("inf")
                if dist2 < min_dist:
                    min_dist = dist2
                    min_time = time2
    
    return min_dist, min_time

def visualize_test_case(x, ob, v, omega, config, result):
    plt.figure()
    plt.plot(x[0], x[1], 'bo', label='Start')
    
    for o in ob:
        circle = plt.Circle((o[0], o[1]), config.obstacle_radius + config.robot_radius, 
                          color='red', alpha=0.3, label='Collision Zone')
        plt.gca().add_patch(circle)
    
    if abs(omega) < 1e-6:
        heading = np.array([np.cos(x[2]), np.sin(x[2])])
        end_point = x[:2] + heading * (v * config.check_time)
        plt.plot([x[0], end_point[0]], [x[1], end_point[1]], 'b--', label='Path')
    else:
        radius = abs(v / omega)
        if omega > 0:
            center = (x[0] - radius * np.sin(x[2]), x[1] + radius * np.cos(x[2]))
        else:
            center = (x[0] + radius * np.sin(x[2]), x[1] - radius * np.cos(x[2]))
        
        circle = plt.Circle(center, radius, color='blue', fill=False, linestyle='--', label='Path')
        plt.gca().add_patch(circle)
        
        start_angle = x[2] - np.pi/2 if omega > 0 else x[2] + np.pi/2
        end_angle = start_angle + (omega * config.check_time)
        
        arc = Arc(center, 2*radius, 2*radius, 
                angle=0, theta1=np.degrees(start_angle), 
                theta2=np.degrees(end_angle), 
                color='blue', linestyle='--')
        plt.gca().add_patch(arc)
    
    if result[0] < float('inf'):
        if abs(omega) < 1e-6:
            heading = np.array([np.cos(x[2]), np.sin(x[2])])
            closest_point = x[:2] + heading * result[0]
        else:
            radius = abs(v / omega)
            if omega > 0:
                center = (x[0] - radius * np.sin(x[2]), x[1] + radius * np.cos(x[2]))
                angle = (x[2] - np.pi/2) + (result[0] / radius)
            else:
                center = (x[0] + radius * np.sin(x[2]), x[1] - radius * np.cos(x[2]))
                angle = (x[2] + np.pi/2) - (result[0] / radius)
            closest_point = (center[0] + radius * np.cos(angle), 
                            center[1] + radius * np.sin(angle))
        
        plt.plot(closest_point[0], closest_point[1], 'rx', markersize=10, label='Closest Collision')
    
    plt.axis('equal')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.title(f"Distance: {result[0]:.1f}m, Time: {result[1]:.1f}s")
    plt.grid(True)
    plt.show()

if True:
    # Test Case 1: Straight line with front obstacle
    print("=== Test Case 1 ===")
    # config = SimpleNamespace(
    #     obstacle_radius=0.5,
    #     robot_type=RobotType.circle,
    #     robot_radius=0.5,
    #     check_time=10
    # )
    result = closest_obstacle_on_curve([0, 0, 0, 1, 0], np.array([[3, 0]]), 1, 0, config)
    print("Result:", result)
    visualize_test_case([0, 0, 0, 1, 0], np.array([[3, 0]]), 1, 0, config, result)

    # Test Case 2: Obstacle just outside collision threshold
    print("\n=== Test Case 2 ===")
    # config = SimpleNamespace(
    #     obstacle_radius=0.5,
    #     robot_type=RobotType.circle,
    #     robot_radius=0.5,
    #     check_time=10
    # )
    result = closest_obstacle_on_curve([0, 0, 0, 1, 0], np.array([[1, 1.1]]), 1, 0, config)
    print("Result:", result)
    visualize_test_case([0, 0, 0, 1, 0], np.array([[1, 1.1]]), 1, 0, config, result)

    # Test Case 3: Circular motion with intersecting obstacle
    print("\n=== Test Case 3 ===")
    # config = SimpleNamespace(
    #     obstacle_radius=0.5,
    #     robot_type=RobotType.circle,
    #     robot_radius=0.5,
    #     check_time=10
    # )
    result = closest_obstacle_on_curve([0, 0, 0, 1, 0.5], np.array([[2, 2]]), 1, 0.5, config)
    print("Result:", result)
    visualize_test_case([0, 0, 0, 1, 0.5], np.array([[2, 2]]), 1, 0.5, config, result)

    # Test Case 4: Obstacle behind robot in straight motion
    print("\n=== Test Case 4 ===")
    # config = SimpleNamespace(
    #     obstacle_radius=0.5,
    #     robot_type=RobotType.circle,
    #     robot_radius=0.5,
    #     check_time=10
    # )
    result = closest_obstacle_on_curve([0, 0, 0, 1, 0], np.array([[-2, 0]]), 1, 0, config)
    print("Result:", result)
    visualize_test_case([0, 0, 0, 1, 0], np.array([[-2, 0]]), 1, 0, config, result)

    # Test Case 5: Multiple obstacles in straight motion
    print("\n=== Test Case 5 ===")
    # config = SimpleNamespace(
    #     obstacle_radius=0.2,
    #     robot_type=RobotType.circle,
    #     robot_radius=0.8,
    #     check_time=10
    # )
    result = closest_obstacle_on_curve([0, 0, np.pi/2, 1, 0], np.array([[0, 2], [0, 1.5]]), 1, 0, config)
    print("Result:", result)
    visualize_test_case([0, 0, np.pi/2, 1, 0], np.array([[0, 2], [0, 1.5]]), 1, 0, config, result)

    # Test Case 6: Circular motion beyond check time span
    print("\n=== Test Case 6 ===")
    # config = SimpleNamespace(
    #     obstacle_radius=0.5,
    #     robot_type=RobotType.circle,
    #     robot_radius=0.5,
    #     check_time=1
    # )
    result = closest_obstacle_on_curve([0, 0, 0, 1, 0.5], np.array([[2, 2]]), 1, 0.5, config)
    print("Result:", result)
    visualize_test_case([0, 0, 0, 1, 0.5], np.array([[2, 2]]), 1, 0.5, config, result)

    # Test Case 7
    print("\n=== Test Case 7 ===")
    # config = SimpleNamespace(
    #     obstacle_radius=0.5,
    #     robot_type=RobotType.circle,
    #     robot_radius=0.5,
    #     check_time=5
    # )
    robot_x = [0, 0, 0]  # 初始朝向东
    obstacles = np.array([
        [1, -1],   # 障碍物①：（1, -1），位于圆弧起点附近
        [-2, -2]   # 障碍物②：（-2, -2），圆心距足够但角度范围外
    ])
    v, omega = 1, 1  # 半径 = 1, 顺时针圆弧
    result = closest_obstacle_on_curve(robot_x, obstacles, v, omega, config)
    print("Result:", result)  # 预期：约1.57 m, 时间1.57 s
    visualize_test_case(robot_x, obstacles, v, omega, config, result)

    # Test Case 8
    print("\n=== Test Case 8 ===")
    # config = SimpleNamespace(
    #     obstacle_radius=0.5,
    #     robot_type=RobotType.circle,
    #     robot_radius=0.5,
    #     check_time=2  # 限制扫描范围
    # )
    robot_x = [0, 0, np.pi/2]  # 初始朝向北
    obstacles = np.array([
        [4, 0]  # 圆心在机器人右侧（逆时针圆心），路径上但需扫描多圈才到达
    ])
    v, omega = 2, -1  # 半径=2, 逆时针
    result = closest_obstacle_on_curve(robot_x, obstacles, v, omega, config)
    print("Result:", result)  # 预期：(inf, inf)
    visualize_test_case(robot_x, obstacles, v, omega, config, result)

    # Test Case 9
    print("\n=== Test Case 9 ===")
    # config = SimpleNamespace(
    #     obstacle_radius=0.2,
    #     robot_type=RobotType.rectangle,
    #     robot_length=1.0,  # 对角线长度 ≈ √(0.5²+0.3²)=0.583
    #     robot_width=0.6,
    #     check_time=4
    # )
    robot_x = [0, 0, np.pi]  # 初始朝西
    obstacles = np.array([
        [0.3, -1.5]  # 位于圆弧路径的切线外缘，未达碰撞阈值
    ])
    v, omega = 1, 0.5  # 半径=2, 顺时针
    result = closest_obstacle_on_curve(robot_x, obstacles, v, omega, config)
    print("Result:", result)  # 预期：(inf, inf)
    visualize_test_case(robot_x, obstacles, v, omega, config, result)

    # Test Case 10
    print("\n=== Test Case 10 ===")
    # config = SimpleNamespace(
    #     obstacle_radius=1.0,
    #     robot_type=RobotType.circle,
    #     robot_radius=0.5,
    #     check_time=10
    # )
    robot_x = [0, 0, 0]  # 朝东
    obstacles = np.array([
        [3, 3]  # 障碍物导致两个交点：一个较近，一个较远
    ])
    v, omega = 2, 0.2  # 半径=10, 顺时针
    result = closest_obstacle_on_curve(robot_x, obstacles, v, omega, config)
    print("Result:", result)  # 应返回较近的碰撞距离
    visualize_test_case(robot_x, obstacles, v, omega, config, result)

    # Test Case 11
    print("\n=== Test Case 11 ===")
    # config = SimpleNamespace(
    #     obstacle_radius=0.3,
    #     robot_type=RobotType.circle,
    #     robot_radius=0.5,
    #     check_time=8
    # )
    robot_x = [0, 0, 0]  # 朝东
    obstacles = np.array([
        [2, 0],      # ① 正前方，碰撞距离≈2 - √(0.8²) =1.2 m 
        [1.5, 0.9],  # ② 横向距离=0.9 > 0.8 (0.3+0.5)
        [-1, 0]      # ③ 后方，不检测
    ])
    v, omega = 1, 0
    result = closest_obstacle_on_curve(robot_x, obstacles, v, omega, config)
    print("Result:", result)  # 预期：(1.2, 1.2)
    visualize_test_case(robot_x, obstacles, v, omega, config, result)

if False:
    def run_test_case(case_num, x, ob, v, omega, config):
        print(f"\n=== Test Case {case_num} ===")
        result = closest_obstacle_on_curve(x, ob, v, omega, config)
        print("Result:", result)
        visualize_test_case(x, ob, v, omega, config, result)
        return result

    if __name__ == "__main__":
        config = Config()
        
        # Test Case 1: Forward collision
        run_test_case(1, 
                    [0, 0, 0],  # [x, y, heading]
                    np.array([[3, 0]]),
                    1.0, 0.0, config)
        
        # Test Case 2: Side obstacle (no collision)
        run_test_case(2,
                    [0, 0, 0],
                    np.array([[2, 1.1]]),
                    1.0, 0.0, config)
        
        # Test Case 3: Arc collision (CCW turn)
        run_test_case(3,
                    [0, 0, math.pi/2],  # Facing north
                    np.array([[-1.0, 3.0]]),
                    1.0, -0.5, config)
        
        # Test Case 4: Behind obstacle (ignored)
        run_test_case(4,
                    [0, 0, 0],
                    np.array([[-2.0, 0]]),
                    1.0, 0.0, config)
        
        # Test Case 5: Multi-obstacle priority
        run_test_case(5,
                    [0, 0, 0],
                    np.array([[1.5, 0], [2.5, -0.3], [3, 0.8]]),
                    1.0, 0.0, config)

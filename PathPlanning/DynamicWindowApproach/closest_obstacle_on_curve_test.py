import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from enum import Enum
# from types import SimpleNamespace
from dwa_paper_with_width import Config, RobotType

config = Config()

# class RobotType(Enum):
#     circle = 1
#     rectangle = 2

def closest_obstacle_on_curve_with_check_time(x, ob, v, omega, config):
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
            if check_span > 2 * math.pi:
                in_span = True
            else:
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

            # if not in_span:
            #     continue
            
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


def closest_obstacle_on_curve(x, ob, v, omega, config):
    """
    Calculate the distance to the closest obstacle that intersects with the curvature
    without time/span limitations - checks the entire trajectory
    
    Parameters:
    x: current state
        [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    ob: obstacle positions
        [[x(m), y(m)], ...]
    v: translational velocity (m/s)
    omega: angular velocity (rad/s)
    config: simulation configuration
    
    Returns:
    dist: distance to the closest obstacle
    t: time to reach the closest obstacle
    """
    start_pos = (x[0], x[1])
    heading = x[2]
    min_dist = float("inf")
    min_time = float("inf")

    if abs(omega) < 1e-6:
        # Straight line motion - check entire line
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
        # Circular arc motion - check entire circle
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
                
            # Calculate intersection points without span checking
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
            
            # Calculate angle differences without span limitation
            if omega > 0:
                angle_diff1 = (angle1 - start_angle) % (2 * math.pi)
                angle_diff2 = (angle2 - start_angle) % (2 * math.pi)
            else:
                angle_diff1 = (start_angle - angle1) % (2 * math.pi)
                angle_diff2 = (start_angle - angle2) % (2 * math.pi)
            
            # Check both intersection points (no span restriction)
            for angle_diff in [angle_diff1, angle_diff2]:
                dist = angle_diff * radius
                time = dist / v if v > 0 else float("inf")
                
                if dist < min_dist:
                    min_dist = dist
                    min_time = time
                    
        return min_dist, min_time


def visualize_test_case(x, ob, v, omega, config, result):
    plt.figure()
    plt.plot(x[0], x[1], 'bo', label='Start')
    
    for o in ob:
        ## Version 1
        if config.robot_type == RobotType.rectangle:
            diagonal = math.sqrt((config.robot_length/2)**2 + (config.robot_width/2)**2)
        else:
            diagonal = config.robot_radius
        circle = plt.Circle((o[0], o[1]), config.obstacle_radius + diagonal, # + config.robot_radius, 
                          color='red', alpha=0.3, label='Collision Zone')
        plt.gca().add_patch(circle)
        # ## Version 2
        # circle = plt.Circle((o[0], o[1]), config.obstacle_radius，# + config.robot_radius, 
        #                   color='red', alpha=0.3, label='Collision Zone')
        # plt.gca().add_patch(circle)

    
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
        # print(closest_point)
        
        plt.plot(closest_point[0], closest_point[1], 'rx', markersize=10, label='Closest Collision')
    
    plt.axis('equal')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.title(f"Distance: {result[0]:.1f}m, Time: {result[1]:.1f}s")
    plt.grid(True)
    plt.show()


# def test_cases():
#     # 初始配置
#     config.obstacle_radius = 0.5
    
#     # Test Case 1：直线路径前方有障碍物（圆形机器人）
#     print("Test Case 1: Straight path with front obstacle (circle)")
#     config.robot_type = RobotType.rectangle
#     config.robot_length = 1.2
#     config.robot_width = 0.5
#     x1 = np.array([0, 0, 0])  # 朝向正东
#     ob1 = np.array([[4.5, 0]])
#     v1 = 1.0
#     omega1 = 0.0
#     res1 = closest_obstacle_on_curve(x1, ob1, v1, omega1, config)
#     visualize_test_case(x1, ob1, v1, omega1, config, res1)

#     # Test Case 2：圆弧左转路径内有障碍物
#     print("Test Case 2: Left turn arc with obstacle")
#     config.robot_type = RobotType.circle
#     x2 = np.array([0, 0, 0])  # 圆心在(0,2)
#     ob2 = np.array([[2, 2.5]])
#     v2 = 2.0
#     omega2 = 1.0  # 左转，半径2
#     res2 = closest_obstacle_on_curve(x2, ob2, v2, omega2, config)
#     visualize_test_case(x2, ob2, v2, omega2, config, res2)

#     # Test Case 3：多障碍物检测最近
#     print("Test Case 3: Multiple obstacles")
#     x3 = np.array([0, 0, np.pi/4])  # 东北方向
#     ob3 = np.array([[3, -3], [3.5, 0], [2, 2]])
#     v3 = 1.0
#     omega3 = 0.0
#     res3 = closest_obstacle_on_curve(x3, ob3, v3, omega3, config)
#     visualize_test_case(x3, ob3, v3, omega3, config, res3)

#     # Test Case 4：矩形机器人右转检测
#     print("Test Case 4: Rectangle robot right turn")
#     config.robot_type = RobotType.rectangle
#     config.robot_length = 1.2
#     config.robot_width = 0.5
#     x4 = np.array([0, 0, 0])
#     ob4 = np.array([[2, -1]])
#     v4 = 1.0
#     omega4 = -0.5  # 右转，半径2
#     res4 = closest_obstacle_on_curve(x4, ob4, v4, omega4, config)
#     visualize_test_case(x4, ob4, v4, omega4, config, res4)

#     # Test Case 5：无碰撞情况
#     print("Test Case 5: No collision")
#     config.robot_type = RobotType.circle
#     x5 = np.array([0, 0, 0])
#     ob5 = np.array([[5, 3]])
#     v5 = 1.0
#     omega5 = 0.0
#     res5 = closest_obstacle_on_curve(x5, ob5, v5, omega5, config)
#     visualize_test_case(x5, ob5, v5, omega5, config, res5)


def test_case_from_exp():
    config.obstacle_radius = 0.5
    config.robot_type = RobotType.rectangle
    config.robot_length = 1.2
    config.robot_width = 0.5
    # x = np.array([26.78515083, 80.50046229, -1.27025063, 0.49000000, -0.25307274])
    x = np.array([36.83082373,56.79979085,-2.16787346,0.01000000,-0.00174533])
    # ob = np.array([[25.0, 79.0], [25.0, 80.0], [26.0, 79.0], [26.0, 80.0]])
    ob = np.array([[35., 55.], [36., 56]])
    v, omega = (0.01000000,-0.00872665)
    
    # from dwa_paper_with_width import motion
    # x = motion(x, [v, omega], config.dt)
    # # print(x)
    # v, omega = (0.01000000,-0.00174533)

    
    res = closest_obstacle_on_curve(x, ob, v, omega, config)
    visualize_test_case(x, ob, v, omega, config, res)
    print(f"Result: Distance = {res[0]:.5f} m, Time = {res[1]:.2f} s")
    print(f"v: {v:.5f}, a: {config.max_accel:.5f}, dt: {config.dt:.5f}\n")

    print(f"v^2: {v**2:.5f}, 2 * a * d: {2 * config.max_accel * res[0]:.5f}")
    print(f"v^2 - 2 * a * d: {v**2 - 2 * config.max_accel * res[0]:.5f}")
    print(f"v^2 + a * v * dt: {v**2 + config.max_accel * v * config.dt:.5f}")
    print(f"v^2 + a * v * dt - 2 * a * d: {v**2 + config.max_accel * v * config.dt - 2 * config.max_accel * res[0]:.8f}")
    # print(f"(v-a*dt)^2 - 2 * a * (d-v*dt): {(v - config.max_accel * config.dt)**2 - 2 * config.max_accel * (res[0] - v * config.dt):.5f}")


if __name__ == '__main__':
    # test_cases()
    test_case_from_exp()

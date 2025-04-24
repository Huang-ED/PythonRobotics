import numpy as np
import math

def check_arc_obstacle_intersection(start_pos, initial_heading, curvature, obstacles, arc_length=100):
    """
    Check if a circular arc intersects with any circular obstacles.
    
    Parameters:
    start_pos (tuple): Starting position (x, y)
    initial_heading (float): Initial heading direction in radians
    curvature (float): Curvature of the arc (1/radius). Positive values mean turning left.
    obstacles (list): List of obstacles represented as (center_x, center_y, radius)
    arc_length (float): Maximum arc length to check
    
    Returns:
    tuple: (bool, float) - Whether an intersection was found and at what distance
    """
    # Extract start position
    x, y = start_pos
    
    # If curvature is close to zero, the path is approximately a straight line
    if abs(curvature) < 1e-10:
        # Handle straight line case
        return check_straight_line_intersection(start_pos, initial_heading, obstacles, arc_length)
    
    # Calculate the center of the arc's circle
    # For a given curvature, the radius is 1/curvature
    radius = 1.0 / abs(curvature)
    # The center is perpendicular to the heading direction
    if curvature > 0:  # turning left
        center_x = x - radius * math.sin(initial_heading)
        center_y = y + radius * math.cos(initial_heading)
    else:  # turning right
        center_x = x + radius * math.sin(initial_heading)
        center_y = y - radius * math.cos(initial_heading)
    
    arc_center = (center_x, center_y)
    
    # Calculate the start angle of the arc
    if curvature > 0:  # turning left
        start_angle = initial_heading - math.pi/2
    else:  # turning right
        start_angle = initial_heading + math.pi/2
    
    # The angular span of the arc is (arc_length / radius)
    angle_span = arc_length * abs(curvature)
    
    # Check intersection with each obstacle
    min_distance = float('inf')
    intersection_found = False
    
    for obstacle in obstacles:
        obs_center_x, obs_center_y, obs_radius = obstacle
        
        # Calculate distance between centers
        dist_between_centers = math.sqrt((center_x - obs_center_x)**2 + (center_y - obs_center_y)**2)
        
        # Quick check: if the distance between centers is greater than the sum of radii plus
        # the obstacle radius, there's no intersection
        if dist_between_centers > radius + obs_radius:
            # Check if the obstacle intersects with the arc segment (not the full circle)
            obs_vector = (obs_center_x - center_x, obs_center_y - center_y)
            obs_angle = math.atan2(obs_vector[1], obs_vector[0])
            
            # Normalize angles to make comparison easier
            while obs_angle < start_angle:
                obs_angle += 2 * math.pi
                
            # Check if obstacle is within the angular span of the arc
            if curvature > 0:  # counterclockwise
                end_angle = start_angle + angle_span
                if obs_angle > end_angle:
                    continue
            else:  # clockwise
                end_angle = start_angle - angle_span
                if obs_angle < end_angle:
                    continue
        
        # Calculate the intersection points between the arc circle and obstacle circle
        if abs(dist_between_centers - (radius - obs_radius)) < 1e-10:
            # Circles touch internally at one point
            # Calculate the point of tangency
            theta = math.atan2(obs_center_y - center_y, obs_center_x - center_x)
            intersection_x = center_x + radius * math.cos(theta)
            intersection_y = center_y + radius * math.sin(theta)
            
            # Calculate the angle of this point on the arc
            intersect_angle = math.atan2(intersection_y - center_y, intersection_x - center_x)
            
            # Check if the intersection point is within the arc's angular span
            in_arc_span = is_angle_in_span(intersect_angle, start_angle, angle_span, curvature > 0)
            
            if in_arc_span:
                # Calculate the distance along the arc
                angle_diff = angle_difference(start_angle, intersect_angle, curvature > 0)
                distance = angle_diff * radius
                
                if distance < min_distance:
                    min_distance = distance
                    intersection_found = True
                    
        elif dist_between_centers < radius + obs_radius:
            # Circles intersect at two points
            # Use the Law of Cosines to find the distance from the center to the line connecting the intersection points
            a = radius
            b = dist_between_centers
            c = obs_radius
            
            if b == 0:  # Concentric circles
                continue
                
            # Distance from center of arc circle to the line connecting intersection points
            d = (a*a + b*b - c*c) / (2*b)
            
            # Height of the triangle formed by the centers and an intersection point
            h = math.sqrt(a*a - d*d)
            
            # Calculate the intersection points
            center_to_obs_unit = ((obs_center_x - center_x) / dist_between_centers, 
                                 (obs_center_y - center_y) / dist_between_centers)
            
            # Calculate perpendicular unit vector
            perp_unit = (-center_to_obs_unit[1], center_to_obs_unit[0])
            
            # First intersection point
            int_point1_x = center_x + d * center_to_obs_unit[0] + h * perp_unit[0]
            int_point1_y = center_y + d * center_to_obs_unit[1] + h * perp_unit[1]
            
            # Second intersection point
            int_point2_x = center_x + d * center_to_obs_unit[0] - h * perp_unit[0]
            int_point2_y = center_y + d * center_to_obs_unit[1] - h * perp_unit[1]
            
            # Calculate angles of these points on the arc
            angle1 = math.atan2(int_point1_y - center_y, int_point1_x - center_x)
            angle2 = math.atan2(int_point2_y - center_y, int_point2_x - center_x)
            
            # Check if these points are within the arc's angular span
            in_arc1 = is_angle_in_span(angle1, start_angle, angle_span, curvature > 0)
            in_arc2 = is_angle_in_span(angle2, start_angle, angle_span, curvature > 0)
            
            # If either point is in the arc, calculate the distance to the first intersection
            if in_arc1 or in_arc2:
                if in_arc1:
                    angle_diff1 = angle_difference(start_angle, angle1, curvature > 0)
                    distance1 = angle_diff1 * radius
                    if distance1 < min_distance:
                        min_distance = distance1
                        intersection_found = True
                        
                if in_arc2:
                    angle_diff2 = angle_difference(start_angle, angle2, curvature > 0)
                    distance2 = angle_diff2 * radius
                    if distance2 < min_distance:
                        min_distance = distance2
                        intersection_found = True
    
    return intersection_found, min_distance if intersection_found else None

def check_straight_line_intersection(start_pos, heading, obstacles, max_distance):
    """
    Check if a straight line intersects with any circular obstacles.
    
    Parameters:
    start_pos (tuple): Starting position (x, y)
    heading (float): Heading direction in radians
    obstacles (list): List of obstacles represented as (center_x, center_y, radius)
    max_distance (float): Maximum distance to check
    
    Returns:
    tuple: (bool, float) - Whether an intersection was found and at what distance
    """
    x, y = start_pos
    heading_vector = (math.cos(heading), math.sin(heading))
    
    min_distance = float('inf')
    intersection_found = False
    
    for obstacle in obstacles:
        obs_center_x, obs_center_y, obs_radius = obstacle
        
        # Vector from start to obstacle center
        to_center = (obs_center_x - x, obs_center_y - y)
        
        # Calculate the projection of to_center onto the heading vector
        projection = to_center[0] * heading_vector[0] + to_center[1] * heading_vector[1]
        
        # If projection is negative, the obstacle is behind the start point
        if projection < 0:
            continue
            
        # Calculate the closest distance from the line to the obstacle center
        closest_approach = math.sqrt(
            (to_center[0] - projection * heading_vector[0])**2 + 
            (to_center[1] - projection * heading_vector[1])**2
        )
        
        # If the closest approach is greater than the obstacle radius, no intersection
        if closest_approach > obs_radius:
            continue
            
        # Calculate the distance to the intersection point
        # Using Pythagorean theorem
        dist_to_intersection = projection - math.sqrt(obs_radius**2 - closest_approach**2)
        
        # Check if the intersection is within the maximum distance
        if 0 <= dist_to_intersection <= max_distance and dist_to_intersection < min_distance:
            min_distance = dist_to_intersection
            intersection_found = True
    
    return intersection_found, min_distance if intersection_found else None

def is_angle_in_span(angle, start_angle, span, counterclockwise=True):
    """
    Check if an angle is within a given angular span.
    
    Parameters:
    angle (float): Angle to check
    start_angle (float): Starting angle of the span
    span (float): Angular span
    counterclockwise (bool): True if the span goes counterclockwise, False otherwise
    
    Returns:
    bool: True if the angle is within the span, False otherwise
    """
    # Normalize angles to [0, 2π)
    angle = (angle % (2 * math.pi))
    start_angle = (start_angle % (2 * math.pi))
    
    if counterclockwise:
        end_angle = (start_angle + span) % (2 * math.pi)
        if start_angle <= end_angle:
            return start_angle <= angle <= end_angle
        else:  # The span crosses the 0 angle
            return angle >= start_angle or angle <= end_angle
    else:
        end_angle = (start_angle - span) % (2 * math.pi)
        if start_angle >= end_angle:
            return end_angle <= angle <= start_angle
        else:  # The span crosses the 0 angle
            return angle <= start_angle or angle >= end_angle

def angle_difference(start_angle, end_angle, counterclockwise=True):
    """
    Calculate the angular difference between two angles in a specific direction.
    
    Parameters:
    start_angle (float): Starting angle
    end_angle (float): Ending angle
    counterclockwise (bool): True if measuring counterclockwise, False for clockwise
    
    Returns:
    float: Angular difference in radians
    """
    # Normalize angles to [0, 2π)
    start_angle = start_angle % (2 * math.pi)
    end_angle = end_angle % (2 * math.pi)
    
    if counterclockwise:
        if end_angle >= start_angle:
            return end_angle - start_angle
        else:
            return 2 * math.pi - (start_angle - end_angle)
    else:
        if start_angle >= end_angle:
            return start_angle - end_angle
        else:
            return 2 * math.pi - (end_angle - start_angle)



def closest_obstacle_on_curve(x, ob, v, omega, config):
    """
    Calculate the distance to the closest obstacle that intersects with the curvature
    using geometric circle-to-arc intersection checks.
    """
    start_pos = (x[0], x[1])
    initial_heading = x[2]
    
    # Calculate curvature (handle near-zero velocity)
    curvature = omega / v if abs(v) > 1e-10 else 0.0
    
    # Prepare obstacles with safety margins based on robot shape
    adjusted_obstacles = []
    for (ox, oy) in ob:
        # Calculate safety margin for robot size
        if config.robot_type == RobotType.circle:
            safety_margin = config.robot_radius
        else:  # rectangle
            half_diagonal = math.hypot(config.robot_length/2, config.robot_width/2)
            safety_margin = half_diagonal
        
        adjusted_radius = config.obstacle_radius + safety_margin
        adjusted_obstacles.append((ox, oy, adjusted_radius))
    
    # Check intersections along the arc
    intersection_found, min_distance = check_arc_obstacle_intersection(
        start_pos=start_pos,
        initial_heading=initial_heading,
        curvature=curvature,
        obstacles=adjusted_obstacles,
        arc_length=abs(v) * config.check_time  # Maximum distance to check
    )
    
    if intersection_found:
        t_collision = min_distance / v if abs(v) > 1e-10 else 0.0
        return min_distance, t_collision
    else:
        return float('inf'), float('inf')



import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc
import numpy as np
import math
from dwa_paper_with_width import Config, RobotType, plot_robot

def visualize_test_case(start_pos, initial_heading, curvature, obstacles, arc_length, result, ax):
    """Visualize a single test case with matplotlib"""
    ax.set_aspect('equal')
    ax.grid(True)
    
    # Draw start position and heading
    ax.plot(start_pos[0], start_pos[1], 'bo', markersize=10)
    ax.arrow(start_pos[0], start_pos[1], 
             math.cos(initial_heading), math.sin(initial_heading),
             head_width=0.5, head_length=0.7, fc='b', ec='b')
    
    # Calculate arc parameters
    if abs(curvature) > 1e-10:
        radius = 1 / abs(curvature)
        if curvature > 0:  # Left turn
            center = (start_pos[0] - radius * math.sin(initial_heading),
                      start_pos[1] + radius * math.cos(initial_heading))
        else:  # Right turn
            center = (start_pos[0] + radius * math.sin(initial_heading),
                      start_pos[1] - radius * math.cos(initial_heading))
        
        # Draw arc circle
        ax.plot(center[0], center[1], 'rx' if result[0] else 'gx')
        ax.add_patch(Circle(center, radius, fill=False, linestyle='--', 
                          color='red' if result[0] else 'green'))
        
        # Calculate arc angles
        start_angle = math.degrees(initial_heading - np.pi/2) if curvature > 0 else \
                      math.degrees(initial_heading + np.pi/2)
        angle_span = math.degrees(arc_length * curvature)
        
        # Draw actual arc
        ax.add_patch(Arc(center, 2*radius, 2*radius,
                       theta1=start_angle, theta2=start_angle + angle_span,
                       color='red' if result[0] else 'green', linewidth=2))
    else:  # Straight line
        end_pos = (start_pos[0] + arc_length * math.cos(initial_heading),
                   start_pos[1] + arc_length * math.sin(initial_heading))
        ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                color='red' if result[0] else 'green', linewidth=2)
    
    # Draw obstacles
    for obs in obstacles:
        color = 'red' if result[0] else 'green'
        if result[0] and obs == obstacles[0]:  # First obstacle
            color = 'purple'
        ax.add_patch(Circle((obs[0], obs[1]), obs[2], color=color, alpha=0.3))
        ax.plot(obs[0], obs[1], 'kx')
    
    ax.set_title(f"Intersection: {result[0]}\nDistance: {result[1]:.2f}")

def test_arc_intersection():
    # Create test cases with various scenarios
    test_cases = [
        {   # Test 1: Left turn with intersection
            'start_pos': (0, 0),
            'initial_heading': 0,
            'curvature': 1/5,  # Radius 5
            'obstacles': [(4, 3, 1)],
            'arc_length': 10
        },
        {   # Test 2: Right turn with no intersection
            'start_pos': (0, 0),
            'initial_heading': np.pi/2,
            'curvature': -1/5,  # Radius 5
            'obstacles': [(5, 5, 1)],
            'arc_length': 10
        },
        {   # Test 3: Straight line collision
            'start_pos': (0, 0),
            'initial_heading': np.pi/4,
            'curvature': 0,
            'obstacles': [(5, 5, 1)],
            'arc_length': 10
        },
        {   # Test 4: Multiple obstacles
            'start_pos': (0, 0),
            'initial_heading': np.pi/2,
            'curvature': 1/8,
            'obstacles': [(3, 5, 1), (4, 2, 0.5), (-2, 3, 1)],
            'arc_length': 15
        },
        {   # Test 5: No obstacles
            'start_pos': (0, 0),
            'initial_heading': 0,
            'curvature': 1/6,
            'obstacles': [],
            'arc_length': 10
        }
    ]

    fig, axs = plt.subplots(1, len(test_cases), figsize=(25, 5))
    
    for i, test_case in enumerate(test_cases):
        # Run the collision check
        print(f"Running test case {i+1}...")
        print(f"start_pos: {test_case['start_pos']}, "
              f"initial_heading: {test_case['initial_heading']}, "
              f"curvature: {test_case['curvature']}, "
              f"obstacles: {test_case['obstacles']}, "
              f"arc_length: {test_case['arc_length']}")
        print("Checking for intersection...")
        result = check_arc_obstacle_intersection(
            start_pos=test_case['start_pos'],
            initial_heading=test_case['initial_heading'],
            curvature=test_case['curvature'],
            obstacles=test_case['obstacles'],
            arc_length=test_case['arc_length']
        )
        
        # Visualize the test case
        visualize_test_case(
            test_case['start_pos'],
            test_case['initial_heading'],
            test_case['curvature'],
            test_case['obstacles'],
            test_case['arc_length'],
            result,
            axs[i] if len(test_cases) > 1 else axs
        )

    plt.tight_layout()
    plt.show()

def test_closest_obstacle():
    # Test robot collision detection with different configurations
    config = Config()
    config.robot_type = RobotType.rectangle
    config.robot_length = 1.2
    config.robot_width = 0.5
    
    test_cases = [
        {   # Circle collision check
            'x': np.array([0, 0, 0, 1, 0]),
            'ob': np.array([[3, 0]]),
            'v': 1,
            'omega': 0,
            'expected': (3 - 0.5 - math.hypot(1.2/2, 0.5/2), 3 - 0.5 - math.hypot(1.2/2, 0.5/2))
        },
        {   # Arc collision check
            'x': np.array([0, 0, np.pi/2, 1, 0.5]),
            'ob': np.array([[2, 3]]),
            'v': 1,
            'omega': 0.5,
            'expected': (True, True)
        }
    ]

    fig, axs = plt.subplots(1, len(test_cases), figsize=(12, 6))
    
    for i, test_case in enumerate(test_cases):
        # Run collision check
        dist, t = closest_obstacle_on_curve(
            test_case['x'],
            test_case['ob'],
            test_case['v'],
            test_case['omega'],
            config
        )
        
        # Visualization
        ax = axs[i] if len(test_cases) > 1 else axs
        ax.set_aspect('equal')
        ax.grid(True)
        
        # Draw robot
        plot_robot(test_case['x'][0], test_case['x'][1], 
                  test_case['x'][2], config)
        
        # Draw obstacles
        for obs in test_case['ob']:
            ax.add_patch(Circle(obs, config.obstacle_radius, color='red', alpha=0.3))
            ax.plot(obs[0], obs[1], 'kx')
        
        ax.set_title(f"Dist: {dist:.2f}, Time: {t:.2f}")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    print("Running arc intersection tests...")
    test_arc_intersection()
    
    print("\nRunning robot collision detection tests...")
    test_closest_obstacle()


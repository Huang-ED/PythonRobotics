import numpy as np

def circles_overlap_with_box(circles, center, length, width, rot):
    """
    Check whether any of the given circles overlap with a rotated rectangular box.
    
    Parameters:
    - circles: 2D numpy array, shape (N, 3), where each row contains the 2D coordinate of the point 
               and the radius of the circle (x, y, radius)
    - center: tuple (cx, cy), the 2D coordinate of the center of the box
    - length: float, length of the box
    - width: float, width of the box
    - rot: float, rotational angle of the box in radians
    
    Returns:
    - Boolean: True if any circle overlaps with the box, False otherwise
    """
    cx, cy = center
    half_length = length / 2
    half_width = width / 2
    
    # Check each circle
    for circle in circles:
        x, y, radius = circle
        
        # Step 1: Translate circle center relative to rectangle center
        dx = x - cx
        dy = y - cy
        
        # Step 2: Rotate the point to align with rectangle's axes
        cos_rot = np.cos(-rot)  # Negative rotation to align with rectangle
        sin_rot = np.sin(-rot)
        
        # Apply rotation matrix
        x_rot = dx * cos_rot - dy * sin_rot
        y_rot = dx * sin_rot + dy * cos_rot
        
        # Step 3: Find the closest point on the axis-aligned rectangle to the circle center
        closest_x = np.clip(x_rot, -half_length, half_length)
        closest_y = np.clip(y_rot, -half_width, half_width)
        
        # Step 4: Calculate distance from the circle center to the closest point
        distance = np.sqrt((x_rot - closest_x)**2 + (y_rot - closest_y)**2)
        
        # Check if the circle overlaps with the rectangle
        if distance <= radius:
            return True
    
    return False

# Example usage:
def test_example():
    circles = np.array([
        [26, 80, 0.5],
    ])
    center = (26.21826, 81.12303)
    length = 1.2
    width = 0.5
    rot = -0.90129
    
    result = circles_overlap_with_box(circles, center, length, width, rot)
    print(f"Do the circle and rectangle overlap? {result}")
    return result

if __name__ == "__main__":
    test_example()
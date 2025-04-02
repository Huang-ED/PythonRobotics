import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

class Config:
    """
    Simulation parameter class
    """
    def __init__(self):
        # robot parameter
        self.robot_radius = 0.5  # [m] for collision check

def get_ship_vertices(pos_x, pos_y, yaw, ship_length, ship_width):
    """
    Generate ship bounding box vertices based on position, orientation and dimensions
    Args:
        pos_x (float): Ship center x-coordinate
        pos_y (float): Ship center y-coordinate
        yaw (float): Yaw angle in radians (counter-clockwise positive)
        ship_length (float): Ship length (along unrotated x-axis)
        ship_width (float): Ship width (along unrotated y-axis)
    Returns:
        np.ndarray: 4x2 array of vertex coordinates
    """
    half_len = ship_length / 2
    half_wid = ship_width / 2
    # Unrotated corner points (relative to center)
    points = np.array([
        [half_len, half_wid],
        [half_len, -half_wid],
        [-half_len, -half_wid],
        [-half_len, half_wid]
    ])
    # Rotation matrix (counter-clockwise)
    rot_mat = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw), np.cos(yaw)]
    ])
    # Apply rotation and translate to center position
    rotated_points = np.dot(points, rot_mat.T)
    rotated_points[:, 0] += pos_x
    rotated_points[:, 1] += pos_y
    return rotated_points

def main():
    config = Config()

    # ----- Set up the map -----
    ## Load the map from image
    image_path = "EnvData/AISData_20240827/land_shapes_sf_crop.png"
    arr = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    arr = cv2.resize(arr, (100, 100))
    _, arr = cv2.threshold(arr, 128, 1, cv2.THRESH_BINARY)

    ## add boundary obstacles
    arr[0, :] = 0  # Top edge
    arr[-1, :] = 0  # Bottom edge
    arr[:, 0] = 0  # Left edge
    arr[:, -1] = 0  # Right edge
    ob = np.argwhere(arr == 0)

    ## imread direction and plot direction are different
    ob[:, [0, 1]] = ob[:, [1, 0]]  # Swap columns to match (x, y)
    ob[:, 1] = arr.shape[0] - ob[:, 1] - 1  # Flip y-axis
    ox, oy = ob[:, 0], ob[:, 1]

    # Map for DWA
    arr = 1 - arr
    eroded_arr = cv2.erode(arr, kernel=np.ones((3, 3), np.uint8), iterations=1)
    arr_dwa = cv2.subtract(arr, eroded_arr)
    arr_dwa = 1 - arr_dwa

    ob_dwa = np.argwhere(arr_dwa == 0)
    ob_dwa[:, [0, 1]] = ob_dwa[:, [1, 0]]  # Swap columns to match (x, y)
    ob_dwa[:, 1] = arr_dwa.shape[0] - ob_dwa[:, 1] - 1  # Flip y-axis
    new_ob = np.array([
        [25., 79.], [25., 80.], [26., 79.], [26., 80.],
        [35., 55.], [36., 56],
        [28., 46.], [27., 47.],
        [10., 19.], [10., 20.], [11., 19.], [11., 20.]
    ])
    ob_dwa = np.append(ob_dwa, new_ob, axis=0)

    # ----- Set up the start and goal positions -----
    sx, sy = 20, 90
    gx, gy = 3, 3

    # ----- Define ships to plot -----
    ships = [
        # Format: (pos_x, pos_y, yaw, length, width, color)
        (26.00318, 81.41164, -0.96587, 1.2, 0.5, 'blue'),
        (26.10839, 81.26560, -0.93358, 1.2, 0.5, 'blue'),
        (26.21826, 81.12303, -0.90129, 1.2, 0.5, 'red'),
    ]

    # ----- Plot the map with ships -----
    plt.figure(figsize=(10, 10))
    
    # Plot obstacles (original code)
    for (x, y) in ob:
        circle = plt.Circle((x, y), config.robot_radius, color="darkgrey")
        plt.gca().add_patch(circle)
    for (x, y) in ob_dwa:
        circle = plt.Circle((x, y), config.robot_radius, color="k")
        plt.gca().add_patch(circle)
    
    # Plot ship bounding boxes
    for ship in ships:
        pos_x, pos_y, yaw, length, width, color = ship
        vertices = get_ship_vertices(pos_x, pos_y, yaw, length, width)
        polygon = plt.Polygon(
            vertices, 
            closed=True, 
            edgecolor=color, 
            fill=False, 
            linewidth=1.5,
            linestyle="--"
        )
        plt.gca().add_patch(polygon)
        # Add ship center marker
        plt.plot(pos_x, pos_y, color=color, markersize=3)
    
    # Plot start and goal (original code)
    plt.plot(sx, sy, "or", zorder=10, label='Start')
    plt.plot(gx, gy, "sr", zorder=10, label='Goal')
    
    plt.grid(True)
    plt.axis("equal")
    plt.title("Navigation Map with Ship Bounding Boxes")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
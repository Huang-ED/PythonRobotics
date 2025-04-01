import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

class Config:
    """
    simulation parameter class
    """
    def __init__(self):
        # robot parameter
        self.robot_radius = 0.5  # [m] for collision check

def draw_ship(pos_x, pos_y, yaw, ship_length, ship_width, color="red"):
    """
    Draw a rotated ship bounding box on the plot.
    
    Parameters:
        pos_x, pos_y (float): Center coordinates of the ship.
        yaw (float): Heading angle in degrees (0 degrees = north, clockwise).
        ship_length (float): Length of the ship along the y-axis at 0 yaw.
        ship_width (float): Width of the ship along the x-axis at 0 yaw.
        color (str): Color of the bounding box.
    """
    # Convert yaw to radians (clockwise rotation)
    theta = np.radians(yaw)
    
    # Half dimensions
    half_length = ship_length / 2
    half_width = ship_width / 2
    
    # Corners relative to center (before rotation)
    corners = np.array([
        [-half_width, half_length],   # Front-left
        [half_width, half_length],    # Front-right
        [half_width, -half_length],   # Rear-right
        [-half_width, -half_length]   # Rear-left
    ])
    
    # Rotation matrix (clockwise)
    rotation_matrix = np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ])
    
    # Apply rotation
    rotated_corners = np.dot(corners, rotation_matrix)
    
    # Translate to global coordinates
    global_corners = rotated_corners + np.array([pos_x, pos_y])
    
    # Draw polygon
    polygon = plt.Polygon(global_corners, closed=True, edgecolor=color, fill=False, linewidth=1.5, zorder=5)
    plt.gca().add_patch(polygon)
    
    # Optional: Draw heading line
    front_local = np.array([0, half_length])
    front_global = np.dot(front_local, rotation_matrix) + [pos_x, pos_y]
    plt.plot([pos_x, front_global[0]], [pos_y, front_global[1]], color=color, linewidth=1.5, zorder=5)

def main():
    config = Config()

    # ----- Set up the map -----
    ## Load the map from image
    image_path = "EnvData/AISData_20240827/land_shapes_sf_crop.png"
    arr = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    arr = cv2.resize(arr, (100, 100))
    _, arr = cv2.threshold(arr, 128, 1, cv2.THRESH_BINARY)

    ## Add boundary obstacles
    arr[0, :] = 0  # Top edge
    arr[-1, :] = 0  # Bottom edge
    arr[:, 0] = 0  # Left edge
    arr[:, -1] = 0  # Right edge
    ob = np.argwhere(arr == 0)

    ## Convert image coordinates to plot coordinates
    ob[:, [0, 1]] = ob[:, [1, 0]]  # Swap to (x, y)
    ob[:, 1] = arr.shape[0] - ob[:, 1] - 1  # Flip y-axis
    ox, oy = ob[:, 0], ob[:, 1]

    # Map for DWA
    arr = 1 - arr
    eroded_arr = cv2.erode(arr, kernel=np.ones((3, 3), np.uint8), iterations=1)
    arr_dwa = cv2.subtract(arr, eroded_arr)
    arr_dwa = 1 - arr_dwa

    ob_dwa = np.argwhere(arr_dwa == 0)
    ob_dwa[:, [0, 1]] = ob_dwa[:, [1, 0]]  # Swap to (x, y)
    ob_dwa[:, 1] = arr_dwa.shape[0] - ob_dwa[:, 1] - 1  # Flip y-axis

    # ----- Set up the start and goal positions -----
    sx, sy = 20, 90
    gx, gy = 3, 3

    # ----- Plot the map -----
    plt.figure(figsize=(10, 10))
    
    # Plot obstacles
    for (x, y) in ob:
        circle = plt.Circle((x, y), config.robot_radius, color="darkgrey")
        plt.gca().add_patch(circle)
    for (x, y) in ob_dwa:
        circle = plt.Circle((x, y), config.robot_radius, color="k")
        plt.gca().add_patch(circle)
    
    # Plot start and goal
    plt.plot(sx, sy, "or", zorder=10)
    plt.plot(gx, gy, "sr", zorder=10)
    
    # Plot ships (Example parameters)
    draw_ship(pos_x=30, pos_y=50, yaw=30, ship_length=10, ship_width=5, color="blue")
    draw_ship(pos_x=70, pos_y=20, yaw=90, ship_length=8, ship_width=4, color="green")
    
    plt.grid(True)
    plt.axis("equal")
    plt.show()

if __name__ == '__main__':
    main()

import os, sys
rpath = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.append(rpath)

import numpy as np
from PathPlanning.DynamicWindowApproach.dwa_paper_with_width \
    import any_circle_overlap_with_box

"""
Check whether any of the given circles overlap with a rotated rectangular box.

Parameters:
- circles: 2D numpy array, shape (N, 3), where each row contains the 2D coordinate of the point and the radius of the circle (x, y, radius)
- center: tuple (cx, cy), the 2D coordinate of the center of the box
- length: float, length of the box
- width: float, width of the box
- rot: float, rotational angle of the box in radians

Returns:
- Boolean: True if any circle overlaps with the box, False otherwise
"""

circles = np.array([
    [26, 80, 0.5],
])
center = (26.21826, 81.12303)
length = 1.2
width = 0.5
rot = -0.90129

# Check for overlap
overlap = any_circle_overlap_with_box(circles, center, length, width, rot)
print(f"Overlap: {overlap}")

exit()

# ----- Plot it Out -----
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 8))

# Plot the circles
for circle in circles:
    x, y, radius = circle
    circle_patch = patches.Circle((x, y), radius, fill=True, color='blue', alpha=0.5)
    ax.add_patch(circle_patch)
    ax.plot(x, y, 'bo')  # Center point

# Calculate rectangle vertices
half_length = length / 2
half_width = width / 2

# Vertices before rotation
vertices = np.array([
    [-half_length, -half_width],
    [half_length, -half_width],
    [half_length, half_width],
    [-half_length, half_width]
])

# Rotation matrix
rot_matrix = np.array([
    [np.cos(rot), -np.sin(rot)],
    [np.sin(rot), np.cos(rot)]
])

# Rotate vertices
rotated_vertices = np.dot(vertices, rot_matrix.T)

# Translate to center
rotated_vertices += np.array(center)

# Create rectangle patch
rect_patch = patches.Polygon(rotated_vertices, closed=True, 
                            fill=True, color='red', alpha=0.5)
ax.add_patch(rect_patch)

# Plot settings
ax.set_xlim(min(rotated_vertices[:,0].min(), circles[:,0].min()) - 2,
            max(rotated_vertices[:,0].max(), circles[:,0].max()) + 2)
ax.set_ylim(min(rotated_vertices[:,1].min(), circles[:,1].min()) - 2,
            max(rotated_vertices[:,1].max(), circles[:,1].max()) + 2)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title(f"Overlap Check (Result: {overlap})")
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")

plt.show()
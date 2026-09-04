import numpy as np
import matplotlib.pyplot as plt
import os

# Create 100x100 array of zeros
arr = np.zeros((100, 100), dtype=np.uint8)

# Draw vertical line at column 30 from row 20 to 80
arr[30:70, 50] = 1

# Save the image without displaying or annotations
# plt.imsave('test-1.png', arr, cmap='binary', vmin=0, vmax=1)

# Save the image in the same directory as the script
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, 'simple1.png')
plt.imsave(image_path, arr, cmap='binary', vmin=0, vmax=1)

import json
import numpy as np
import os
import argparse

# Import the provided manager
# Ensure map_manager.py is in the same directory
from map_manager import MapManager

class MockConfig:
    """Mock configuration object to satisfy MapManager's __init__ requirement."""
    def __init__(self):
        # Default radius if not specified in JSON (used in map_manager.py logic)
        self.obstacle_radius = 0.5

class HeadlessMapManager(MapManager):
    """
    A subclass of MapManager that ignores image loading.
    This allows testing JSON logic without needing the actual PNG map files.
    """
    def load_map_from_image(self, image_path: str, map_size: tuple = (100, 100)) -> None:
        print(f"  [Info] Headless Mode: Skipping image load for '{image_path}'.")
        # Initialize empty arrays to prevent crashes in get_current_obstacles
        self.static_obstacles = np.empty((0, 2))
        self.astar_obstacles = np.empty((0, 2))
        self.boundary_obstacles = np.empty((0, 2))
        # Note: map_size is still respected for coordinate logic if needed later

def simulate_and_export(json_file_path, start_iter, end_iter, dt=0.1, output_file="obstacle_data.txt"):
    """
    Simulates dynamic obstacles and records their state within a specific iteration range.
    """
    if not os.path.exists(json_file_path):
        print(f"Error: File '{json_file_path}' not found.")
        return

    # 1. Initialize Manager
    config = MockConfig()
    manager = HeadlessMapManager(config)
    
    # 2. Load the Map Config (Dynamic Obstacles)
    print(f"Loading configuration from: {json_file_path}")
    manager.load_map_config(json_file_path)
    
    num_obstacles = len(manager.dynamic_obstacles)
    print(f"Loaded {num_obstacles} dynamic obstacles.")

    # 3. Open Output File
    with open(output_file, 'w') as f:
        # Write CSV Header
        header = "Iteration, Obstacle_ID, Pos_X, Pos_Y, Vel_X, Vel_Y"
        f.write(header + "\n")
        print("-" * 60)
        print(header)
        print("-" * 60)

        current_iter = 0

        # 4. PRE-ROLL: Fast forward to the start_iteration
        # We must simulate the steps 0 to start_iter-1 to get the correct state
        if start_iter > 0:
            print(f"Pre-rolling simulation to iteration {start_iter}...")
            while current_iter < start_iter:
                manager.update_dynamic_obstacles(dt)
                current_iter += 1

        # 5. RECORDING: Loop from start_iter to end_iter
        while current_iter <= end_iter:
            # Get current states
            positions = manager.get_dynamic_obstacles_pos()
            velocities = manager.get_dynamic_obstacles_vel()
            
            # If no obstacles, break or write empty
            if len(positions) == 0:
                print(f"Iteration {current_iter}: No dynamic obstacles active.")
            else:
                for i in range(len(positions)):
                    pos = positions[i]
                    vel = velocities[i]
                    
                    # Format: Iteration, ID, X, Y, Vx, Vy
                    line_str = f"{current_iter}, {i}, {pos[0]:.4f}, {pos[1]:.4f}, {vel[0]:.4f}, {vel[1]:.4f}"
                    
                    # Output to file and console
                    f.write(line_str + "\n")
                    print(line_str)

            # Update for next step
            manager.update_dynamic_obstacles(dt)
            current_iter += 1
            
    print("-" * 60)
    print(f"Simulation complete. Data saved to {output_file}")


if __name__ == "__main__":
    # --- CONFIGURATION ---
    # You can change these values or parse them from command line arguments
    JSON_FILE = "PathPlanning/DWAT_v4_st/map_config/map_config_video1.1.json" 
    START_ITERATION = 860   # The range starts here (inclusive)
    END_ITERATION = 880     # The range ends here (inclusive)
    TIME_STEP = 0.1        # dt in seconds
    OUTPUT_FILENAME = "dynamic_obs_log.txt"

    simulate_and_export(JSON_FILE, START_ITERATION, END_ITERATION, TIME_STEP, OUTPUT_FILENAME)
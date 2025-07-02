import numpy as np
import cv2
import json
from typing import Tuple, List, Dict, Any

class MapManager:
    def __init__(self, config: Any):
        self.config = config
        self.static_obstacles = None
        self.dynamic_obstacles = None
        self.astar_obstacles = None
        self.boundary_obstacles = None
        self.road_map = None
        
    def load_map_from_image(self, image_path: str, map_size: Tuple[int, int] = (100, 100)) -> None:
        """Load map from image file and process it"""
        # Load and process the base map
        arr = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        arr = cv2.resize(arr, map_size)
        _, arr = cv2.threshold(arr, 128, 1, cv2.THRESH_BINARY)
        
        # Add boundary obstacles
        arr[0, :] = 0  # Top edge
        arr[-1, :] = 0  # Bottom edge
        arr[:, 0] = 0  # Left edge
        arr[:, -1] = 0  # Right edge
        
        # Extract obstacles
        self.static_obstacles = np.argwhere(arr == 0)
        self._process_obstacle_coordinates(self.static_obstacles, arr.shape)
        
        # Process for A* and DWA
        self._process_astar_map(arr)
        self._process_boundary_map(arr)
        
    def _process_obstacle_coordinates(self, obstacles: np.ndarray, arr_shape: Tuple[int, int]) -> None:
        """Process obstacle coordinates to match plot orientation"""
        obstacles[:, [0, 1]] = obstacles[:, [1, 0]]  # Swap columns to match (x, y)
        obstacles[:, 1] = arr_shape[0] - obstacles[:, 1] - 1  # Flip y-axis
        
    def _process_astar_map(self, arr: np.ndarray) -> None:
        """Process map specifically for A* planning"""
        arr_astar = cv2.erode(arr, kernel=np.ones((3, 3), np.uint8), iterations=1)
        self.astar_obstacles = np.argwhere(arr_astar == 0)
        self._process_obstacle_coordinates(self.astar_obstacles, arr_astar.shape)
        
    def _process_boundary_map(self, arr: np.ndarray) -> None:
        """Process map specifically for DWA planning"""
        arr = 1 - arr
        eroded_arr = cv2.erode(arr, kernel=np.ones((3, 3), np.uint8), iterations=1)
        arr_dwa = cv2.subtract(arr, eroded_arr)
        arr_dwa = 1 - arr_dwa
        
        self.boundary_obstacles = np.argwhere(arr_dwa == 0)
        self._process_obstacle_coordinates(self.boundary_obstacles, arr_dwa.shape)
        
    def add_dynamic_obstacles(self, obstacles: List[Tuple[float, float]]) -> None:
        """Add dynamic obstacles to the map"""
        if self.dynamic_obstacles is None:
            self.dynamic_obstacles = np.array(obstacles)
        else:
            self.dynamic_obstacles = np.vstack((self.dynamic_obstacles, obstacles))
            
    def get_current_obstacles(self) -> np.ndarray:
        """Get current obstacles including both static and dynamic"""
        if self.dynamic_obstacles is None:
            return self.boundary_obstacles
        return np.vstack((self.boundary_obstacles, self.dynamic_obstacles))
    
    def set_road_map(self, road_map: np.ndarray) -> None:
        """Set the global path (road map) for navigation"""
        self.road_map = road_map
        
    def save_map_config(self, file_path: str) -> None:
        """Save map configuration to file"""
        map_data = {
            'static_obstacles': self.static_obstacles.tolist() if self.static_obstacles is not None else None,
            'astar_obstacles': self.astar_obstacles.tolist() if self.astar_obstacles is not None else None,
            'dwa_obstacles': self.boundary_obstacles.tolist() if self.boundary_obstacles is not None else None,
            'road_map': self.road_map.tolist() if self.road_map is not None else None
        }
        with open(file_path, 'w') as f:
            json.dump(map_data, f)
            
    def load_map_config(self, file_path: str) -> None:
        """Load map configuration from file"""
        with open(file_path, 'r') as f:
            map_data = json.load(f)
            
        self.static_obstacles = np.array(map_data['static_obstacles']) if map_data['static_obstacles'] else None
        self.astar_obstacles = np.array(map_data['astar_obstacles']) if map_data['astar_obstacles'] else None
        self.boundary_obstacles = np.array(map_data['dwa_obstacles']) if map_data['dwa_obstacles'] else None
        self.road_map = np.array(map_data['road_map']) if map_data['road_map'] else None

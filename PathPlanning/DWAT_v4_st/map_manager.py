import numpy as np
import cv2
import json
from typing import Tuple, List, Dict, Any
import os


class DynamicObstacle:
    def __init__(self, waypoints: List[List[float]], speed: float, radius: float = 0.5):
        """
        Initialize a dynamic obstacle
        
        Args:
            waypoints: List of [x,y] coordinates the obstacle will follow
            speed: Speed at which the obstacle moves (m/s)
            radius: Collision radius of the obstacle
        """
        self.waypoints = np.array(waypoints, dtype=float)
        self.speed = float(speed)
        self.radius = float(radius)
        self.current_position = self.waypoints[0].copy()
        self.current_waypoint_index = 0
        self.distance_to_next = 0
        self.total_distance = self._calculate_total_path_length()
        
    def _calculate_total_path_length(self) -> float:
        """Calculate total path length of all waypoints"""
        if len(self.waypoints) < 2:
            return 0.0
        return sum(np.linalg.norm(self.waypoints[i+1] - self.waypoints[i]) 
                  for i in range(len(self.waypoints)-1))
    
    def update_position(self, dt: float) -> None:
        """Update obstacle position based on elapsed time"""
        if len(self.waypoints) < 2:
            return
            
        distance_to_move = self.speed * dt
        
        while distance_to_move > 0:
            # Get current and next waypoint (using modulo for cyclic path)
            current_wp = self.waypoints[self.current_waypoint_index]
            next_index = (self.current_waypoint_index + 1) % len(self.waypoints)
            next_wp = self.waypoints[next_index]
            
            segment_vector = next_wp - current_wp
            segment_length = np.linalg.norm(segment_vector)
            
            # Handle zero-length segments
            if segment_length < 1e-6:
                self.current_waypoint_index = next_index
                self.distance_to_next = 0
                continue
                
            segment_direction = segment_vector / segment_length
            remaining_in_segment = segment_length - self.distance_to_next
            
            if distance_to_move <= remaining_in_segment:
                # Move within current segment
                self.distance_to_next += distance_to_move
                self.current_position = current_wp + segment_direction * self.distance_to_next
                distance_to_move = 0
            else:
                # Move to next segment
                distance_to_move -= remaining_in_segment
                self.current_waypoint_index = next_index
                self.distance_to_next = 0



class MapManager:
    def __init__(self, config: Any):
        self.config = config
        self.static_obstacles = np.empty((0, 2))  # All static obstacles from image
        self.dynamic_obstacles = []  # Now a list of DynamicObstacle objects
        self.astar_obstacles = np.empty((0, 2))  # Obstacles for A*
        self.boundary_obstacles = np.empty((0, 2)) # Static obstacles for DWA (e.g., walls)
        self.road_map = None
        self.start_position = None
        self.goal_position = None
        
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
        # self._process_astar_map(arr)  # Add buffer for Global Planning
        self.astar_obstacles = self.static_obstacles.copy()  # Do not add buffer for Global Planning
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
        
    def add_dynamic_obstacle(self, waypoints: List[Tuple[float, float]], speed: float, radius: float = 0.5) -> None:
        """Add a dynamic obstacle with waypoints and speed"""
        self.dynamic_obstacles.append(DynamicObstacle(waypoints, speed, radius))
        
    def update_dynamic_obstacles(self, dt: float) -> None:
        """Update positions of all dynamic obstacles"""
        for obstacle in self.dynamic_obstacles:
            obstacle.update_position(dt)
            
    # --- Methods for Split-Cost Function ---
    
    def get_static_obstacles(self) -> np.ndarray:
        """Returns only the static boundary obstacles."""
        return self.boundary_obstacles
    
    def get_static_obstacle_radii(self) -> List[float]:
        """Returns radii for static boundary obstacles."""
        # Static obstacles all use the default radius from config
        return [self.config.obstacle_radius] * len(self.boundary_obstacles)

    def get_dynamic_obstacles_pos(self) -> np.ndarray:
        """Returns only the current positions of dynamic obstacles."""
        if not self.dynamic_obstacles:
            return np.empty((0, 2))
        return np.array([obs.current_position for obs in self.dynamic_obstacles])

    def get_dynamic_obstacle_radii(self) -> List[float]:
        """Returns the individual radii of dynamic obstacles."""
        return [obs.radius for obs in self.dynamic_obstacles]

    # --- Methods for Final Collision Check (Combined) ---
            
    def get_current_obstacles(self) -> np.ndarray:
        """Get current obstacles including both static and dynamic"""
        # Get static obstacles
        all_obstacles = self.boundary_obstacles.copy()
        
        # Add dynamic obstacles
        dynamic_pos = self.get_dynamic_obstacles_pos()
        if dynamic_pos.shape[0] > 0:
            if all_obstacles.shape[0] == 0:
                all_obstacles = dynamic_pos
            else:
                all_obstacles = np.vstack((all_obstacles, dynamic_pos))
                
        return all_obstacles
    
    def get_obstacle_radii(self) -> List[float]:
        """Get radii for all obstacles (static and dynamic)"""
        # Static obstacles use default radius
        static_radii = self.get_static_obstacle_radii()
        
        # Dynamic obstacles use their own radii
        dynamic_radii = self.get_dynamic_obstacle_radii()
        
        return static_radii + dynamic_radii
    
    def set_road_map(self, road_map: np.ndarray) -> None:
        """Set the global path (road map) for navigation"""
        self.road_map = road_map
        
    def load_map_config(self, file_path: str) -> None:
        """Load map configuration from file including dynamic obstacles"""
        with open(file_path, 'r') as f:
            map_data = json.load(f)
        
        # Load basic map data
        self.load_map_from_image(map_data['image_path'], tuple(map_data['map_size']))
        # image_dir = os.path.dirname(file_path)
        # image_path_full = os.path.join(image_dir, map_data['image_path'])
        # self.load_map_from_image(image_path_full, tuple(map_data['map_size']))
        
        # Load start and goal positions
        self.start_position = np.array(map_data['start_position'])
        self.goal_position = np.array(map_data['goal_position'])
        
        # Load dynamic obstacles if they exist
        if 'dynamic_obstacles' in map_data:
            for obs_data in map_data['dynamic_obstacles']:
                self.add_dynamic_obstacle(
                    waypoints=obs_data['waypoints'],
                    speed=obs_data['speed'],
                    radius=obs_data.get('radius', self.config.obstacle_radius)
                )
 
    def save_map_config(self, file_path: str) -> None:
        """Save map configuration including dynamic obstacles"""
        map_data = {
            'image_path': 'relative/path/to/image.png', # Placeholder
            'map_size': [100, 100], # Placeholder
            'start_position': self.start_position.tolist() if self.start_position is not None else None,
            'goal_position': self.goal_position.tolist() if self.goal_position is not None else None,
            'dynamic_obstacles': [
                {
                    'waypoints': obstacle.waypoints.tolist(),
                    'speed': obstacle.speed,
                    'radius': obstacle.radius
                }
                for obstacle in self.dynamic_obstacles
            ] if self.dynamic_obstacles else []
        }
        with open(file_path, 'w') as f:
            json.dump(map_data, f, indent=4)
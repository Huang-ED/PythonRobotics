import numpy as np
import cv2
import json
from typing import Tuple, List, Dict, Any
import os


class DynamicObstacle:
    def __init__(self,
                 waypoints: List[List[float]] = None,
                 speed: float = 0.5,
                 radius: float = 0.5,
                 motion_type: str = "waypoint",
                 center: List[float] = None,
                 circle_radius: float = None,
                 initial_angle: float = 0.0,
                 direction: int = 1,
                 trajectory: np.ndarray = None,
                 times: np.ndarray = None,
                 velocities: np.ndarray = None,
                 start_time: float = 0.0,
                 on_finish: str = "hold",
                 track_id: str = None,
                 on_start: str = "hold_wrong_speed"):
        """
        Initialize a dynamic obstacle.

        Args:
            waypoints:     (waypoint mode) List of [x,y] coordinates to follow.
            speed:         Tangential / linear speed (m/s).
            radius:        Collision radius of the obstacle.
            motion_type:   "waypoint" (default), "circle" or "trajectory".
            center:        (circle mode) [cx, cy] – centre of the circular orbit.
            circle_radius: (circle mode) Radius of the circular orbit (m).
            initial_angle: (circle mode) Starting angle in radians (default 0).
            direction:     (circle mode) +1 for CCW, -1 for CW (default +1).
            trajectory:    (trajectory mode) (K, 2) array of recorded [x, y] samples.
            times:         (trajectory mode) (K,) array of sample times in seconds.
            velocities:    (trajectory mode) optional (K, 2) array. If omitted it is
                           derived as v[k] = (p[k+1] - p[k]) / (t[k+1] - t[k]), v[-1] = 0,
                           i.e. the piecewise-constant velocity that reproduces the
                           linear interpolation between samples.
            start_time:    (trajectory mode) simulation time offset at t=0 (s).
            on_finish:     (trajectory mode) "hold" (freeze at the last sample, velocity 0)
                           or "loop" (restart the recording).
            on_start:      (trajectory mode) where the vessel is before its first
                           sample, for a track that starts after the run does:
                           "extrapolate" -- at p[0] + v[0] * (t - t_start), reporting
                           v[0], so it arrives at its first fix on time;
                           "hold" -- parked at p[0] with velocity 0;
                           "hold_wrong_speed" (default) -- parked at p[0] but
                           reporting v[0]. DEPRECATED: position and velocity
                           disagree, so a planner extrapolates a stationary ship
                           sailing away. Kept only so runs before 2026-10-01
                           reproduce; to be removed.
            track_id:      Optional label carried through for logging/plotting.
        """
        self.motion_type = motion_type
        self.speed = float(speed)
        self.radius = float(radius)
        self.velocity = np.array([0.0, 0.0])
        self.track_id = track_id

        if self.motion_type == "trajectory":
            self._init_trajectory(trajectory, times, velocities, start_time, on_finish,
                                  on_start)
        elif self.motion_type == "circle":
            if center is None or circle_radius is None:
                raise ValueError("'center' and 'circle_radius' are required for circle motion.")
            self.center = np.array(center, dtype=float)
            self.circle_radius = float(circle_radius)
            self.angle = float(initial_angle)
            self.direction = int(direction)
            # Derived angular speed (rad/s)
            self.angular_speed = self.speed / self.circle_radius
            # Set initial position and velocity
            self.current_position = self._circle_position()
            self._update_circle_velocity()
            # Unused in circle mode but kept for API consistency
            self.waypoints = np.empty((0, 2))
            self.total_distance = 0.0
            self.current_waypoint_index = 0
            self.distance_to_next = 0.0
        else:  # waypoint mode
            self.waypoints = np.array(waypoints, dtype=float)
            self.current_position = self.waypoints[0].copy()
            self.current_waypoint_index = 0
            self.distance_to_next = 0
            self.total_distance = self._calculate_total_path_length()
            self._update_velocity_vector()

    # ------------------------------------------------------------------
    # Trajectory-mode helpers
    # ------------------------------------------------------------------

    def _init_trajectory(self, trajectory, times, velocities, start_time, on_finish,
                         on_start="hold_wrong_speed") -> None:
        """Set up a replayed, time-parameterised recording."""
        if trajectory is None:
            raise ValueError("'trajectory' is required for trajectory motion.")

        self.trajectory = np.asarray(trajectory, dtype=float).reshape(-1, 2)
        if self.trajectory.shape[0] == 0:
            raise ValueError("'trajectory' must contain at least one sample.")

        if times is None:
            raise ValueError("'times' is required for trajectory motion.")
        self.times = np.asarray(times, dtype=float).reshape(-1)
        if self.times.shape[0] != self.trajectory.shape[0]:
            raise ValueError(
                f"'times' ({self.times.shape[0]}) and 'trajectory' "
                f"({self.trajectory.shape[0]}) must have the same length."
            )
        if np.any(np.diff(self.times) < 0):
            raise ValueError("'times' must be non-decreasing.")

        if velocities is None:
            self.velocities = self._derive_velocities()
        else:
            self.velocities = np.asarray(velocities, dtype=float).reshape(-1, 2)
            if self.velocities.shape[0] != self.trajectory.shape[0]:
                raise ValueError("'velocities' must have the same length as 'trajectory'.")

        self.on_finish = on_finish
        if self.on_finish not in ("hold", "loop"):
            raise ValueError("'on_finish' must be 'hold' or 'loop'.")
        self.on_start = on_start
        if self.on_start not in ("extrapolate", "hold", "hold_wrong_speed"):
            raise ValueError("'on_start' must be 'extrapolate', 'hold' or 'hold_wrong_speed'.")

        self.t = float(start_time)
        self.t_start = float(self.times[0])
        self.t_end = float(self.times[-1])
        self.duration = self.t_end - self.t_start

        # Unused in trajectory mode but kept for API consistency
        self.waypoints = np.empty((0, 2))
        self.total_distance = float(
            np.linalg.norm(np.diff(self.trajectory, axis=0), axis=1).sum()
        )
        self.current_waypoint_index = 0
        self.distance_to_next = 0.0

        self._sync_trajectory_state()

    def _derive_velocities(self) -> np.ndarray:
        """v[k] = (p[k+1] - p[k]) / (t[k+1] - t[k]), with v[-1] = 0."""
        velocities = np.zeros_like(self.trajectory)
        if self.trajectory.shape[0] < 2:
            return velocities
        dt = np.diff(self.times)
        # Guard against duplicated timestamps producing division by zero.
        safe_dt = np.where(dt > 1e-12, dt, np.inf)
        velocities[:-1] = np.diff(self.trajectory, axis=0) / safe_dt[:, None]
        return velocities

    def _query_time(self) -> float:
        """Map simulation time onto the recording's own timeline."""
        if self.on_finish == "loop" and self.duration > 1e-12:
            return self.t_start + (self.t - self.t_start) % self.duration
        return float(np.clip(self.t, self.t_start, self.t_end))

    def _sync_trajectory_state(self) -> None:
        """Recompute position and velocity for the current simulation time."""
        tq = self._query_time()

        self.current_position = np.array([
            np.interp(tq, self.times, self.trajectory[:, 0]),
            np.interp(tq, self.times, self.trajectory[:, 1]),
        ])

        # Piecewise-constant velocity: the segment that starts at or before tq.
        # This is exactly the velocity implied by the linear interpolation above.
        k = int(np.clip(np.searchsorted(self.times, tq, side="right") - 1,
                        0, self.velocities.shape[0] - 1))
        self.velocity = self.velocities[k].copy()

        # Past the end of a "hold" recording the obstacle is parked.
        if self.on_finish == "hold" and self.t >= self.t_end:
            self.velocity = np.array([0.0, 0.0])

        # Before the first sample (a "loop" recording never is). The clamp in
        # _query_time parks the vessel at p[0], and the segment lookup above
        # hands it v[0] -- "hold_wrong_speed", the historical behaviour.
        if self.on_finish != "loop" and self.t < self.t_start:
            if self.on_start == "hold":
                self.velocity = np.array([0.0, 0.0])
            elif self.on_start == "extrapolate":
                self.velocity = self.velocities[0].copy()
                self.current_position = self.trajectory[0] + self.velocity * (self.t - self.t_start)

    def _update_trajectory_position(self, dt: float) -> None:
        self.t += dt
        self._sync_trajectory_state()

    @property
    def is_finished(self) -> bool:
        """True once a 'hold' recording has run out of real samples."""
        return self.motion_type == "trajectory" and self.on_finish == "hold" \
            and self.t >= self.t_end

    # ------------------------------------------------------------------
    # Circle-mode helpers
    # ------------------------------------------------------------------

    def _circle_position(self) -> np.ndarray:
        """Return position on the circle for the current angle."""
        return np.array([
            self.center[0] + self.circle_radius * np.cos(self.angle),
            self.center[1] + self.circle_radius * np.sin(self.angle)
        ])

    def _update_circle_velocity(self) -> None:
        """Set velocity tangent to the circle at the current angle."""
        # Tangent direction for CCW: (-sin θ, cos θ); CW: (sin θ, -cos θ)
        self.velocity = np.array([
            -np.sin(self.angle) * self.speed * self.direction,
             np.cos(self.angle) * self.speed * self.direction
        ])

    # ------------------------------------------------------------------
    # Waypoint-mode helpers
    # ------------------------------------------------------------------

    def _calculate_total_path_length(self) -> float:
        """Calculate total path length of all waypoints"""
        if len(self.waypoints) < 2:
            return 0.0
        return sum(np.linalg.norm(self.waypoints[i+1] - self.waypoints[i]) 
                  for i in range(len(self.waypoints)-1))
    
    def _update_velocity_vector(self) -> None:
        """Helper to calculate current velocity vector based on target waypoint"""
        if len(self.waypoints) < 2:
            self.velocity = np.array([0.0, 0.0])
            return

        current_wp = self.waypoints[self.current_waypoint_index]
        next_index = (self.current_waypoint_index + 1) % len(self.waypoints)
        next_wp = self.waypoints[next_index]

        segment_vector = next_wp - current_wp
        segment_length = np.linalg.norm(segment_vector)

        if segment_length < 1e-6:
            self.velocity = np.array([0.0, 0.0])
        else:
            # Velocity = Direction * Speed
            self.velocity = (segment_vector / segment_length) * self.speed
    
    def update_position(self, dt: float) -> None:
        """Update obstacle position based on elapsed time."""
        if self.motion_type == "trajectory":
            self._update_trajectory_position(dt)
            return
        if self.motion_type == "circle":
            self._update_circle_position(dt)
            return
        self._update_waypoint_position(dt)

    def _update_circle_position(self, dt: float) -> None:
        """Advance angle smoothly and recompute position + velocity."""
        self.angle += self.angular_speed * dt * self.direction
        # Keep angle in [-π, π] to avoid float drift over long runs
        self.angle = (self.angle + np.pi) % (2 * np.pi) - np.pi
        self.current_position = self._circle_position()
        self._update_circle_velocity()

    def _update_waypoint_position(self, dt: float) -> None:
        """Original waypoint-following logic."""
        if len(self.waypoints) < 2:
            return

        distance_to_move = self.speed * dt

        while distance_to_move > 0:
            # Get current and next waypoint
            current_wp = self.waypoints[self.current_waypoint_index]
            next_index = (self.current_waypoint_index + 1) % len(self.waypoints)
            next_wp = self.waypoints[next_index]
            
            segment_vector = next_wp - current_wp
            segment_length = np.linalg.norm(segment_vector)
            
            # Handle zero-length segments
            if segment_length < 1e-6:
                self.current_waypoint_index = next_index
                self.distance_to_next = 0
                self._update_velocity_vector() 
                continue
                
            segment_direction = segment_vector / segment_length
            remaining_in_segment = segment_length - self.distance_to_next
            
            if distance_to_move <= remaining_in_segment:
                # Move within current segment
                self.distance_to_next += distance_to_move
                self.current_position = current_wp + segment_direction * self.distance_to_next
                distance_to_move = 0
                
                # --- NEW: Ensure velocity is current ---
                self.velocity = segment_direction * self.speed
            else:
                # Move to next segment
                distance_to_move -= remaining_in_segment
                self.current_waypoint_index = next_index
                self.distance_to_next = 0
                
                # --- NEW: Update Velocity for the new segment ---
                self._update_velocity_vector()


class MapManager:
    def __init__(self, config: Any):
        self.config = config
        self.static_obstacles = np.empty((0, 2))  # All static obstacles from image
        self.dynamic_obstacles = []  # Now a list of DynamicObstacle objects
        self.astar_obstacles = np.empty((0, 2))  # Obstacles for A*
        self.boundary_obstacles = np.empty((0, 2)) # Static obstacles for DWA (e.g., walls)
        self.map_size = None  # (width, height) of the map in pixels / coordinate units
        self.road_map = None
        self.start_position = None
        self.goal_position = None
        # Populated only when the map comes from an occupancy grid
        self.occupancy = None
        self.occupancy_shape = None
        self.occupancy_resolution = None
        self.occupancy_origin = None
        self.occupancy_cell_anchor = None
        
    def load_map_from_image(self, image_path: str, map_size: Tuple[int, int] = (100, 100)) -> None:
        """Load map from image file and process it"""
        # Load and process the base map
        arr = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        arr = cv2.resize(arr, map_size)
        _, arr = cv2.threshold(arr, 128, 1, cv2.THRESH_BINARY)
        self.map_size = map_size  # store (width, height) for downstream use

        arr_for_planning = arr.copy()
        if self.config.enable_map_boundary_obstacles:
            arr_for_planning[0, :] = 0  # Top edge
            arr_for_planning[-1, :] = 0  # Bottom edge
            arr_for_planning[:, 0] = 0  # Left edge
            arr_for_planning[:, -1] = 0  # Right edge
        
        # Extract obstacles
        self.static_obstacles = np.argwhere(arr_for_planning == 0)
        self._process_obstacle_coordinates(self.static_obstacles, arr_for_planning.shape)
        
        # Process for A* and DWA
        # self._process_astar_map(arr)  # Add buffer for Global Planning
        self.astar_obstacles = self.static_obstacles.copy()  # Do not add buffer for Global Planning
        self._process_boundary_map(arr_for_planning)
        
    def load_map_from_occupancy(self,
                                occupancy: np.ndarray,
                                resolution: float = 1.0,
                                origin: Tuple[float, float] = (0.0, 0.0),
                                cell_anchor: str = "center") -> None:
        """Load the static map from an occupancy grid instead of an image.

        Args:
            occupancy:   (nx, ny) array where a non-zero entry marks an occupied
                         cell. Note the first axis is x and the second is y --
                         this is the convention used by the exported scene data,
                         not the row/column convention of an image.
            resolution:  Side length of one cell in metres.
            origin:      World coordinates of the (0, 0) cell's lower-left corner.
            cell_anchor: "center" places an obstacle point at the centre of each
                         occupied cell, i.e. ((i + 0.5) * res, (j + 0.5) * res);
                         "corner" places it at (i * res, j * res). Use "center"
                         for data that documents the cell-centre convention.

        Unlike load_map_from_image there is no resampling step, so the grid is
        consumed exactly as authored -- important when the accompanying
        trajectories are certified collision-free against this very grid.
        """
        occ = (np.asarray(occupancy) != 0).astype(np.uint8)
        if occ.ndim != 2:
            raise ValueError(f"occupancy must be 2-D, got shape {occ.shape}")

        if cell_anchor not in ("center", "corner"):
            raise ValueError("cell_anchor must be 'center' or 'corner'")
        offset = 0.5 if cell_anchor == "center" else 0.0

        # map_size is consumed downstream as the WORLD extent (plot limits,
        # planner bounds), not as a cell count. For an image map one pixel is
        # one world unit so the two coincide; for an occupancy grid they only
        # coincide at 1 m cells, so convert explicitly.
        self.map_size = (origin[0] + occ.shape[0] * float(resolution),
                         origin[1] + occ.shape[1] * float(resolution))
        self.occupancy = occ
        self.occupancy_shape = (occ.shape[0], occ.shape[1])
        self.occupancy_resolution = float(resolution)
        self.occupancy_origin = np.array(origin, dtype=float)
        self.occupancy_cell_anchor = cell_anchor

        occ_for_planning = occ.copy()
        if self.config.enable_map_boundary_obstacles:
            occ_for_planning[0, :] = 1
            occ_for_planning[-1, :] = 1
            occ_for_planning[:, 0] = 1
            occ_for_planning[:, -1] = 1

        def to_world(indices: np.ndarray) -> np.ndarray:
            if indices.shape[0] == 0:
                return np.empty((0, 2))
            return (indices.astype(float) + offset) * float(resolution) + np.array(origin, dtype=float)

        # All occupied cells.
        self.static_obstacles = to_world(np.argwhere(occ_for_planning == 1))
        self.astar_obstacles = self.static_obstacles.copy()

        # Only the rim of each occupied region, mirroring _process_boundary_map:
        # erode the obstacle mask and keep what the erosion removed.
        eroded = cv2.erode(occ_for_planning, kernel=np.ones((3, 3), np.uint8), iterations=1)
        rim = cv2.subtract(occ_for_planning, eroded)
        self.boundary_obstacles = to_world(np.argwhere(rim == 1))

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
        
    def add_dynamic_obstacle(self,
                              waypoints: List[Tuple[float, float]] = None,
                              speed: float = 0.5,
                              radius: float = 0.5,
                              motion_type: str = "waypoint",
                              center: List[float] = None,
                              circle_radius: float = None,
                              initial_angle: float = 0.0,
                              direction: int = 1,
                              trajectory: np.ndarray = None,
                              times: np.ndarray = None,
                              velocities: np.ndarray = None,
                              start_time: float = 0.0,
                              on_finish: str = "hold",
                              track_id: str = None,
                              on_start: str = "hold_wrong_speed") -> None:
        """Add a dynamic obstacle (waypoint, circle or replayed trajectory)."""
        self.dynamic_obstacles.append(DynamicObstacle(
            waypoints=waypoints,
            speed=speed,
            radius=radius,
            motion_type=motion_type,
            center=center,
            circle_radius=circle_radius,
            initial_angle=initial_angle,
            direction=direction,
            trajectory=trajectory,
            times=times,
            velocities=velocities,
            start_time=start_time,
            on_finish=on_finish,
            track_id=track_id,
            on_start=on_start,
        ))
        
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
    
    def get_dynamic_obstacles_vel(self) -> np.ndarray:
        """
        Returns only the current velocity vectors (vx, vy) of dynamic obstacles.
        Simulates reading SOG/COG from AIS data.
        """
        if not self.dynamic_obstacles:
            return np.empty((0, 2))
        
        # Directly retrieve the pre-calculated velocity from each obstacle object
        return np.array([obs.velocity for obs in self.dynamic_obstacles])

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

        self.config.enable_map_boundary_obstacles = map_data.get(
            'enable_map_boundary_obstacles',
            self.config.enable_map_boundary_obstacles
        )

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
                motion_type = obs_data.get('motion_type', 'waypoint')
                if motion_type == 'circle':
                    self.add_dynamic_obstacle(
                        speed=obs_data['speed'],
                        radius=obs_data.get('radius', self.config.obstacle_radius),
                        motion_type='circle',
                        center=obs_data['center'],
                        circle_radius=obs_data['circle_radius'],
                        initial_angle=obs_data.get('initial_angle', 0.0),
                        direction=obs_data.get('direction', 1)
                    )
                else:
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
            'enable_map_boundary_obstacles': self.config.enable_map_boundary_obstacles,
            'start_position': self.start_position.tolist() if self.start_position is not None else None,
            'goal_position': self.goal_position.tolist() if self.goal_position is not None else None,
            'dynamic_obstacles': [
                (
                    {
                        'motion_type': 'trajectory',
                        'trajectory': obstacle.trajectory.tolist(),
                        'times': obstacle.times.tolist(),
                        'velocities': obstacle.velocities.tolist(),
                        'on_finish': obstacle.on_finish,
                        'track_id': obstacle.track_id,
                        'radius': obstacle.radius
                    }
                    if obstacle.motion_type == 'trajectory'
                    else
                    {
                        'motion_type': 'circle',
                        'center': obstacle.center.tolist(),
                        'circle_radius': obstacle.circle_radius,
                        'initial_angle': obstacle.angle,
                        'direction': obstacle.direction,
                        'speed': obstacle.speed,
                        'radius': obstacle.radius
                    }
                    if obstacle.motion_type == 'circle'
                    else
                    {
                        'motion_type': 'waypoint',
                        'waypoints': obstacle.waypoints.tolist(),
                        'speed': obstacle.speed,
                        'radius': obstacle.radius
                    }
                )
                for obstacle in self.dynamic_obstacles
            ] if self.dynamic_obstacles else []
        }
        with open(file_path, 'w') as f:
            json.dump(map_data, f, indent=4)
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class RuntimeOptions:
    show_animation: bool = True
    save_animation_to_figs: bool = True
    show_global_planner_search: bool = False
    plot_predicted_dynamic_obstacles: bool = True
    plot_candidate_trajectories: bool = True
    candidate_plot_stride: int = 20
    candidate_plot_max_count: int = 120
    output_tag: str = "figs_v20.7.5-modular"


@dataclass
class MapContext:
    map_manager: Any
    start: np.ndarray
    goal: np.ndarray
    road_map: np.ndarray
    rx: List[float]
    ry: List[float]


@dataclass
class CollisionEvent:
    iteration: int
    obstacle_index: int
    distance_to_center: float
    robot_shape_info: str
    obstacle_position: List[float]
    obstacle_radius: float
    collision_threshold: float


@dataclass
class SimulationResult:
    trajectory: np.ndarray
    log_entries: List[Dict[str, Any]]
    reached_goal: bool
    collision_event: Optional[CollisionEvent]
    error: Optional[str] = None
    traceback_text: Optional[str] = None


@dataclass
class OutputArtifacts:
    fig_dir: Optional[Path]
    log_file: Optional[Path]

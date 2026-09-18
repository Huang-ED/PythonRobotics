import math
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
    output_tag: str = ""
    # Hard ceiling on control iterations. 0 = unlimited, which is the historical
    # behaviour and stays the default so existing runs reproduce.
    #
    # The control loop has no natural bound: it exits on reaching the goal, a
    # collision, or _choose_local_goal straying outside dist_localgoal. A ship
    # that stalls just outside catch_goal_dist satisfies none of those and spins
    # forever -- owenc_cross_stbd_090 at v21.1 burned 11h49m and 9,996
    # iterations without moving. Set this and that becomes a clear failure in
    # minutes instead.
    max_sim_steps: int = 0
    # Seconds of wall-clock between progress lines on stdout. 0 = silent.
    # Printing changes no trajectory, so this is not a config-restorable
    # concern; it defaults ON because a diagnostic that is off by default
    # helps nobody. Time-based rather than step-based so it self-tunes to
    # whatever the config costs per iteration.
    progress_every_s: float = 30.0
    # Trailing window, in steps, used to decide whether the ego is making
    # headway. Must be long enough to contain a full minimum-radius circle:
    # at max_yaw_rate the circumference is 2*pi*v/omega, ~330 steps for the
    # v21.x ego, so 600 leaves room. 0 disables the check.
    stall_window_steps: int = 600


@dataclass
class MapContext:
    map_manager: Any
    start: np.ndarray
    goal: np.ndarray
    road_map: np.ndarray
    rx: List[float]
    ry: List[float]
    # Initial heading (radians). Convention: 0 = +x (east), pi/2 = +y (north).
    # Defaults to the historical hardcoded value so existing scenarios are unchanged.
    start_yaw: float = -math.pi / 8.0


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

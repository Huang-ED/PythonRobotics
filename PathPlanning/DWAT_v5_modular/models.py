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
    # Where per-frame PNGs go, and whether they outlive the run.
    #
    # Frames are an INTERMEDIATE -- run_paths documents the video as "assembled
    # afterwards by ffmpeg" -- but they have always been written into the run
    # directory and never deleted. Measured 2026-09-21: ~55 KB a frame, ~85 MB
    # for a 1540-step run, and 52 GB across 358 frames/ directories already in
    # runs/, which on this machine is inside a OneDrive folder and therefore
    # syncing to the cloud.
    #
    # frames_root = ""   frames go to <run dir>/frames. The historical
    #                    behaviour, and the default, so nothing changes for a
    #                    config that does not set this.
    # frames_root = path frames go to <path>/<run tag>/ instead. Point it at a
    #                    WSL-native path: ext4 also saves the ~16% DrvFs write
    #                    penalty.
    #
    # NOTHING in this repository deletes them. Scratch is the caller's to clear
    # -- keeping it out of the run means no code path can ever throw away the
    # frames of a run someone still wanted. tools/purge_scratch_frames.py
    # sweeps a frames_root when asked.
    #
    # None of this changes a trajectory, so it is not a config-restorable
    # question in the sense of ../configs/algorithm/README.md.
    frames_root: str = ""
    # Log the DWA candidate set every N control steps, 0 = off (the default).
    #
    # The candidate fan is the one thing a video needs that log_details.json
    # does not carry -- predicted obstacle tracks reconstruct exactly from the
    # logged positions as pos + vel*t, and the global-planner search is a
    # one-off before the loop. Logging it lets tools/render_run_video.py
    # produce a full investigation video WITHOUT rendering inline, which costs
    # ~85 MB of PNGs and ~+65% wall clock.
    #
    # Sizes measured on a 1540-step head_on run, 120 candidates, float32, with
    # candidate_point_stride = 10:
    #
    #   every step     115.5 MB        every 10 steps    11.5 MB
    #   every 20 steps   5.8 MB
    #
    # 10 is the useful default when on: 6.5 frames/s of decision fan at dt=0.1.
    log_candidates_every: int = 0
    # Keep every Nth point of each logged candidate. The fan is for seeing
    # WHERE the options went, so full time resolution is waste.
    candidate_point_stride: int = 10
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

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from PathPlanning.DWAT_v4_st import theta_star
from PathPlanning.DWAT_v4_st.map_manager import MapManager

from .models import MapContext


def _prepare_global_search_canvas(map_manager: MapManager, start: np.ndarray, goal: np.ndarray, config: Any) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10))

    if map_manager.static_obstacles.shape[0] > 0:
        for (x_s, y_s) in map_manager.static_obstacles:
            circle = plt.Circle((x_s, y_s), config.robot_radius, color="darkgrey")
            plt.gca().add_patch(circle)

    if map_manager.boundary_obstacles.shape[0] > 0:
        for (x_b, y_b) in map_manager.boundary_obstacles:
            circle = plt.Circle((x_b, y_b), config.robot_radius, color="k")
            plt.gca().add_patch(circle)

    plt.plot(start[0], start[1], "or", zorder=10)
    plt.plot(goal[0], goal[1], "sr", zorder=10)
    plt.grid(True)
    plt.axis("equal")
    plt.xlim(-1, map_manager.map_size[0])
    plt.ylim(-1, map_manager.map_size[1])


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_scenario_path(scenario_path: str) -> Path:
    candidate = Path(scenario_path)
    if candidate.is_absolute():
        return candidate

    root_candidate = (_repo_root() / candidate).resolve()
    if root_candidate.exists():
        return root_candidate

    local_candidate = (Path(__file__).resolve().parent / candidate).resolve()
    if local_candidate.exists():
        return local_candidate

    raise FileNotFoundError(f"Scenario config not found: {scenario_path}")


def resolve_image_path(image_path: str, scenario_file: Path) -> Path:
    path = Path(image_path)
    if path.is_absolute() and path.exists():
        return path

    root_candidate = (_repo_root() / path).resolve()
    if root_candidate.exists():
        return root_candidate

    scenario_relative = (scenario_file.parent / path).resolve()
    if scenario_relative.exists():
        return scenario_relative

    raise FileNotFoundError(
        f"Map image not found. Tried absolute/root/scenario-relative for: {image_path}"
    )


def _add_dynamic_obstacles(map_manager: MapManager, dynamic_obstacles: List[Dict[str, Any]]) -> None:
    for obs_data in dynamic_obstacles:
        motion_type = obs_data.get("motion_type", "waypoint")
        if motion_type == "circle":
            map_manager.add_dynamic_obstacle(
                speed=obs_data["speed"],
                radius=obs_data.get("radius", map_manager.config.obstacle_radius),
                motion_type="circle",
                center=obs_data["center"],
                circle_radius=obs_data["circle_radius"],
                initial_angle=obs_data.get("initial_angle", 0.0),
                direction=obs_data.get("direction", 1),
            )
        else:
            map_manager.add_dynamic_obstacle(
                waypoints=obs_data["waypoints"],
                speed=obs_data["speed"],
                radius=obs_data.get("radius", map_manager.config.obstacle_radius),
                motion_type="waypoint",
            )


def _plan_global_path(
    config: Any,
    map_manager: MapManager,
    start: np.ndarray,
    goal: np.ndarray,
    show_global_search_animation: bool,
) -> Tuple[np.ndarray, List[float], List[float]]:
    if map_manager.astar_obstacles.shape[0] > 0:
        ob_min_x = map_manager.astar_obstacles[:, 0].min()
        ob_min_y = map_manager.astar_obstacles[:, 1].min()
        ob_max_x = map_manager.astar_obstacles[:, 0].max()
        ob_max_y = map_manager.astar_obstacles[:, 1].max()
    else:
        ob_min_x = ob_min_y = 0.0
        ob_max_x = float(map_manager.map_size[0] - 1)
        ob_max_y = float(map_manager.map_size[1] - 1)

    astar_ob = (
        map_manager.astar_obstacles
        if map_manager.astar_obstacles.shape[0] > 0
        else np.array([[-999.0, -999.0]])
    )

    planner = theta_star.ThetaStarPlanner(
        ob=astar_ob,
        resolution=1.0,
        rr=max(config.robot_width, config.robot_length),
        min_x=min(ob_min_x, start[0] - 2, goal[0] - 2),
        min_y=min(ob_min_y, start[1] - 2, goal[1] - 2),
        max_x=max(ob_max_x, start[0] + 2, goal[0] + 2),
        max_y=max(ob_max_y, start[1] + 2, goal[1] + 2),
        save_animation_to_figs=False,
        fig_dir=None,
    )

    previous_show_animation = theta_star.show_animation
    theta_star.show_animation = bool(show_global_search_animation)
    try:
        rx, ry = planner.planning(start[0], start[1], goal[0], goal[1], curr_i_fig=0)
    finally:
        theta_star.show_animation = previous_show_animation

    road_map = np.array([rx, ry]).transpose()[::-1]
    return road_map, rx, ry


def create_map_context(
    config: Any,
    scenario_path: str,
    show_global_search_animation: bool = False,
) -> MapContext:
    scenario_file = resolve_scenario_path(scenario_path)
    scenario = _load_json(scenario_file)

    image_file = resolve_image_path(scenario["image_path"], scenario_file)

    map_manager = MapManager(config)
    map_manager.config.enable_map_boundary_obstacles = scenario.get(
        "enable_map_boundary_obstacles", map_manager.config.enable_map_boundary_obstacles
    )
    map_manager.load_map_from_image(str(image_file), tuple(scenario["map_size"]))

    map_manager.start_position = np.array(scenario["start_position"], dtype=float)
    map_manager.goal_position = np.array(scenario["goal_position"], dtype=float)

    dynamic_obstacles = scenario.get("dynamic_obstacles", [])
    _add_dynamic_obstacles(map_manager, dynamic_obstacles)

    if show_global_search_animation:
        # Keep global-search and local-search updates on the same figure.
        _prepare_global_search_canvas(
            map_manager=map_manager,
            start=map_manager.start_position,
            goal=map_manager.goal_position,
            config=config,
        )

    road_map, rx, ry = _plan_global_path(
        config=config,
        map_manager=map_manager,
        start=map_manager.start_position,
        goal=map_manager.goal_position,
        show_global_search_animation=show_global_search_animation,
    )
    map_manager.set_road_map(road_map)

    return MapContext(
        map_manager=map_manager,
        start=map_manager.start_position,
        goal=map_manager.goal_position,
        road_map=road_map,
        rx=rx,
        ry=ry,
    )

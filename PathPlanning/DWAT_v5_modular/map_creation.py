import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

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


def find_config_by_name(name: str, search_root: Path) -> Optional[Path]:
    """Locate a config by bare filename anywhere under search_root.

    Configs are grouped into subdirectories (scenarios/encounters/,
    algorithm/v20.7.8-2/, ...), so a path written before that grouping -- or a
    bare name like "ow_sequence" -- no longer resolves directly. Searching by
    filename keeps both working, and means the grouping can be rearranged again
    without invalidating every command in the READMEs and notebooks.

    Returns None if there is no match. Raises if the name is ambiguous, rather
    than silently picking one of two different configs.
    """
    rel = Path(name)
    if not rel.name.endswith(".json"):
        rel = rel.with_name(rel.name + ".json")

    # A directory-qualified name ("v20.7.9/predT5-ramped",
    # "encounters/owenc_head_on") is already unambiguous. Honour it before
    # falling back to the bare-filename search, so that the "Pass a fuller
    # path" advice below is actually actionable -- the search discards the
    # directory, so without this a fuller path could not resolve an ambiguity.
    if rel.parent != Path("."):
        direct = search_root / rel
        if direct.is_file():
            return direct.resolve()

    stem = rel.name
    hits = sorted(search_root.rglob(stem))
    if not hits:
        return None
    if len(hits) > 1:
        joined = ", ".join(str(h.relative_to(search_root)) for h in hits)
        raise FileNotFoundError(
            f"Ambiguous config name {stem!r}; matches {joined}. Pass a fuller path."
        )
    return hits[0]


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

    found = find_config_by_name(
        scenario_path, Path(__file__).resolve().parent / "configs" / "scenarios"
    )
    if found is not None:
        return found

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


def resolve_data_path(data_path: str, scenario_file: Path) -> Path:
    """Resolve a data asset the same way resolve_image_path resolves images."""
    path = Path(data_path)
    if path.is_absolute() and path.exists():
        return path

    root_candidate = (_repo_root() / path).resolve()
    if root_candidate.exists():
        return root_candidate

    scenario_relative = (scenario_file.parent / path).resolve()
    if scenario_relative.exists():
        return scenario_relative

    raise FileNotFoundError(
        f"Data file not found. Tried absolute/root/scenario-relative for: {data_path}"
    )


def _load_track(obs_data: Dict[str, Any], scenario_file: Path, track_cache: Dict[str, Any]):
    """Pull one recorded track out of a bundle .npz, or read it inline.

    Bundle layout (see adapters/scene_format.py):
        positions (n_obs, K, 2) | lengths (n_obs,) | times (K,) or (n_obs, K)
        radii (n_obs,) | velocities (n_obs, K, 2), optional | ids (n_obs,), optional
    """
    if "track_file" not in obs_data:
        trajectory = np.asarray(obs_data["trajectory"], dtype=float)
        times = np.asarray(obs_data["times"], dtype=float)
        velocities = obs_data.get("velocities")
        if velocities is not None:
            velocities = np.asarray(velocities, dtype=float)
        return trajectory, times, velocities, obs_data.get("radius"), obs_data.get("track_id")

    track_file = str(resolve_data_path(obs_data["track_file"], scenario_file))
    if track_file not in track_cache:
        track_cache[track_file] = np.load(track_file, allow_pickle=False)
    bundle = track_cache[track_file]

    k = int(obs_data["track_index"])
    n = int(bundle["lengths"][k])
    trajectory = bundle["positions"][k, :n]
    times = bundle["times"][:n] if bundle["times"].ndim == 1 else bundle["times"][k, :n]
    velocities = bundle["velocities"][k, :n] if "velocities" in bundle.files else None
    radius = obs_data.get("radius", float(bundle["radii"][k]))

    track_id = obs_data.get("track_id")
    if track_id is None and "ids" in bundle.files:
        track_id = str(bundle["ids"][k])

    return trajectory, times, velocities, radius, track_id


def _add_dynamic_obstacles(
    map_manager: MapManager,
    dynamic_obstacles: List[Dict[str, Any]],
    scenario_file: Path,
) -> None:
    # Trajectory obstacles usually share one bundle file; open it once.
    track_cache: Dict[str, Any] = {}

    for obs_data in dynamic_obstacles:
        motion_type = obs_data.get("motion_type", "waypoint")
        if motion_type == "trajectory":
            trajectory, times, velocities, radius, track_id = _load_track(
                obs_data, scenario_file, track_cache
            )
            map_manager.add_dynamic_obstacle(
                motion_type="trajectory",
                trajectory=trajectory,
                times=times,
                velocities=velocities,
                radius=radius if radius is not None else map_manager.config.obstacle_radius,
                start_time=obs_data.get("start_time", 0.0),
                on_finish=obs_data.get("on_finish", "hold"),
                track_id=track_id,
            )
        elif motion_type == "circle":
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
    planner_resolution: float = 1.0,
) -> Tuple[np.ndarray, List[float], List[float]]:
    """Plan the sparse global route with Theta*.

    ``planner_resolution`` is the global search grid spacing in world units. It
    is a property of the world, not of the planner tuning: hand-authored
    scenarios live on a 1-unit grid, but a scenario in real metres spans
    kilometres, where a 1 m search grid is both far finer than the route needs
    and large enough to exhaust memory. Scenarios set it via
    ``global_planner_resolution``; the default reproduces the historical value.
    """
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
        resolution=planner_resolution,
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

    map_manager = MapManager(config)
    map_manager.config.enable_map_boundary_obstacles = scenario.get(
        "enable_map_boundary_obstacles", map_manager.config.enable_map_boundary_obstacles
    )

    # The static map comes either from a raster image (hand-authored scenarios)
    # or from an occupancy grid exported by an external dataset. Prefer the grid
    # when present: resizing an image resamples it, and imported obstacle
    # trajectories are certified against the grid, not against the picture.
    if "occupancy_path" in scenario:
        occupancy_file = resolve_data_path(scenario["occupancy_path"], scenario_file)
        map_manager.load_map_from_occupancy(
            np.load(str(occupancy_file)),
            resolution=scenario.get("occupancy_resolution", 1.0),
            origin=tuple(scenario.get("occupancy_origin", (0.0, 0.0))),
            cell_anchor=scenario.get("occupancy_cell_anchor", "center"),
        )
    else:
        image_file = resolve_image_path(scenario["image_path"], scenario_file)
        map_manager.load_map_from_image(str(image_file), tuple(scenario["map_size"]))

    map_manager.start_position = np.array(scenario["start_position"], dtype=float)
    map_manager.goal_position = np.array(scenario["goal_position"], dtype=float)

    # Optional initial heading (radians). 0 = east (+x), pi/2 = north (+y).
    start_yaw = float(scenario.get("start_yaw", MapContext.start_yaw))

    dynamic_obstacles = scenario.get("dynamic_obstacles", [])
    _add_dynamic_obstacles(map_manager, dynamic_obstacles, scenario_file)

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
        planner_resolution=float(scenario.get("global_planner_resolution", 1.0)),
    )
    map_manager.set_road_map(road_map)

    return MapContext(
        map_manager=map_manager,
        start=map_manager.start_position,
        goal=map_manager.goal_position,
        road_map=road_map,
        rx=rx,
        ry=ry,
        start_yaw=start_yaw,
    )

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Dict

from PathPlanning.DWAT_v4_st.dwa_average_weighted_side import Config, RobotType


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_config_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path

    root_candidate = (_repo_root() / path).resolve()
    if root_candidate.exists():
        return root_candidate

    local_candidate = (Path(__file__).resolve().parent / path).resolve()
    if local_candidate.exists():
        return local_candidate

    raise FileNotFoundError(f"Config file not found: {path_text}")


def apply_algorithm_overrides(config: Config, overrides: Dict[str, Any]) -> Config:
    for key, value in overrides.items():
        if key == "robot_type":
            if isinstance(value, str):
                config.robot_type = RobotType[value]
            else:
                raise TypeError("'robot_type' in algorithm config must be string enum name")
            continue

        if not hasattr(config, key):
            raise AttributeError(f"Unknown algorithm config key: {key}")
        setattr(config, key, value)

    return config


def load_runtime_options(overrides: Dict[str, Any]) -> Any:
    RuntimeOptions = importlib.import_module(
        "PathPlanning.DWAT_v5_modular.models"
    ).RuntimeOptions

    runtime = RuntimeOptions()
    for key, value in overrides.items():
        if not hasattr(runtime, key):
            raise AttributeError(f"Unknown runtime option key: {key}")
        setattr(runtime, key, value)
    return runtime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Modular DWA + Theta* runner")
    parser.add_argument(
        "--scenario",
        default="PathPlanning/DWAT_v5_modular/configs/scenarios/scenario_3.json",
        help="Scenario JSON path.",
    )
    parser.add_argument(
        "--algorithm-config",
        default="PathPlanning/DWAT_v5_modular/configs/algorithm/default.json",
        help="Algorithm config JSON path.",
    )
    parser.add_argument(
        "--run-config",
        default="PathPlanning/DWAT_v5_modular/configs/run/default.json",
        help="Runtime options JSON path.",
    )
    parser.add_argument(
        "--no-animation",
        action="store_true",
        help="Disable matplotlib animation.",
    )
    parser.add_argument(
        "--no-save-frames",
        action="store_true",
        help="Disable per-frame PNG saving.",
    )
    return parser


def main() -> int:
    algorithm_mod = importlib.import_module("PathPlanning.DWAT_v5_modular.algorithm")
    map_creation_mod = importlib.import_module("PathPlanning.DWAT_v5_modular.map_creation")
    output_mod = importlib.import_module("PathPlanning.DWAT_v5_modular.output_display")

    run_simulation = algorithm_mod.run_simulation
    create_map_context = map_creation_mod.create_map_context
    resolve_scenario_path = map_creation_mod.resolve_scenario_path
    SimulationRenderer = output_mod.SimulationRenderer
    build_output_artifacts = output_mod.build_output_artifacts
    save_log = output_mod.save_log

    args = build_parser().parse_args()

    scenario_path = resolve_scenario_path(args.scenario)
    algorithm_config = _load_json(_resolve_config_path(args.algorithm_config))
    run_config = _load_json(_resolve_config_path(args.run_config))

    config = apply_algorithm_overrides(Config(), algorithm_config)
    runtime = load_runtime_options(run_config)

    if args.no_animation:
        runtime.show_animation = False
    if args.no_save_frames:
        runtime.save_animation_to_figs = False

    map_ctx = create_map_context(
        config=config,
        scenario_path=str(scenario_path),
        show_global_search_animation=runtime.show_global_planner_search and runtime.show_animation,
    )

    renderer = None
    if runtime.show_animation:
        renderer = SimulationRenderer(runtime)
        renderer.initialize_scene(
            map_ctx=map_ctx,
            config=config,
            base_dir=Path(__file__).resolve().parent,
            reuse_existing_figure=runtime.show_global_planner_search,
        )

    result = run_simulation(config=config, map_ctx=map_ctx, runtime=runtime, renderer=renderer)

    log_file = save_log(
        config=config,
        runtime=runtime,
        result=result,
        logs_root=_repo_root() / "Logs",
    )

    artifacts = build_output_artifacts(renderer=renderer, log_file=log_file)

    if result.error is not None:
        print(f"Simulation ended with error: {result.error}")
        if result.traceback_text:
            print(result.traceback_text)

    if result.collision_event is not None:
        ce = result.collision_event
        print(
            "Collision detected | "
            f"iter={ce.iteration} obs_idx={ce.obstacle_index} "
            f"dist={ce.distance_to_center:.3f} threshold={ce.collision_threshold:.3f}"
        )

    if result.reached_goal:
        print("Goal reached.")
    else:
        print("Goal not reached.")

    if artifacts.log_file is not None:
        print(f"Log file: {artifacts.log_file}")
    if artifacts.fig_dir is not None:
        print(f"Frames directory: {artifacts.fig_dir}")

    if renderer is not None:
        renderer.show()

    return 0 if result.error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())

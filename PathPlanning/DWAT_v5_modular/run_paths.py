"""Where a run's output goes.

One run == one directory, holding everything that run produced:

    runs/<version>/<group>/<scenario>__<timestamp>/
        run.json          what was run (versions, paths, resolved config)
        log_details.json  the simulation log
        frames/           frame_0.png, frame_1.png, ...
        video.mp4         assembled afterwards by ffmpeg

This replaces the older split where frames went to
``DWAT_v5_modular/results/figs_<algo>-modular-<scenario>/`` (no timestamp, so a
rerun overwrote them) while the log went to ``Logs/<same tag>_<timestamp>/``
(timestamped, so a rerun added one). Pairing a video back to the numbers that
produced it was guesswork; now they sit in the same folder.

The three path components come from what you already pass on the command line:

    --algorithm-config configs/algorithm/v20.7.8-2.json   -> version
    --scenario         configs/scenarios/ow_sequence.json -> scenario
    scenario JSON's "group" key                           -> group
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Used when a scenario JSON does not declare a "group".
DEFAULT_GROUP = "misc"

ILLEGAL_PATH_CHARS = '\\/:*?"<>|'


def safe_component(name: str, fallback: str = "unnamed") -> str:
    """Make one path component safe to store on Windows / OneDrive.

    A leading or trailing space breaks OneDrive sync, and NTFS silently drops a
    trailing one -- so two names the code treats as different would land in the
    same directory. Strip both rather than trust config authors.
    """
    cleaned = "".join(
        "_" if (c in ILLEGAL_PATH_CHARS or ord(c) < 32) else c for c in name
    )
    cleaned = cleaned.strip().rstrip(".")
    return cleaned or fallback


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_runs_root() -> Path:
    return _repo_root() / "runs"


def version_from_config(algorithm_config_path: Path) -> str:
    """Derive the version path from where the algorithm config sits.

    The config tree and the run tree use the same shape, so a config's location
    is its version::

        configs/algorithm/v20.9.json            -> v20.9
        configs/algorithm/v20.7.8-2/predT10.json -> v20.7.8-2/predT10

    A config outside the tree falls back to the older flat convention, where a
    single underscore marked a variant (``v20.7.8-2_predT10`` ->
    ``v20.7.8-2/predT10``), so paths written before the grouping still map to
    the same run directory they always did.
    """
    path = Path(algorithm_config_path).resolve()
    root = (Path(__file__).resolve().parent / "configs" / "algorithm").resolve()
    try:
        rel = path.relative_to(root)
    except ValueError:
        stem = path.stem
        parent, sep, variant = stem.partition("_")
        return f"{parent}/{variant}" if sep and variant else stem
    return rel.with_suffix("").as_posix()


def group_from_scenario(scenario: Dict[str, Any]) -> str:
    """Read the family label out of a scenario dict."""
    group = str(scenario.get("group", "") or "").strip().strip("/")
    return safe_component(group, DEFAULT_GROUP) if group else DEFAULT_GROUP


def build_run_dir(
    runs_root: Path,
    version: str,
    group: str,
    scenario_name: str,
    timestamp: Optional[str] = None,
) -> Path:
    """Return (and create) the directory for one run.

    The timestamp lives in the leaf name rather than in a level of its own, so
    a scenario's reruns sort together in one listing instead of each hiding
    behind its own folder.
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Sanitise each level separately: version may legitimately contain "/"
    # (v20.7.8-2/predT10), so it cannot be cleaned as a single component.
    version = "/".join(safe_component(part) for part in str(version).split("/") if part)
    group = safe_component(group, DEFAULT_GROUP)
    scenario_name = safe_component(scenario_name)

    run_dir = Path(runs_root) / version / group / f"{scenario_name}__{timestamp}"

    # A second run inside the same clock second would otherwise collide.
    if run_dir.exists():
        suffix = 2
        while True:
            candidate = run_dir.with_name(f"{run_dir.name}-{suffix}")
            if not candidate.exists():
                run_dir = candidate
                break
            suffix += 1

    run_dir.mkdir(parents=True)
    return run_dir


def write_run_manifest(
    run_dir: Path,
    version: str,
    group: str,
    scenario_name: str,
    scenario_path: Path,
    algorithm_config_path: Path,
    run_config_path: Path,
    config: Any,
    runtime: Any,
    road_map: Any = None,
) -> Path:
    """Record what was run, so the folder is self-describing.

    Paths are stored repo-relative where possible; an absolute path outside the
    repo is kept as-is rather than forced into a misleading relative form.

    ``road_map`` is the global planner's output -- the polyline the ego is
    meant to follow. Without it a run cannot be scored for cross-track error
    except by guessing the path, which is how ``|x - 50|`` came to be used as
    "cross-track" on a route that runs (70,90) -> (60,10). It is a handful of
    vertices, so it costs nothing to keep.
    """
    root = _repo_root()

    def rel(path: Path) -> str:
        try:
            return str(Path(path).resolve().relative_to(root))
        except ValueError:
            return str(path)

    manifest = {
        "version": version,
        "group": group,
        "scenario": scenario_name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "scenario": rel(scenario_path),
            "algorithm_config": rel(algorithm_config_path),
            "run_config": rel(run_config_path),
        },
        "config": _attrs(config),
        "runtime": _attrs(runtime),
    }
    if road_map is not None:
        manifest["road_map"] = _jsonable(road_map)

    manifest_path = run_dir / "run.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def _attrs(obj: Any) -> Dict[str, Any]:
    """Every public, non-callable attribute -- class-level ones included.

    ``vars()`` alone would miss ``Config.robot_type``, which is declared on the
    class rather than assigned in ``__init__``; log_details.json has always had
    that gap. run.json is meant to be a complete record of what was run, so it
    reads the class attributes too.
    """
    out = {}
    for name in dir(obj):
        if name.startswith("_"):
            continue
        value = getattr(obj, name, None)
        if callable(value):
            continue
        out[name] = _jsonable(value)
    return out


def _jsonable(value: Any) -> Any:
    """Coerce config values (enums, numpy scalars/arrays) into plain JSON."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "name") and hasattr(value, "value"):  # enum.Enum
        return value.name
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)

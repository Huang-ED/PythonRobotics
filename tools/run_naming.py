"""Parse the historical ``figs_*`` run names into (version, scenario, timestamp).

Every run this project has ever produced was named by one of two schemes:

    figs_<version>-modular-<scenario>[_<timestamp>]   the v5 modular runner
    figs_<version>-<scenario>[_<timestamp>]           everything before it

The second is ambiguous on its own, because both the version and the scenario
may contain a hyphen::

    figs_v11.10.4-scenario1-1     -> version v11.10.4,   scenario scenario1-1
    figs_v20.7.8-2-video3         -> version v20.7.8-2,  scenario video3

They are told apart by what follows the first hyphen: a bare integer belongs to
the version (it is a revision suffix), anything else starts the scenario name.
"""

import re
from pathlib import Path
from typing import NamedTuple, Optional, Set

ILLEGAL_PATH_CHARS = '\\/:*?"<>|'

TIMESTAMP_RE = re.compile(r"_(\d{8}_\d{6})$")
VERSION_ONLY_RE = re.compile(r"^v[\d.]+$")
MODULAR_SEP = "-modular-"

# Used when a run name records a version but never named its scenario.
UNNAMED_SCENARIO = "unnamed"


def safe_component(name: str, fallback: str = "unnamed") -> str:
    """Make one path component safe to store on Windows / OneDrive.

    Historical run names were not written with that in mind:
    ``figs_temp - Compare`` splits into version ``"temp "`` and scenario
    ``" Compare"``, and a leading or trailing space is not storable. NTFS
    silently *drops* a trailing space, so ``temp /`` and ``temp/`` become the
    same directory while the code still believes they differ; a leading space
    survives instead and breaks OneDrive sync. Both are stripped here so the
    two failure modes cannot arise.
    """
    cleaned = "".join(
        "_" if (c in ILLEGAL_PATH_CHARS or ord(c) < 32) else c for c in name
    )
    cleaned = cleaned.strip().rstrip(".")
    return cleaned or fallback


class RunName(NamedTuple):
    version: str          # e.g. "v20.7.8-2" or "v20.7.8-2_predT10"
    scenario: str         # e.g. "owenc_head_on"
    timestamp: Optional[str]
    modular: bool         # produced by the v5 runner


def parse_run_name(name: str) -> Optional[RunName]:
    """Return the parts of a ``figs_*`` directory name, or None if it is junk."""
    if not name.startswith("figs_"):
        return None
    body = name[len("figs_"):]

    timestamp = None
    m = TIMESTAMP_RE.search(body)
    if m:
        timestamp = m.group(1)
        body = body[: m.start()]

    if MODULAR_SEP in body:
        version, scenario = body.split(MODULAR_SEP, 1)
        return RunName(safe_component(version), safe_component(scenario), timestamp, True)

    if "-" not in body:
        # Older LocalGlobal names separate with "_" (figs_v7.1_video1) or carry
        # no scenario at all (figs_v7.3.1). Keep both rather than drop them.
        if VERSION_ONLY_RE.match(body):
            return RunName(safe_component(body), UNNAMED_SCENARIO, timestamp, False)
        head, sep, tail = body.partition("_")
        if sep and VERSION_ONLY_RE.match(head):
            return RunName(safe_component(head), safe_component(tail), timestamp, False)
        return RunName(safe_component(body), UNNAMED_SCENARIO, timestamp, False)

    tokens = body.split("-")
    version_tokens = [tokens[0]]
    i = 1
    # A bare integer after the version is a revision suffix (v20.7.8-2),
    # not the start of the scenario name (v11.10.4-scenario1-1).
    while i < len(tokens) and tokens[i].isdigit():
        version_tokens.append(tokens[i])
        i += 1
    if i >= len(tokens):
        return RunName(safe_component("-".join(version_tokens)), UNNAMED_SCENARIO, timestamp, False)
    return RunName(safe_component("-".join(version_tokens)),
                   safe_component("-".join(tokens[i:])), timestamp, False)


def version_dir(version: str, known_parents: Set[str]) -> str:
    """Nest a variant config under its parent, when the parent really exists.

    ``v20.7.8-2_predT10`` -> ``v20.7.8-2/predT10`` because ``v20.7.8-2`` is a
    real algorithm config. ``P3_predT10_v1.0`` is left whole, because ``P3`` is
    not -- splitting there would invent a parent that never existed.
    """
    parent, sep, variant = version.partition("_")
    if sep and variant and parent in known_parents:
        return f"{parent}/{variant}"
    return version


def known_algorithm_versions(repo_root: Path) -> Set[str]:
    d = repo_root / "PathPlanning/DWAT_v5_modular/configs/algorithm"
    return {p.stem for p in d.glob("*.json")} if d.is_dir() else set()


def group_for(scenario: str) -> str:
    """The family a scenario belongs to; mirrors the "group" key in new configs."""
    if scenario.startswith(("owenc_", "v3enc_")):
        return "encounters"
    if scenario in ("ow_sequence", "ow_simultaneous", "v3seq"):
        return "combined"
    if scenario.startswith("scene_export"):
        return "imported"
    if scenario.startswith("video"):
        return "video"
    if scenario.startswith("openwater"):
        return "baseline"
    if scenario.startswith("scenario"):
        return "synthetic"
    return "misc"

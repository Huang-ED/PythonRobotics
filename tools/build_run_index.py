"""Regenerate runs/INDEX.md -- a searchable table of every run on disk.

Folders answer "show me this version's encounter runs". They do not answer
"which runs collided" or "which ones never reached the goal", because that
lives inside each log. This walks the tree once and writes those answers to a
single file, so finding a run is a text search rather than a click hunt.

Usage::

    python tools/build_run_index.py
    python tools/build_run_index.py --runs-root runs --out runs/INDEX.md
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

MISSING = "-"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _human(size: int) -> str:
    for unit in ("B", "K", "M", "G"):
        if size < 1024 or unit == "G":
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size:.1f}G"


def read_outcome(run_dir: Path) -> Dict[str, Any]:
    """Pull the summary fields out of a run's log, tolerating older formats."""
    log = run_dir / "log_details.json"
    out = {"outcome": MISSING, "steps": MISSING}
    if not log.is_file():
        return out
    try:
        with log.open("r", encoding="utf-8") as f:
            doc = json.load(f)
    except (json.JSONDecodeError, OSError, MemoryError):
        out["outcome"] = "unreadable"
        return out

    if not isinstance(doc, dict):
        return out

    entries = doc.get("log_entries")
    if isinstance(entries, list):
        out["steps"] = str(len(entries))

    # Logs written before these keys existed simply omit them.
    if doc.get("error"):
        out["outcome"] = "error"
    elif doc.get("collision_event"):
        out["outcome"] = "collision"
    elif doc.get("reached_goal") is True:
        out["outcome"] = "goal"
    elif doc.get("reached_goal") is False:
        out["outcome"] = "no goal"
    return out


def scan(runs_root: Path, with_sizes: bool = False):
    """Yield one record per run directory (a leaf holding a log, frames or video).

    Walks with os.walk and prunes ``frames/`` rather than using rglob: a full
    recursive listing visits every frame PNG -- hundreds of thousands of files
    across the archive -- and that dominates the runtime on this mount.
    """
    for dirpath, dirnames, filenames in os.walk(runs_root):
        path = Path(dirpath)
        names = set(filenames)
        has_log = "log_details.json" in names
        has_frames = "frames" in dirnames
        # Two videos, not interchangeable (see tools/render_run_video.py):
        # video.mp4 is built from the run's own frames/, video_from_log.mp4 is
        # reconstructed afterwards from log_details.json.
        has_video = "video.mp4" in names
        has_log_video = "video_from_log.mp4" in names

        if has_frames:
            dirnames.remove("frames")     # never descend into a run's frames

        if not (has_log or has_frames or has_video or has_log_video):
            continue
        dirnames[:] = []                  # a run directory holds no run directories

        rel = path.relative_to(runs_root)
        parts = rel.parts
        leaf = parts[-1] if parts else ""
        group = parts[-2] if len(parts) >= 2 else MISSING
        version = "/".join(parts[:-2]) if len(parts) >= 3 else MISSING
        scenario, _, timestamp = leaf.rpartition("__")
        if not scenario:
            scenario, timestamp = leaf, MISSING

        artifacts = []
        if has_log:
            artifacts.append("log")
        if has_frames:
            n = sum(1 for _ in (path / "frames").glob("*.png"))
            artifacts.append(f"frames({n})")
        if has_video:
            artifacts.append("video")
        if has_log_video:
            artifacts.append("video_from_log")

        rec = {
            "version": version,
            "group": group,
            "scenario": scenario,
            "timestamp": timestamp or MISSING,
            "artifacts": " ".join(artifacts),
            # Sizing stats every frame of every run; opt in when you want it.
            "size": _human(_dir_size(path)) if with_sizes else MISSING,
            "path": rel.as_posix(),
        }
        rec.update(read_outcome(path))
        yield rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--out", default=None, help="default: <runs-root>/INDEX.md")
    ap.add_argument("--sizes", action="store_true",
                    help="measure each run's size (slow: stats every frame)")
    args = ap.parse_args()

    root = repo_root()
    runs_root = (root / args.runs_root).resolve()
    if not runs_root.is_dir():
        print(f"no such directory: {runs_root}")
        return 1
    out_path = Path(args.out).resolve() if args.out else runs_root / "INDEX.md"

    records = list(scan(runs_root, with_sizes=args.sizes))
    current = [r for r in records if not r["path"].startswith("_archive/")]
    archived = [r for r in records if r["path"].startswith("_archive/")]

    lines = [
        "# Run index",
        "",
        f"{len(records)} runs "
        f"({len(current)} current, {len(archived)} archived). "
        "Regenerate with `python tools/build_run_index.py`.",
        "",
        "`outcome` comes from each run's `log_details.json`; `-` means the log "
        "predates that field or is absent.",
        "",
    ]

    def table(title: str, rows) -> None:
        lines.append(f"## {title}")
        lines.append("")
        if not rows:
            lines.extend(["_none_", ""])
            return
        lines.append("| version | group | scenario | timestamp | outcome | steps | artifacts | size | path |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            lines.append(
                f"| {r['version']} | {r['group']} | {r['scenario']} | {r['timestamp']} "
                f"| {r['outcome']} | {r['steps']} | {r['artifacts']} | {r['size']} | `{r['path']}` |"
            )
        lines.append("")

    table("Current", sorted(current, key=lambda r: (r["version"], r["group"], r["scenario"], r["timestamp"])))
    table("Archive", sorted(archived, key=lambda r: (r["path"])))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_path.relative_to(root)}  ({len(records)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

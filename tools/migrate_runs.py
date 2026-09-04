"""Move historical run output into the runs/ tree.

Before: output for one run was split across two places, and the frame half had
no timestamp, so reruns overwrote frames while adding logs::

    PathPlanning/DWAT_v5_modular/results/figs_v20.7.8-2-modular-ow_sequence/
    Logs/figs_v20.7.8-2-modular-ow_sequence_20260904_152000/

After: one run, one directory::

    runs/v20.7.8-2/combined/ow_sequence__20260904_152000/
        log_details.json
        frames/
        video.mp4

Runs from the current (v5 modular) pipeline land in ``runs/``. Everything older
lands under ``runs/_archive/<generation>/`` in the same version/group/run shape,
so the archive browses identically instead of being a dumping ground.

Frame directories carry no timestamp, so they are attached to the *latest* log
sharing their version and scenario: frames are overwritten on rerun, so the
surviving PNGs are from the most recent run. Where no log matches, the run
directory is suffixed ``__notime``. Every such decision is written to the
manifest, so nothing is silently guessed.

Usage::

    python tools/migrate_runs.py --dry-run     # plan only, touches nothing
    python tools/migrate_runs.py --apply       # perform the moves
"""

import argparse
import csv
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_naming import (  # noqa: E402
    UNNAMED_SCENARIO,
    group_for,
    known_algorithm_versions,
    parse_run_name,
    version_dir,
)

# (source directory, which runs to take, destination root)
#   "modular"    -- produced by the v5 runner (name contains -modular-)
#   "premodular" -- everything else in that source
#   "all"        -- every figs* entry
TIERS = [
    ("Logs",                                 "modular",    "runs"),
    ("PathPlanning/DWAT_v5_modular/results", "all",        "runs"),
    ("Logs",                                 "premodular", "runs/_archive/v4_st"),
    ("PathPlanning/DWAT_v4_st",              "all",        "runs/_archive/v4_st"),
    ("Logs_split",                           "all",        "runs/_archive/v3_split"),
    ("PathPlanning/DWAT_v3_split",           "all",        "runs/_archive/v3_split"),
    ("PathPlanning/DWAT_v2_split",           "all",        "runs/_archive/v2_split"),
    ("PathPlanning/DWAT_v3",                 "all",        "runs/_archive/v3"),
    ("PathPlanning/DWAT_v2",                 "all",        "runs/_archive/v2"),
    ("PathPlanning/DWAT",                    "all",        "runs/_archive/v1"),
    ("PathPlanning/LocalGlobal",             "all",        "runs/_archive/local_global"),
]

# Sources whose naming carries no scenario at all; moved verbatim.
WHOLESALE = [("Logs_LG", "runs/_archive/local_global/logs")]

# Loose helper scripts sitting among the run folders.
SCRIPT_DEST = "tools/legacy"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def collect(root: Path):
    """Build the move plan: list of (source, destination, kind, note)."""
    known = known_algorithm_versions(root)
    # key -> {"logs": [(ts, path)], "frames": [path], "videos": [path]}
    registry = defaultdict(lambda: {"logs": [], "frames": [], "videos": []})

    for src_name, which, dest_root in TIERS:
        src = root / src_name
        if not src.is_dir():
            continue
        for entry in sorted(os.listdir(src)):
            if not entry.startswith("figs"):
                continue
            path = src / entry
            is_video = entry.endswith(".mp4")
            base = entry[:-4] if is_video else entry
            parsed = parse_run_name(base)
            if parsed is None:
                continue
            if which == "modular" and not parsed.modular:
                continue
            if which == "premodular" and parsed.modular:
                continue

            key = (dest_root, version_dir(parsed.version, known), parsed.scenario)
            if is_video:
                registry[key]["videos"].append(path)
            elif parsed.timestamp:
                registry[key]["logs"].append((parsed.timestamp, path))
            elif path.is_dir():
                registry[key]["frames"].append(path)

    moves = []
    for (dest_root, version, scenario), items in sorted(registry.items()):
        group = group_for(scenario)
        base = root / dest_root / version / group

        logs = sorted(items["logs"])
        for ts, log_path in logs:
            run_dir = base / f"{scenario}__{ts}"
            # A log directory's *contents* become the run directory's contents.
            for child in sorted(os.listdir(log_path)):
                moves.append((log_path / child, run_dir / child, "log", ""))

        # Frames and videos have no timestamp of their own.
        latest_ts = logs[-1][0] if logs else None
        note = "" if latest_ts else "no matching log"
        if latest_ts and len(logs) > 1:
            note = f"attached to latest of {len(logs)} logs"
        target = base / f"{scenario}__{latest_ts}" if latest_ts else base / f"{scenario}__notime"

        for frames in items["frames"]:
            moves.append((frames, target / "frames", "frames", note))
        for video in items["videos"]:
            moves.append((video, target / "video.mp4", "video", note))

    for src_name, dest in WHOLESALE:
        src = root / src_name
        if src.is_dir():
            for entry in sorted(os.listdir(src)):
                moves.append((src / entry, root / dest / entry, "wholesale", ""))

    for src_name in ("Logs", "Logs_split"):
        src = root / src_name
        if not src.is_dir():
            continue
        for entry in sorted(os.listdir(src)):
            if entry.endswith(".py"):
                moves.append((src / entry, root / SCRIPT_DEST / entry, "script", ""))

    return moves


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="print the plan, move nothing")
    mode.add_argument("--apply", action="store_true", help="perform the moves")
    ap.add_argument("--manifest", default="tools/migration_manifest.csv",
                    help="where to record every move (for reversal)")
    args = ap.parse_args()

    root = repo_root()
    moves = collect(root)

    kinds = defaultdict(int)
    for _, _, kind, _ in moves:
        kinds[kind] += 1
    print(f"planned moves: {len(moves)}  {dict(kinds)}")

    notes = [(s, d, n) for s, d, _, n in moves if n]
    if notes:
        print(f"\ndecisions worth recording ({len(notes)}):")
        for s, d, n in notes[:15]:
            print(f"  {n:32s} {s.relative_to(root)}")

    if args.dry_run:
        print("\nsample of the resulting tree:")
        for _, dest, _, _ in moves[:12]:
            print(f"  {dest.relative_to(root)}")
        print("\n(dry run -- nothing was moved)")
        return 0

    manifest_path = root / args.manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    moved = skipped = 0
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "destination", "kind", "note"])
        for src, dest, kind, note in moves:
            if not src.exists():
                skipped += 1
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                print(f"  SKIP (destination exists): {dest.relative_to(root)}")
                skipped += 1
                continue
            shutil.move(str(src), str(dest))
            w.writerow([str(src.relative_to(root)), str(dest.relative_to(root)), kind, note])
            moved += 1

    print(f"\nmoved {moved}, skipped {skipped}")
    print(f"manifest: {manifest_path.relative_to(root)}")

    # Moving a log directory's contents leaves the wrapper behind; drop it.
    for src_name, _, _ in TIERS:
        src = root / src_name
        if not src.is_dir():
            continue
        for entry in sorted(os.listdir(src)):
            d = src / entry
            if d.is_dir() and entry.startswith("figs") and not os.listdir(d):
                d.rmdir()

    # Remove the now-empty shells left behind.
    for src_name, _, _ in TIERS:
        p = root / src_name
        if p.is_dir() and not os.listdir(p):
            p.rmdir()
            print(f"  removed empty {src_name}")
    for src_name, _ in WHOLESALE:
        p = root / src_name
        if p.is_dir() and not os.listdir(p):
            p.rmdir()
            print(f"  removed empty {src_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

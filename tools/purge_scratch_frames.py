"""Remove orphaned scratch frame directories left by killed runs.

``--frames-root`` deletes its own directory in a ``finally``, but a run killed
with SIGKILL never runs it. This sweeps whatever is left.

    python tools/purge_scratch_frames.py /tmp/dwat-frames            # list
    python tools/purge_scratch_frames.py /tmp/dwat-frames --delete   # remove
    python tools/purge_scratch_frames.py /tmp/dwat-frames --delete --older-than 2

Refuses to touch anything under the repository, so it can never be pointed at
runs/ by accident.
"""
import argparse
import shutil
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="the frames_root that was passed to --frames-root")
    ap.add_argument("--delete", action="store_true", help="actually remove (default: list only)")
    ap.add_argument("--older-than", type=float, default=0.0,
                    help="only those untouched for this many hours (default: all)")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    repo = Path(__file__).resolve().parents[1]
    if root == repo or repo in root.parents or root in repo.parents:
        print(f"refusing: {root} is inside or contains the repository at {repo}")
        return 2
    if not root.is_dir():
        print(f"nothing to do: {root} does not exist")
        return 0

    cutoff = time.time() - args.older_than * 3600.0
    total_n = total_b = 0
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        pngs = list(d.glob("*.png"))
        if not pngs:
            continue
        if args.older_than and max(p.stat().st_mtime for p in pngs) > cutoff:
            continue
        size = sum(p.stat().st_size for p in pngs)
        total_n += len(pngs)
        total_b += size
        print(f"  {'removing' if args.delete else 'would remove'} {d}  "
              f"({len(pngs)} png, {size/1e6:.1f} MB)")
        if args.delete:
            shutil.rmtree(d, ignore_errors=True)
    verb = "removed" if args.delete else "would remove"
    print(f"{verb} {total_n} frames, {total_b/1e6:.1f} MB"
          + ("" if args.delete else "   (pass --delete to do it)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

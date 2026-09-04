"""Find path names OneDrive / Windows cannot store.

This tree lives inside OneDrive, so a name that is legal on Linux can still
break sync or, worse, be silently altered. The two that actually bit this
repository:

  - A **trailing space** is silently dropped by NTFS, so ``temp /`` and
    ``temp/`` become the same directory while the code still believes they are
    different. Nothing errors; files just land somewhere unexpected.
  - A **leading space** survives on disk but stops OneDrive syncing that item,
    and it is easy to miss in a directory listing.

Both came from one historical run folder, ``figs_temp - Compare``, which parsed
into version ``"temp "`` and scenario ``" Compare"``.

Usage::

    python tools/check_names.py              # scan the repo, exit 1 on any hit
    python tools/check_names.py --fix        # rename offenders to a safe form
    python tools/check_names.py --root runs  # scan one subtree
"""

import argparse
import os
import sys
from pathlib import Path

ILLEGAL_CHARS = set('\\/:*?"<>|')
RESERVED = (
    {"CON", "PRN", "AUX", "NUL", "CLOCK$"}
    | {f"COM{i}" for i in range(10)}
    | {f"LPT{i}" for i in range(10)}
)
SKIP_DIRS = {".git"}


def problems(name: str):
    """Every reason this single path component is unsafe on Windows/OneDrive."""
    found = []
    bad = sorted({c for c in name if c in ILLEGAL_CHARS or ord(c) < 32})
    if bad:
        found.append("illegal character " + " ".join(repr(c) for c in bad))
    if name[:1].isspace():
        found.append("leading space")
    if name[-1:].isspace():
        found.append("trailing space (NTFS drops this silently)")
    if name.endswith(".") and name not in (".", ".."):
        found.append("trailing dot")
    if name.split(".")[0].upper() in RESERVED:
        found.append("reserved device name")
    if name.startswith("~$"):
        found.append("leading ~$ (treated as an Office lock file)")
    if name.lower() == ".lock" or "_vti_" in name.lower():
        found.append("name reserved by OneDrive")
    if len(name.encode("utf-8")) > 255:
        found.append("component longer than 255 bytes")
    return found


def safe_name(name: str, fallback: str = "unnamed") -> str:
    cleaned = "".join(
        "_" if (c in ILLEGAL_CHARS or ord(c) < 32) else c for c in name
    )
    cleaned = cleaned.strip().rstrip(".")
    return cleaned or fallback


def scan(root: Path):
    """Yield (parent, name, is_dir, reasons), deepest first.

    Deepest-first matters for --fix: renaming a parent before its children
    would invalidate the child paths still queued.
    """
    hits = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in dirnames:
            r = problems(name)
            if r:
                hits.append((Path(dirpath), name, True, r))
        for name in filenames:
            r = problems(name)
            if r:
                hits.append((Path(dirpath), name, False, r))
    hits.sort(key=lambda h: len(h[0].parts), reverse=True)
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default=".", help="subtree to scan (default: repo root)")
    ap.add_argument("--fix", action="store_true", help="rename offenders to a safe form")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"no such directory: {root}")
        return 2

    hits = scan(root)
    if not hits:
        print(f"{root}: no unsafe names found")
        return 0

    print(f"{len(hits)} unsafe name(s) under {root}:\n")
    renamed = failed = 0
    for parent, name, is_dir, reasons in hits:
        kind = "dir " if is_dir else "file"
        print(f"  [{kind}] {parent / name}")
        for r in reasons:
            print(f"          - {r}")

        if not args.fix:
            continue

        target = safe_name(name)
        if target == name:
            print("          ! cannot derive a safe name; rename by hand")
            failed += 1
            continue
        dest = parent / target
        if dest.exists():
            print(f"          ! {target!r} already exists; rename by hand")
            failed += 1
            continue
        os.rename(parent / name, dest)
        print(f"          -> renamed to {target!r}")
        renamed += 1

    if args.fix:
        print(f"\nrenamed {renamed}, needing manual attention {failed}")
        return 1 if failed else 0
    print("\nRe-run with --fix to rename these automatically.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Mirror reco/ and decode/ ROOT files from the CI_build_lar_ci_* DQM dCache
area to a local directory via xrootd (xrdcp), preserving the source
directory structure.

Source layout (one directory per CI build), e.g. under
/pnfs/sbnd/scratch/ci_validation/dqm/v09_93_01_02/:
    CI_build_lar_ci_<N>/reco/*.root
    CI_build_lar_ci_<N>/decode/*.root
Subdirectory names are configurable via --subdirs in case your area actually
uses e.g. raw_decode instead of decode (this project's data/README.md
documents raw_decode elsewhere -- confirm which your v09_93_01_02 area
actually has with `ls` before running for real).

Listing/globbing under /pnfs is normal POSIX filesystem access (dCache's
NFS4 namespace mount -- `ls`/`find`/glob work fine there); only the actual
bulk file copy goes through xrootd (xrdcp), which is the correct way to move
real data off dCache rather than a plain `cp`.

IMPORTANT -- you must supply the xrootd door/redirector for your site
(--xrootd-door). I have no way to confirm the current correct value for your
dCache instance from here -- check your experiment's data-handling docs or
an existing xrdcp invocation from a colleague's script/job. A common
Fermilab dCache URL *format* looks like:
    --xrootd-door root://fndca1.fnal.gov:1094
but treat that as an example of the FORMAT only, not a confirmed working
hostname -- verify it yourself (e.g. `xrdcp <that-door>//pnfs/sbnd/... /tmp/test.root`
on one small file) before pointing this at the full dataset.

Re-running the same command is safe and resumes automatically: files already
present at the destination are skipped unless --overwrite is given, so a
partially-failed run can just be re-launched.

Usage
-----
Dry run first -- lists what WOULD be copied and the exact xrdcp commands,
copies nothing:
    python scripts/xrootd_mirror_ci_data.py \\
        --source-glob '/pnfs/sbnd/scratch/ci_validation/dqm/v09_93_01_02/CI_build_lar_ci*' \\
        --dest /exp/sbnd/data/users/<you>/DQM/ci_mirror \\
        --xrootd-door root://fndca1.fnal.gov:1094 \\
        --dry-run

Then for real:
    python scripts/xrootd_mirror_ci_data.py \\
        --source-glob '/pnfs/sbnd/scratch/ci_validation/dqm/v09_93_01_02/CI_build_lar_ci*' \\
        --dest /exp/sbnd/data/users/<you>/DQM/ci_mirror \\
        --xrootd-door root://fndca1.fnal.gov:1094

Only the raw_decode-style subdirectory, more parallel transfers:
    python scripts/xrootd_mirror_ci_data.py \\
        --source-glob '/pnfs/sbnd/scratch/ci_validation/dqm/v09_93_01_02/CI_build_lar_ci*' \\
        --dest /exp/sbnd/data/users/<you>/DQM/ci_mirror \\
        --xrootd-door root://fndca1.fnal.gov:1094 \\
        --subdirs raw_decode --max-workers 8

Destination layout mirrors the source: for a matched top-level directory
CI_build_lar_ci_12 under the common parent of --source-glob's matches, files
land at <dest>/CI_build_lar_ci_12/reco/... and <dest>/CI_build_lar_ci_12/decode/....
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger("xrootd_mirror_ci_data")


def discover_files(
    source_glob: str, subdirs: List[str], exclude_dirs: Optional[List[str]] = None,
) -> List[Path]:
    """Find every .root file under <matched top dir>/<subdir>/ (recursively),
    for every top-level directory `source_glob` expands to.

    Files under any directory component named in `exclude_dirs` (default:
    just "log") are skipped -- e.g. <subdir>/log/whatever.root, or a nested
    .../reco/some_job/log/foo.root, are excluded regardless of how deep the
    "log" component sits in the path.
    """
    exclude = set(exclude_dirs) if exclude_dirs is not None else {"log"}
    top_dirs = sorted({Path(p) for p in glob.glob(source_glob) if Path(p).is_dir()})
    if not top_dirs:
        raise ValueError(f"No directories matched {source_glob!r}")
    logger.info("Matched %d top-level directories", len(top_dirs))

    files: List[Path] = []
    for top in top_dirs:
        for sub in subdirs:
            subdir = top / sub
            if not subdir.is_dir():
                logger.warning("  %s: no '%s' subdirectory -- skipping", top, sub)
                continue
            found = sorted(subdir.rglob("*.root"))
            kept = [
                f for f in found
                if not (exclude & set(f.relative_to(subdir).parts[:-1]))
            ]
            n_excluded = len(found) - len(kept)
            if n_excluded:
                logger.info("  %s/%s: excluding %d file(s) under %s",
                            top.name, sub, n_excluded, sorted(exclude))
            logger.info("  %s/%s: %d .root file(s)", top.name, sub, len(kept))
            files.extend(kept)
    return files


def common_source_root(source_glob: str) -> Path:
    """The shared parent of every directory source_glob matches -- mirrored
    paths under --dest start at the matched directory name (e.g.
    CI_build_lar_ci_12/...), not the full /pnfs/... prefix.
    """
    top_dirs = sorted({Path(p) for p in glob.glob(source_glob) if Path(p).is_dir()})
    if not top_dirs:
        raise ValueError(f"No directories matched {source_glob!r}")
    return Path(os.path.commonpath([str(p.parent) for p in top_dirs]))


def dest_path_for(source_file: Path, source_root: Path, dest_root: Path) -> Path:
    """Mirror source_file's path relative to source_root under dest_root."""
    rel = source_file.relative_to(source_root)
    return dest_root / rel


def xrootd_url(local_pnfs_path: Path, xrootd_door: str) -> str:
    """Absolute /pnfs/... path -> root://door/pnfs/... URL.

    dCache convention: the xrootd URL's path component is the same absolute
    pnfs namespace path used for POSIX access -- only the root://door:port/
    prefix changes.
    """
    door = xrootd_door.rstrip("/")
    path = str(local_pnfs_path)
    if not path.startswith("/"):
        path = "/" + path
    return f"{door}/{path.lstrip('/')}"


def copy_one(
    source_file: Path, dest_file: Path, xrootd_door: str, overwrite: bool, dry_run: bool,
) -> Tuple[Path, bool, str]:
    """Copy one file via xrdcp. Returns (source_file, ok, message)."""
    if dest_file.exists() and not overwrite:
        return (source_file, True, "skipped (already exists)")
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    src_url = xrootd_url(source_file, xrootd_door)
    cmd = ["xrdcp"]
    if overwrite:
        cmd.append("-f")
    cmd += [src_url, str(dest_file)]
    if dry_run:
        return (source_file, True, f"[dry-run] {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    except FileNotFoundError:
        return (source_file, False, "xrdcp not found on PATH -- is XRootD client set up in this environment?")
    except subprocess.TimeoutExpired:
        return (source_file, False, "timed out after 3600s")
    if result.returncode != 0:
        return (source_file, False, f"xrdcp failed (rc={result.returncode}): {result.stderr.strip()[-500:]}")
    return (source_file, True, "copied")


def mirror(
    source_glob: str,
    dest: str,
    xrootd_door: str,
    subdirs: List[str],
    overwrite: bool,
    dry_run: bool,
    max_workers: int,
    exclude_dirs: Optional[List[str]] = None,
) -> None:
    files = discover_files(source_glob, subdirs, exclude_dirs=exclude_dirs)
    if not files:
        logger.warning("No .root files found under any matched directory's %s -- nothing to do.", subdirs)
        return

    source_root = common_source_root(source_glob)
    dest_root = Path(dest)
    logger.info("%d file(s) to mirror from %s to %s", len(files), source_root, dest_root)
    if dry_run:
        logger.info("DRY RUN -- no files will actually be copied")

    n_ok = n_fail = n_skip = 0
    failed: List[Tuple[Path, str]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(copy_one, f, dest_path_for(f, source_root, dest_root), xrootd_door, overwrite, dry_run): f
            for f in files
        }
        for i, fut in enumerate(as_completed(futures), 1):
            src, ok, msg = fut.result()
            if ok and msg.startswith("skipped"):
                n_skip += 1
            elif ok:
                n_ok += 1
                if dry_run:
                    logger.info("  %s", msg)
            else:
                n_fail += 1
                failed.append((src, msg))
                logger.error("FAILED: %s -- %s", src, msg)
            if i % 25 == 0 or i == len(files):
                logger.info("  progress: %d/%d (%d copied, %d skipped, %d failed)",
                             i, len(files), n_ok, n_skip, n_fail)

    logger.info("Done: %d copied, %d skipped (already existed), %d failed", n_ok, n_skip, n_fail)
    if failed:
        logger.warning(
            "%d file(s) failed -- re-run the exact same command to retry just those "
            "(already-copied files are skipped automatically unless --overwrite).", len(failed))


def _parse_args(argv):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-glob", required=True,
                     help="Glob matching the per-build top-level directories, e.g. "
                          "'/pnfs/sbnd/scratch/ci_validation/dqm/v09_93_01_02/CI_build_lar_ci*'")
    ap.add_argument("--dest", required=True, help="Local destination root directory")
    ap.add_argument("--xrootd-door", required=True,
                     help="e.g. root://fndca1.fnal.gov:1094 -- verify this for your site, see module docstring")
    ap.add_argument("--subdirs", nargs="+", default=["reco", "decode"],
                     help="Subdirectory name(s) under each matched build directory to mirror (default: reco decode)")
    ap.add_argument("--exclude-dirs", nargs="+", default=["log"],
                     help="Directory name(s) to skip anywhere in the path under a subdir, e.g. "
                          "reco/log/foo.root is excluded when 'log' is in this list (default: log)")
    ap.add_argument("--overwrite", action="store_true",
                     help="Re-copy files that already exist at the destination (default: skip them)")
    ap.add_argument("--dry-run", action="store_true",
                     help="List what would be copied and the exact xrdcp commands, copy nothing")
    ap.add_argument("--max-workers", type=int, default=4,
                     help="Parallel xrdcp transfers (default 4 -- network-bound; too high can overload the door)")
    return ap.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    mirror(
        source_glob=args.source_glob, dest=args.dest, xrootd_door=args.xrootd_door,
        subdirs=args.subdirs, overwrite=args.overwrite, dry_run=args.dry_run,
        max_workers=args.max_workers, exclude_dirs=args.exclude_dirs,
    )


if __name__ == "__main__":
    main()

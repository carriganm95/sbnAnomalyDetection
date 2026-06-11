"""Helpers for resolving ROOT file inputs.

Supports inline ROOT paths, glob patterns, and manifest files containing one
ROOT path per line. This is used by the training and inference CLIs so users
can define reusable train/test file lists on disk.
"""

from __future__ import annotations

import os
import glob as _glob
from urllib.parse import urlparse
from typing import Iterable


def resolve_root_files(inputs: Iterable[str]) -> list[str]:
    """Expand ROOT file inputs into a flat list of paths.

    Each element in *inputs* may be:
    - a literal ROOT file path
    - a glob pattern (e.g. ``/data/*.root``)
    - a manifest file (``.txt``/``.lst``/``.csv``) containing one ROOT path
      per line, with ``#`` comments allowed

    Empty lines and comments are ignored. Manifest entries may themselves be
    glob patterns or literal paths.
    """

    resolved: list[str] = []
    for item in inputs:
        resolved.extend(_resolve_single_root_input(item))
    return resolved


def _resolve_single_root_input(item: str) -> list[str]:
    parsed = urlparse(item)

    if not parsed.scheme and os.path.isfile(item) and item.lower().endswith((".txt", ".lst", ".csv")):
        return _resolve_manifest_file(item)

    matches = _glob.glob(item)
    if matches:
        return matches

    return [item]


def _resolve_manifest_file(path: str) -> list[str]:
    resolved: list[str] = []
    with open(path) as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            resolved.extend(_resolve_single_root_input(line))
    return resolved
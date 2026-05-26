"""Shared dataset-IO helpers.

This module exists to centralize cluster-specific I/O quirks (multiple
serializer formats for the same logical data) without forcing every
dataset loader to duplicate the dispatch logic.
"""
from __future__ import annotations

import os
import os.path as osp
from typing import Mapping, Optional

import numpy as np


# Valid values for the per-dataset `data_format` kwarg.
DATA_FORMAT_AUTO = "auto"
DATA_FORMAT_NPZ = "npz"
DATA_FORMAT_SAFETENSOR = "safetensor"
_VALID_FORMATS = {DATA_FORMAT_AUTO, DATA_FORMAT_NPZ, DATA_FORMAT_SAFETENSOR}


def _load_safetensor(path: str) -> Mapping[str, np.ndarray]:
    # Local import keeps the dataset module importable in environments
    # without safetensors (e.g. CPU-only smoke without HF stack installed).
    from safetensors.numpy import load_file
    return load_file(path)


def load_cam_params(
    stem_path: str,
    data_format: str = DATA_FORMAT_AUTO,
) -> Mapping[str, np.ndarray]:
    """Load per-frame camera params, supporting `.npz` and `.safetensor`.

    Args:
        stem_path: Path without extension. The loader probes
                   `<stem>.npz` and `<stem>.safetensor` based on `data_format`.
        data_format: 'auto' (try npz first, fall back to safetensor),
                     'npz' (only npz; raise if missing),
                     'safetensor' (only safetensor; raise if missing).

    Returns:
        Dict-like mapping from key (e.g. 'intrinsics', 'pose') to ndarray.
        Both formats produce structurally equivalent outputs since they
        encode the same logical content.
    """
    if data_format not in _VALID_FORMATS:
        raise ValueError(
            f"data_format must be one of {sorted(_VALID_FORMATS)}, "
            f"got {data_format!r}"
        )

    npz = stem_path + ".npz"
    sft = stem_path + ".safetensor"

    if data_format == DATA_FORMAT_NPZ:
        return np.load(npz, allow_pickle=True)
    if data_format == DATA_FORMAT_SAFETENSOR:
        return _load_safetensor(sft)
    # auto: prefer canonical .npz, fall back to .safetensor.
    if osp.exists(npz):
        return np.load(npz, allow_pickle=True)
    if osp.exists(sft):
        return _load_safetensor(sft)
    raise FileNotFoundError(
        f"camera params: neither {npz} nor {sft} exists "
        f"(data_format={data_format!r})"
    )


def list_subdirs(path: str) -> list[str]:
    """Return sorted names of subdirectories of `path` (one level deep)."""
    try:
        entries = os.listdir(path)
    except (FileNotFoundError, PermissionError):
        return []
    return sorted(
        e for e in entries if osp.isdir(osp.join(path, e))
    )


def walk_nested_scenes(
    root: str,
    depth: int,
    join_sep: str = "/",
) -> list[str]:
    """Enumerate scene ids that live `depth` directory levels under `root`.

    For depth=1 returns ['scene1', 'scene2', ...] -- the upstream "flat" pattern.
    For depth=2 returns ['cat/inst', ...] -- e.g. OmniObject3D.
    For depth=3 returns ['scene/weather/cam', ...] -- e.g. VKITTI 2.

    Each returned id is a `join_sep`-joined path component string suitable for
    `osp.join(root, scene_id)` to reach the leaf directory.
    """
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")
    if depth == 1:
        return list_subdirs(root)
    out: list[str] = []
    for parent in list_subdirs(root):
        sub_ids = walk_nested_scenes(osp.join(root, parent), depth - 1, join_sep)
        out.extend(join_sep.join((parent, s)) for s in sub_ids)
    return out


def detect_layout_depth(
    root: str,
    leaf_markers: tuple[str, ...] = ("rgb", "depth", "cam"),
    max_depth: int = 4,
) -> Optional[int]:
    """Detect how many directory levels separate `root` from leaf scenes.

    A "leaf scene" is identified by the presence of any of `leaf_markers`
    as subdirectories. Returns the depth (1-based) at which the markers
    first appear, or None if `max_depth` is exhausted.

    Useful for auto-detecting whether this cluster's data has the upstream
    flat layout (depth=1) or an extra category/instance level (depth=2+).
    """
    def has_markers(p: str) -> bool:
        return any(osp.isdir(osp.join(p, m)) for m in leaf_markers)

    if has_markers(root):
        return 0  # root itself is a leaf
    queue = [(root, 1)]
    while queue:
        path, d = queue.pop(0)
        if d > max_depth:
            return None
        for sub in list_subdirs(path):
            sp = osp.join(path, sub)
            if has_markers(sp):
                return d
            queue.append((sp, d + 1))
    return None

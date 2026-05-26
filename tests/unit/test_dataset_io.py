"""Unit tests for src/dust3r/datasets/_io.py.

Imports the module directly via importlib so the package's __init__.py
(which depends on accelerate, torch, etc.) doesn't have to load. Keeps
these tests runnable on a slim Python without the training stack.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_io():
    spec = importlib.util.spec_from_file_location(
        "_io",
        Path(__file__).resolve().parents[2] / "src" / "dust3r" / "datasets" / "_io.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


io = _load_io()


# ---------- load_cam_params -----------------------------------------


def test_load_cam_params_prefers_npz(tmp_path):
    stem = tmp_path / "frame00"
    np.savez(stem.with_suffix(".npz"), intrinsics=np.eye(3), pose=np.eye(4))
    d = io.load_cam_params(str(stem))
    assert np.array_equal(d["intrinsics"], np.eye(3))


def test_load_cam_params_falls_back_to_safetensor(tmp_path):
    pytest.importorskip("safetensors")
    from safetensors.numpy import save_file

    stem = tmp_path / "frame01"
    save_file(
        {"intrinsics": np.eye(3, dtype=np.float32), "pose": np.eye(4, dtype=np.float32)},
        str(stem) + ".safetensor",
    )
    d = io.load_cam_params(str(stem))  # auto: npz absent, falls through
    assert np.array_equal(d["intrinsics"], np.eye(3, dtype=np.float32))


def test_load_cam_params_explicit_safetensor_misses_npz_only(tmp_path):
    pytest.importorskip("safetensors")
    stem = tmp_path / "frame02"
    np.savez(stem.with_suffix(".npz"), x=np.zeros(1))
    with pytest.raises(FileNotFoundError):
        io.load_cam_params(str(stem), data_format="safetensor")


def test_load_cam_params_missing_both_raises(tmp_path):
    stem = tmp_path / "nope"
    with pytest.raises(FileNotFoundError):
        io.load_cam_params(str(stem))


def test_load_cam_params_invalid_format_raises(tmp_path):
    with pytest.raises(ValueError):
        io.load_cam_params(str(tmp_path / "x"), data_format="pickle")


# ---------- walk_nested_scenes --------------------------------------


def _mk_tree(root: Path, levels: list[list[str]]):
    """Recursively create a directory tree from a list-of-lists spec."""
    def _build(p, depth):
        if depth == len(levels):
            return
        for name in levels[depth]:
            sub = p / name
            sub.mkdir(parents=True, exist_ok=True)
            _build(sub, depth + 1)
    _build(root, 0)


def test_walk_depth1_matches_listdir(tmp_path):
    _mk_tree(tmp_path, [["scene1", "scene2", "scene3"]])
    assert io.walk_nested_scenes(str(tmp_path), depth=1) == ["scene1", "scene2", "scene3"]


def test_walk_depth2_joins_with_slash(tmp_path):
    _mk_tree(tmp_path, [["anise", "apple"], ["_001", "_002"]])
    # Yields sorted by parent-then-child.
    got = io.walk_nested_scenes(str(tmp_path), depth=2)
    assert got == ["anise/_001", "anise/_002", "apple/_001", "apple/_002"]


def test_walk_depth3_three_levels(tmp_path):
    _mk_tree(tmp_path, [["Scene01"], ["sunset", "morning"], ["Camera_0", "Camera_1"]])
    got = io.walk_nested_scenes(str(tmp_path), depth=3)
    assert got == [
        "Scene01/morning/Camera_0",
        "Scene01/morning/Camera_1",
        "Scene01/sunset/Camera_0",
        "Scene01/sunset/Camera_1",
    ]


def test_walk_zero_depth_rejected():
    with pytest.raises(ValueError):
        io.walk_nested_scenes("/tmp", depth=0)


# ---------- detect_layout_depth -------------------------------------


def test_detect_layout_depth_flat(tmp_path):
    # ROOT/<scene>/{rgb,depth,cam}/  — depth 1
    _mk_tree(tmp_path, [["sceneA"], ["rgb", "depth", "cam"]])
    assert io.detect_layout_depth(str(tmp_path)) == 1


def test_detect_layout_depth_nested_cat_inst(tmp_path):
    # ROOT/<cat>/<inst>/{rgb,depth,cam}/  — depth 2 (e.g. OmniObject3D)
    _mk_tree(tmp_path, [["anise"], ["anise_001"], ["rgb", "depth", "cam"]])
    assert io.detect_layout_depth(str(tmp_path)) == 2


def test_detect_layout_depth_returns_none_when_absent(tmp_path):
    _mk_tree(tmp_path, [["foo"], ["bar"]])
    assert io.detect_layout_depth(str(tmp_path), max_depth=3) is None

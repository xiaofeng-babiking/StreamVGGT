"""Loads and validates workflows/cluster.yaml into typed dataclasses."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SshConfig:
    user: str
    identity_file: str
    connect_timeout_s: int = 5


@dataclass(frozen=True)
class DockerConfig:
    image: str
    shm_size: str = "16G"


@dataclass(frozen=True)
class PathsConfig:
    repo: str
    venv: str
    jfs: str


@dataclass(frozen=True)
class DiscoveryConfig:
    idle_threshold_mib: int = 500
    min_idle_gpus_per_node: int = 1


@dataclass(frozen=True)
class NcclConfig:
    ib_hca: str
    socket_ifname: str
    ib_disable: int = 0
    async_error_handling: int = 1
    debug: str = "WARN"
    ib_timeout: int = 23
    net_gdr_level: str = "PIX"


@dataclass(frozen=True)
class TrainConfig:
    entrypoint: str = "src/train.py"
    config_name: str = "train"
    main_port: int = 26902


@dataclass(frozen=True)
class ClusterConfig:
    ssh: SshConfig
    candidate_hosts: list[str]
    docker: DockerConfig
    paths: PathsConfig
    discovery: DiscoveryConfig
    nccl: NcclConfig
    train: TrainConfig


def _require(d: dict, key: str) -> Any:
    if key not in d:
        raise KeyError(f"missing required field '{key}' in cluster.yaml")
    return d[key]


def load_cluster_config(path: Path | str) -> ClusterConfig:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"cluster.yaml must be a mapping, got {type(raw).__name__}")

    ssh_raw = _require(raw, "ssh")
    paths_raw = _require(raw, "paths")
    docker_raw = _require(raw, "docker")
    nccl_raw = _require(raw, "nccl")

    return ClusterConfig(
        ssh=SshConfig(
            user=_require(ssh_raw, "user"),
            identity_file=_require(ssh_raw, "identity_file"),
            connect_timeout_s=int(ssh_raw.get("connect_timeout_s", 5)),
        ),
        candidate_hosts=list(_require(raw, "candidate_hosts")),
        docker=DockerConfig(
            image=_require(docker_raw, "image"),
            shm_size=docker_raw.get("shm_size", "16G"),
        ),
        paths=PathsConfig(
            repo=_require(paths_raw, "repo"),
            venv=_require(paths_raw, "venv"),
            jfs=_require(paths_raw, "jfs"),
        ),
        discovery=DiscoveryConfig(
            idle_threshold_mib=int(raw.get("discovery", {}).get("idle_threshold_mib", 500)),
            min_idle_gpus_per_node=int(raw.get("discovery", {}).get("min_idle_gpus_per_node", 1)),
        ),
        nccl=NcclConfig(
            ib_hca=_require(nccl_raw, "ib_hca"),
            socket_ifname=_require(nccl_raw, "socket_ifname"),
            ib_disable=int(nccl_raw.get("ib_disable", 0)),
            async_error_handling=int(nccl_raw.get("async_error_handling", 1)),
            debug=nccl_raw.get("debug", "WARN"),
            ib_timeout=int(nccl_raw.get("ib_timeout", 23)),
            net_gdr_level=nccl_raw.get("net_gdr_level", "PIX"),
        ),
        train=TrainConfig(
            entrypoint=raw.get("train", {}).get("entrypoint", "src/train.py"),
            config_name=raw.get("train", {}).get("config_name", "train"),
            main_port=int(raw.get("train", {}).get("main_port", 26902)),
        ),
    )

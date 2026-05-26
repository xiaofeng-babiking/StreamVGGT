"""Loads and validates workflows/cluster.yaml into typed dataclasses."""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass(frozen=True)
class SshConfig:
    """SSH connection settings.

    Auth precedence (when both set): asyncssh tries client_keys first, then
    falls back to password. At least one of `identity_file` or `password`
    must be configured.
    """
    user: str
    identity_file: Optional[str] = None
    # repr=False keeps the resolved password out of dataclass __repr__ output,
    # so accidental print(cfg) / logger.debug(cfg) won't leak it.
    password: Optional[str] = field(default=None, repr=False)
    connect_timeout_s: int = 5

    def __post_init__(self) -> None:
        if not self.identity_file and not self.password:
            raise ValueError(
                "cluster.yaml#ssh: must set at least one of "
                "identity_file or password"
            )


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


_ENV_VAR_PATTERN = re.compile(r"^\$\{([A-Z_][A-Z0-9_]*)\}$")


def _resolve_secret(value: Optional[str]) -> Optional[str]:
    """Resolve a config value that may reference an environment variable.

    A value of exactly "${VAR_NAME}" is replaced with os.environ["VAR_NAME"];
    anything else is returned verbatim. Missing env vars raise RuntimeError
    instead of producing an empty string.
    """
    if value is None:
        return None
    m = _ENV_VAR_PATTERN.match(value)
    if m:
        env_name = m.group(1)
        if env_name not in os.environ:
            raise RuntimeError(
                f"cluster.yaml references env var ${{{env_name}}} but it is not set"
            )
        return os.environ[env_name]
    return value


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

    raw_password = ssh_raw.get("password")
    resolved_password = _resolve_secret(raw_password)
    if (
        resolved_password is not None
        and raw_password == resolved_password
        and not _ENV_VAR_PATTERN.match(raw_password)
    ):
        # Literal password (not env-var-interpolated). Warn but don't fail —
        # the user explicitly asked to put it in the yaml.
        sys.stderr.write(
            "[cluster.yaml] WARN: ssh.password is a literal string in plaintext yaml. "
            "Consider `password: \"${SSH_PASSWORD}\"` plus an env var, "
            "or .gitignore cluster.yaml to keep it local.\n"
        )

    return ClusterConfig(
        ssh=SshConfig(
            user=_require(ssh_raw, "user"),
            identity_file=ssh_raw.get("identity_file"),
            password=resolved_password,
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


def asyncssh_connect_kwargs(cfg: ClusterConfig) -> dict[str, Any]:
    """Build the kwargs dict passed to `asyncssh.connect(host, **kwargs)`.

    Centralizes the auth wiring so the six call sites (discover, launch's
    _start_one / _kill_all / _docker_ps_one, supervise._tail_one,
    monitor._sample_one) stay in sync.

    Returns:
        A dict containing `username`, `known_hosts=None`, `connect_timeout`,
        plus whichever of `client_keys` and `password` are configured.
    """
    kwargs: dict[str, Any] = {
        "username": cfg.ssh.user,
        "known_hosts": None,  # trust-on-first-use; cluster-internal
        "connect_timeout": cfg.ssh.connect_timeout_s,
    }
    if cfg.ssh.identity_file:
        kwargs["client_keys"] = [str(Path(cfg.ssh.identity_file).expanduser())]
    if cfg.ssh.password:
        kwargs["password"] = cfg.ssh.password
    return kwargs

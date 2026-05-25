"""Build the NCCL_* environment for worker containers from the cluster config."""
from __future__ import annotations

from typing import Protocol


class _NcclLike(Protocol):
    ib_hca: str
    socket_ifname: str
    ib_disable: int
    async_error_handling: int
    debug: str
    ib_timeout: int
    net_gdr_level: str


def build_nccl_env(cfg: _NcclLike) -> dict[str, str]:
    return {
        "NCCL_IB_HCA": cfg.ib_hca,
        "NCCL_IB_DISABLE": str(int(cfg.ib_disable)),
        "NCCL_SOCKET_IFNAME": cfg.socket_ifname,
        "NCCL_ASYNC_ERROR_HANDLING": str(int(cfg.async_error_handling)),
        "NCCL_DEBUG": cfg.debug,
        "NCCL_IB_TIMEOUT": str(cfg.ib_timeout),
        "NCCL_NET_GDR_LEVEL": cfg.net_gdr_level,
    }

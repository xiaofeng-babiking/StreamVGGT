from dataclasses import dataclass

from svggt_orch.nccl_env import build_nccl_env


@dataclass(frozen=True)
class _Cfg:
    ib_hca: str = "mlx5_1"
    socket_ifname: str = "bond0"
    ib_disable: int = 0
    async_error_handling: int = 1
    debug: str = "WARN"
    ib_timeout: int = 23
    net_gdr_level: str = "PIX"


def test_build_nccl_env_defaults():
    env = build_nccl_env(_Cfg())
    assert env["NCCL_IB_HCA"] == "mlx5_1"
    assert env["NCCL_IB_DISABLE"] == "0"
    assert env["NCCL_SOCKET_IFNAME"] == "bond0"
    assert env["NCCL_ASYNC_ERROR_HANDLING"] == "1"
    assert env["NCCL_DEBUG"] == "WARN"
    assert env["NCCL_IB_TIMEOUT"] == "23"
    assert env["NCCL_NET_GDR_LEVEL"] == "PIX"


def test_build_nccl_env_disable_ib():
    env = build_nccl_env(_Cfg(ib_disable=1))
    assert env["NCCL_IB_DISABLE"] == "1"


def test_build_nccl_env_keys_are_all_strings():
    env = build_nccl_env(_Cfg())
    for k, v in env.items():
        assert isinstance(k, str)
        assert isinstance(v, str), f"value for {k} is {type(v).__name__}, expected str"

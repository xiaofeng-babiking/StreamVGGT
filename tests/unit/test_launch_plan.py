from pathlib import Path

from svggt_orch.config import (
    ClusterConfig, SshConfig, DockerConfig, PathsConfig,
    DiscoveryConfig, NcclConfig, TrainConfig,
)
from svggt_orch.launch import build_launch_plan, render_launch_plan_for_print
from svggt_orch.types import NodeInfo


def _cfg() -> ClusterConfig:
    return ClusterConfig(
        ssh=SshConfig(user="jing.feng", identity_file="~/.ssh/id_ed25519"),
        candidate_hosts=["gpu003", "gpu004"],
        docker=DockerConfig(image="babiking/ubuntu:12.8.1-cudnn-devel-ubuntu24.04"),
        paths=PathsConfig(
            repo="/jfs/jing.feng/codebases/StreamVGGT",
            venv="/jfs/jing.feng/codebases/StreamVGGT/.venv",
            jfs="/jfs",
        ),
        discovery=DiscoveryConfig(),
        nccl=NcclConfig(ib_hca="mlx5_1", socket_ifname="bond0"),
        train=TrainConfig(),
    )


def test_launch_plan_world_size_and_ranks():
    nodes = [
        NodeInfo("gpu003", "172.31.208.5", "10.10.100.5", [0, 1], "t"),
        NodeInfo("gpu004", "172.31.208.6", "10.10.100.6", [0, 1, 2], "t"),
    ]
    plan = build_launch_plan(_cfg(), nodes, job_id="20260525-162300", exp_name="x")
    assert plan.world_size == 5
    assert plan.master_host == "gpu003"
    assert plan.master_ip == "10.10.100.5"
    assert [s.machine_rank for s in plan.workers] == [0, 1]
    assert [s.nproc_per_node for s in plan.workers] == [2, 3]
    assert plan.container_name == "svggt-20260525-162300"


def test_launch_plan_cuda_visible_devices():
    nodes = [NodeInfo("gpu003", "172.31.208.5", "10.10.100.5", [2, 4], "t")]
    plan = build_launch_plan(_cfg(), nodes, job_id="J", exp_name="x")
    assert plan.workers[0].cuda_visible_devices == "2,4"


def test_launch_plan_falls_back_to_bond_ip_when_no_ib():
    nodes = [
        NodeInfo("gpu003", "172.31.208.5", None, [0], "t"),  # no ib_ip
        NodeInfo("gpu004", "172.31.208.6", "10.10.100.6", [0], "t"),
    ]
    plan = build_launch_plan(_cfg(), nodes, job_id="J", exp_name="x")
    # Falls back to bond_ip of nodes[0]
    assert plan.master_ip == "172.31.208.5"


def test_render_launch_plan_for_print_matches_golden():
    nodes = [
        NodeInfo("gpu003", "172.31.208.5", "10.10.100.5", [0, 1], "t"),
        NodeInfo("gpu004", "172.31.208.6", "10.10.100.6", [0, 1], "t"),
    ]
    plan = build_launch_plan(_cfg(), nodes, job_id="20260525-162300", exp_name="StreamVGGT_smoke")
    out = render_launch_plan_for_print(plan)
    golden = (Path(__file__).resolve().parents[1] / "golden" / "launch.txt").read_text()
    assert out.rstrip() == golden.rstrip()


def test_config_name_defaults_to_cluster_yaml():
    nodes = [NodeInfo("gpu003", "172.31.208.5", "10.10.100.5", [0], "t")]
    plan = build_launch_plan(_cfg(), nodes, job_id="J", exp_name="x")
    # _cfg() uses default TrainConfig() → config_name = "train"
    assert plan.config_name == "train"


def test_config_name_override_takes_effect():
    nodes = [NodeInfo("gpu003", "172.31.208.5", "10.10.100.5", [0], "t")]
    plan = build_launch_plan(
        _cfg(), nodes, job_id="J", exp_name="x", config_name="train_smoke"
    )
    assert plan.config_name == "train_smoke"


def test_config_name_override_propagates_to_docker_env():
    # The override must reach the CONFIG_NAME env var in the docker run command.
    from svggt_orch.launch import _build_docker_cmd
    nodes = [NodeInfo("gpu003", "172.31.208.5", "10.10.100.5", [0], "t")]
    plan = build_launch_plan(
        _cfg(), nodes, job_id="J", exp_name="x", config_name="finetune"
    )
    cmd = _build_docker_cmd(_cfg(), plan, plan.workers[0])
    assert "CONFIG_NAME=finetune" in cmd
    assert "CONFIG_NAME=train" not in cmd  # cfg.train.config_name was overridden


def test_gpu_flag_uses_nested_quoting():
    """Regression: docker errors with 'cannot set both Count and DeviceIDs'
    when --gpus is passed without nested quoting around the device list.
    The build must always emit `--gpus '"device=..."'` (outer single,
    inner double) so docker's parser sees one literal device-spec argument.
    """
    from svggt_orch.launch import _build_docker_cmd
    nodes = [NodeInfo("gpu003", "172.31.208.5", "10.10.100.5", [0, 1, 2, 3], "t")]
    plan = build_launch_plan(_cfg(), nodes, job_id="J", exp_name="x")
    cmd = _build_docker_cmd(_cfg(), plan, plan.workers[0])
    # Exactly the nested-quoted form must be present.
    assert """--gpus '"device=0,1,2,3"'""" in cmd
    # The bare un-nested form must NOT be present (regression guard).
    assert '--gpus "device=0,1,2,3" ' not in cmd

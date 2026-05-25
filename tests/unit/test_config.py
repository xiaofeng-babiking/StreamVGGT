from pathlib import Path

import pytest

from svggt_orch.config import ClusterConfig, load_cluster_config


SAMPLE_YAML = """
ssh:
  user: jing.feng
  identity_file: ~/.ssh/id_ed25519
  connect_timeout_s: 5
candidate_hosts:
  - gpu003
  - gpu004
docker:
  image: babiking/ubuntu:12.8.1-cudnn-devel-ubuntu24.04
  shm_size: 16G
paths:
  repo: /jfs/jing.feng/codebases/StreamVGGT
  venv: /jfs/jing.feng/codebases/StreamVGGT/.venv
  jfs: /jfs
discovery:
  idle_threshold_mib: 500
  min_idle_gpus_per_node: 1
nccl:
  ib_hca: mlx5_1
  socket_ifname: bond0
  ib_disable: 0
  async_error_handling: 1
  debug: WARN
  ib_timeout: 23
  net_gdr_level: PIX
train:
  entrypoint: src/train.py
  config_name: train
  main_port: 26902
"""


def test_load_cluster_config(tmp_path: Path):
    p = tmp_path / "cluster.yaml"
    p.write_text(SAMPLE_YAML)
    cfg = load_cluster_config(p)
    assert isinstance(cfg, ClusterConfig)
    assert cfg.ssh.user == "jing.feng"
    assert cfg.candidate_hosts == ["gpu003", "gpu004"]
    assert cfg.docker.shm_size == "16G"
    assert cfg.paths.repo == "/jfs/jing.feng/codebases/StreamVGGT"
    assert cfg.discovery.idle_threshold_mib == 500
    assert cfg.nccl.ib_hca == "mlx5_1"
    assert cfg.train.main_port == 26902


def test_load_cluster_config_missing_required(tmp_path: Path):
    # Provide every required top-level mapping EXCEPT candidate_hosts so the
    # loader's error message pins down exactly which field is missing.
    yaml_missing_hosts = """
ssh: {user: x, identity_file: x}
docker: {image: x}
paths: {repo: /x, venv: /x, jfs: /jfs}
nccl: {ib_hca: mlx5_1, socket_ifname: bond0}
"""
    p = tmp_path / "cluster.yaml"
    p.write_text(yaml_missing_hosts)
    with pytest.raises(KeyError) as exc:
        load_cluster_config(p)
    assert "candidate_hosts" in str(exc.value)


def test_load_cluster_config_defaults_applied(tmp_path: Path):
    minimal = """
ssh:
  user: jing.feng
  identity_file: ~/.ssh/id_ed25519
candidate_hosts: [gpu003]
docker:
  image: x
paths:
  repo: /x
  venv: /x/.venv
  jfs: /jfs
nccl:
  ib_hca: mlx5_1
  socket_ifname: bond0
"""
    p = tmp_path / "cluster.yaml"
    p.write_text(minimal)
    cfg = load_cluster_config(p)
    assert cfg.ssh.connect_timeout_s == 5
    assert cfg.docker.shm_size == "16G"
    assert cfg.discovery.idle_threshold_mib == 500
    assert cfg.train.main_port == 26902

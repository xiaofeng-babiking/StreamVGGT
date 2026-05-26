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


def _yaml_with_ssh(ssh_block: str) -> str:
    """Build a minimal-but-valid cluster.yaml around a custom ssh block."""
    return (
        f"ssh:\n{ssh_block}\n"
        "candidate_hosts: [gpu003]\n"
        "docker: {image: x}\n"
        "paths: {repo: /x, venv: /x/.venv, jfs: /jfs}\n"
        "nccl: {ib_hca: mlx5_1, socket_ifname: bond0}\n"
    )


def test_password_only_config_is_valid(tmp_path, capsys):
    p = tmp_path / "cluster.yaml"
    p.write_text(_yaml_with_ssh('  user: x\n  password: "literal-pw"'))
    cfg = load_cluster_config(p)
    assert cfg.ssh.password == "literal-pw"
    assert cfg.ssh.identity_file is None
    # Literal password should print a warning to stderr.
    assert "literal string" in capsys.readouterr().err


def test_env_var_password_is_resolved(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("MY_SSH_PASSWORD", "from-env-12345")
    p = tmp_path / "cluster.yaml"
    p.write_text(_yaml_with_ssh('  user: x\n  password: "${MY_SSH_PASSWORD}"'))
    cfg = load_cluster_config(p)
    assert cfg.ssh.password == "from-env-12345"
    # Env-var interpolated values should NOT trigger the plaintext warning.
    assert "literal string" not in capsys.readouterr().err


def test_missing_env_var_password_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("DEFINITELY_NOT_SET", raising=False)
    p = tmp_path / "cluster.yaml"
    p.write_text(_yaml_with_ssh('  user: x\n  password: "${DEFINITELY_NOT_SET}"'))
    with pytest.raises(RuntimeError) as exc:
        load_cluster_config(p)
    assert "DEFINITELY_NOT_SET" in str(exc.value)


def test_neither_key_nor_password_raises(tmp_path):
    p = tmp_path / "cluster.yaml"
    p.write_text(_yaml_with_ssh("  user: x"))
    with pytest.raises(ValueError) as exc:
        load_cluster_config(p)
    assert "identity_file or password" in str(exc.value)


def test_password_excluded_from_repr(tmp_path):
    p = tmp_path / "cluster.yaml"
    p.write_text(_yaml_with_ssh('  user: x\n  password: "super-secret-123"'))
    cfg = load_cluster_config(p)
    # SshConfig's __repr__ must not leak the password.
    assert "super-secret-123" not in repr(cfg.ssh)


def test_asyncssh_connect_kwargs_uses_password(tmp_path):
    from svggt_orch.config import asyncssh_connect_kwargs

    p = tmp_path / "cluster.yaml"
    p.write_text(_yaml_with_ssh('  user: x\n  password: "pw"'))
    cfg = load_cluster_config(p)
    kw = asyncssh_connect_kwargs(cfg)
    assert kw["username"] == "x"
    assert kw["password"] == "pw"
    assert kw["known_hosts"] is None
    assert "client_keys" not in kw


def test_asyncssh_connect_kwargs_uses_key(tmp_path):
    from svggt_orch.config import asyncssh_connect_kwargs

    p = tmp_path / "cluster.yaml"
    p.write_text(_yaml_with_ssh("  user: x\n  identity_file: ~/.ssh/id_ed25519"))
    cfg = load_cluster_config(p)
    kw = asyncssh_connect_kwargs(cfg)
    assert "password" not in kw
    assert kw["client_keys"][0].endswith("id_ed25519")

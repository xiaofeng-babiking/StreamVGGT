"""Launch a multi-node DDP training job via SSH + docker.

Two layers:
  * Pure planning (`build_launch_plan`, `render_launch_plan_for_print`) — unit-tested.
  * `run_launch` async coroutine — fans out `docker run` over asyncssh.

asyncssh is imported lazily inside the async functions so the planning
helpers stay importable for unit tests in environments without asyncssh.
"""
from __future__ import annotations

import asyncio
import datetime as _dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import ClusterConfig
    from .types import NodeInfo


@dataclass
class WorkerSpec:
    machine_rank: int
    host: str
    nproc_per_node: int
    cuda_visible_devices: str  # "0,1,2"


@dataclass
class LaunchPlan:
    job_id: str
    exp_name: str
    world_size: int
    master_host: str
    master_ip: str
    main_port: int
    container_name: str
    image: str
    workers: list[WorkerSpec]
    nodes: "list[NodeInfo]"


def build_launch_plan(
    cfg: "ClusterConfig",
    nodes: "list[NodeInfo]",
    job_id: str,
    exp_name: str,
) -> LaunchPlan:
    if not nodes:
        raise ValueError("build_launch_plan: nodes list is empty")
    workers: list[WorkerSpec] = []
    for rank, n in enumerate(nodes):
        workers.append(
            WorkerSpec(
                machine_rank=rank,
                host=n.host,
                nproc_per_node=len(n.free_gpus),
                cuda_visible_devices=",".join(str(g) for g in n.free_gpus),
            )
        )
    master = nodes[0]
    master_ip = master.ib_ip if master.ib_ip else master.bond_ip
    return LaunchPlan(
        job_id=job_id,
        exp_name=exp_name,
        world_size=sum(w.nproc_per_node for w in workers),
        master_host=master.host,
        master_ip=master_ip,
        main_port=cfg.train.main_port,
        container_name=f"svggt-{job_id}",
        image=cfg.docker.image,
        workers=workers,
        nodes=nodes,
    )


def render_launch_plan_for_print(plan: LaunchPlan) -> str:
    lines = [
        "launch (dry-run)",
        f"  job_id: {plan.job_id}",
        f"  exp_name: {plan.exp_name}",
        f"  world_size: {plan.world_size}",
        f"  master: {plan.master_host} @ {plan.master_ip}:{plan.main_port}",
        f"  container: {plan.container_name}",
        f"  image: {plan.image}",
        "  workers:",
    ]
    for w in plan.workers:
        lines.append(
            f"    rank={w.machine_rank} host={w.host} "
            f"nproc={w.nproc_per_node} cuda={w.cuda_visible_devices}"
        )
    return "\n".join(lines)


def _shquote(s: str) -> str:
    if all(c.isalnum() or c in "_-.,:/=" for c in s):
        return s
    return "'" + s.replace("'", "'\\''") + "'"


def _build_docker_cmd(cfg: "ClusterConfig", plan: LaunchPlan, w: WorkerSpec) -> str:
    """Render the docker run command to execute on the worker node via ssh."""
    from .nccl_env import build_nccl_env

    nccl = build_nccl_env(cfg.nccl)
    env_args: list[str] = []
    extra_env = {
        "JOB_ID": plan.job_id,
        "EXP_NAME": plan.exp_name,
        "NUM_MACHINES": str(len(plan.workers)),
        "MACHINE_RANK": str(w.machine_rank),
        "NUM_PROCESSES": str(plan.world_size),
        "NPROC_PER_NODE": str(w.nproc_per_node),
        "MAIN_PROCESS_IP": plan.master_ip,
        "MAIN_PROCESS_PORT": str(plan.main_port),
        "CONFIG_NAME": cfg.train.config_name,
        "CUDA_VISIBLE_DEVICES": w.cuda_visible_devices,
    }
    for k, v in {**nccl, **extra_env}.items():
        env_args.append(f"--env {k}={_shquote(v)}")

    gpu_flag = f'--gpus "device={w.cuda_visible_devices}"'

    # Enter repo first so $(pwd) at lib/docker_env.sh sourcing time is the
    # repo root (the helper bind-mounts $(pwd) -> /workspace).
    docker_run = (
        f"cd {_shquote(cfg.paths.repo)} && "
        f"source scripts/lib/docker_env.sh && "
        f"docker run -d --name {plan.container_name} "
        f"{gpu_flag} "
        + " ".join(env_args)
        + " "
        + '"${DOCKER_RUN_FLAGS[@]}" '
        + f"{cfg.docker.image} "
        f"/workspace/scripts/worker_entry.sh"
    )
    return docker_run


async def _start_one(
    host: str, docker_cmd: str, cfg: "ClusterConfig"
) -> "tuple[str, int, str, str]":
    import asyncssh  # lazy

    async with asyncssh.connect(
        host,
        username=cfg.ssh.user,
        client_keys=[str(Path(cfg.ssh.identity_file).expanduser())],
        known_hosts=None,
        connect_timeout=cfg.ssh.connect_timeout_s,
    ) as conn:
        r = await conn.run(f"bash -lc {_shquote(docker_cmd)}", check=False, timeout=120)
        return host, r.exit_status, str(r.stdout or ""), str(r.stderr or "")


async def _kill_all(plan: LaunchPlan, cfg: "ClusterConfig") -> None:
    import asyncssh  # lazy

    async def _one(host: str) -> None:
        try:
            async with asyncssh.connect(
                host,
                username=cfg.ssh.user,
                client_keys=[str(Path(cfg.ssh.identity_file).expanduser())],
                known_hosts=None,
                connect_timeout=cfg.ssh.connect_timeout_s,
            ) as conn:
                await conn.run(f"docker rm -f {plan.container_name}", check=False, timeout=30)
        except Exception:
            pass

    await asyncio.gather(*[_one(w.host) for w in plan.workers])


def write_manifest(plan: LaunchPlan, manifest_dir: Path) -> Path:
    from .types import JobManifest

    started_at = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    m = JobManifest(
        job_id=plan.job_id,
        exp_name=plan.exp_name,
        nodes=plan.nodes,
        master_host=plan.master_host,
        master_ip=plan.master_ip,
        main_port=plan.main_port,
        world_size=plan.world_size,
        container_name=plan.container_name,
        started_at=started_at,
    )
    manifest_dir.mkdir(parents=True, exist_ok=True)
    p = manifest_dir / "manifest.json"
    p.write_text(m.to_json())
    return p


def run_launch(cfg_path: Path, dry_run: bool, exp_name: "str | None") -> int:
    from .config import load_cluster_config
    from .types import NodeInfo

    cfg = load_cluster_config(cfg_path)
    nodes_path = Path(cfg_path).parent / "nodes.json"
    if not nodes_path.exists():
        print(f"launch: {nodes_path} missing. Run `discover` first.", flush=True)
        return 2
    nodes_raw = json.loads(nodes_path.read_text())
    nodes = [NodeInfo.from_dict(d) for d in nodes_raw]
    if not nodes:
        print("launch: nodes.json is empty", flush=True)
        return 2

    job_id = _dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    plan = build_launch_plan(
        cfg, nodes, job_id=job_id, exp_name=exp_name or "StreamVGGT_run",
    )

    if dry_run:
        print(render_launch_plan_for_print(plan))
        return 0

    manifest_dir = Path(cfg_path).parent / "jobs" / job_id
    write_manifest(plan, manifest_dir)
    print(f"launch: starting {plan.container_name} on {len(plan.workers)} nodes")

    async def _go() -> int:
        coros = [_start_one(w.host, _build_docker_cmd(cfg, plan, w), cfg) for w in plan.workers]
        results = await asyncio.gather(*coros, return_exceptions=True)
        ok = True
        for r in results:
            if isinstance(r, BaseException):
                ok = False
                print(f"launch: docker run raised: {r!r}", flush=True)
                continue
            host, code, out, err = r
            if code != 0:
                ok = False
                print(f"launch: docker run failed on {host} (exit {code}):\n{err}", flush=True)
        if not ok:
            print("launch: aborting all started containers", flush=True)
            await _kill_all(plan, cfg)
            return 3
        print(f"launch: started ok. Manifest: {manifest_dir / 'manifest.json'}", flush=True)
        return 0

    return asyncio.run(_go())

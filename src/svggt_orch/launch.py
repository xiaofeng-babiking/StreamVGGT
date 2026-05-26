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
    config_name: str  # Hydra config name passed to train.py via --config-name
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
    config_name: "str | None" = None,
) -> LaunchPlan:
    """Build the per-worker plan.

    Args:
        config_name: optional override for cfg.train.config_name. When None,
            falls back to the value in cluster.yaml. This is what the
            `launch --config-name X` CLI flag plumbs through.
    """
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
        config_name=config_name or cfg.train.config_name,
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
        f"  config_name: {plan.config_name}",
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
        "CONFIG_NAME": plan.config_name,
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

    from .config import asyncssh_connect_kwargs

    async with asyncssh.connect(host, **asyncssh_connect_kwargs(cfg)) as conn:
        r = await conn.run(f"bash -lc {_shquote(docker_cmd)}", check=False, timeout=120)
        return host, r.exit_status, str(r.stdout or ""), str(r.stderr or "")


async def _kill_all(plan: LaunchPlan, cfg: "ClusterConfig") -> None:
    import asyncssh  # lazy

    from .config import asyncssh_connect_kwargs

    async def _one(host: str) -> None:
        try:
            async with asyncssh.connect(host, **asyncssh_connect_kwargs(cfg)) as conn:
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


def run_launch(
    cfg_path: Path,
    dry_run: bool,
    exp_name: "str | None",
    config_name: "str | None" = None,
) -> int:
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
        cfg, nodes,
        job_id=job_id,
        exp_name=exp_name or "StreamVGGT_run",
        config_name=config_name,  # None -> use cfg.train.config_name
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


async def _docker_ps_one(
    host: str, container: str, cfg: "ClusterConfig"
) -> "tuple[str, str]":
    import asyncssh  # lazy

    from .config import asyncssh_connect_kwargs

    try:
        async with asyncssh.connect(host, **asyncssh_connect_kwargs(cfg)) as conn:
            r = await conn.run(
                f"docker inspect -f '{{{{.State.Status}}}} {{{{.State.ExitCode}}}}' {container}",
                check=False,
                timeout=10,
            )
            return host, (str(r.stdout).strip() if r.exit_status == 0 else "absent 0")
    except Exception as e:
        return host, f"error {e}"


def kill_job(cfg_path: Path, job_id: str) -> int:
    from .config import load_cluster_config
    from .types import JobManifest

    cfg = load_cluster_config(cfg_path)
    manifest_path = Path(cfg_path).parent / "jobs" / job_id / "manifest.json"
    if not manifest_path.exists():
        print(f"kill: manifest not found at {manifest_path}", flush=True)
        return 1
    m = JobManifest.from_json(manifest_path.read_text())
    # Reconstruct a minimal LaunchPlan-like view so _kill_all can iterate.
    plan_like = LaunchPlan(
        job_id=m.job_id,
        exp_name=m.exp_name,
        config_name="(from-manifest)",
        world_size=m.world_size,
        master_host=m.master_host,
        master_ip=m.master_ip,
        main_port=m.main_port,
        container_name=m.container_name,
        image="(from-manifest)",
        workers=[
            WorkerSpec(
                machine_rank=i,
                host=n.host,
                nproc_per_node=len(n.free_gpus),
                cuda_visible_devices=",".join(str(g) for g in n.free_gpus),
            )
            for i, n in enumerate(m.nodes)
        ],
        nodes=m.nodes,
    )
    asyncio.run(_kill_all(plan_like, cfg))

    # Flip status to "killed" on disk.
    m_killed = JobManifest.from_json(manifest_path.read_text())
    m_killed.status = "killed"
    manifest_path.write_text(m_killed.to_json())
    print(f"kill: containers removed across {len(m.nodes)} nodes", flush=True)
    return 0


def status_snapshot(cfg_path: Path, job_id: str) -> int:
    from .config import load_cluster_config
    from .types import JobManifest

    cfg = load_cluster_config(cfg_path)
    manifest_path = Path(cfg_path).parent / "jobs" / job_id / "manifest.json"
    if not manifest_path.exists():
        print(f"status: manifest not found at {manifest_path}", flush=True)
        return 1
    m = JobManifest.from_json(manifest_path.read_text())
    print(f"job_id     : {m.job_id}")
    print(f"exp_name   : {m.exp_name}")
    print(f"started_at : {m.started_at}")
    print(f"container  : {m.container_name}")
    print(f"world_size : {m.world_size}")
    print(f"status     : {m.status}")
    print("per-host:")

    async def _go() -> "list[tuple[str, str]]":
        coros = [_docker_ps_one(n.host, m.container_name, cfg) for n in m.nodes]
        return await asyncio.gather(*coros)

    rows = asyncio.run(_go())
    for host, state in rows:
        print(f"  {host:20s} {state}")
    return 0

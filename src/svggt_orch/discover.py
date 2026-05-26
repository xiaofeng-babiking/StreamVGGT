"""GPU discovery across the cluster.

Two layers:
  * Pure parsing helpers + dry-run summary — unit-tested, no I/O.
  * `discover_all()` async coroutine — runs the asyncssh fan-out.

asyncssh is imported lazily inside the async functions so the parsing
helpers remain importable in environments where asyncssh is unavailable
(e.g. CI machines without the orchestrator extra installed).
"""
from __future__ import annotations

import asyncio
import datetime as _dt
import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .config import ClusterConfig
    from .types import NodeInfo


def parse_nvidia_smi_csv(text: str) -> list[tuple[int, int]]:
    """Parse `nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits`.

    Returns a list of (gpu_index, memory_used_mib).
    """
    out: list[tuple[int, int]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        idx_str, mem_str = (p.strip() for p in line.split(","))
        out.append((int(idx_str), int(mem_str)))
    return out


def filter_idle_gpus(samples: list[tuple[int, int]], threshold_mib: int) -> list[int]:
    """Return GPU indices whose memory.used is strictly less than threshold_mib."""
    return [idx for idx, mem in samples if mem < threshold_mib]


def parse_ip_addr_show(json_text: str) -> Optional[str]:
    """Extract the first IPv4 address from `ip -j addr show <iface>` output."""
    arr = json.loads(json_text)
    if not arr:
        return None
    for entry in arr:
        for ai in entry.get("addr_info", []):
            if ai.get("family") == "inet" and "local" in ai:
                return ai["local"]
    return None


def dry_run_summary(
    hosts: list[str],
    idle_threshold_mib: int,
    ssh_user: str,
    identity_file: str,
    timeout_s: int,
) -> str:
    lines = [
        "discover (dry-run)",
        f"  ssh: {ssh_user}@<host>  -i {identity_file}  timeout={timeout_s}s",
        f"  hosts: {hosts}",
        "  per-host commands:",
        "    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits",
        "    ip -j addr show ibs110",
        f"  idle_threshold_mib: {idle_threshold_mib}",
        "  output: workflows/nodes.json",
    ]
    return "\n".join(lines)


async def _probe_one_host(host: str, cfg: "ClusterConfig") -> "NodeInfo | None":
    """SSH into `host`, query nvidia-smi and ibs110 IP. Return None on any failure."""
    import asyncssh  # lazy: keep module importable without asyncssh

    from .config import asyncssh_connect_kwargs
    from .types import NodeInfo

    try:
        async with asyncssh.connect(host, **asyncssh_connect_kwargs(cfg)) as conn:
            smi = await conn.run(
                "nvidia-smi --query-gpu=index,memory.used "
                "--format=csv,noheader,nounits",
                check=False,
                timeout=10,
            )
            if smi.exit_status != 0:
                return None
            ipres = await conn.run("ip -j addr show ibs110", check=False, timeout=5)
            bond = await conn.run(
                "ip -j addr show bond0 || ip -j addr show eth0",
                check=False,
                timeout=5,
            )

            samples = parse_nvidia_smi_csv(str(smi.stdout))
            free = filter_idle_gpus(samples, cfg.discovery.idle_threshold_mib)
            if len(free) < cfg.discovery.min_idle_gpus_per_node:
                return None

            ib_ip = parse_ip_addr_show(str(ipres.stdout)) if ipres.exit_status == 0 else None
            bond_ip = parse_ip_addr_show(str(bond.stdout)) if bond.exit_status == 0 else ""

            return NodeInfo(
                host=host,
                bond_ip=bond_ip or "",
                ib_ip=ib_ip,
                free_gpus=free,
                detected_at=_dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
    except (OSError, asyncio.TimeoutError, asyncssh.Error):
        return None


async def discover_all(cfg: "ClusterConfig") -> "list[NodeInfo]":
    """Probe every candidate host concurrently. Drop unreachable / no-idle-GPU hosts."""
    coros = [_probe_one_host(h, cfg) for h in cfg.candidate_hosts]
    results = await asyncio.gather(*coros, return_exceptions=False)
    return [r for r in results if r is not None]


def write_nodes_json(nodes: "list[NodeInfo]", path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([n.to_dict() for n in nodes], indent=2, sort_keys=True))


def run_discover(cfg_path: Path, dry_run: bool) -> int:
    from .config import load_cluster_config
    from .types import NodeInfo  # noqa: F401  imported for from_dict downstream

    cfg = load_cluster_config(cfg_path)
    if dry_run:
        print(dry_run_summary(
            hosts=cfg.candidate_hosts,
            idle_threshold_mib=cfg.discovery.idle_threshold_mib,
            ssh_user=cfg.ssh.user,
            identity_file=cfg.ssh.identity_file,
            timeout_s=cfg.ssh.connect_timeout_s,
        ))
        return 0

    nodes = asyncio.run(discover_all(cfg))
    if not nodes:
        print("discover: no reachable nodes with idle GPUs", flush=True)
        return 2

    out = Path(cfg_path).parent / "nodes.json"
    write_nodes_json(nodes, out)
    total_gpus = sum(len(n.free_gpus) for n in nodes)
    print(f"discover: wrote {out} — {len(nodes)} nodes, {total_gpus} idle GPUs", flush=True)
    return 0

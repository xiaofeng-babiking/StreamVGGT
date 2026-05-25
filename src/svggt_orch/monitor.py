"""Live TUI dashboard for a running multi-node job.

Polls each node's dmon-<host>.csv (written by worker_entry.sh) and the IB
counter sysfs files, plus a heartbeat freshness check. Refreshes every 2s.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from rich.console import Console
from rich.live import Live
from rich.table import Table

from .heartbeat import is_stale

if TYPE_CHECKING:
    from .config import ClusterConfig
    from .types import JobManifest


@dataclass
class HostSample:
    host: str
    gpu_util: list[int]
    gpu_mem_used_mib: list[int]
    gpu_temp_c: list[int]
    ib_tx_GBps: Optional[float]
    ib_rx_GBps: Optional[float]
    error: Optional[str] = None


_DMON_TAIL_CMD = (
    "tail -n 200 /workspace/workflows/jobs/{job}/dmon-$(hostname -s).csv 2>/dev/null"
)

_IB_COUNTERS_CMD = (
    "cat /sys/class/infiniband/mlx5_1/ports/1/counters/port_xmit_data "
    "/sys/class/infiniband/mlx5_1/ports/1/counters/port_rcv_data 2>/dev/null"
)


def _parse_dmon_tail(text: str) -> "tuple[list[int], list[int], list[int]]":
    """Parse the most recent line per GPU from `nvidia-smi dmon` output.

    Header-driven: the first '#' line that contains a column called 'gpu'
    defines the column index map. We extract sm (util %), fb (MB) and gtemp
    (C) by NAME so a future dmon version reordering columns won't silently
    mislabel data.

    Returns (utils, mems, temps) lists ordered by ascending gpu index.
    Missing/non-integer values become -1.
    """
    col_idx: dict[str, int] = {}
    latest: dict[int, list[str]] = {}
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith("#"):
            # Header line. The first one with 'gpu' as a token wins.
            head = s.lstrip("#").split()
            if "gpu" in head and not col_idx:
                col_idx = {name: i for i, name in enumerate(head)}
            continue
        if not col_idx:
            continue  # haven't seen a header yet; skip orphan data
        toks = s.split()
        try:
            gpu = int(toks[col_idx["gpu"]])
        except (ValueError, IndexError, KeyError):
            continue
        latest[gpu] = toks

    def _at(toks: list[str], name: str) -> int:
        if name not in col_idx:
            return -1
        try:
            return int(toks[col_idx[name]])
        except (ValueError, IndexError):
            return -1

    utils: list[int] = []
    mems: list[int] = []
    temps: list[int] = []
    for g in sorted(latest):
        toks = latest[g]
        utils.append(_at(toks, "sm"))
        mems.append(_at(toks, "fb"))
        temps.append(_at(toks, "gtemp"))
    return utils, mems, temps


async def _sample_one(
    host: str,
    job_id: str,
    cfg: "ClusterConfig",
    prev_ib: "dict[str, tuple[float, int, int]]",
) -> HostSample:
    import asyncssh  # lazy

    try:
        async with asyncssh.connect(
            host,
            username=cfg.ssh.user,
            client_keys=[str(Path(cfg.ssh.identity_file).expanduser())],
            known_hosts=None,
            connect_timeout=cfg.ssh.connect_timeout_s,
        ) as conn:
            dmon = await conn.run(
                _DMON_TAIL_CMD.format(job=job_id), check=False, timeout=10
            )
            ib = await conn.run(_IB_COUNTERS_CMD, check=False, timeout=5)
            utils, mems, temps = _parse_dmon_tail(str(dmon.stdout or ""))
            now = time.time()
            tx: Optional[float] = None
            rx: Optional[float] = None
            ib_tokens = str(ib.stdout or "").split()
            if len(ib_tokens) == 2:
                xmit_now = int(ib_tokens[0])
                rcv_now = int(ib_tokens[1])
                if host in prev_ib:
                    t0, xmit0, rcv0 = prev_ib[host]
                    dt = max(now - t0, 1e-6)
                    # IB counters are in units of 4 bytes (lanes); 1 GB = 10**9 bytes
                    tx = (xmit_now - xmit0) * 4 / dt / 1e9
                    rx = (rcv_now - rcv0) * 4 / dt / 1e9
                prev_ib[host] = (now, xmit_now, rcv_now)
            return HostSample(host, utils, mems, temps, tx, rx)
    except Exception as e:
        return HostSample(host, [], [], [], None, None, error=str(e))


def _render(
    job: "JobManifest", samples: "list[HostSample]", elapsed_s: float, stale: bool
) -> Table:
    title = (
        f"job={job.job_id}  exp={job.exp_name}  "
        f"elapsed={int(elapsed_s)//60:02d}:{int(elapsed_s)%60:02d}  ws={job.world_size}"
    )
    t = Table(title=title, title_style="bold cyan" if not stale else "bold red")
    t.add_column("host")
    t.add_column("gpu utils %", justify="right")
    t.add_column("gpu mem MiB", justify="right")
    t.add_column("temp C", justify="right")
    t.add_column("IB tx/rx GB/s", justify="right")
    t.add_column("status")
    for s in samples:
        if s.error:
            t.add_row(s.host, "-", "-", "-", "-", f"[red]err: {s.error[:30]}[/red]")
            continue
        if s.ib_tx_GBps is not None and s.ib_rx_GBps is not None:
            ib = f"{s.ib_tx_GBps:.1f} / {s.ib_rx_GBps:.1f}"
        else:
            ib = "  - / -"
        t.add_row(
            s.host,
            " ".join(str(u) for u in s.gpu_util),
            " ".join(str(m) for m in s.gpu_mem_used_mib),
            " ".join(str(c) for c in s.gpu_temp_c),
            ib,
            "[red]STALE HEARTBEAT[/red]" if stale else "ok",
        )
    return t


def run_monitor(cfg_path: Path, job_id: str) -> int:
    from .config import load_cluster_config
    from .types import JobManifest

    cfg = load_cluster_config(cfg_path)
    job_dir = Path(cfg_path).parent / "jobs" / job_id
    m_path = job_dir / "manifest.json"
    if not m_path.exists():
        print(f"monitor: manifest not found at {m_path}", flush=True)
        return 1
    job = JobManifest.from_json(m_path.read_text())
    heartbeat_path = (
        Path(cfg.paths.repo) / "checkpoints" / job.exp_name / "heartbeat"
    )

    prev_ib: dict[str, tuple[float, int, int]] = {}
    started = time.time()

    async def _go() -> None:
        with Live(
            _render(job, [], 0, False), refresh_per_second=1, screen=False
        ) as live:
            while True:
                coros = [
                    _sample_one(n.host, job.job_id, cfg, prev_ib) for n in job.nodes
                ]
                samples = await asyncio.gather(*coros)
                stale = is_stale(heartbeat_path, threshold_s=600.0)
                live.update(_render(job, samples, time.time() - started, stale))
                await asyncio.sleep(2.0)

    try:
        asyncio.run(_go())
    except KeyboardInterrupt:
        pass
    return 0

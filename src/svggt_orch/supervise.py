"""Stream per-rank `docker logs -f` from each worker node, demux to stdout."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.text import Text

if TYPE_CHECKING:
    from .config import ClusterConfig


_RANK_COLORS = ["cyan", "magenta", "yellow", "green", "blue", "red", "white"]


async def _tail_one(
    rank: int,
    host: str,
    container: str,
    job_dir: Path,
    cfg: "ClusterConfig",
    console: Console,
) -> None:
    import asyncssh  # lazy

    color = _RANK_COLORS[rank % len(_RANK_COLORS)]
    prefix = f"[r={rank:02d} h={host}] "
    out_file = job_dir / f"rank-{rank:02d}.log"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        async with asyncssh.connect(
            host,
            username=cfg.ssh.user,
            client_keys=[str(Path(cfg.ssh.identity_file).expanduser())],
            known_hosts=None,
            connect_timeout=cfg.ssh.connect_timeout_s,
        ) as conn:
            with out_file.open("a", buffering=1) as fp:
                proc = await conn.create_process(
                    f"docker logs -f --since=0s {container} 2>&1"
                )
                async for line in proc.stdout:
                    s = line.rstrip("\n")
                    fp.write(s + "\n")
                    console.print(Text(prefix + s, style=color))
    except (OSError, asyncssh.Error) as e:
        console.print(Text(f"{prefix}[supervise] disconnected: {e}", style="red"))


def run_tail(cfg_path: Path, job_id: str) -> int:
    from .config import load_cluster_config
    from .types import JobManifest

    cfg = load_cluster_config(cfg_path)
    job_dir = Path(cfg_path).parent / "jobs" / job_id
    m_path = job_dir / "manifest.json"
    if not m_path.exists():
        print(f"tail: manifest not found at {m_path}", flush=True)
        return 1
    manifest = JobManifest.from_json(m_path.read_text())
    console = Console()
    console.print(
        f"[bold]tail[/bold] {job_id}  containers={manifest.container_name}"
    )

    async def _go() -> None:
        coros = [
            _tail_one(rank, n.host, manifest.container_name, job_dir, cfg, console)
            for rank, n in enumerate(manifest.nodes)
        ]
        await asyncio.gather(*coros)

    try:
        asyncio.run(_go())
    except KeyboardInterrupt:
        console.print("[yellow]tail interrupted.[/yellow]")
    return 0

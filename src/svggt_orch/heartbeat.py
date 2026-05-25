"""Heartbeat freshness watchdog for the orchestrator.

Reads <output_dir>/heartbeat and emits a red banner via `rich` if mtime is
older than HANG_THRESHOLD_S (default 600). Notify only — no auto-kill.
"""
from __future__ import annotations

import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel


def is_stale(path: Path, threshold_s: float, now: "float | None" = None) -> bool:
    if not path.exists():
        return True
    age = (now or time.time()) - path.stat().st_mtime
    return age > threshold_s


def render_banner(path: Path, threshold_s: float) -> Panel:
    age = time.time() - path.stat().st_mtime if path.exists() else float("inf")
    msg = (
        f"[bold red]HEARTBEAT STALE[/bold red]\n"
        f"file: {path}\n"
        f"age:  {age:.0f}s (threshold {threshold_s:.0f}s)\n"
        f"Investigate with `svggt-orch tail` or `svggt-orch status`."
    )
    return Panel(msg, border_style="red")


def watch_heartbeat(
    output_dir: Path,
    threshold_s: float = 600.0,
    poll_s: float = 30.0,
) -> None:
    """Blocking watcher; prints a banner once each time the file becomes stale."""
    path = output_dir / "heartbeat"
    console = Console()
    was_stale = False
    while True:
        stale = is_stale(path, threshold_s)
        if stale and not was_stale:
            console.print(render_banner(path, threshold_s))
        was_stale = stale
        time.sleep(poll_s)

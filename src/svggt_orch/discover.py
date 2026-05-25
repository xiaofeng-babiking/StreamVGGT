"""GPU discovery across the cluster.

Two layers:
  * Pure parsing helpers (this file) — unit-tested.
  * `discover_all()` async coroutine — runs the asyncssh fan-out (added next task).
"""
from __future__ import annotations

import json
from typing import Optional


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

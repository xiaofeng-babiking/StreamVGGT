"""Launch TensorBoard against a job's logs and print the SSH port-forward command."""
from __future__ import annotations

import os
import socket
import subprocess
from pathlib import Path


def run_tb(cfg_path: Path, job_id: str, port: int) -> int:
    from .config import load_cluster_config
    from .types import JobManifest

    cfg = load_cluster_config(cfg_path)
    job_dir = Path(cfg_path).parent / "jobs" / job_id
    m_path = job_dir / "manifest.json"
    if not m_path.exists():
        print(f"tb: manifest not found at {m_path}", flush=True)
        return 1
    job = JobManifest.from_json(m_path.read_text())
    logdir = Path(cfg.paths.repo) / "checkpoints" / job.exp_name / "logs"
    if not logdir.exists():
        print(f"tb: logs dir not found yet: {logdir}", flush=True)
        print("    (rank 0 will create it once training starts)", flush=True)

    host = socket.gethostname()
    print(f"tb: spawning tensorboard --logdir {logdir} --port {port}")
    print("tb: on your laptop, run:")
    print(f"    ssh -L {port}:localhost:{port} {os.getenv('USER', 'jing.feng')}@{host}")
    print(f"    then open http://localhost:{port}")
    try:
        subprocess.run(
            [
                "tensorboard",
                "--logdir", str(logdir),
                "--port", str(port),
                "--bind_all",
            ],
            check=False,
        )
    except FileNotFoundError:
        print("tb: `tensorboard` binary not found. Install via `uv add tensorboard`.", flush=True)
        return 1
    return 0

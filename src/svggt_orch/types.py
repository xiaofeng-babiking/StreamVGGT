"""Data models shared across orchestrator modules."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class NodeInfo:
    host: str
    bond_ip: str
    ib_ip: Optional[str]
    free_gpus: list[int]
    detected_at: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "NodeInfo":
        return cls(
            host=d["host"],
            bond_ip=d["bond_ip"],
            ib_ip=d.get("ib_ip"),
            free_gpus=list(d["free_gpus"]),
            detected_at=d["detected_at"],
        )


@dataclass
class JobManifest:
    job_id: str
    exp_name: str
    nodes: list[NodeInfo]
    master_host: str
    master_ip: str
    main_port: int
    world_size: int
    container_name: str
    started_at: str
    status: str = "running"
    exit_codes: dict[str, int] = field(default_factory=dict)

    def to_json(self) -> str:
        payload = {
            "job_id": self.job_id,
            "exp_name": self.exp_name,
            "nodes": [n.to_dict() for n in self.nodes],
            "master_host": self.master_host,
            "master_ip": self.master_ip,
            "main_port": self.main_port,
            "world_size": self.world_size,
            "container_name": self.container_name,
            "started_at": self.started_at,
            "status": self.status,
            "exit_codes": self.exit_codes,
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "JobManifest":
        d = json.loads(s)
        return cls(
            job_id=d["job_id"],
            exp_name=d["exp_name"],
            nodes=[NodeInfo.from_dict(n) for n in d["nodes"]],
            master_host=d["master_host"],
            master_ip=d["master_ip"],
            main_port=int(d["main_port"]),
            world_size=int(d["world_size"]),
            container_name=d["container_name"],
            started_at=d["started_at"],
            status=d.get("status", "running"),
            exit_codes=dict(d.get("exit_codes", {})),
        )

import json
from svggt_orch.types import NodeInfo, JobManifest


def test_nodeinfo_roundtrip():
    n = NodeInfo(
        host="gpu003",
        bond_ip="172.31.208.5",
        ib_ip="10.10.100.5",
        free_gpus=[0, 1, 2, 3, 4, 5],
        detected_at="2026-05-25T16:23:00Z",
    )
    d = n.to_dict()
    n2 = NodeInfo.from_dict(d)
    assert n2 == n


def test_nodeinfo_ib_ip_optional():
    n = NodeInfo(
        host="gpu003",
        bond_ip="172.31.208.5",
        ib_ip=None,
        free_gpus=[0],
        detected_at="2026-05-25T16:23:00Z",
    )
    n2 = NodeInfo.from_dict(n.to_dict())
    assert n2.ib_ip is None


def test_jobmanifest_json_roundtrip():
    m = JobManifest(
        job_id="20260525-162300",
        exp_name="StreamVGGT_smoke",
        nodes=[
            NodeInfo("gpu003", "172.31.208.5", "10.10.100.5", [0, 1], "t"),
            NodeInfo("gpu004", "172.31.208.6", "10.10.100.6", [0, 1], "t"),
        ],
        master_host="gpu003",
        master_ip="10.10.100.5",
        main_port=26902,
        world_size=4,
        container_name="svggt-20260525-162300",
        started_at="2026-05-25T16:23:00Z",
    )
    s = m.to_json()
    m2 = JobManifest.from_json(s)
    assert m2 == m
    assert json.loads(s)["world_size"] == 4


def test_jobmanifest_default_status():
    m = JobManifest(
        job_id="x", exp_name="x", nodes=[], master_host="x",
        master_ip="x", main_port=0, world_size=0,
        container_name="x", started_at="x",
    )
    assert m.status == "running"
    assert m.exit_codes == {}

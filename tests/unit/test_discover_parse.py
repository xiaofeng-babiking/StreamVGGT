from pathlib import Path

from svggt_orch.discover import (
    parse_nvidia_smi_csv,
    filter_idle_gpus,
    parse_ip_addr_show,
    dry_run_summary,
)


def test_parse_nvidia_smi_csv_six_gpus():
    text = "\n".join([
        "0, 12",
        "1, 8",
        "2, 80432",
        "3, 32",
        "4, 16",
        "5, 11",
    ])
    out = parse_nvidia_smi_csv(text)
    assert out == [(0, 12), (1, 8), (2, 80432), (3, 32), (4, 16), (5, 11)]


def test_parse_nvidia_smi_csv_strips_whitespace():
    text = "  0 ,   500   "
    assert parse_nvidia_smi_csv(text) == [(0, 500)]


def test_parse_nvidia_smi_csv_empty_lines_ignored():
    text = "\n\n0, 12\n\n1, 8\n"
    assert parse_nvidia_smi_csv(text) == [(0, 12), (1, 8)]


def test_filter_idle_gpus_default_threshold():
    samples = [(0, 12), (1, 8), (2, 80432), (3, 32), (4, 16), (5, 11)]
    idle = filter_idle_gpus(samples, threshold_mib=500)
    assert idle == [0, 1, 3, 4, 5]


def test_filter_idle_gpus_strict_threshold():
    samples = [(0, 100), (1, 500), (2, 499)]
    # threshold is exclusive (memory < threshold)
    assert filter_idle_gpus(samples, threshold_mib=500) == [0, 2]


def test_parse_ip_addr_show_extracts_inet():
    # Trimmed `ip -j addr show ibs110` output
    j = """[{"ifname":"ibs110","addr_info":[{"family":"inet","local":"10.10.100.5","prefixlen":21},{"family":"inet6","local":"fe80::1","prefixlen":64}]}]"""
    assert parse_ip_addr_show(j) == "10.10.100.5"


def test_parse_ip_addr_show_missing_returns_none():
    assert parse_ip_addr_show("[]") is None
    j = """[{"ifname":"ibs110","addr_info":[{"family":"inet6","local":"fe80::1"}]}]"""
    assert parse_ip_addr_show(j) is None


def test_dry_run_summary_format():
    summary = dry_run_summary(
        hosts=["gpu003", "gpu004"],
        idle_threshold_mib=500,
        ssh_user="jing.feng",
        identity_file="~/.ssh/id_ed25519",
        timeout_s=5,
    )
    golden = Path(__file__).resolve().parents[1] / "golden" / "discover.txt"
    assert summary.rstrip() == golden.read_text().rstrip()

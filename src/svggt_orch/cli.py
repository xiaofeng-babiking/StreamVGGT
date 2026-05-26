"""svggt-orch CLI entry point.

Subcommands are wired into this module one by one in subsequent tasks.
For now we ship the click group and `--help`, plus stubs that exit cleanly.
"""
from __future__ import annotations

from pathlib import Path

import click

DEFAULT_CONFIG = Path("workflows/cluster.yaml")


@click.group()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=False, dir_okay=False, path_type=Path),
    default=DEFAULT_CONFIG,
    show_default=True,
    help="Path to cluster.yaml",
)
@click.pass_context
def main(ctx: click.Context, config_path: Path) -> None:
    """StreamVGGT multi-node training orchestrator."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path


@main.command()
@click.option("--dry-run", is_flag=True, help="Print the plan; do not SSH anywhere.")
@click.pass_context
def discover(ctx: click.Context, dry_run: bool) -> None:
    """Scan candidate hosts for idle GPUs and write workflows/nodes.json."""
    from svggt_orch.discover import run_discover
    code = run_discover(ctx.obj["config_path"], dry_run=dry_run)
    ctx.exit(code)


@main.command()
@click.option("--dry-run", is_flag=True)
@click.option("--exp-name", default=None, help="Override exp_name (else uses train.yaml default)")
@click.option(
    "--config-name",
    "config_name",
    default=None,
    help=(
        "Hydra config under config/ (e.g. train, train_smoke, finetune). "
        "Overrides cluster.yaml#train.config_name; defaults to the cluster.yaml value."
    ),
)
@click.pass_context
def launch(
    ctx: click.Context,
    dry_run: bool,
    exp_name: "str | None",
    config_name: "str | None",
) -> None:
    """Launch the distributed training job."""
    from svggt_orch.launch import run_launch
    code = run_launch(
        ctx.obj["config_path"],
        dry_run=dry_run,
        exp_name=exp_name,
        config_name=config_name,
    )
    ctx.exit(code)


@main.command()
@click.option("--job", "job_id", required=True, help="Job id from launch")
@click.pass_context
def tail(ctx: click.Context, job_id: str) -> None:
    """Stream per-rank logs from all worker containers."""
    from svggt_orch.supervise import run_tail
    code = run_tail(ctx.obj["config_path"], job_id)
    ctx.exit(code)


@main.command()
@click.option("--job", "job_id", required=True)
@click.pass_context
def monitor(ctx: click.Context, job_id: str) -> None:
    """Live TUI dashboard of GPU and IB stats."""
    from svggt_orch.monitor import run_monitor
    code = run_monitor(ctx.obj["config_path"], job_id)
    ctx.exit(code)


@main.command()
@click.option("--job", "job_id", required=True)
@click.option("--port", default=6006, show_default=True)
@click.pass_context
def tb(ctx: click.Context, job_id: str, port: int) -> None:
    """Launch TensorBoard against this job's logs."""
    from svggt_orch.tb import run_tb
    code = run_tb(ctx.obj["config_path"], job_id, port)
    ctx.exit(code)


@main.command()
@click.option("--job", "job_id", required=True)
@click.pass_context
def status(ctx: click.Context, job_id: str) -> None:
    """One-shot status snapshot for a job."""
    from svggt_orch.launch import status_snapshot
    code = status_snapshot(ctx.obj["config_path"], job_id)
    ctx.exit(code)


@main.command()
@click.option("--job", "job_id", required=True)
@click.confirmation_option(prompt="Kill all containers for this job?")
@click.pass_context
def kill(ctx: click.Context, job_id: str) -> None:
    """Kill worker containers across all nodes for a job."""
    from svggt_orch.launch import kill_job
    code = kill_job(ctx.obj["config_path"], job_id)
    ctx.exit(code)


if __name__ == "__main__":
    main()

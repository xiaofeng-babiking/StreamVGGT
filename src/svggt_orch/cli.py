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
@click.pass_context
def discover(ctx: click.Context) -> None:
    """Scan candidate hosts for idle GPUs and write workflows/nodes.json."""
    click.echo("discover: not implemented yet", err=True)
    ctx.exit(2)


@main.command()
@click.pass_context
def launch(ctx: click.Context) -> None:
    """Launch the distributed training job."""
    click.echo("launch: not implemented yet", err=True)
    ctx.exit(2)


@main.command()
@click.option("--job", "job_id", required=True, help="Job id from launch")
@click.pass_context
def tail(ctx: click.Context, job_id: str) -> None:
    """Stream per-rank logs from all worker containers."""
    click.echo(f"tail {job_id}: not implemented yet", err=True)
    ctx.exit(2)


@main.command()
@click.option("--job", "job_id", required=True)
@click.pass_context
def monitor(ctx: click.Context, job_id: str) -> None:
    """Live TUI dashboard of GPU and IB stats."""
    click.echo(f"monitor {job_id}: not implemented yet", err=True)
    ctx.exit(2)


@main.command()
@click.option("--job", "job_id", required=True)
@click.option("--port", default=6006, show_default=True)
@click.pass_context
def tb(ctx: click.Context, job_id: str, port: int) -> None:
    """Launch TensorBoard against this job's logs."""
    click.echo(f"tb {job_id}: not implemented yet", err=True)
    ctx.exit(2)


@main.command()
@click.option("--job", "job_id", required=True)
@click.pass_context
def status(ctx: click.Context, job_id: str) -> None:
    """One-shot status snapshot for a job."""
    click.echo(f"status {job_id}: not implemented yet", err=True)
    ctx.exit(2)


@main.command()
@click.option("--job", "job_id", required=True)
@click.confirmation_option(prompt="Kill all containers for this job?")
@click.pass_context
def kill(ctx: click.Context, job_id: str) -> None:
    """Kill worker containers across all nodes for a job."""
    click.echo(f"kill {job_id}: not implemented yet", err=True)
    ctx.exit(2)


if __name__ == "__main__":
    main()

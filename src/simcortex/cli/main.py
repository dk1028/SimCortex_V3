from __future__ import annotations

import shlex
import subprocess
import sys
from collections.abc import Sequence

import typer


app = typer.Typer(
    help="Fixed-initialization MRI-only cortical surface deformation."
)

FORWARD_CONTEXT_SETTINGS = {
    "allow_extra_args": True,
    "ignore_unknown_options": True,
}


def _format_command(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def run_module(
    module: str,
    overrides: Sequence[str] | None = None,
    *,
    torchrun: bool = False,
    nproc_per_node: int = 1,
) -> int:
    if nproc_per_node < 1:
        raise typer.BadParameter(
            f"--nproc-per-node must be >= 1, got {nproc_per_node}."
        )

    forwarded_args = list(overrides or [])

    if torchrun:
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={nproc_per_node}",
            "-m",
            module,
        ]
    else:
        cmd = [sys.executable, "-m", module]

    cmd.extend(forwarded_args)

    typer.echo(f"[fixed-init] Running: {_format_command(cmd)}")
    completed = subprocess.run(cmd, check=False)
    return completed.returncode


@app.command(
    "train",
    help="Train the MRI-only fixed-initialization deformation model.",
    context_settings=FORWARD_CONTEXT_SETTINGS,
)
def train(
    ctx: typer.Context,
    torchrun: bool = typer.Option(
        False,
        "--torchrun",
        help="Launch training with torch.distributed.run.",
    ),
    nproc_per_node: int = typer.Option(
        1,
        "--nproc-per-node",
        min=1,
        help="Number of processes/GPUs per node with --torchrun.",
    ),
) -> None:
    raise typer.Exit(
        run_module(
            "simcortex.deform.train",
            ctx.args,
            torchrun=torchrun,
            nproc_per_node=nproc_per_node,
        )
    )


@app.command(
    "infer",
    help="Run MRI-only fixed-template deformation inference.",
    context_settings=FORWARD_CONTEXT_SETTINGS,
)
def infer(ctx: typer.Context) -> None:
    raise typer.Exit(
        run_module(
            "simcortex.deform.inference",
            ctx.args,
        )
    )


@app.command(
    "eval",
    help="Evaluate predicted cortical surfaces and optional collision metrics.",
    context_settings=FORWARD_CONTEXT_SETTINGS,
)
def evaluate(ctx: typer.Context) -> None:
    raise typer.Exit(
        run_module(
            "simcortex.deform.eval",
            ctx.args,
        )
    )


if __name__ == "__main__":
    app()

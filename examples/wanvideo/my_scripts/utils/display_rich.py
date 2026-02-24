"""
display_rich.py  –  Run a shell command and display its output using rich.

Public API
----------
    from utils.display_rich import run_command, run_batch

    # single stage
    ok = run_command(command, description, stage_num, total_stages)  # -> bool

    # batch of tasks
    results = run_batch(tasks, overall_description, max_log_lines)   # -> list[bool]

Behaviour
---------
  • Captures stdout + stderr together.
  • Shows a live rolling window of the last WINDOW_LINES lines so tqdm/progress
    bars from the subprocess never flood the terminal.
  • Colour-highlights progress bars, loss lines, warnings and errors.
  • Prints a start panel and a completion/failure panel around the run.
  • Returns True on exit-code 0, False otherwise.
"""

from __future__ import annotations

import subprocess
import sys
import time
from collections import deque
from datetime import datetime

from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from utils.display_common import _highlight

# ── tunables ──────────────────────────────────────────────────────────────────
WINDOW_LINES   = 20   # how many recent lines to keep visible
REFRESH_RATE   = 6    # live-display fps

# ── module-level console (shared within one process) ─────────────────────────
_con = Console()


def _make_live_renderable(buf: deque, line_count: int, elapsed: float) -> Table:
    """Build the Table that rich.Live re-renders on each refresh."""
    stats = Text()
    stats.append(f"lines: {line_count}  |  ", style="dim")
    stats.append(f"elapsed: {elapsed:.1f}s", style="bold blue")

    inner = Table(
        show_header=False,
        box=box.SIMPLE,
        padding=(0, 1),
        collapse_padding=True,
        expand=True,
    )
    inner.add_column("out")
    for l in buf:
        inner.add_row(_highlight(l))

    outer = Table.grid(expand=True)
    outer.add_row(stats)
    outer.add_row("")
    outer.add_row(inner)
    return outer


# ── public ────────────────────────────────────────────────────────────────────

def run_command(
    command: str,
    description: str = "",
    stage_num: int = 1,
    total_stages: int = 1,
) -> bool:
    """
    Run *command* in a subprocess and display output via a rich rolling window.

    Parameters
    ----------
    command       : shell command to execute
    description   : human-readable label
    stage_num     : current stage number (for display)
    total_stages  : total number of stages  (for display)

    Returns
    -------
    True if exit code == 0, False otherwise.
    """
    start = time.time()

    _con.print()
    _con.print(Panel(
        f"[bold cyan]{description}[/bold cyan]\n"
        f"[dim]Stage {stage_num}/{total_stages}  •  "
        f"started {datetime.now().strftime('%H:%M:%S')}[/dim]",
        title=f"[bold green]▶  Stage {stage_num}[/bold green]",
        border_style="cyan",
        box=box.DOUBLE,
    ))
    short = command[:160] + "…" if len(command) > 160 else command
    _con.print(f"\n[dim]CMD:[/dim] [yellow]{short}[/yellow]\n")

    buf        = deque(maxlen=WINDOW_LINES)
    line_count = 0

    proc = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
    )

    with Live(console=_con, refresh_per_second=REFRESH_RATE) as live:
        try:
            for raw in proc.stdout:
                line = raw.rstrip()
                if line:
                    line_count += 1
                    buf.append(line)
                    live.update(_make_live_renderable(
                        buf, line_count, time.time() - start
                    ))
        except KeyboardInterrupt:
            _con.print("\n[bold red]⚠  Interrupted by user[/bold red]")
            proc.terminate()
            return False

    proc.wait()
    elapsed = time.time() - start

    if proc.returncode == 0:
        _con.print(Panel(
            f"[bold green]✓  Completed[/bold green]\n"
            f"[dim]{elapsed:.1f}s  ({elapsed/60:.1f} min)  •  "
            f"lines: {line_count}[/dim]",
            title=f"[bold green]Stage {stage_num} done[/bold green]",
            border_style="green",
            box=box.DOUBLE,
        ))
        _con.print()
        return True

    _con.print(Panel(
        f"[bold red]✗  Failed  (exit {proc.returncode})[/bold red]\n"
        f"[dim]{elapsed:.1f}s[/dim]",
        title=f"[bold red]Stage {stage_num} failed[/bold red]",
        border_style="red",
        box=box.DOUBLE,
    ))
    _con.print()
    return False


# ── batch runner ──────────────────────────────────────────────────────────────

def run_batch(
    tasks: list[tuple[str, str]],
    overall_description: str = "Processing...",
    max_log_lines: int = 20,
) -> list[bool]:
    """
    Run a list of (label, command) pairs sequentially.

    Shows an overall rich Progress bar across all tasks, plus a Live
    rolling-window panel for each running subprocess.

    Parameters
    ----------
    tasks                : list of (label, command) tuples
    overall_description  : label shown in the overall progress bar
    max_log_lines        : rolling window size for subprocess output

    Returns
    -------
    list[bool] — True if exit code == 0, one entry per task.
    """
    from rich.progress import (
        BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn,
    )

    total   = len(tasks)
    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=_con,
    ) as progress:
        overall = progress.add_task(f"[cyan]{overall_description}", total=total)

        for idx, (label, command) in enumerate(tasks, 1):
            _con.print(f"\n[bold cyan]→[/bold cyan]  [{idx}/{total}]: {label}")
            progress.update(overall, description=f"[cyan][{idx}/{total}]: {label}")

            short = command[:160] + "…" if len(command) > 160 else command
            _con.print(f"[dim]CMD:[/dim] [yellow]{short}[/yellow]\n")

            buf = deque(maxlen=max_log_lines)

            proc = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            )

            with Live(console=_con, refresh_per_second=REFRESH_RATE) as live:
                try:
                    for raw in proc.stdout:
                        line = raw.rstrip()
                        if line:
                            buf.append(line)
                            live.update(Panel(
                                "\n".join(_highlight(l) for l in buf),
                                title=f"[bold green]{label}[/bold green]",
                                subtitle=f"[dim]last {len(buf)} lines[/dim]",
                                border_style="blue",
                            ))
                except KeyboardInterrupt:
                    _con.print("\n[bold red]⚠  Interrupted by user[/bold red]")
                    proc.terminate()
                    results.append(False)
                    progress.advance(overall)
                    continue

            proc.wait()
            ok = proc.returncode == 0
            if ok:
                _con.print(f"[green]✓[/green]  Completed: {label}")
            else:
                _con.print(f"[red]✗[/red]  Failed: {label} (exit code: {proc.returncode})")
            results.append(ok)
            progress.advance(overall)

    passed = sum(results)
    _con.print(f"\n[bold green]Done:[/bold green] {passed}/{total} succeeded, "
               f"[bold red]{total - passed}[/bold red] failed.\n")
    return results

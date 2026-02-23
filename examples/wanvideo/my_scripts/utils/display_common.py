"""
display_common.py  –  Shared display helpers used by all display backends
                      and automation scripts.

Exports
-------
    _highlight(line)              colour-markup a single log line (rich syntax)
    print_config(args_or_dict)    pretty-print an argparse Namespace or dict
    print_summary(rows, title)    pretty-print a completion summary table
    die(msg)                      print an error message and sys.exit(1)
    run_command_plain(...)        bare-bones subprocess runner, no formatting
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime


# ══════════════════════════════════════════════════════════════════════════════
# Line highlighter  (shared by display_rich and display_textual)
# ══════════════════════════════════════════════════════════════════════════════

def _highlight(line: str) -> str:
    """
    Return a rich-markup-wrapped version of *line* based on its content.
    Safe to use inside both rich.Live and textual.widgets.RichLog.
    """
    lo   = line.lower()
    safe = line.replace("[", r"\[")   # escape brackets so rich won't mis-parse
    if "%" in line or "it/s" in line or "epoch" in lo:
        return f"[bold cyan]{safe}[/bold cyan]"
    if "error" in lo or "fail" in lo or "traceback" in lo:
        return f"[bold red]{safe}[/bold red]"
    if "warn" in lo:
        return f"[yellow]{safe}[/yellow]"
    if "loss" in lo or "step" in lo or "lr" in lo:
        return f"[green]{safe}[/green]"
    return safe


# ══════════════════════════════════════════════════════════════════════════════
# Config / summary printers  (rich with plain fallback)
# ══════════════════════════════════════════════════════════════════════════════

def print_config(args) -> None:
    """
    Print a formatted configuration table.

    Parameters
    ----------
    args : argparse.Namespace  or  dict
    """
    data = vars(args) if hasattr(args, "__dict__") else dict(args)
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
        t = Table(title="[bold]Run configuration[/bold]",
                  box=box.ROUNDED, border_style="blue")
        t.add_column("Argument", style="cyan",   no_wrap=True)
        t.add_column("Value",    style="yellow")
        for k, v in data.items():
            t.add_row(str(k), str(v))
        con = Console()
        con.print()
        con.print(t)
        con.print()
    except ImportError:
        print("\n-- Configuration --")
        for k, v in data.items():
            print(f"  {k}: {v}")
        print()


def print_summary(rows: list[tuple[str, str]], title: str = "✓  Completed") -> None:
    """
    Print a two-column completion summary table.

    Parameters
    ----------
    rows  : list of (label, value) tuples
    title : table title string
    """
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
        t = Table(title=f"[bold green]{title}[/bold green]",
                  box=box.DOUBLE_EDGE, border_style="green")
        t.add_column("Stage",  style="cyan",   no_wrap=True)
        t.add_column("Output", style="yellow")
        for label, value in rows:
            t.add_row(label, value)
        con = Console()
        con.print()
        con.print(t)
        con.print()
    except ImportError:
        print(f"\n{title}")
        for label, value in rows:
            print(f"  {label}  →  {value}")
        print()


# ══════════════════════════════════════════════════════════════════════════════
# Error helper
# ══════════════════════════════════════════════════════════════════════════════

def die(msg: str) -> None:
    """Print *msg* as a bold-red error and exit with code 1."""
    try:
        from rich.console import Console
        Console().print(f"[bold red]❌  {msg}[/bold red]")
    except ImportError:
        print(f"❌  {msg}", file=sys.stderr)
    sys.exit(1)

def success(msg: str) -> None:
    """Print *msg* as a bold-green success message."""
    try:
        from rich.console import Console
        Console().print(f"[bold green]✅  {msg}[/bold green]")
    except ImportError:
        print(f"✅  {msg}")


# ══════════════════════════════════════════════════════════════════════════════
# Plain runner  (no rich / textual dependency)
# ══════════════════════════════════════════════════════════════════════════════

def run_command_plain(
    command: str,
    description: str = "",
    stage_num: int = 1,
    total_stages: int = 1,
) -> bool:
    """
    Run *command* with raw stdout passthrough and minimal formatting.
    Safe for non-TTY environments (CI, log files, etc.).

    Returns True on exit code 0, False otherwise.
    """
    sep = "=" * 80
    ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{sep}")
    print(f"[{ts}]  Stage {stage_num}/{total_stages}: {description}")
    print(f"{sep}\nCMD: {command}\n")

    proc = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
    )
    try:
        for line in proc.stdout:
            print(line, end="")
            sys.stdout.flush()
    except KeyboardInterrupt:
        print("\n⚠  Interrupted by user.")
        proc.terminate()
        return False

    proc.wait()
    ok = proc.returncode == 0
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{sep}")
    print(f"[{ts}]  {'✓  Done' if ok else f'✗  Failed (exit {proc.returncode})'}  –  {description}")
    print(f"{sep}\n")
    return ok

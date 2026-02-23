"""
display_textual.py  –  Run a shell command inside a full-screen Textual TUI.

Public API
----------
    from utils.display_textual import run_command

    ok = run_command(
        command      = "python train.py",
        description  = "Training",
        stage_num    = 1,
        total_stages = 2,
    )  # -> bool

Behaviour
---------
  • Takes over the full terminal for a proper TUI.
  • A RichLog pane accumulates ALL output lines – fully scrollable while the
    process is still running (mouse wheel or arrow keys).
  • Header shows stage info and a wall-clock timer that updates every second.
  • Footer shows key bindings.
  • Auto-closes once the subprocess finishes; press q / Ctrl-C to interrupt.
  • Falls back to display_rich.run_command if textual is not installed.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime


def run_command(
    command: str,
    description: str = "",
    stage_num: int = 1,
    total_stages: int = 1,
) -> bool:
    """
    Run *command* inside a full-screen Textual TUI with a scrollable log.

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
    try:
        from textual.app import App, ComposeResult
        from textual.binding import Binding
        from textual.widgets import Footer, RichLog, Static
        from textual import work
    except ImportError:
        print("⚠  textual not installed (pip install textual). Falling back to rich.")
        from utils.display_rich import run_command as _rich_run
        return _rich_run(command, description, stage_num, total_stages)

    from utils.display_common import _highlight as _hl

    # ── Textual app ───────────────────────────────────────────────────────────
    class RunApp(App):

        CSS = """
        Screen { layout: vertical; }

        #header {
            height: 3;
            background: $surface;
            border: double cyan;
            padding: 0 2;
            content-align: left middle;
        }

        #stats {
            height: 1;
            background: $boost;
            padding: 0 2;
            color: $text-muted;
        }

        #log {
            height: 1fr;
            border: solid $primary;
        }
        """

        BINDINGS = [
            Binding("ctrl+c", "interrupt", "Interrupt"),
            Binding("q",      "interrupt", "Quit / close"),
        ]

        TITLE = f"Stage {stage_num}/{total_stages}  –  {description}"

        def __init__(self):
            super().__init__()
            self._rc:    int | None = None
            self._start: float     = time.time()
            self._proc:  asyncio.subprocess.Process | None = None
            self._lines: int = 0

        # ── layout ────────────────────────────────────────────────────────────
        def compose(self) -> ComposeResult:
            yield Static(
                f"[bold cyan]Stage {stage_num}/{total_stages}[/bold cyan]  "
                f"[white]{description}[/white]  "
                f"[dim]started {datetime.now().strftime('%H:%M:%S')}[/dim]",
                id="header",
            )
            yield Static("Starting process…", id="stats")
            yield RichLog(id="log", highlight=True, markup=True, wrap=True,
                          auto_scroll=True)
            yield Footer()

        # ── startup ───────────────────────────────────────────────────────────
        def on_mount(self) -> None:
            self.set_interval(1.0, self._tick)
            self._stream()

        def _tick(self) -> None:
            """Update clock in stats bar every second."""
            elapsed = time.time() - self._start
            running = self._rc is None
            status  = "[yellow]running…[/yellow]" if running else (
                "[bold green]✓ done[/bold green]" if self._rc == 0
                else f"[bold red]✗ failed (exit {self._rc})[/bold red]"
            )
            self.query_one("#stats", Static).update(
                f"{status}  |  lines: {self._lines}  |  "
                f"elapsed: {elapsed:.0f}s  |  "
                f"[dim]scroll ↑↓ / mouse  •  q or Ctrl-C to close[/dim]"
            )

        @work(exclusive=True)
        async def _stream(self) -> None:
            log = self.query_one("#log", RichLog)

            self._proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

            async for raw in self._proc.stdout:
                line = raw.decode(errors="replace").rstrip()
                if line:
                    self._lines += 1
                    log.write(_hl(line))

            await self._proc.wait()
            self._rc = self._proc.returncode

            elapsed = time.time() - self._start
            if self._rc == 0:
                log.write(
                    f"\n[bold green]✓  Completed in "
                    f"{elapsed:.1f}s ({elapsed/60:.1f} min)[/bold green]  "
                    f"[dim]– press q to close[/dim]"
                )
            else:
                log.write(
                    f"\n[bold red]✗  Failed (exit {self._rc}) "
                    f"after {elapsed:.1f}s[/bold red]  "
                    f"[dim]– press q to close[/dim]"
                )
            self._tick()       # force immediate stats refresh
            self.exit()        # auto-close so the pipeline continues

        # ── actions ───────────────────────────────────────────────────────────
        def action_interrupt(self) -> None:
            if self._proc and self._proc.returncode is None:
                self._proc.terminate()
                self._rc = -1
            self.exit()

    app = RunApp()
    app.run()
    return app._rc == 0

"""
Rich-based progress tracker that monitors rendered frames by polling
output directories.

Since child Blender processes are opaque subprocesses, we can't read their
internal progress.  Instead we periodically count ``frame*.png`` files in
each output directory and compare against the expected total.

Falls back gracefully to plain-text logging when ``rich`` is not installed.

Usage
-----
::

    tracker = FrameProgressTracker(
        dir_expectations={"./out/Walk/cam_0": 100, "./out/Walk/cam_1": 100},
        poll_interval=2.0,
    )
    tracker.start()          # starts background polling thread

    # ... run your ThreadPoolExecutor jobs here ...

    tracker.stop()           # final update + stop polling
    tracker.print_summary()  # optional failure report
"""

import glob
import os
import threading
import time
from collections import defaultdict

try:
    from rich.progress import (
        Progress,
        BarColumn,
        MofNCompleteColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
        SpinnerColumn,
    )
    from rich.console import Console
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


# ---------------------------------------------------------------------------
# Frame counter (shared between rich and fallback paths)
# ---------------------------------------------------------------------------

def count_rendered_frames(directory: str) -> int:
    """Count frame*.png files in *directory*."""
    try:
        return len(glob.glob(os.path.join(directory, "frame*.png")))
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Rich progress tracker
# ---------------------------------------------------------------------------

class FrameProgressTracker:
    """
    Background-polling progress display using ``rich``.

    Parameters
    ----------
    dir_expectations : dict[str, int]
        Mapping from output directory path to the total number of frames
        expected in that directory (across all chunks that write there).
    poll_interval : float
        Seconds between directory polls.
    """

    def __init__(self, dir_expectations: dict, poll_interval: float = 2.0):
        self.dir_expectations = dir_expectations
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._thread = None
        self._progress = None
        self._tasks = {}           # out_dir  -> rich task id
        self._overall_task = None
        self._job_results = []     # appended externally

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Create the rich Progress display and begin polling."""
        if not HAS_RICH:
            print("[progress] rich not installed — using plain-text output",
                  flush=True)
            return

        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            expand=False,
        )
        self._progress.start()

        total_expected = 0
        for out_dir in sorted(self.dir_expectations):
            expected = self.dir_expectations[out_dir]
            # Short human-readable label: last two path components
            parts = out_dir.rstrip("/").split("/")
            label = "/".join(parts[-2:]) if len(parts) >= 2 else out_dir
            task = self._progress.add_task(f"[cyan]{label}", total=expected)
            self._tasks[out_dir] = task
            total_expected += expected

        self._overall_task = self._progress.add_task(
            "[bold green]Overall", total=total_expected,
        )

        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the polling thread and perform one final update."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        # Final accurate count
        self._update_once()
        if self._progress:
            self._progress.stop()

    def record_job_result(self, label: str, returncode: int, log_path: str):
        """Called by the main thread after each job finishes."""
        self._job_results.append({
            "label": label,
            "returncode": returncode,
            "log_path": log_path,
        })

    def print_summary(self):
        """Print a final summary of succeeded / failed jobs."""
        total = len(self._job_results)
        failed = [r for r in self._job_results if r["returncode"] != 0]
        succeeded = total - len(failed)

        if HAS_RICH:
            console = Console()
            console.print(
                f"\n[bold green]Finished:[/] {succeeded}/{total} succeeded",
            )
            if failed:
                console.print("[bold red]Failed jobs:[/]")
                for r in failed:
                    console.print(f"  • {r['label']}  (log: {r['log_path']})")
        else:
            print(f"\nFinished: {succeeded}/{total} succeeded", flush=True)
            if failed:
                print("Failed jobs:", flush=True)
                for r in failed:
                    print(f"  - {r['label']}  (log: {r['log_path']})", flush=True)

        return len(failed)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _update_once(self):
        if not self._progress:
            return
        total_done = 0
        for out_dir, task_id in self._tasks.items():
            count = count_rendered_frames(out_dir)
            expected = self.dir_expectations[out_dir]
            completed = min(count, expected)
            self._progress.update(task_id, completed=completed)
            total_done += completed
        self._progress.update(self._overall_task, completed=total_done)

    def _poll_loop(self):
        while not self._stop_event.is_set():
            self._update_once()
            self._stop_event.wait(self.poll_interval)


# ---------------------------------------------------------------------------
# Fallback plain-text progress (no rich)
# ---------------------------------------------------------------------------

class PlainProgressTracker:
    """Same interface as FrameProgressTracker but uses plain print()."""

    def __init__(self, dir_expectations: dict, poll_interval: float = 5.0):
        self.dir_expectations = dir_expectations
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._thread = None
        self._job_results = []

    def start(self):
        total = sum(self.dir_expectations.values())
        print(f"[progress] Tracking {len(self.dir_expectations)} directories, "
              f"{total} total expected frames", flush=True)
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        self._print_status()

    def record_job_result(self, label, returncode, log_path):
        self._job_results.append({
            "label": label, "returncode": returncode, "log_path": log_path,
        })

    def print_summary(self):
        total = len(self._job_results)
        failed = [r for r in self._job_results if r["returncode"] != 0]
        print(f"\nFinished: {total - len(failed)}/{total} succeeded", flush=True)
        if failed:
            print("Failed jobs:", flush=True)
            for r in failed:
                print(f"  - {r['label']}  (log: {r['log_path']})", flush=True)
        return len(failed)

    def _print_status(self):
        total_expected = sum(self.dir_expectations.values())
        total_done = sum(
            min(count_rendered_frames(d), exp)
            for d, exp in self.dir_expectations.items()
        )
        print(f"[progress] {total_done}/{total_expected} frames rendered",
              flush=True)

    def _poll_loop(self):
        while not self._stop_event.is_set():
            self._print_status()
            self._stop_event.wait(self.poll_interval)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_tracker(dir_expectations: dict, poll_interval: float = 2.0):
    """
    Return a FrameProgressTracker (rich) or PlainProgressTracker (fallback).
    """
    if HAS_RICH:
        print("[progress] Using rich for progress display", flush=True)
        return FrameProgressTracker(dir_expectations, poll_interval)
    else:
        print("[progress] rich not installed — using plain-text output", flush=True)
        return PlainProgressTracker(dir_expectations, poll_interval)

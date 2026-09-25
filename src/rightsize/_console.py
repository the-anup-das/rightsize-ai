"""Terminal output: colour, tables and progress bars when `rich` is installed and stdout is a
terminal; plain text otherwise. `rich` is optional (extras `cli` and `llamacpp`); the core never
imports it at startup. Honours NO_COLOR and --no-color.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


def _rich_available() -> bool:
    try:
        import rich  # noqa: F401
    except ImportError:
        return False
    return True


class Progress:
    """One progress bar. `total` may arrive later (set_total) when the tool announces it."""

    def __init__(self, console: Console, desc: str, total: int | None) -> None:
        self.console = console
        self.desc = desc
        self.total = total
        self.current = 0
        self.t0 = time.perf_counter()
        self._last_pct = -1
        self._rich = None
        self._task = None
        if console.use_rich:
            from rich.progress import (
                BarColumn,
                MofNCompleteColumn,
                SpinnerColumn,
                TextColumn,
                TimeElapsedColumn,
            )
            from rich.progress import (
                Progress as RichProgress,
            )

            self._rich = RichProgress(
                SpinnerColumn(),
                TextColumn("[bold]{task.description}"),
                BarColumn(bar_width=28),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TextColumn("{task.fields[note]}", style="dim"),
                console=console.rich,
                transient=False,
            )
            self._rich.start()
            self._task = self._rich.add_task(desc, total=total, note="")
        elif not console.quiet:
            console.out(f"{desc} ...")

    def set_total(self, total: int) -> None:
        self.total = total
        if self._rich is not None:
            self._rich.update(self._task, total=total)

    def update(self, current: int, note: str | None = None) -> None:
        self.current = current
        if self._rich is not None:
            self._rich.update(self._task, completed=current, **({"note": note} if note else {}))
            return
        if self.console.quiet or not self.total:
            return
        pct = int(current * 100 / self.total)
        if pct // 10 > self._last_pct // 10:
            self._last_pct = pct
            self.console.out(f"  {self.desc}: {pct}% ({current}/{self.total})")

    def close(self, ok: bool = True) -> None:
        elapsed = time.perf_counter() - self.t0
        if self._rich is not None:
            if ok and self.total:
                self._rich.update(self._task, completed=self.total)
            self._rich.stop()
            return
        if not self.console.quiet:
            self.console.out(f"  {self.desc}: {'done' if ok else 'FAILED'} in {elapsed:.0f}s")


class Console:
    def __init__(
        self, *, color: bool | None = None, verbose: bool = False, quiet: bool = False
    ) -> None:
        self.verbose = verbose
        self.quiet = quiet
        if color is None:
            color = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
        self.use_rich = bool(color) and _rich_available()
        self.rich = None
        if self.use_rich:
            from rich.console import Console as RichConsole

            self.rich = RichConsole(highlight=False)

    # ---- plain lines
    def out(self, text: str = "") -> None:
        if self.rich is not None:
            self.rich.print(text, markup=False)
        else:
            print(text, flush=True)

    def _styled(self, text: str, style: str) -> None:
        if self.quiet:
            return
        if self.rich is not None:
            self.rich.print(text, style=style, markup=False)
        else:
            print(text, flush=True)

    def title(self, text: str) -> None:
        if self.rich is not None and not self.quiet:
            self.rich.rule(text, style="cyan")
        elif not self.quiet:
            print(f"== {text} ==", flush=True)

    def info(self, text: str) -> None:
        self._styled(text, "")

    def ok(self, text: str) -> None:
        self._styled(text, "green")

    def warn(self, text: str) -> None:
        self._styled(text, "yellow")

    def fail(self, text: str) -> None:
        # failures always print, even in quiet mode
        if self.rich is not None:
            self.rich.print(text, style="bold red", markup=False)
        else:
            print(text, file=sys.stderr, flush=True)

    def debug(self, text: str) -> None:
        if self.verbose:
            self._styled(text, "dim")

    # ---- tables
    def table(
        self, header: list[str], rows: list[list[Any]], *, styles: list[str | None] | None = None
    ) -> None:
        if self.quiet:
            return
        if self.rich is not None:
            from rich.table import Table

            t = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
            for h in header:
                t.add_column(h)
            for i, row in enumerate(rows):
                style = styles[i] if styles and i < len(styles) else None
                t.add_row(*[str(c) for c in row], style=style)
            self.rich.print(t)
            return
        widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
        fmt = "  ".join("{:<" + str(w) + "}" for w in widths)
        print(fmt.format(*header), flush=True)
        print(fmt.format(*["-" * w for w in widths]), flush=True)
        for r in rows:
            print(fmt.format(*[str(c) for c in r]), flush=True)

    # ---- progress
    @contextmanager
    def progress(self, desc: str, total: int | None = None) -> Iterator[Progress]:
        p = Progress(self, desc, total)
        ok = True
        try:
            yield p
        except BaseException:
            ok = False
            raise
        finally:
            p.close(ok)


def verdict_style(verdict: str) -> str | None:
    return {"pass": "green", "fits": "green", "warn": "yellow", "tight": "yellow",
            "offload": "yellow", "fail": "red", "no_fit": "red"}.get(verdict)  # fmt: skip


def error_style(error_pct: float | None) -> str | None:
    if error_pct is None:
        return None
    a = abs(error_pct)
    return "green" if a <= 10 else "yellow" if a <= 25 else "red"

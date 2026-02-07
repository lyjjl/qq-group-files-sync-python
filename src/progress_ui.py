from __future__ import annotations

from rich.console import Console
from rich.progress import Progress, ProgressColumn, Task, TextColumn, TimeElapsedColumn
from rich.text import Text


class MbProgressColumn(ProgressColumn):
    def render(self, task: Task) -> Text:
        total = float(task.total or 0)
        completed = float(task.completed or 0)
        if total <= 0:
            return Text(f"{completed / 1024 / 1024:.1f} MB")
        return Text(f"{completed / 1024 / 1024:.1f}/{total / 1024 / 1024:.1f} MB")


class MbSpeedColumn(ProgressColumn):
    def render(self, task: Task) -> Text:
        speed = float(task.speed or 0)
        return Text(f"{speed / 1024 / 1024:.1f} MB/s")


class DualPhaseBarColumn(ProgressColumn):
    def __init__(
        self,
        *,
        bar_width: int,
        phase1_style: str = "cyan",
        phase2_style: str = "green",
        back_style: str = "dim",
    ) -> None:
        super().__init__()
        self.bar_width = max(10, int(bar_width))
        self.phase1_style = phase1_style
        self.phase2_style = phase2_style
        self.back_style = back_style

    def render(self, task: Task) -> Text:
        total = float(task.total or 0)
        phase2_ratio = 0.0 if total <= 0 else min(1.0, max(0.0, float(task.completed) / total))
        phase1_total = float(task.fields.get("phase1_total", 0) or 0)
        phase1_done = float(task.fields.get("phase1_done", 0) or 0)
        phase1_ratio = 0.0 if phase1_total <= 0 else min(1.0, max(0.0, phase1_done / phase1_total))

        width = self.bar_width
        phase1_cells = int(round(width * phase1_ratio))
        phase2_cells = int(round(width * phase2_ratio))

        bar = Text()
        for i in range(width):
            if i < phase2_cells:
                bar.append("━", style=self.phase2_style)
            elif i < phase1_cells:
                bar.append("━", style=self.phase1_style)
            else:
                bar.append("━", style=self.back_style)
        return bar


def create_progress(console: Console, *, description: str = "进度") -> Progress:
    bar_width = max(20, min(60, int(console.width) - 40)) if getattr(console, "width", None) else 40
    return Progress(
        TextColumn(f"{description}"),
        DualPhaseBarColumn(bar_width=bar_width),
        MbProgressColumn(),
        MbSpeedColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )

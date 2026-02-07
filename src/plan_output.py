from __future__ import annotations

from typing import Iterable

from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text


def print_plan_panel(
    console: Console,
    summary_rows: Iterable[tuple[str, int | str, str]],
    detail_rows: list[tuple[str, str, str]],
    *,
    style_map: dict[str, str],
    title: str = "操作计划",
    limit: int = 200,
) -> None:
    summary_rows = list(summary_rows)
    if not summary_rows and not detail_rows:
        console.print(Panel("没有需要进行的操作。", title=title, expand=False))
        return

    def _strlen(value: int | str) -> int:
        return len(str(value))

    action_header = "动作"
    count_header = "数量"
    path_header = "路径"
    note_header = "说明"

    max_action = max(
        [_strlen(action_header)]
        + [_strlen(r[0]) for r in summary_rows]
        + [_strlen(r[0]) for r in detail_rows]
    )
    max_col2 = max(
        [_strlen(count_header), _strlen(path_header)]
        + [_strlen(r[1]) for r in summary_rows]
        + [_strlen(r[1]) for r in detail_rows]
    )
    max_note = max(
        [_strlen(note_header)]
        + [_strlen(r[2]) for r in summary_rows]
        + [_strlen(r[2]) for r in detail_rows]
    )

    console_width = max(int(getattr(console, "width", 100) or 100), 60)
    action_w = max(9, max_action + 6)
    col2_w = max(11, max_col2)
    note_w = max(12, max_note)

    total = action_w + col2_w + note_w + 8
    if total > console_width:
        overflow = total - console_width
        reduce = min(overflow, max(0, note_w - 12))
        note_w -= reduce
        overflow -= reduce
        if overflow:
            reduce = min(overflow, max(0, col2_w - 10))
            col2_w -= reduce
            overflow -= reduce
        if overflow:
            reduce = min(overflow, max(0, action_w - 6))
            action_w -= reduce

    summary = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
    summary.add_column(action_header, width=action_w, no_wrap=True, overflow="ellipsis")
    summary.add_column(count_header, width=col2_w, justify="left", no_wrap=True, overflow="ellipsis")
    summary.add_column(note_header, width=note_w, no_wrap=True, overflow="ellipsis")
    for action, count, note in summary_rows:
        summary.add_row(action, str(count), note)

    details = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
    details.add_column(action_header, width=action_w, no_wrap=True, overflow="ellipsis")
    details.add_column(path_header, width=col2_w, no_wrap=True, overflow="ellipsis")
    details.add_column(note_header, width=note_w, no_wrap=True, overflow="ellipsis")

    show = detail_rows[:limit]
    for action, path, note in show:
        st = style_map.get(action, "")
        details.add_row(Text(action, style=st), path, note)
    if len(detail_rows) > limit:
        details.add_row("...", f"...（已截断，剩余 {len(detail_rows) - limit} 项）", "")

    body = Group(summary, Rule(style="dim"), details)
    console.print(Panel(body, title=title, expand=False, padding=(0, 1)))

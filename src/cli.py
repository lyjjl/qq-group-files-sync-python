from __future__ import annotations

import asyncio
import logging
import re
import signal
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import aiofiles
import httpx
import typer
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

import websockets

from config import load_config, is_placeholder_config, find_duplicate_group_ids
from dashboard import generate_dashboard
from filesystem import FileSystemManager
from indexer import GroupFileIndexer
from onebot import OneBotWsClient, extract_plain_text, parse_group_numeric_id
from pusher import GroupFilePusher
from progress_ui import create_count_progress, create_progress, create_search_progress
from syncer import GroupFileSyncer
from ignore_rules import IgnoreMatcher
from onebot_common import require_ok as _require_ok
from ui_console import console

app = typer.Typer(add_completion=False, no_args_is_help=True)


_LEVEL_MAP: dict[str, int] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARNING,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "none": logging.CRITICAL + 10,
}


class _HttpxNoiseFilter(logging.Filter):

    def __init__(self, console_level: int):
        super().__init__()
        self._console_level = console_level

    def filter(self, record: logging.LogRecord) -> bool:
        if self._console_level <= logging.DEBUG:
            return True
        if record.levelno != logging.INFO:
            return True
        name = record.name or ""
        if name.startswith("httpx") or name.startswith("httpcore"):
            return False
        return True


class _HttpxLevelDowngradeFilter(logging.Filter):

    def filter(self, record: logging.LogRecord) -> bool:
        name = record.name or ""
        if record.levelno == logging.INFO and (name.startswith("httpx") or name.startswith("httpcore")):
            record.levelno = logging.DEBUG
            record.levelname = "DEBUG"
        return True


def _setup_logging(log_file: str, log_level: str) -> int:
    lvl = (log_level or "").strip().lower()
    if lvl == "none":
        # Disable all logging output (console + files).
        root = logging.getLogger()
        for h in list(root.handlers):
            try:
                root.removeHandler(h)
            except Exception:
                pass
        logging.basicConfig(level=logging.CRITICAL + 10, handlers=[logging.NullHandler()])
        logging.disable(logging.CRITICAL)
        return logging.CRITICAL + 10

    level = _LEVEL_MAP.get(lvl, logging.INFO)

    # Console handler (Rich)
    # IMPORTANT: RichHandler already renders time/level; keep console formatter minimal to avoid duplication.
    console_handler = RichHandler(
        console=console,
        rich_tracebacks=(level <= logging.DEBUG),
        show_time=True,
        show_level=True,
        show_path=False,
    )
    console_handler.setLevel(level)
    console_handler.addFilter(_HttpxNoiseFilter(level))
    console_handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))

    # File handlers
    log_path = Path(log_file or "./logs/main.log").expanduser().resolve()
    log_dir = log_path.parent
    log_dir.mkdir(parents=True, exist_ok=True)
    err_file = log_dir / "error.log"

    main_handler = logging.FileHandler(log_path, encoding="utf-8")
    main_handler.setLevel(logging.DEBUG)
    main_handler.addFilter(_HttpxLevelDowngradeFilter())
    main_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )

    # error.log 包含 warn 和 err
    error_handler = logging.FileHandler(err_file, encoding="utf-8")
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )

    root = logging.getLogger()
    for h in list(root.handlers):
        try:
            root.removeHandler(h)
        except Exception:
            pass
    root.setLevel(logging.DEBUG)
    root.addHandler(console_handler)
    root.addHandler(main_handler)
    root.addHandler(error_handler)

    if level > logging.DEBUG:
        logging.getLogger("httpx").setLevel(logging.INFO)
        logging.getLogger("httpcore").setLevel(logging.INFO)

    return level


async def _send_group_text(bot: OneBotWsClient, group_id: int, text: str) -> None:
    try:
        await bot.call_api("send_group_msg", {"group_id": group_id, "message": text})
    except Exception:
        logging.getLogger(__name__).exception("failed to send group message")


def _fmt_group_id_str(group_id_num: int) -> str:
    return f"QQ-Group:{group_id_num}"


def _fmt_group_display(group_id: str, group_name: str | None) -> str:
    name = str(group_name or "").strip()
    gid = str(group_id or "").strip()
    if gid and name:
        return f"{name} ({gid})"
    if name:
        return name
    if gid:
        return f"未知群名 ({gid})"
    return gid


def _fmt_ts_local(ts: int | None) -> str:
    try:
        n = int(ts or 0)
    except Exception:
        n = 0
    if n <= 0:
        return "-"
    try:
        return datetime.fromtimestamp(n).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "-"


def _fmt_size(size: int | None) -> str:
    try:
        v = int(size or 0)
    except Exception:
        v = 0
    unit = 1024.0
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    value = float(max(0, v))
    idx = 0
    while value >= unit and idx < len(units) - 1:
        value /= unit
        idx += 1
    if idx == 0:
        return f"{int(value)} {units[idx]}"
    return f"{value:.1f} {units[idx]}"


def _safe_download_name(name: str) -> str:
    text = str(name or "").strip()
    if not text:
        return "file"
    text = re.sub(r"[\\/:*?\"<>|\x00-\x1f]", "_", text)
    text = text.strip().strip(".")
    return text or "file"


def _dedupe_download_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    i = 1
    while True:
        candidate = parent / f"{stem} ({i}){suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def _parse_get_file_target_tokens(text: str) -> list[str]:
    out: list[str] = []
    for part in str(text or "").split(","):
        token = part.strip()
        if token:
            out.append(token)
    if not out:
        raise ValueError("请至少提供一个下载目标")
    return out


def _print_index_update_stats(stats: dict[str, Any], *, title: str) -> None:
    table = Table(show_header=True, header_style="bold", box=None)
    table.add_column("群组")
    table.add_column("远端文件", justify="right")
    table.add_column("新增", justify="right")
    table.add_column("更新", justify="right")
    table.add_column("未变化", justify="right")
    table.add_column("删除", justify="right")
    table.add_column("状态")
    table.add_column("错误")

    total_remote = 0
    total_inserted = 0
    total_updated = 0
    total_unchanged = 0
    total_deleted = 0
    total_failed = 0

    rows: list[tuple[int, str, int, int, int, int, int, bool, str]] = []
    for gid, s in stats.items():
        failed = bool(getattr(s, "failed", False))
        if failed:
            total_failed += 1
        remote = int(getattr(s, "total_remote", 0))
        inserted = int(getattr(s, "inserted", 0))
        updated = int(getattr(s, "updated", 0))
        unchanged = int(getattr(s, "unchanged", 0))
        deleted = int(getattr(s, "deleted", 0))
        total_remote += remote
        total_inserted += inserted
        total_updated += updated
        total_unchanged += unchanged
        total_deleted += deleted

        change_size = inserted + updated + deleted
        if (not failed) and change_size <= 0:
            continue
        rows.append(
            (
                change_size,
                gid,
                remote,
                inserted,
                updated,
                unchanged,
                deleted,
                failed,
                str(getattr(s, "error", "") or "-"),
            )
        )

    rows.sort(key=lambda x: (-x[0], x[1]))
    for _change_size, gid, remote, inserted, updated, unchanged, deleted, failed, error in rows:
        table.add_row(
            gid,
            str(remote),
            str(inserted),
            str(updated),
            str(unchanged),
            str(deleted),
            ("失败" if failed else "正常"),
            error,
        )

    if stats:
        table.add_section()
        table.add_row(
            "合计",
            str(total_remote),
            str(total_inserted),
            str(total_updated),
            str(total_unchanged),
            str(total_deleted),
            (f"失败 {total_failed}" if total_failed else "正常"),
            "-",
        )
    console.print(Panel(table, title=title, expand=False))


def _print_search_results(query_raw: str, results: list[Any], elapsed_ms: float | None = None) -> None:
    row_limit = 200
    rows: list[tuple[str, str, str, str, str, str, str]] = []

    for r in results:
        folder_path = str(getattr(r, "folder_path", "") or "").strip().strip("/")
        folder_show = "/" if not folder_path else f"/{folder_path}"
        uploader_name = str(getattr(r, "uploader_name", "") or "")
        uploader_id = int(getattr(r, "uploader_id", 0) or 0)
        if uploader_id > 0 and uploader_name:
            uploader_show = f"{uploader_name} (QQ:{uploader_id})"
        elif uploader_id > 0:
            uploader_show = f"QQ:{uploader_id}"
        else:
            uploader_show = uploader_name or "-"
        rows.append(
            (
                _fmt_group_display(str(getattr(r, "group_id", "")), str(getattr(r, "group_name", "") or "")),
                str(getattr(r, "file_name", "")),
                uploader_show,
                _fmt_ts_local(getattr(r, "modify_time", 0)),
                _fmt_size(getattr(r, "file_size", 0)),
                folder_show,
                str(getattr(r, "short_id", "")),
            )
        )

    timing = f" | 耗时 {elapsed_ms:.1f} ms" if elapsed_ms is not None else ""
    if not results:
        console.print(Panel(f"查询 `{query_raw}` 没有命中。", title=f"搜索结果{timing}", expand=False))
    else:
        # 与 --plan 一致
        columns = (
            ("群组", 12, 28),
            ("文件名", 16, 40),
            ("上传者", 12, 24),
            ("修改时间", 16, 19),
            ("大小", 8, 12),
            ("文件夹", 12, 32),
            ("ID", 6, 10),
        )
        widths: dict[str, int] = {}
        for name, min_w, pref_w in columns:
            widths[name] = pref_w
        console_width = max(int(getattr(console, "width", 100) or 100), 60)
        total = sum(widths.values()) + len(columns) * 3 + 4
        overflow = max(0, total - console_width)
        for name in ("文件名", "文件夹", "群组", "上传者", "修改时间", "大小", "ID"):
            if overflow <= 0:
                break
            min_w = next(min_w for col, min_w, _ in columns if col == name)
            can_reduce = max(0, widths[name] - min_w)
            reduce_by = min(overflow, can_reduce)
            widths[name] -= reduce_by
            overflow -= reduce_by

        table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
        table.add_column("群组", width=widths["群组"], no_wrap=True, overflow="ellipsis")
        table.add_column("文件名", width=widths["文件名"], no_wrap=True, overflow="ellipsis")
        table.add_column("上传者", width=widths["上传者"], no_wrap=True, overflow="ellipsis")
        table.add_column("修改时间", width=widths["修改时间"], no_wrap=True, overflow="ellipsis")
        table.add_column("大小", width=widths["大小"], no_wrap=True, overflow="ellipsis", justify="right")
        table.add_column("文件夹", width=widths["文件夹"], no_wrap=True, overflow="ellipsis")
        table.add_column("ID", width=widths["ID"], no_wrap=True, overflow="ellipsis")

        show_rows = rows[:row_limit]
        for row in show_rows:
            table.add_row(*row)
        if len(rows) > row_limit:
            table.add_row("...", f"...（已截断，剩余 {len(rows) - row_limit} 项）", "", "", "", "", "")
        console.print(Panel(table, title=f"搜索结果：{query_raw}（{len(results)}）{timing}", expand=False))


async def _sync_all(
    cfg,
    fs: FileSystemManager,
    bot: OneBotWsClient,
    syncer: GroupFileSyncer,
    build_dashboard: bool,
    mirror: bool,
    plan: bool,
) -> None:
    if not cfg.groups:
        console.print("配置文件中没有配置群组列表，请在 config.toml 中添加 groups 配置")
        return

    # pull all: skip groups with no_pull=true
    groups_to_run = [g for g in cfg.groups if not bool(getattr(g, "no_pull", False))]
    console.print(f"即将开始同步 {len(groups_to_run)} 个群组的文件")

    ok = 0
    failed: list[str] = []
    for g in groups_to_run:
        gid = g.id
        alias = g.alias or gid
        try:
            console.print(f"开始同步群组: {gid} ({alias})")
            await syncer.sync_group(bot, gid, mirror=mirror, plan=plan)
            ok += 1
        except Exception as e:
            logging.getLogger(__name__).exception("sync failed")
            failed.append(alias)

    if build_dashboard and not plan:
        try:
            generate_dashboard(cfg, fs)
        except Exception:
            logging.getLogger(__name__).exception("failed to build dashboard")

    msg = ("对比完成！" if plan else "同步全部完成！") + f"成功: {ok} 个群组"
    if failed:
        msg += f"，失败: {len(failed)} 个群组 ({', '.join(failed)})"

    # invalid files are not treated as errors, but we still report them
    invalid_counts = syncer.get_invalid_counts()
    if invalid_counts:
        total_invalid = 0
        msg += "\n失效文件统计:"
        for g in groups_to_run:
            gid = g.id
            n = int(invalid_counts.get(gid, 0))
            if n <= 0:
                continue
            total_invalid += n
            msg += f"\n- {gid}: {n}"
        msg += f"\n总失效文件: {total_invalid}"
        msg += f"\n详见 {getattr(cfg, 'invalid_files_log', 'invalidFiles.log')}"

    if not plan:
        msg += f"\n文件已保存到: {fs.base_path}"
    console.print(Panel(msg, title=("对比完成" if plan else "同步完成"), expand=False))


async def _interactive(cfg, fs: FileSystemManager, bot: OneBotWsClient, syncer: GroupFileSyncer) -> None:
    console.print("小海豹QQ群文件同步器")
    console.print("可用指令:")
    console.print(".同步当前")
    console.print("    同步当前群的群文件，完成后自动生成展示页面")
    console.print(".同步文件 QQ-Group:群号")
    console.print("    指定一个群进行同步，账号应该在群内")
    console.print(".同步全部")
    console.print("    同步 config.toml 中设置的所有群")
    console.print(".展示页面")
    console.print("    强制重新生成展示页面")
    console.print("等待消息中... 使用 Ctrl+C 退出")

    stop = asyncio.Event()

    def _stop(*_: Any) -> None:
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _stop)
        except NotImplementedError:
            signal.signal(sig, lambda *_a, **_k: _stop())

    async def handle(evt: dict[str, Any]) -> None:
        if evt.get("post_type") != "message":
            return
        if evt.get("message_type") != "group":
            return

        user_id = evt.get("user_id")
        self_id = evt.get("self_id")
        if user_id is not None and self_id is not None and str(user_id) == str(self_id):
            return

        group_id_num = int(evt.get("group_id"))
        group_id_str = _fmt_group_id_str(group_id_num)
        text = extract_plain_text(evt.get("raw_message") or evt.get("message"))
        if not text:
            return

        if text.startswith(".同步当前"):
            await _send_group_text(bot, group_id_num, "开始同步本群文件...")
            try:
                await syncer.sync_group(bot, group_id_str)
                generate_dashboard(cfg, fs)
                await _send_group_text(bot, group_id_num, f"同步完成。文件已保存到: {fs.base_path}")
            except Exception:
                logging.getLogger(__name__).exception("sync current failed")
                await _send_group_text(bot, group_id_num, "同步失败，详情请查看日志")
            return

        if text.startswith(".同步文件"):
            parts = text.split()
            if len(parts) >= 2:
                target = parts[1].strip()
                try:
                    target_num = parse_group_numeric_id(target)
                    target_str = _fmt_group_id_str(target_num)
                except Exception:
                    await _send_group_text(bot, group_id_num, "群号格式不正确，应为 QQ-Group:12345 或纯数字")
                    return

                await _send_group_text(bot, group_id_num, f"开始同步 {target_str}...")
                try:
                    await syncer.sync_group(bot, target_str)
                    generate_dashboard(cfg, fs)
                    await _send_group_text(bot, group_id_num, "同步完成")
                except Exception:
                    logging.getLogger(__name__).exception("sync target failed")
                    await _send_group_text(bot, group_id_num, "同步失败，详情请查看日志")
            return

        if text.startswith(".同步全部"):
            groups_to_run = [g for g in cfg.groups if not bool(getattr(g, "no_pull", False))]
            await _send_group_text(bot, group_id_num, f"即将开始同步 {len(groups_to_run)} 个群组的文件")
            await _sync_all(cfg, fs, bot, syncer, build_dashboard=True, mirror=False, plan=False)
            await _send_group_text(bot, group_id_num, "同步全部已结束")
            return

        if text.startswith(".展示页面"):
            try:
                out, _data = generate_dashboard(cfg, fs)
                await _send_group_text(bot, group_id_num, f"展示页面已生成: {out}")
            except Exception:
                logging.getLogger(__name__).exception("dashboard failed")
                await _send_group_text(bot, group_id_num, "生成展示页面失败，详情请查看日志")
            return

    async def consume() -> None:
        async for evt in bot.events():
            await handle(evt)

    consumer = asyncio.create_task(consume())
    await stop.wait()
    consumer.cancel()
    try:
        await consumer
    except asyncio.CancelledError:
        pass


def _connect_hint(cfg) -> None:
    msg = (
        f"无法连接到 OneBot11 正向WS：{cfg.onebot11.ws_url}\n"
        "请确认 OneBot 实现已启动并监听该地址/端口；如需 token 请配置 onebot11.access_token。\n"
        "详细堆栈请查看 error.log（与 logFile 同目录），或将 logLevel 设为 debug。"
    )
    console.print(Panel(msg, title="连接失败", expand=False))


def _load_ignore_matcher(ignore_file: str | None) -> IgnoreMatcher | None:
    """从文件加载忽略规则"""

    if ignore_file:
        p = Path(ignore_file).expanduser()
        if not p.exists() or not p.is_file():
            console.print(f"[yellow]WARN[/yellow] 忽略规则文件不存在：{ignore_file}（将不启用忽略）")
            return None
        console.print(f"已启用忽略规则：{p}")
        return IgnoreMatcher.from_file(p)

    p = Path(".ignore")
    if p.exists() and p.is_file():
        console.print(f"已启用忽略规则：{p}")
        return IgnoreMatcher.from_file(p)
    return None


def _ensure_unique_group_ids(cfg) -> None:
    dupes = find_duplicate_group_ids(cfg)
    if dupes:
        console.print("[red]配置错误：groups 中存在重复的 id[/red]")
        console.print("请修改 config.toml，确保每个 [[groups]] 的 id 唯一。")
        console.print("重复的 id：")
        for gid in dupes:
            console.print(f"- {gid}")
        raise typer.Exit(code=2)


def _load_cfg_and_logging(config_path: str):
    cfg = load_config(config_path)
    _ensure_unique_group_ids(cfg)
    console_level = _setup_logging(cfg.log_file, cfg.log_level)
    return cfg, console_level


def _build_fs_and_ignore(cfg, ignore_file: str | None) -> tuple[FileSystemManager, IgnoreMatcher | None]:
    fs = FileSystemManager(cfg.file_system.local_path)
    ignore = _load_ignore_matcher(ignore_file)
    return fs, ignore


def _run_with_ws(console_level: int, cfg, runner) -> None:
    try:
        asyncio.run(runner())
    except (
        ConnectionRefusedError,
        OSError,
        websockets.InvalidURI,
        websockets.InvalidHandshake,
    ) as e:
        if console_level <= logging.DEBUG:
            logging.getLogger(__name__).exception("failed to connect onebot ws")
            raise
        logging.getLogger(__name__).error("failed to connect onebot ws: %s", e)
        _connect_hint(cfg)
        raise typer.Exit(code=2)
    except Exception as e:
        logging.getLogger(__name__).exception("unhandled error")
        if console_level <= logging.DEBUG:
            raise
        console.print(f"运行失败：{e}")
        console.print("将 logLevel 调为 debug 可查看更详细堆栈。")
        raise typer.Exit(code=1)


@app.command(help="拉取群文件（增量备份）")
def pull(
    target: str = typer.Argument(..., help="all 或群号：QQ-Group:123456 / 纯数字"),
    config: str = typer.Option("config.toml", "--config", help="配置文件路径（推荐 TOML）"),
    concurrency: int | None = typer.Option(None, "--concurrency", min=1, max=64, help="下载并发数（N，默认使用配置文件）"),
    url_workers: int | None = typer.Option(None, "--url-workers", min=1, max=64, help="获取资源链接并发数（M，默认使用配置文件）"),
    ignore_invalid_record: bool = typer.Option(False, "--ignore-invalid-record", help="忽略失效文件记录（不跳过）"),
    reset_invalid_record: bool = typer.Option(False, "--reset-invalid-record", help="重置失效文件记录"),
    web: bool = typer.Option(False, "-w", "--web", help="同步完成后生成/更新展示页面"),
    mirror: bool = typer.Option(False, "--mirror", help="镜像模式：覆盖不同文件、删除本地多余文件（按文件名+大小判断）"),
    plan: bool = typer.Option(False, "--plan", help="只对比并输出将执行的操作，不做任何修改"),
    ignore_file: str | None = typer.Option(None, "--ignore-file", help="忽略规则文件路径（相对路径）。默认读取 ./.ignore（如存在）"),
) -> None:
    cfg, console_level = _load_cfg_and_logging(config)
    fs, ignore = _build_fs_and_ignore(cfg, ignore_file)

    pull_all = (target or "").strip().lower() == "all"

    # When pulling all, refuse to run with untouched template config.
    if pull_all and is_placeholder_config(cfg):
        logging.getLogger(__name__).warning("refuse to run: placeholder config")
        console.print("检测到示例配置尚未修改。")
        console.print("请编辑 config.toml：填写真实的 groups 列表，并确认 onebot11.ws_url 可连接。")
        raise typer.Exit(code=2)

    async def runner() -> None:
        async with OneBotWsClient(cfg.onebot11.ws_url, cfg.onebot11.access_token) as bot:
            download_workers = int(cfg.sync.download_workers) if concurrency is None else int(concurrency)
            url_w = int(cfg.sync.url_workers) if url_workers is None else int(url_workers)
            syncer = GroupFileSyncer(
                cfg,
                fs,
                concurrency=download_workers,
                url_workers=url_w,
                ignore_invalid_record=ignore_invalid_record,
                reset_invalid_record=reset_invalid_record,
                ignore=ignore,
            )
            if pull_all:
                await _sync_all(cfg, fs, bot, syncer, build_dashboard=web, mirror=mirror, plan=plan)
                return

            # 单群
            try:
                gid_num = parse_group_numeric_id(target)
                gid_str = _fmt_group_id_str(gid_num)
            except Exception:
                logging.getLogger(__name__).warning("invalid group id: %s", target)
                console.print("群号格式不正确，应为 QQ-Group:123456 或纯数字。")
                raise typer.Exit(code=2)

            # 特殊处理没有配置群组的情况
            if not any((g.id or "").strip() == gid_str for g in cfg.groups):
                from config import GroupConfig

                cfg.groups.append(GroupConfig(id=gid_str, alias=gid_str, description=""))

            await syncer.sync_group(bot, gid_str, mirror=mirror, plan=plan)
            if web and (not plan):
                try:
                    generate_dashboard(cfg, fs)
                except Exception:
                    logging.getLogger(__name__).exception("failed to build dashboard")

            title = "对比完成" if plan else "同步完成"
            body = f"{gid_str}"
            if not plan:
                body += f"\n文件已保存到: {fs.base_path}"
            invalid_n = int(syncer.get_invalid_counts().get(gid_str, 0))
            if invalid_n:
                body += f"\n失效文件: {invalid_n}（详见 {getattr(cfg, 'invalid_files_log', 'invalidFiles.log')}）"
            if web and (not plan):
                body += "\n展示页面：已生成/更新"
            console.print(Panel(body, title=title, expand=False))

    _run_with_ws(console_level, cfg, runner)


@app.command(help="推送本地文件到群文件（仅补齐远端缺失）")
def push(
    target: str = typer.Argument(..., help="all 或群号：QQ-Group:123456 / 纯数字"),
    config: str = typer.Option("config.toml", "--config", help="配置文件路径（推荐 TOML）"),
    concurrency: int = typer.Option(2, "--concurrency", min=1, max=16, help="并发上传数"),
    url_workers: int | None = typer.Option(None, "--url-workers", min=1, max=64, help="获取资源链接并发数（M，默认使用配置文件）"),
    mirror: bool = typer.Option(False, "--mirror", help="镜像模式：覆盖不同文件、删除远端多余文件（按文件名+大小判断）"),
    plan: bool = typer.Option(False, "--plan", help="只对比并输出将执行的操作，不做任何修改"),
    ignore_file: str | None = typer.Option(None, "--ignore-file", help="忽略规则文件路径（相对路径）。默认读取 ./.ignore（如存在）"),
) -> None:
    cfg, console_level = _load_cfg_and_logging(config)
    fs, ignore = _build_fs_and_ignore(cfg, ignore_file)
    _ = url_workers  # keep CLI signature consistent; push doesn't fetch resource URLs
    push_all = (target or "").strip().lower() == "all"

    if push_all and is_placeholder_config(cfg):
        logging.getLogger(__name__).warning("refuse to run: placeholder config")
        console.print("检测到示例配置尚未修改。")
        console.print("请编辑 config.toml：填写真实的 groups 列表，并确认 onebot11.ws_url 可连接。")
        raise typer.Exit(code=2)

    async def runner() -> None:
        async with OneBotWsClient(cfg.onebot11.ws_url, cfg.onebot11.access_token) as bot:
            pusher = GroupFilePusher(fs, concurrency=concurrency, ignore=ignore, cfg=cfg)

            if push_all:
                if not cfg.groups:
                    console.print("配置文件中没有配置群组列表，请在 config.toml 中添加 groups 配置")
                    return
                ok = 0
                failed: list[str] = []
                # push all: skip groups with no_push=true
                groups_to_run = [g for g in cfg.groups if not bool(getattr(g, "no_push", False))]
                for g in groups_to_run:
                    gid = g.id
                    alias = g.alias or gid
                    try:
                        console.print(f"开始推送群组: {gid} ({alias})")
                        await pusher.push_group(bot, gid, mirror=mirror, plan=plan)
                        ok += 1
                    except Exception:
                        logging.getLogger(__name__).exception("push failed")
                        failed.append(alias)

                msg = ("对比完成！" if plan else "推送全部完成！") + f"成功: {ok} 个群组"
                if failed:
                    msg += f"，失败: {len(failed)} 个群组 ({', '.join(failed)})"
                msg += f"\n本地数据目录: {fs.base_path}"
                console.print(Panel(msg, title=("对比完成" if plan else "推送完成"), expand=False))
                return

            # 单群
            try:
                gid_num = parse_group_numeric_id(target)
                gid_str = _fmt_group_id_str(gid_num)
            except Exception:
                logging.getLogger(__name__).warning("invalid group id: %s", target)
                console.print("群号格式不正确，应为 QQ-Group:123456 或纯数字。")
                raise typer.Exit(code=2)

            await pusher.push_group(bot, gid_str, mirror=mirror, plan=plan)
            title = "对比完成" if plan else "推送完成"
            body = f"{gid_str}\n本地数据目录: {fs.base_path}"
            console.print(Panel(body, title=title, expand=False))

    _run_with_ws(console_level, cfg, runner)


@app.command(help="进入交互等待模式（可在群里发指令触发同步）")
def watch(
    config: str = typer.Option("config.toml", "--config", help="配置文件路径"),
    concurrency: int | None = typer.Option(None, "--concurrency", min=1, max=64, help="下载并发数（N，默认使用配置文件）"),
    url_workers: int | None = typer.Option(None, "--url-workers", min=1, max=64, help="获取资源链接并发数（M，默认使用配置文件）"),
    ignore_invalid_record: bool = typer.Option(False, "--ignore-invalid-record", help="忽略失效文件记录（不跳过）"),
    reset_invalid_record: bool = typer.Option(False, "--reset-invalid-record", help="重置失效文件记录"),
    ignore_file: str | None = typer.Option(None, "--ignore-file", help="忽略规则文件路径（相对路径）。默认读取 ./.ignore（如存在）"),
) -> None:
    cfg, console_level = _load_cfg_and_logging(config)
    fs, ignore = _build_fs_and_ignore(cfg, ignore_file)

    if is_placeholder_config(cfg):
        logging.getLogger(__name__).warning("refuse to run: placeholder config")
        console.print("检测到示例配置尚未修改。")
        console.print("请编辑 config.toml：填写真实的 groups 列表，并确认 onebot11.ws_url 可连接。")
        raise typer.Exit(code=2)

    async def runner() -> None:
        async with OneBotWsClient(cfg.onebot11.ws_url, cfg.onebot11.access_token) as bot:
            download_workers = int(cfg.sync.download_workers) if concurrency is None else int(concurrency)
            url_w = int(cfg.sync.url_workers) if url_workers is None else int(url_workers)
            syncer = GroupFileSyncer(
                cfg,
                fs,
                concurrency=download_workers,
                url_workers=url_w,
                ignore_invalid_record=ignore_invalid_record,
                reset_invalid_record=reset_invalid_record,
                ignore=ignore,
            )
            await _interactive(cfg, fs, bot, syncer)

    _run_with_ws(console_level, cfg, runner)


@app.command(help="更新本地索引缓存（默认 all）")
def update_index(
    target: str = typer.Argument("all", help="all 或群号：QQ-Group:123456 / 纯数字"),
    config: str = typer.Option("config.toml", "--config", help="配置文件路径（推荐 TOML）"),
) -> None:
    cfg, console_level = _load_cfg_and_logging(config)
    fs = FileSystemManager(cfg.file_system.local_path)
    indexer = GroupFileIndexer(cfg, fs)

    async def runner() -> None:
        async with OneBotWsClient(cfg.onebot11.ws_url, cfg.onebot11.access_token) as bot:
            with create_count_progress(console, description="索引更新中", transient=False) as progress:
                task_id = progress.add_task("索引更新中", total=1, phase1_total=1, phase1_done=1)
                total_groups = 1

                def on_group_done(done: int, total: int, gid: str, ok: bool) -> None:
                    nonlocal total_groups
                    total_groups = max(1, int(total or 0))
                    status = "OK" if ok else "FAIL"
                    progress.update(
                        task_id,
                        total=total_groups,
                        completed=int(done),
                        phase1_total=1,
                        phase1_done=1,
                        description=f"索引更新中 [{status}] {gid}",
                    )

                stats = await indexer.update_index(
                    bot,
                    target=target,
                    no_cache=True,
                    on_group_done=on_group_done,
                )
                progress.update(
                    task_id,
                    total=total_groups,
                    completed=total_groups,
                    phase1_total=1,
                    phase1_done=1,
                    description="索引更新完成",
                )
        _print_index_update_stats(stats, title="索引更新完成")
        console.print(f"索引库: {indexer.db_path}")
        console.print(f"总记录数: {indexer.count_records()}")

    try:
        _run_with_ws(console_level, cfg, runner)
    finally:
        indexer.close()


@app.command(name="index-info", help="查看本地索引统计信息")
def index_info(
    config: str = typer.Option("config.toml", "--config", help="配置文件路径（推荐 TOML）"),
    id: str | None = typer.Option(None, "--id", help="短 ID（来自 search 结果）"),
    detail: bool = typer.Option(False, "--detail", help="显示群组索引明细（默认不显示）"),
) -> None:
    cfg, _console_level = _load_cfg_and_logging(config)
    fs = FileSystemManager(cfg.file_system.local_path)
    indexer = GroupFileIndexer(cfg, fs)
    try:
        if id:
            rec = indexer.get_indexed_record_by_short_id(id)
            if rec is None:
                console.print(f"[red]未找到 ID[/red]: {id}")
                raise typer.Exit(code=2)
            folder_path = str(rec.folder_path or "").strip().strip("/")
            folder_show = "/" if not folder_path else f"/{folder_path}"
            rel_path = rec.file_name if folder_show == "/" else f"{folder_show}/{rec.file_name}"
            table = Table(show_header=False, box=None)
            table.add_column("项", style="bold")
            table.add_column("值")
            table.add_row("ID", rec.short_id)
            table.add_row("群组", _fmt_group_display(rec.group_id, rec.group_name))
            table.add_row("文件名", rec.file_name)
            table.add_row("文件夹", folder_show)
            table.add_row("完整路径", rel_path)
            table.add_row("文件ID", rec.file_id)
            table.add_row("文件类型(busid)", str(rec.busid))
            table.add_row("文件大小", _fmt_size(rec.file_size))
            table.add_row("上传者", f"{rec.uploader_name} (QQ:{rec.uploader_id})" if rec.uploader_id else (rec.uploader_name or "-"))
            table.add_row("修改时间", _fmt_ts_local(rec.modify_time))
            table.add_row("过期时间", _fmt_ts_local(rec.dead_time))
            table.add_row("下载次数", str(rec.download_times))
            table.add_row("md5", rec.md5 or "-")
            table.add_row("alias", rec.alias or "-")
            console.print(Panel(table, title=f"索引详情：{rec.short_id}", expand=False))
            return

        info = indexer.get_index_info()
        try:
            db_size = int(indexer.db_path.stat().st_size)
        except Exception:
            db_size = 0

        table = Table(show_header=False, box=None)
        table.add_column("项", style="bold")
        table.add_column("值")
        table.add_row("索引库路径", str(indexer.db_path))
        table.add_row("索引群组数", str(info.total_groups))
        table.add_row("总索引量（文件）", str(info.total_records))
        table.add_row("估计索引文件总大小", _fmt_size(info.total_file_size))
        table.add_row("平均文件大小", _fmt_size(info.avg_file_size))
        table.add_row("索引库文件大小", _fmt_size(db_size))
        table.add_row("最早修改时间", _fmt_ts_local(info.earliest_modify_time))
        table.add_row("最近修改时间", _fmt_ts_local(info.latest_modify_time))
        table.add_row("FTS5 状态", "启用" if bool(getattr(indexer, "_fts_available", False)) else "不可用")
        console.print(Panel(table, title="索引信息", expand=False))

        group_rows = [g for g in indexer.list_indexed_groups_info() if int(g.file_count or 0) > 0]
        if detail and group_rows:
            gtable = Table(show_header=True, header_style="bold", box=None)
            gtable.add_column("群组")
            gtable.add_column("文件数", justify="right")
            gtable.add_column("总大小", justify="right")
            gtable.add_column("更新时间")
            for g in group_rows:
                gtable.add_row(
                    _fmt_group_display(g.group_id, g.group_name),
                    str(g.file_count),
                    _fmt_size(g.total_file_size),
                    _fmt_ts_local(g.updated_at),
                )
            console.print(Panel(gtable, title=f"群组索引明细（{len(group_rows)}）", expand=False))
    finally:
        indexer.close()


@app.command(name="get-file", help="按 group_id/file_id 下载群文件到 Download")
def get_file(
    targets: str = typer.Argument(..., help="group_id/file_id 或短ID，可逗号分隔"),
    config: str = typer.Option("config.toml", "--config", help="配置文件路径（推荐 TOML）"),
) -> None:
    cfg, console_level = _load_cfg_and_logging(config)
    fs = FileSystemManager(cfg.file_system.local_path)
    indexer = GroupFileIndexer(cfg, fs)
    try:
        try:
            target_tokens = _parse_get_file_target_tokens(targets)
        except Exception as e:
            console.print(f"[red]参数错误[/red]: {e}")
            raise typer.Exit(code=2)

        async def runner() -> None:
            download_root = Path("Download").resolve()
            download_root.mkdir(parents=True, exist_ok=True)
            rows: list[tuple[str, str, str, str]] = []
            resolved_targets: list[tuple[str, str]] = []
            seen_targets: set[tuple[str, str]] = set()

            for token in target_tokens:
                if "/" in token:
                    group_part, file_part = token.split("/", 1)
                    group_part = group_part.strip()
                    file_part = file_part.strip()
                    if not group_part or not file_part:
                        rows.append((token, token, "失败", "无效目标（应为 group_id/file_id）"))
                        continue
                    try:
                        gid = _fmt_group_id_str(parse_group_numeric_id(group_part))
                    except Exception:
                        rows.append((token, token, "失败", f"无效群号: {group_part}"))
                        continue
                    fid = file_part if file_part.startswith("/") else f"/{file_part}"
                    key = (gid, fid)
                    if key in seen_targets:
                        continue
                    seen_targets.add(key)
                    resolved_targets.append(key)
                    continue

                # 没有 "/" 的目标按短 ID 处理
                rec = indexer.get_indexed_record_by_short_id(token)
                if rec is None:
                    rows.append((token, token, "失败", "无效短ID，且不符合 group_id/file_id"))
                    continue
                key = (rec.group_id, rec.file_id)
                if key in seen_targets:
                    continue
                seen_targets.add(key)
                resolved_targets.append(key)

            resolved_records: list[Any] = []
            for gid, fid in resolved_targets:
                rec = indexer.get_indexed_record_by_group_file(gid, fid)
                if rec is None:
                    rows.append((gid, fid, "失败", "索引中不存在该文件（请先 update-index）"))
                    continue
                resolved_records.append(rec)

            if resolved_records:
                progress_ctx = create_progress(console, description="文件下载中", transient=False)
                progress_ctx.start()
                task_id = progress_ctx.add_task("文件下载中", total=0, phase1_total=1, phase1_done=1)
            else:
                progress_ctx = None
                task_id = None

            try:
                async with OneBotWsClient(cfg.onebot11.ws_url, cfg.onebot11.access_token) as bot:
                    limits = httpx.Limits(max_connections=4, max_keepalive_connections=4)
                    async with httpx.AsyncClient(follow_redirects=True, timeout=httpx.Timeout(60.0), limits=limits) as client:
                        progress_total_bytes = 0
                        for rec in resolved_records:
                            gid = rec.group_id
                            fid = rec.file_id
                            status = "FAIL"

                            try:
                                group_num = int(rec.group_id_num or parse_group_numeric_id(gid))
                                url_res = await bot.call_api(
                                    "get_group_file_url",
                                    {"group_id": group_num, "file_id": rec.file_id, "busid": int(rec.busid or 0)},
                                )
                                _require_ok("get_group_file_url", url_res)
                                url = str(((url_res.data or {}).get("url")) or "").strip()
                                if not url:
                                    raise RuntimeError("OneBot 未返回下载链接")

                                group_dir = download_root / gid.replace(":", "_")
                                group_dir.mkdir(parents=True, exist_ok=True)
                                safe_name = _safe_download_name(rec.file_name or rec.file_id.strip("/"))
                                dst = _dedupe_download_path(group_dir / safe_name)
                                tmp = dst.with_suffix(dst.suffix + ".part")
                                downloaded_this = 0
                                planned_bytes = 0

                                async with client.stream("GET", url) as resp:
                                    resp.raise_for_status()
                                    header_len = int(resp.headers.get("Content-Length") or 0)
                                    if header_len > 0:
                                        planned_bytes = header_len
                                    else:
                                        planned_bytes = max(int(rec.file_size or 0), 0)
                                    if planned_bytes > 0:
                                        progress_total_bytes += planned_bytes
                                        if progress_ctx is not None and task_id is not None:
                                            progress_ctx.update(
                                                task_id,
                                                total=progress_total_bytes,
                                                phase1_total=1,
                                                phase1_done=1,
                                            )
                                    async with aiofiles.open(tmp, "wb") as af:
                                        async for chunk in resp.aiter_bytes():
                                            await af.write(chunk)
                                            chunk_len = len(chunk)
                                            downloaded_this += chunk_len
                                            if progress_ctx is not None and task_id is not None and chunk_len > 0:
                                                progress_ctx.update(
                                                    task_id,
                                                    advance=chunk_len,
                                                    phase1_total=1,
                                                    phase1_done=1,
                                                )
                                if planned_bytes > downloaded_this:
                                    progress_total_bytes -= (planned_bytes - downloaded_this)
                                    if progress_ctx is not None and task_id is not None:
                                        completed_now = float(progress_ctx.tasks[task_id].completed or 0)
                                        progress_ctx.update(
                                            task_id,
                                            total=max(progress_total_bytes, int(completed_now)),
                                            phase1_total=1,
                                            phase1_done=1,
                                        )
                                elif downloaded_this > planned_bytes:
                                    progress_total_bytes += (downloaded_this - planned_bytes)
                                    if progress_ctx is not None and task_id is not None:
                                        progress_ctx.update(
                                            task_id,
                                            total=progress_total_bytes,
                                            phase1_total=1,
                                            phase1_done=1,
                                        )
                                tmp.replace(dst)
                                rows.append((gid, rec.file_name, "成功", str(dst)))
                                status = "OK"
                            except Exception as e:
                                rows.append((gid, rec.file_name or fid, "失败", str(e)))
                            finally:
                                if progress_ctx is not None and task_id is not None:
                                    progress_ctx.update(
                                        task_id,
                                        phase1_total=1,
                                        phase1_done=1,
                                        description=f"文件下载中 [{status}] {gid}",
                                    )
            finally:
                if progress_ctx is not None and task_id is not None:
                    progress_ctx.update(
                        task_id,
                        phase1_total=1,
                        phase1_done=1,
                        description="文件下载完成",
                    )
                    progress_ctx.stop()

            table = Table(show_header=True, header_style="bold", box=None)
            table.add_column("群组")
            table.add_column("文件")
            table.add_column("结果")
            table.add_column("详情")
            ok = 0
            for gid, name, status, detail in rows:
                if status == "成功":
                    ok += 1
                table.add_row(gid, name, status, detail)
            console.print(Panel(table, title=f"下载结果：成功 {ok}/{len(rows)}", expand=False))
            console.print(f"下载目录: {download_root}")

        _run_with_ws(console_level, cfg, runner)
    finally:
        indexer.close()


@app.command(help="在索引缓存中搜索群文件")
def search(
    keyword: list[str] = typer.Argument(None, help="查询表达式，可传多个"),
    config: str = typer.Option("config.toml", "--config", help="配置文件路径（推荐 TOML）"),
    refresh: bool = typer.Option(False, "--refresh", help="先通过 API 增量刷新索引，再执行搜索"),
    strict: bool = typer.Option(False, "--strict", help="严格模式：禁用模糊搜索（仅精确匹配）"),
) -> None:
    cfg, console_level = _load_cfg_and_logging(config)
    fs = FileSystemManager(cfg.file_system.local_path)
    indexer = GroupFileIndexer(cfg, fs)
    keyword = keyword or []

    def _split_compound_query(raw: str) -> list[str]:
        s = str(raw or "").strip()
        if not s:
            return []
        parts = [p.strip() for p in s.split(",") if p.strip()]
        # 兼容常见写法：name::xxx,uploader::yyy
        if len(parts) > 1 and any("::" in p for p in parts[1:]):
            return parts
        return [s]

    query_groups: list[tuple[str, list[Any]]] = []
    flat_queries: list[Any] = []
    for raw in keyword:
        parts = _split_compound_query(raw)
        if not parts:
            continue
        parsed_group: list[Any] = []
        for part in parts:
            try:
                parsed = indexer.parse_query(part)
            except Exception as e:
                console.print(f"[red]查询语法错误[/red] `{part}`: {e}")
                indexer.close()
                raise typer.Exit(code=2)
            parsed_group.append(parsed)
            flat_queries.append(parsed)
        query_groups.append((",".join(parts), parsed_group))

    async def refresh_runner() -> None:
        async with OneBotWsClient(cfg.onebot11.ws_url, cfg.onebot11.access_token) as bot:
            if not flat_queries:
                stats = await indexer.update_index(bot, target="all", no_cache=True)
                _print_index_update_stats(stats, title="索引刷新完成")
                return

            group_filters = sorted({q.group_id for q in flat_queries if getattr(q, "group_id", None)})
            all_scoped = bool(flat_queries) and all(bool(getattr(q, "group_id", None)) for q in flat_queries)
            if group_filters and all_scoped:
                merged: dict[str, Any] = {}
                for gid in group_filters:
                    part = await indexer.update_index(bot, target=gid, no_cache=True)
                    merged.update(part)
                _print_index_update_stats(merged, title="索引刷新完成")
            else:
                stats = await indexer.update_index(bot, target="all", no_cache=True)
                _print_index_update_stats(stats, title="索引刷新完成")

    try:
        if refresh:
            _run_with_ws(console_level, cfg, refresh_runner)

        if not query_groups:
            if refresh:
                console.print(f"索引库: {indexer.db_path}")
                console.print(f"总记录数: {indexer.count_records()}")
                return
            console.print("请提供查询表达式，或使用 --refresh 先刷新索引。")
            raise typer.Exit(code=2)

        total_queries = max(1, len(query_groups))
        render_events: list[tuple[str, Any]] = []
        with create_search_progress(console, description="搜索中", transient=False) as progress:
            def _label(text: str, idx: int | None = None) -> str:
                if total_queries <= 1:
                    return text
                if idx is None:
                    return text
                return f"查询 {idx}/{total_queries} | {text}"

            task_id = progress.add_task(
                "搜索中",
                total=total_queries,
                completed=0,
                phase1_total=100,
                phase1_done=0,
                inner_label=_label("准备中", 0),
            )
            done = 0
            for group_idx, (group_raw, group_queries) in enumerate(query_groups, start=1):
                status = "OK"
                progress.update(
                    task_id,
                    total=total_queries,
                    completed=done,
                    phase1_total=100,
                    phase1_done=0,
                    inner_label=_label(f"准备: {group_raw}", group_idx),
                )
                try:
                    t0 = time.perf_counter()
                    group_results: list[list[Any]] = []
                    for sub_idx, q in enumerate(group_queries, start=1):
                        def on_internal_progress(stage: str, inner_done: int, inner_total: int) -> None:
                            progress.update(
                                task_id,
                                total=total_queries,
                                completed=done,
                                phase1_total=max(1, int(inner_total)),
                                phase1_done=min(max(0, int(inner_done)), max(1, int(inner_total))),
                                inner_label=_label(
                                    f"条件 {sub_idx}/{len(group_queries)} {stage} {inner_done}/{inner_total}",
                                    group_idx,
                                ),
                            )

                        group_results.append(
                            indexer.search(
                                q,
                                strict=strict,
                                min_results=int(cfg.search.min_results),
                                on_progress=on_internal_progress,
                            )
                        )

                    if not group_results:
                        results = []
                    elif len(group_results) == 1:
                        results = group_results[0]
                    else:
                        # 组合查询采用加权融合而不是硬交集：
                        # 全命中优先，同时保留主要条件强命中 + 次要条件弱命中/未命中的候选
                        cond_total = len(group_results)
                        agg: dict[int, dict[str, float | int]] = {}
                        row_obj: dict[int, Any] = {}
                        for cond_idx, one in enumerate(group_results):
                            total_one = max(1, len(one))
                            for rank, item in enumerate(one, start=1):
                                rid = int(getattr(item, "row_id", 0) or 0)
                                if rid <= 0:
                                    continue
                                # 位置越靠前权重越高
                                rank_score = 1.0 - float(rank - 1) / float(total_one)
                                cond_weight = 1.15 if cond_idx == 0 else 1.0
                                score_gain = (100.0 + 25.0 * rank_score) * cond_weight
                                st = agg.get(rid)
                                if st is None:
                                    st = {"hits": 0, "score": 0.0, "rank_sum": 0.0}
                                    agg[rid] = st
                                st["hits"] = int(st["hits"]) + 1
                                st["score"] = float(st["score"]) + float(score_gain)
                                st["rank_sum"] = float(st["rank_sum"]) + float(rank_score)

                                prev = row_obj.get(rid)
                                if prev is None:
                                    row_obj[rid] = item
                                else:
                                    prev_tag = str(getattr(prev, "match_tag", "") or "")
                                    cur_tag = str(getattr(item, "match_tag", "") or "")
                                    if prev_tag != "Exact" and cur_tag == "Exact":
                                        row_obj[rid] = item

                        ranked: list[tuple[Any, float, int, float]] = []
                        for rid, st in agg.items():
                            hits = int(st["hits"])
                            coverage = float(hits) / float(max(1, cond_total))
                            score = float(st["score"]) + 80.0 * coverage
                            if hits == cond_total:
                                score += 40.0
                            ranked.append((row_obj[rid], score, hits, float(st["rank_sum"])))

                        ranked.sort(
                            key=lambda x: (
                                -float(x[1]),
                                -int(x[2]),
                                -float(x[3]),
                                str(getattr(x[0], "file_name", "") or "").lower(),
                            )
                        )
                        full_hits: list[Any] = []
                        partial_hits: list[Any] = []
                        for item, _score, hits, _rank_sum in ranked:
                            if int(hits) >= cond_total:
                                full_hits.append(item)
                            else:
                                partial_hits.append(item)

                        threshold_local = max(1, int(cfg.search.min_results))
                        results = list(full_hits)
                        if len(results) < threshold_local:
                            need = threshold_local - len(results)
                            results.extend(partial_hits[:need])
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    render_events.append(("result", (group_raw, results, elapsed_ms)))
                except re.error as e:
                    status = "REGEX_ERR"
                    render_events.append(("error", f"[red]正则错误[/red] `{group_raw}`: {e}"))
                except Exception as e:
                    status = "FAIL"
                    logging.getLogger(__name__).exception("search failed: query=%s", group_raw)
                    render_events.append(("error", f"[red]搜索失败[/red] `{group_raw}`: {e}"))
                finally:
                    done += 1
                    progress.update(
                        task_id,
                        total=total_queries,
                        completed=done,
                        phase1_total=100,
                        phase1_done=100,
                        inner_label=_label(f"完成 [{status}]", group_idx),
                    )
            progress.update(
                task_id,
                total=total_queries,
                completed=total_queries,
                phase1_total=100,
                phase1_done=100,
                inner_label=_label("搜索完成", total_queries if total_queries > 1 else None),
            )
        for kind, payload in render_events:
            if kind == "result":
                raw, results, elapsed_ms = payload
                _print_search_results(str(raw), list(results), elapsed_ms=float(elapsed_ms))
            else:
                console.print(str(payload))
    finally:
        indexer.close()


if __name__ == "__main__":
    app()

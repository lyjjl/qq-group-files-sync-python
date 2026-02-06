from __future__ import annotations

import asyncio
import logging
import signal
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel

import websockets

from config import load_config, is_placeholder_config, find_duplicate_group_ids
from dashboard import generate_dashboard
from filesystem import FileSystemManager
from onebot import OneBotWsClient, extract_plain_text, parse_group_numeric_id
from pusher import GroupFilePusher
from syncer import GroupFileSyncer
from ignore_rules import IgnoreMatcher

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


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
    concurrency: int = typer.Option(4, "--concurrency", min=1, max=32, help="并发下载数"),
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
            syncer = GroupFileSyncer(cfg, fs, concurrency=concurrency, ignore=ignore)
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
    mirror: bool = typer.Option(False, "--mirror", help="镜像模式：覆盖不同文件、删除远端多余文件（按文件名+大小判断）"),
    plan: bool = typer.Option(False, "--plan", help="只对比并输出将执行的操作，不做任何修改"),
    ignore_file: str | None = typer.Option(None, "--ignore-file", help="忽略规则文件路径（相对路径）。默认读取 ./.ignore（如存在）"),
) -> None:
    cfg, console_level = _load_cfg_and_logging(config)
    fs, ignore = _build_fs_and_ignore(cfg, ignore_file)
    push_all = (target or "").strip().lower() == "all"

    if push_all and is_placeholder_config(cfg):
        logging.getLogger(__name__).warning("refuse to run: placeholder config")
        console.print("检测到示例配置尚未修改。")
        console.print("请编辑 config.toml：填写真实的 groups 列表，并确认 onebot11.ws_url 可连接。")
        raise typer.Exit(code=2)

    async def runner() -> None:
        async with OneBotWsClient(cfg.onebot11.ws_url, cfg.onebot11.access_token) as bot:
            pusher = GroupFilePusher(fs, concurrency=concurrency, ignore=ignore)

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
    concurrency: int = typer.Option(4, "--concurrency", min=1, max=32, help="并发下载数"),
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
            syncer = GroupFileSyncer(cfg, fs, concurrency=concurrency, ignore=ignore)
            await _interactive(cfg, fs, bot, syncer)

    _run_with_ws(console_level, cfg, runner)


if __name__ == "__main__":
    app()

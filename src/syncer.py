from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from urllib.parse import urlparse

import aiofiles
import httpx
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from config import AppConfig
from filesystem import FileSystemManager
from group_paths import (
    group_root_dir,
    group_status_file_path,
    group_timestamp_file_path,
    group_relative_file_path,
    sanitize_component,
)
from onebot import OneBotWsClient, parse_group_numeric_id
from ignore_rules import IgnoreMatcher

console = Console()


@dataclass
class GroupFileInfo:
    file_id: str
    file_name: str
    busid: int
    file_size: int
    upload_time: int | None = None
    modify_time: int | None = None
    dead_time: int | None = None
    download_times: int | None = None
    uploader: int | None = None
    uploader_name: str | None = None
    folder_path: str = ""


@dataclass
class GroupFolderInfo:
    folder_id: str
    folder_name: str


@dataclass
class FileTimestampRecord:
    file_path: str
    file_size: int
    modify_time: int | None
    upload_time: int | None
    file_id: str


@dataclass
class GroupFileStatus:
    group_id: str
    last_update: int
    total_files: int
    total_folders: int
    files: list[dict[str, Any]]
    folders: list[dict[str, Any]]
    download_path: str


@dataclass
class SyncPrediction:
    total_files: int
    total_folders: int
    total_size: int
    existing_files: int
    existing_folders: int
    existing_size: int
    files_to_update: int
    folders_to_create: int
    update_size: int
    files_to_delete: int
    folders_to_delete: int
    delete_size: int
    update_file_map: dict[str, GroupFileInfo]
    extra_local_files: list[str]
    status: GroupFileStatus
    ignored_empty_remote: list[str]
    ignored_empty_local: list[str]


def _format_size(size: int) -> str:
    unit = 1024
    if size < unit:
        return f"{size} B"
    exp = 0
    div = float(unit)
    n = float(size)
    while n / unit >= unit:
        n /= unit
        exp += 1
    # recompute with power
    div = float(unit) ** (exp + 1)
    return f"{size / div:.1f} {'KMGTPE'[exp]}B"



def _is_valid_group_file_url(url: str) -> bool:
    try:
        p = urlparse(url)
    except Exception:
        return False
    if p.scheme not in {"http", "https"}:
        return False
    if not p.netloc:
        return False
    if not p.path.startswith("/ftn_handler/"):
        return False
    if not p.netloc.endswith("ftn.qq.com"):
        return False
    return True


class GroupFileSyncer:
    def __init__(
        self,
        cfg: AppConfig,
        fs: FileSystemManager,
        *,
        concurrency: int = 4,
        ignore: IgnoreMatcher | None = None,
    ):
        self.cfg = cfg
        self.fs = fs
        self.concurrency = max(1, int(concurrency))
        self._timestamp_cache: dict[str, dict[str, FileTimestampRecord]] = {}
        self.ignore = ignore

        # invalid files (你妈的sbtx，封文件好玩吗)
        self._invalid_files: dict[str, set[str]] = defaultdict(set)
        self._invalid_log_lock = asyncio.Lock()

    def get_invalid_counts(self) -> dict[str, int]:
        return {gid: len(paths) for gid, paths in self._invalid_files.items() if paths}

    def get_invalid_total(self) -> int:
        return sum(len(v) for v in self._invalid_files.values())

    async def _record_invalid_file(self, group_id_str: str, rel_path: str, *, reason: str = "") -> None:
        rel_path = (rel_path or "").strip().replace("\\", "/")
        if not rel_path:
            return

        if rel_path in self._invalid_files[group_id_str]:
            return
        self._invalid_files[group_id_str].add(rel_path)

        try:
            raw_path = (getattr(self.cfg, "invalid_files_log", "") or "").strip()
            if raw_path:
                log_path = Path(raw_path).expanduser()
                if str(raw_path).endswith(("/", "\\")) or (log_path.exists() and log_path.is_dir()):
                    log_path = log_path / "invalidFiles.log"
                log_path = log_path.resolve()
            else:
                log_path = Path(self.cfg.log_file or "./logs/main.log").expanduser().resolve().parent / "invalidFiles.log"

            log_dir = log_path.parent
            log_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            reason = (reason or "").strip()
            line = f"{ts}\t{group_id_str}\t{rel_path}\t{reason}\n"
            async with self._invalid_log_lock:
                async with aiofiles.open(log_path, "a", encoding="utf-8") as af:
                    await af.write(line)
        except Exception:
            logging.getLogger(__name__).exception("failed to write invalidFiles.log")

    async def get_complete_file_list(self, bot: OneBotWsClient, group_id_str: str) -> tuple[list[GroupFileInfo], list[GroupFolderInfo]]:
        group_id_num = parse_group_numeric_id(group_id_str)
        files: list[GroupFileInfo] = []
        folders: list[GroupFolderInfo] = []
        seen_file: set[str] = set()
        seen_folder: set[str] = set()

        async def fetch(folder_id: str, folder_path: str) -> None:
            if folder_id:
                res = await bot.call_api(
                    "get_group_files_by_folder",
                    {"group_id": group_id_num, "folder_id": folder_id},
                )
            else:
                res = await bot.call_api("get_group_root_files", {"group_id": group_id_num})

            data = res.data or {}
            for f in data.get("files", []) or []:
                fid = str(f.get("file_id", ""))
                if not fid or fid in seen_file:
                    continue
                seen_file.add(fid)
                fname = str(f.get("file_name") or "").strip()
                files.append(
                    GroupFileInfo(
                        file_id=fid,
                        file_name=fname,
                        busid=int(f.get("busid", 0)),
                        file_size=int(f.get("file_size", 0)),
                        upload_time=_safe_int(f.get("upload_time")),
                        modify_time=_safe_int(f.get("modify_time")),
                        dead_time=_safe_int(f.get("dead_time")),
                        download_times=_safe_int(f.get("download_times")),
                        uploader=_safe_int(f.get("uploader")),
                        uploader_name=_safe_str(f.get("uploader_name")),
                        folder_path=folder_path,
                    )
                )

            for d in data.get("folders", []) or []:
                did = str(d.get("folder_id", ""))
                if not did or did in seen_folder:
                    continue
                seen_folder.add(did)
                folders.append(GroupFolderInfo(folder_id=did, folder_name=str(d.get("folder_name", ""))))

            for d in data.get("folders", []) or []:
                did = str(d.get("folder_id", ""))
                name = sanitize_component(str(d.get("folder_name", "")))
                next_path = str(Path(folder_path) / name) if folder_path else name
                try:
                    await fetch(did, next_path)
                except Exception:
                    continue

        await fetch("", "")
        return files, folders

    async def generate_sync_prediction(self, bot: OneBotWsClient, group_id_str: str, *, mirror: bool = False) -> SyncPrediction:
        remote_files_all, remote_folders = await self.get_complete_file_list(bot, group_id_str)

        # ignore
        if self.ignore:
            remote_files: list[GroupFileInfo] = []
            for f in remote_files_all:
                rel_key = group_relative_file_path(f.folder_path, f.file_name)
                if self.ignore.is_ignored(rel_key):
                    continue
                remote_files.append(f)
        else:
            remote_files = remote_files_all

        group_root = group_root_dir(group_id_str)

        existing_files_list = self.fs.list_files_under(group_root)
        existing_set: set[str] = set()
        existing_size_map: dict[str, int] = {}
        empty_local_full: set[str] = set()
        ignored_empty_local: list[str] = []

        root_prefix = f"{group_root}/".replace("\\", "/")
        for pth in existing_files_list:
            ps = str(pth).replace("\\", "/")
            if not ps.startswith(root_prefix):
                continue
            rel = ps[len(root_prefix):]
            if self.ignore and self.ignore.is_ignored(rel):
                continue
            existing_set.add(ps)
            try:
                sz = int((self.fs.base_path / Path(ps)).stat().st_size)
                existing_size_map[ps] = sz
                if sz == 0:
                    empty_local_full.add(ps)

                    if ps.startswith(root_prefix):
                        ignored_empty_local.append(ps[len(root_prefix):])
            except Exception:
                existing_size_map[ps] = 0

        update_map: dict[str, GroupFileInfo] = {}

        # 在打印预测结果时，“已存在”的含义：
        # 在mirror模式下表示已匹配
        # 在增量模式下表示已经存在
        present_files = 0
        present_size = 0
        matched_files = 0
        matched_size = 0

        ignored_empty_remote: list[str] = []

        for f in remote_files:
            rel = group_relative_file_path(f.folder_path, f.file_name)
            full_rel = f"{group_root}/{rel}".replace("\\", "/")

            local_exists = (full_rel in existing_set)
            local_size = int(existing_size_map.get(full_rel, -1))

            if local_exists:
                present_files += 1
                present_size += int(existing_size_map.get(full_rel, 0))

            # 忽略远程空文件：
            # - 不对它们进行下载或替换
            # - 但如果本地文件也为 0B，则仍将其视为“已存在”或“已匹配”
            if int(f.file_size) == 0:
                ignored_empty_remote.append(rel)
                if mirror and local_exists and local_size == 0:
                    matched_files += 1
                    matched_size += 0
                continue

            if local_exists:
                if mirror:
                    if local_size == int(f.file_size):
                        matched_files += 1
                        matched_size += int(f.file_size)
                    else:
                        update_map[full_rel] = f
                else:
                    # 增量模式下不覆写不同的文件
                    pass
            else:
                update_map[full_rel] = f

        files_to_update = len(update_map)
        update_size = sum(int(f.file_size) for f in update_map.values())

        expected = {f"{group_root}/{group_relative_file_path(f.folder_path, f.file_name)}".replace("\\", "/") for f in remote_files}

        files_to_delete = 0
        delete_size = 0
        extra_local_files: list[str] = []
        if mirror:
            extra = [p for p in existing_set if p not in expected and p not in empty_local_full]
            files_to_delete = len(extra)
            delete_size = sum(int(existing_size_map.get(p, 0)) for p in extra)
            for p in extra:
                ps = str(p).replace("\\", "/")
                if ps.startswith(root_prefix):
                    extra_local_files.append(ps[len(root_prefix) :])
                else:
                    extra_local_files.append(ps)
            extra_local_files.sort()

        status = GroupFileStatus(
            group_id=group_id_str,
            last_update=int(time.time()),
            total_files=len(remote_files),
            total_folders=len(remote_folders),
            files=[self._file_to_dict(x) for x in remote_files],
            folders=[{"folder_id": x.folder_id, "folder_name": x.folder_name} for x in remote_folders],
            download_path=str((self.fs.base_path / group_root).resolve()),
        )

        if mirror:
            existing_files = matched_files
            existing_size = matched_size
        else:
            existing_files = present_files
            existing_size = present_size

        total_size = sum(int(f.file_size) for f in remote_files)

        return SyncPrediction(
            total_files=len(remote_files),
            total_folders=len(remote_folders),
            total_size=int(total_size),
            existing_files=int(existing_files),
            existing_folders=len(remote_folders),
            existing_size=int(existing_size),
            files_to_update=int(files_to_update),
            folders_to_create=0,
            update_size=int(update_size),
            files_to_delete=int(files_to_delete),
            folders_to_delete=0,
            delete_size=int(delete_size),
            update_file_map=update_map,
            extra_local_files=extra_local_files,
                        status=status,
            ignored_empty_remote=sorted(ignored_empty_remote),
            ignored_empty_local=sorted(set(ignored_empty_local)),
        )

    async def sync_group(self, bot: OneBotWsClient, group_id_str: str, *, mirror: bool = False, plan: bool = False) -> None:
        self._preload_timestamps(group_id_str)
        pred = await self.generate_sync_prediction(bot, group_id_str, mirror=mirror)
        self.print_prediction(pred)

        group_root = group_root_dir(group_id_str)
        self.print_plan(pred, group_root, mirror=mirror, plan=plan)
        if plan:
            return

        if pred.ignored_empty_remote:
            logging.getLogger(__name__).warning("ignored %d empty remote file(s) (0B) for group=%s", len(pred.ignored_empty_remote), group_id_str)
            console.print(f"[yellow]WARN[/yellow]: 已忽略 {len(pred.ignored_empty_remote)} 个远端空文件(0B)，不会下载/替换。")

        self.fs.mkdir_all(group_root)

        await self._download_updates(bot, group_id_str, group_root, pred.update_file_map, mirror=mirror)

        if mirror:
            await self._cleanup_extra_files(pred.status, group_root)

        self.fs.write_text(
            group_status_file_path(group_id_str),
            json.dumps(pred.status.__dict__, ensure_ascii=False, indent=2),
        )

        self._flush_timestamps(group_id_str)

    def print_prediction(self, p: SyncPrediction) -> None:
        table = Table(show_header=True, header_style="bold", box=None)
        table.add_column("项目")
        table.add_column("数量", justify="right")
        table.add_column("大小", justify="right")

        table.add_row(
            "群内",
            f"{p.total_folders} 文件夹 / {p.total_files} 文件",
            _format_size(p.total_size),
        )
        table.add_row(
            "本地已存",
            f"{p.existing_folders} 文件夹 / {p.existing_files} 文件",
            _format_size(p.existing_size),
        )
        table.add_row("需要更新", f"{p.files_to_update} 文件", _format_size(p.update_size))
        if p.folders_to_create:
            table.add_row("需要创建", f"{p.folders_to_create} 文件夹", "-")
        table.add_row(
            "需要删除",
            f"{p.files_to_delete} 文件 / {p.folders_to_delete} 文件夹",
            _format_size(p.delete_size),
        )
        if getattr(p, "ignored_empty_remote", None):
            table.add_row("忽略空文件(远端)", f"{len(p.ignored_empty_remote)}", "-")

        console.print(Panel(table, title="同步预测结果", expand=False))

    def print_plan(self, p: SyncPrediction, group_root: str, *, mirror: bool, plan: bool = False) -> None:

        if not plan:
            return

        root_prefix = f"{group_root}/".replace("\\", "/")

        # 更新文件分为 download 和 replace
        download_files: list[str] = []
        replace_files: list[str] = []
        for full_rel in sorted(p.update_file_map.keys()):
            rel = str(full_rel).replace("\\", "/")
            if rel.startswith(root_prefix):
                rel = rel[len(root_prefix) :]
            if self.fs.exists(full_rel):
                replace_files.append(rel)
            else:
                download_files.append(rel)

        dirs_to_create: set[str] = set()
        for rel in download_files + replace_files:
            parent = str(Path(rel).parent).replace("\\", "/")
            if parent in {".", ""}:
                continue
            dpath = self.fs.base_path / Path(group_root) / Path(parent)
            if not dpath.exists():
                dirs_to_create.add(parent)

        extra_files = p.extra_local_files if mirror else []

        dirs_to_delete: list[str] = []
        if mirror:
            expected_files: set[str] = set()
            for f in p.status.files:
                folder_path = str(f.get("folder_path") or "")
                name = str(f.get("file_name") or "")
                expected_files.add(group_relative_file_path(folder_path, name))

            local_kept, local_ignored = self._list_local_files_rel(group_root)
            kept_files = set(expected_files) | set(local_ignored) | set(getattr(p, "ignored_empty_local", []) or [])

            keep_dirs: set[str] = set()
            for fp in kept_files:
                for d in self._all_parent_dirs(fp):
                    keep_dirs.add(d)

            for d in self._list_local_dirs_rel(group_root):
                if d in keep_dirs:
                    continue
                if self.ignore and self.ignore.is_ignored(d):
                    continue
                dirs_to_delete.append(d)
            dirs_to_delete.sort(key=lambda x: (x.count("/"), x), reverse=True)

        action_width = 12
        detail_width = 52
        note_width = 16

        summary = Table(show_header=True, header_style="bold", box=None)
        summary.add_column("动作", width=action_width, no_wrap=True)
        summary.add_column("数量", width=detail_width)
        summary.add_column("说明", width=note_width)

        def add_summary(action: str, n: int, note: str) -> None:
            if n:
                summary.add_row(action, str(n), note)

        add_summary("忽略空文件", len(getattr(p, "ignored_empty_remote", []) or []), "0B：不会下载/替换")
        add_summary("保留空文件", len(getattr(p, "ignored_empty_local", []) or []), "0B：不参与删除")
        add_summary("创建文件夹", len(dirs_to_create), "只在需要时创建")
        add_summary("替换文件", len(replace_files), "删除后下载")
        add_summary("下载缺失", len(download_files), "仅缺失项")
        add_summary("删除多余", len(extra_files), "仅镜像模式")
        add_summary("清理空文件夹", len(dirs_to_delete), "推测")

        details = Table(show_header=True, header_style="bold", box=None)
        details.add_column("动作", width=action_width, no_wrap=True)
        details.add_column("路径", width=detail_width)
        details.add_column("说明", width=note_width)

        rows: list[tuple[str, str, str]] = []

        for it in sorted(getattr(p, "ignored_empty_remote", []) or []):
            rows.append(("IGNORE_EMPTY_REMOTE", it, "0B：自动忽略"))
        for it in sorted(getattr(p, "ignored_empty_local", []) or []):
            rows.append(("IGNORE_EMPTY_LOCAL", it, "0B：保留，不参与删除"))

        for it in sorted(dirs_to_create):
            rows.append(("MKDIR", it, "创建本地文件夹"))
        for it in replace_files:
            rows.append(("REPLACE", it, "删除后下载"))
        for it in download_files:
            rows.append(("DOWNLOAD", it, "下载缺失"))

        for it in extra_files:
            rows.append(("DELETE", it, "镜像：删除本地多余"))
        for it in dirs_to_delete:
            rows.append(("RMDIR", it, "镜像：清理空文件夹(推测)"))

        style_map: dict[str, str] = {
            "IGNORE_EMPTY_REMOTE": "yellow",
            "IGNORE_EMPTY_LOCAL": "yellow",
            "MKDIR": "cyan",
            "REPLACE": "magenta",
            "DOWNLOAD": "green",
            "DELETE": "red",
            "RMDIR": "red",
        }
        for action, path, note in rows:
            st = style_map.get(action, "")
            details.add_row(Text(action, style=st), path, note)
        if len(rows) > 200:
            details.add_row("...", f"...（已截断，剩余 {len(rows) - 200} 项）", "")

        divider = Text("─" * (action_width + detail_width + note_width + 6), style="dim")
        console.print(Panel(Group(summary, divider, details), title="操作计划", expand=False))


    def _build_plan_table(self, rows: list[tuple[str, str, str]], *, limit: int = 200) -> Table:
        table = Table(show_header=True, header_style="bold", box=None)
        table.add_column("动作", no_wrap=True)
        table.add_column("路径")
        table.add_column("说明")
        show = rows[:limit]
        for action, path, note in show:
            table.add_row(action, path, note)
        if len(rows) > limit:
            table.add_row("...", f"...（已截断，剩余 {len(rows) - limit} 项）", "")
        return table

    def _list_local_files_rel(self, group_root: str) -> tuple[set[str], set[str]]:

        kept: set[str] = set()
        ignored: set[str] = set()

        root_prefix = f"{group_root}/".replace("\\", "/")
        for p in self.fs.list_files_under(group_root):
            ps = str(p).replace("\\", "/")
            if not ps.startswith(root_prefix):
                continue
            rel = ps[len(root_prefix) :]
            if not rel:
                continue
            if self.ignore and self.ignore.is_ignored(rel):
                ignored.add(rel)
            else:
                kept.add(rel)
        return kept, ignored

    def _list_local_dirs_rel(self, group_root: str) -> list[str]:
        root = self.fs.base_path / Path(group_root)
        if not root.exists() or not root.is_dir():
            return []
        out: list[str] = []
        for d in root.rglob("*"):
            if not d.is_dir():
                continue
            rel = d.relative_to(root).as_posix()
            if rel:
                out.append(rel)
        return out

    @staticmethod
    def _all_parent_dirs(rel_file: str) -> list[str]:
        p = Path(rel_file)
        out: list[str] = []
        cur = p.parent
        while str(cur) not in {".", ""}:
            s = str(cur).replace("\\", "/")
            out.append(s)
            cur = cur.parent
        return out

    async def _download_updates(
        self,
        bot: OneBotWsClient,
        group_id_str: str,
        group_root: str,
        update_map: dict[str, GroupFileInfo],
        *,
        mirror: bool,
    ) -> None:
        if not update_map:
            return

        group_id_num = parse_group_numeric_id(group_id_str)
        semaphore = asyncio.Semaphore(self.concurrency)
        ts_map = self._timestamp_cache.setdefault(group_id_str, {})

        async with httpx.AsyncClient(follow_redirects=True, timeout=httpx.Timeout(60.0)) as client:

            async def download_one(full_rel: str, f: GroupFileInfo) -> None:
                async with semaphore:
                    try:
                        safe_name = (f.file_name or "").strip()
                        if not safe_name:
                            logging.getLogger(__name__).warning(
                                "skip abnormal file record (empty file_name): group=%s folder_path=%r file_id=%s busid=%s",
                                group_id_str,
                                f.folder_path,
                                f.file_id,
                                f.busid,
                            )
                            return

                        rel_path = group_relative_file_path(f.folder_path, safe_name)
                        try:
                            url_res = await bot.call_api(
                                "get_group_file_url",
                                {"group_id": group_id_num, "file_id": f.file_id, "busid": f.busid},
                            )
                        except asyncio.TimeoutError:
                            logging.getLogger(__name__).debug(
                                "group file url timeout, skipped: group=%s path=%s file_id=%s busid=%s",
                                group_id_str,
                                rel_path,
                                f.file_id,
                                f.busid,
                            )
                            await self._record_invalid_file(group_id_str, rel_path, reason="url_timeout")
                            return
                        data = url_res.data or {}
                        raw_url = data.get("url")
                        url = str(raw_url).strip() if raw_url is not None else ""
                        if not url or not _is_valid_group_file_url(url):
                            logging.getLogger(__name__).debug(
                                "group file invalid (no url), skipped: group=%s path=%s file_id=%s busid=%s url=%r",
                                group_id_str,
                                rel_path,
                                f.file_id,
                                f.busid,
                                raw_url,
                            )
                            await self._record_invalid_file(group_id_str, rel_path, reason="invalid_url")
                            return

                        target = self.fs.base_path / Path(full_rel)
                        target.parent.mkdir(parents=True, exist_ok=True)

                        # 先下载到临时文件，然后进行替换
                        tmp_suffix = sanitize_component(str(f.file_id).strip("/") or "file")[:16]
                        tmp = target.with_name(f"{target.name}.part.{tmp_suffix}")

                        logging.getLogger(__name__).debug("download request: %s", url)
                        start_ts = time.monotonic()
                        total_bytes = 0
                        async with client.stream("GET", url) as resp:
                            logging.getLogger(__name__).debug(
                                "download response: url=%s status=%s headers=%s",
                                url,
                                resp.status_code,
                                dict(resp.headers),
                            )
                            if resp.status_code in {403, 404, 410}:
                                logging.getLogger(__name__).debug(
                                    "group file invalid (http %s), skipped: group=%s path=%s full_rel=%s",
                                    resp.status_code,
                                    group_id_str,
                                    rel_path,
                                    full_rel,
                                )
                                await self._record_invalid_file(
                                    group_id_str,
                                    rel_path,
                                    reason=f"http_{resp.status_code}",
                                )
                                return
                            resp.raise_for_status()
                            async with aiofiles.open(tmp, "wb") as af:
                                async for chunk in resp.aiter_bytes():
                                    total_bytes += len(chunk)
                                    await af.write(chunk)
                        elapsed_ms = int((time.monotonic() - start_ts) * 1000)
                        logging.getLogger(__name__).debug(
                            "download complete: url=%s bytes=%s elapsed_ms=%s",
                            url,
                            total_bytes,
                            elapsed_ms,
                        )

                        if target.exists() and target.is_dir():
                            if mirror:
                                shutil.rmtree(target)
                            else:
                                raise IsADirectoryError(f"target exists as directory: {target}")

                        os.replace(tmp, target)

                        rel = target.relative_to(self.fs.base_path).as_posix()
                        self.fs.set_file_times(rel, upload_time=f.upload_time, modify_time=f.modify_time)

                        rec = FileTimestampRecord(
                            file_path=rel,
                            file_size=f.file_size,
                            modify_time=f.modify_time,
                            upload_time=f.upload_time,
                            file_id=f.file_id,
                        )
                        ts_map[rec.file_path] = rec
                    except Exception:
                        try:
                            if 'tmp' in locals() and isinstance(tmp, Path) and tmp.exists():
                                tmp.unlink(missing_ok=True)
                        except Exception:
                            pass
                        logging.getLogger(__name__).exception(
                            "download failed: group=%s file=%s full_rel=%s",
                            group_id_str,
                            f.file_name,
                            full_rel,
                        )
                        raise

            tasks = [asyncio.create_task(download_one(k, v)) for k, v in update_map.items()]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        failures = [r for r in results if isinstance(r, Exception)]
        if failures:
            logging.getLogger(__name__).warning(
                "download finished with %d error(s) for group=%s",
                len(failures),
                group_id_str,
            )
            console.print(
                Panel(
                    f"下载过程中出现 {len(failures)} 个错误。\n详情请查看 error.log。",
                    title="下载错误",
                    expand=False,
                )
            )

    async def _cleanup_extra_files(self, status: GroupFileStatus, group_root: str) -> None:
        expected = set()
        for f in status.files:
            folder_path = str(f.get("folder_path") or "")
            rel = group_relative_file_path(folder_path, str(f.get("file_name") or ""))
            expected.add(f"{group_root}/{rel}".replace("\\", "/"))

        existing_full = self.fs.list_files_under(group_root)
        existing: set[str] = set()
        root_prefix = f"{group_root}/".replace("\\", "/")
        for p in existing_full:
            ps = str(p).replace("\\", "/")
            if not ps.startswith(root_prefix):
                continue
            rel = ps[len(root_prefix) :]
            if self.ignore and self.ignore.is_ignored(rel):
                continue
            try:
                if (self.fs.base_path / Path(ps)).stat().st_size == 0:
                    continue
            except Exception:
                pass
            existing.add(ps)

        extra = [p for p in existing if p not in expected]
        for p in extra:
            self.fs.remove_file(p)

        self._remove_empty_dirs_with_ignore(group_root)

    def _remove_empty_dirs_with_ignore(self, group_root: str) -> None:

        root = self.fs.base_path / Path(group_root)
        if not root.exists() or not root.is_dir():
            return

        dirs = [p for p in root.rglob("*") if p.is_dir()]
        dirs.sort(key=lambda x: len(x.parts), reverse=True)
        for d in dirs:
            rel = d.relative_to(root).as_posix()
            if rel and self.ignore and self.ignore.is_ignored(rel):
                continue
            try:
                if not any(d.iterdir()):
                    d.rmdir()
            except OSError:
                pass

    def _preload_timestamps(self, group_id_str: str) -> None:
        if group_id_str in self._timestamp_cache:
            return
        ts_file = group_timestamp_file_path(group_id_str)
        if not self.fs.exists(ts_file):
            self._timestamp_cache[group_id_str] = {}
            return

        ts_map: dict[str, FileTimestampRecord] = {}
        raw = self.fs.read_bytes(ts_file).decode("utf-8", errors="ignore")
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rec = FileTimestampRecord(
                    file_path=str(obj.get("file_path") or obj.get("filePath") or ""),
                    file_size=int(obj.get("file_size") or obj.get("fileSize") or 0),
                    modify_time=_safe_int(obj.get("modify_time") or obj.get("modifyTime")),
                    upload_time=_safe_int(obj.get("upload_time") or obj.get("uploadTime")),
                    file_id=str(obj.get("file_id") or obj.get("fileID") or obj.get("fileId") or ""),
                )
                if rec.file_path:
                    ts_map[rec.file_path] = rec
            except Exception:
                continue

        self._timestamp_cache[group_id_str] = ts_map

    def _flush_timestamps(self, group_id_str: str) -> None:
        ts_map = self._timestamp_cache.get(group_id_str) or {}
        ts_file = group_timestamp_file_path(group_id_str)
        lines = []
        for rec in ts_map.values():
            lines.append(
                json.dumps(
                    {
                        "file_path": rec.file_path,
                        "file_size": rec.file_size,
                        "modify_time": rec.modify_time,
                        "upload_time": rec.upload_time,
                        "file_id": rec.file_id,
                    },
                    ensure_ascii=False,
                )
            )
        self.fs.write_text(ts_file, "\n".join(lines) + ("\n" if lines else ""))

    @staticmethod
    def _is_same(rec: FileTimestampRecord | None, f: GroupFileInfo) -> bool:
        if rec is None:
            return False
        if rec.file_id and rec.file_id != f.file_id:
            return False
        if rec.file_size != f.file_size:
            return False
        if rec.modify_time is not None and f.modify_time is not None and rec.modify_time != f.modify_time:
            return False
        return True

    @staticmethod
    def _file_to_dict(f: GroupFileInfo) -> dict[str, Any]:
        return {
            "file_id": f.file_id,
            "file_name": f.file_name,
            "busid": f.busid,
            "file_size": f.file_size,
            "upload_time": f.upload_time,
            "modify_time": f.modify_time,
            "dead_time": f.dead_time,
            "download_times": f.download_times,
            "uploader": f.uploader,
            "uploader_name": f.uploader_name,
            "folder_path": f.folder_path,
        }


def _safe_int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def _safe_str(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v)
    return s if s else None

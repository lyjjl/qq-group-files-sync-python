from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from urllib.parse import urlparse

import aiofiles
import httpx
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from config import AppConfig
from filesystem import FileSystemManager
from group_paths import (
    group_root_dir,
    group_status_file_path,
    group_timestamp_file_path,
    group_invalid_url_count_file_path,
    group_relative_file_path,
    sanitize_component,
)
from onebot import OneBotWsClient, parse_group_numeric_id
from ignore_rules import IgnoreMatcher
from local_files import list_group_files, list_group_files_rel
from plan_output import print_plan_panel
from progress_ui import create_progress
from ui_console import console



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
    folder_path: str = ""


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
    folder_renames: list[tuple[str, str]]
    status: GroupFileStatus
    ignored_empty_remote: list[str]
    ignored_empty_local: list[str]
    dirs_to_create: list[str] = field(default_factory=list)


def _format_size(size: int) -> str:
    unit = 1024.0
    units = ["B", "KB", "MB", "GB", "TB", "PB", "EB"]
    value = float(size)
    idx = 0
    while value >= unit and idx < len(units) - 1:
        value /= unit
        idx += 1
    if idx == 0:
        return f"{int(size)} B"
    return f"{value:.1f} {units[idx]}"



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


def _require_ok(action: str, result) -> None:
    rc = getattr(result, "retcode", None)
    st = getattr(result, "status", None)
    if rc != 0 or (st not in {"ok", "OK", "", None}):
        msg = getattr(result, "message", None)
        wording = getattr(result, "wording", None)
        raise RuntimeError(
            "OneBot API failed: "
            f"{action} retcode={rc} status={st} message={msg!r} wording={wording!r}"
        )


class GroupFileSyncer:
    def __init__(
        self,
        cfg: AppConfig,
        fs: FileSystemManager,
        *,
        concurrency: int = 4,
        url_workers: int | None = None,
        ignore_invalid_record: bool = False,
        reset_invalid_record: bool = False,
        ignore: IgnoreMatcher | None = None,
    ):
        self.cfg = cfg
        self.fs = fs
        self.download_workers = max(1, int(concurrency))
        self.url_workers = max(1, int(cfg.sync.url_workers if url_workers is None else url_workers))
        self._timestamp_cache: dict[str, dict[str, FileTimestampRecord]] = {}
        self.ignore = ignore
        self.ignore_invalid_record = bool(ignore_invalid_record)
        self.reset_invalid_record = bool(reset_invalid_record)
        self.invalid_url_threshold = max(1, int(cfg.sync.invalid_url_threshold))
        self._invalid_url_counts: dict[str, dict[str, int]] = {}
        self._invalid_url_dirty: set[str] = set()
        self._invalid_count_lock = asyncio.Lock()

        # invalid files (你妈的sbtx，封文件好玩吗)
        self._invalid_files: dict[str, set[str]] = defaultdict(set)
        self._invalid_log_lock = asyncio.Lock()

    def get_invalid_counts(self) -> dict[str, int]:
        return {gid: len(paths) for gid, paths in self._invalid_files.items() if paths}

    def get_invalid_total(self) -> int:
        return sum(len(v) for v in self._invalid_files.values())

    def _invalid_url_count_path(self, group_id_str: str) -> str:
        return group_invalid_url_count_file_path(group_id_str)

    def _preload_invalid_url_counts(self, group_id_str: str) -> None:
        if group_id_str in self._invalid_url_counts:
            return
        path = self._invalid_url_count_path(group_id_str)
        if not self.fs.exists(path):
            self._invalid_url_counts[group_id_str] = {}
            return
        try:
            raw = self.fs.read_bytes(path).decode("utf-8", errors="ignore")
            data = json.loads(raw) if raw.strip() else {}
            if isinstance(data, dict):
                counts: dict[str, int] = {}
                for k, v in data.items():
                    if not k:
                        continue
                    try:
                        counts[str(k)] = int(v)
                    except Exception:
                        continue
                self._invalid_url_counts[group_id_str] = counts
                return
        except Exception:
            logging.getLogger(__name__).exception("failed to load invalid url counts: group=%s", group_id_str)
        self._invalid_url_counts[group_id_str] = {}

    def _flush_invalid_url_counts(self, group_id_str: str) -> None:
        if group_id_str not in self._invalid_url_dirty:
            return
        counts = self._invalid_url_counts.get(group_id_str, {})
        path = self._invalid_url_count_path(group_id_str)
        payload = json.dumps(counts, ensure_ascii=False, indent=2)
        self.fs.write_text(path, payload)
        self._invalid_url_dirty.discard(group_id_str)

    def _clear_invalid_url_counts(self, group_id_str: str) -> None:
        self._invalid_url_counts[group_id_str] = {}
        self._invalid_url_dirty.add(group_id_str)

    def _should_skip_invalid_url(self, group_id_str: str, rel_path: str) -> bool:
        if self.ignore_invalid_record:
            return False
        self._preload_invalid_url_counts(group_id_str)
        counts = self._invalid_url_counts.get(group_id_str, {})
        return int(counts.get(rel_path, 0)) >= int(self.invalid_url_threshold)

    async def _record_invalid_url_count(self, group_id_str: str, rel_path: str) -> None:
        self._preload_invalid_url_counts(group_id_str)
        async with self._invalid_count_lock:
            counts = self._invalid_url_counts.setdefault(group_id_str, {})
            counts[rel_path] = int(counts.get(rel_path, 0)) + 1
            self._invalid_url_dirty.add(group_id_str)

    async def _reset_invalid_url_count(self, group_id_str: str, rel_path: str) -> None:
        self._preload_invalid_url_counts(group_id_str)
        async with self._invalid_count_lock:
            counts = self._invalid_url_counts.setdefault(group_id_str, {})
            if rel_path in counts:
                counts.pop(rel_path, None)
                self._invalid_url_dirty.add(group_id_str)

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

        if reason == "invalid_url":
            await self._record_invalid_url_count(group_id_str, rel_path)

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
            _require_ok("get_group_files", res)

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
                name = sanitize_component(str(d.get("folder_name", "")))
                next_path = str(Path(folder_path) / name) if folder_path else name
                seen_folder.add(did)
                folders.append(GroupFolderInfo(folder_id=did, folder_name=name, folder_path=next_path))

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

        existing_rel_files = list_group_files(self.fs, group_root)
        existing_set: set[str] = set()
        existing_size_map: dict[str, int] = {}
        empty_local_full: set[str] = set()
        ignored_empty_local: list[str] = []
        root_prefix = f"{group_root}/".replace("\\", "/")
        for rel in existing_rel_files:
            if self.ignore and self.ignore.is_ignored(rel):
                continue
            full_rel = f"{group_root}/{rel}".replace("\\", "/")
            existing_set.add(full_rel)
            try:
                sz = int(self.fs.resolve(full_rel).stat().st_size)
                existing_size_map[full_rel] = sz
                if sz == 0:
                    empty_local_full.add(full_rel)
                    ignored_empty_local.append(rel)
            except Exception:
                existing_size_map[full_rel] = 0

        local_dirs: set[str] = set()
        for d in self._list_local_dirs_rel(group_root):
            if self.ignore and self.ignore.is_ignored(d):
                continue
            local_dirs.add(d)

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

            if self._should_skip_invalid_url(group_id_str, rel):
                continue

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

        local_total_files = len(existing_set)
        local_total_size = sum(int(existing_size_map.get(p, 0)) for p in existing_set)

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
        folder_renames: list[tuple[str, str]] = []
        if mirror:
            local_rel_size_map = self._build_local_rel_size_map(existing_set, existing_size_map, empty_local_full, root_prefix)
            remote_rel_size_map = {
                group_relative_file_path(f.folder_path, f.file_name): int(f.file_size)
                for f in remote_files
                if int(f.file_size) > 0
            }
            local_all = set(local_rel_size_map.keys())
            remote_all = set(remote_rel_size_map.keys())
            extra_local_set = set(extra_local_files)
            added_remote_set = remote_all - local_all
            raw_renames = self._detect_folder_renames(
                local_all=local_all,
                local_removed=extra_local_set,
                local_size_map=local_rel_size_map,
                remote_all=remote_all,
                remote_added=added_remote_set,
                remote_size_map=remote_rel_size_map,
                min_overlap_ratio=float(self.cfg.sync.folder_rename_similarity),
            )
            folder_renames = self._filter_local_renames(group_root, raw_renames)
            if folder_renames:
                (
                    update_map,
                    extra_local_files,
                    resolved_files,
                    resolved_size,
                    extra_sizes,
                ) = self._apply_folder_renames_to_prediction(
                    group_root,
                    update_map,
                    extra_local_files,
                    local_rel_size_map,
                    remote_rel_size_map,
                    folder_renames,
                )
                files_to_update = len(update_map)
                update_size = sum(int(f.file_size) for f in update_map.values())
                files_to_delete = len(extra_local_files)
                delete_size = sum(extra_sizes.get(p, 0) for p in extra_local_files)
                matched_files += resolved_files
                matched_size += resolved_size

        remote_folder_paths = [f.folder_path for f in remote_folders if f.folder_path]
        dirs_to_create: list[str] = []
        if remote_folder_paths:
            dirs_to_create = self._missing_remote_dirs(group_root, remote_folder_paths)

        status = GroupFileStatus(
            group_id=group_id_str,
            last_update=int(time.time()),
            total_files=len(remote_files),
            total_folders=len(remote_folders),
            files=[self._file_to_dict(x) for x in remote_files],
            folders=[{"folder_id": x.folder_id, "folder_name": x.folder_name, "folder_path": x.folder_path} for x in remote_folders],
            download_path=str((self.fs.base_path / group_root).resolve()),
        )

        existing_files = local_total_files
        existing_size = local_total_size

        total_size = sum(int(f.file_size) for f in remote_files)

        return SyncPrediction(
            total_files=len(remote_files),
            total_folders=len(remote_folders),
            total_size=int(total_size),
            existing_files=int(existing_files),
            existing_folders=len(local_dirs),
            existing_size=int(existing_size),
            files_to_update=int(files_to_update),
            folders_to_create=len(dirs_to_create),
            update_size=int(update_size),
            files_to_delete=int(files_to_delete),
            folders_to_delete=0,
            delete_size=int(delete_size),
            update_file_map=update_map,
            extra_local_files=extra_local_files,
            folder_renames=folder_renames,
            dirs_to_create=dirs_to_create,
            status=status,
            ignored_empty_remote=sorted(ignored_empty_remote),
            ignored_empty_local=sorted(set(ignored_empty_local)),
        )

    async def sync_group(self, bot: OneBotWsClient, group_id_str: str, *, mirror: bool = False, plan: bool = False) -> None:
        self._preload_timestamps(group_id_str)
        self._preload_invalid_url_counts(group_id_str)
        if self.reset_invalid_record:
            self._clear_invalid_url_counts(group_id_str)
        pred = await self.generate_sync_prediction(bot, group_id_str, mirror=mirror)
        self.print_prediction(pred)

        group_root = group_root_dir(group_id_str)
        self.print_plan(pred, group_root, mirror=mirror, plan=plan)
        self._flush_invalid_url_counts(group_id_str)
        if plan:
            return

        if pred.ignored_empty_remote:
            logging.getLogger(__name__).warning("ignored %d empty file(s) (0B) for group=%s", len(pred.ignored_empty_remote), group_id_str)
            console.print(f"[yellow]WARN[/yellow]: 已忽略 {len(pred.ignored_empty_remote)} 个空文件(0B)，不会下载/替换。")

        self.fs.mkdir_all(group_root)

        if pred.folder_renames:
            self._apply_local_folder_renames(group_root, pred.folder_renames)

        if getattr(pred, "dirs_to_create", None):
            for rel in pred.dirs_to_create:
                if not rel:
                    continue
                self.fs.mkdir_all(f"{group_root}/{rel}")

        await self._download_updates(bot, group_id_str, group_root, pred.update_file_map, mirror=mirror)

        if mirror:
            await self._cleanup_extra_files(pred.status, group_root)

        self.fs.write_text(
            group_status_file_path(group_id_str),
            json.dumps(pred.status.__dict__, ensure_ascii=False, indent=2),
        )

        self._flush_timestamps(group_id_str)
        self._flush_invalid_url_counts(group_id_str)

    def print_prediction(self, p: SyncPrediction) -> None:
        table = Table(show_header=True, header_style="bold", box=None)
        table.add_column("项目")
        table.add_column("数量", justify="right")
        table.add_column("大小", justify="right")

        table.add_row(
            "来源",
            f"{p.total_folders} 文件夹 / {p.total_files} 文件",
            _format_size(p.total_size),
        )
        table.add_row(
            "已存",
            f"{p.existing_folders} 文件夹 / {p.existing_files} 文件",
            _format_size(p.existing_size),
        )
        table.add_row("需要更新", f"{p.files_to_update} 文件", _format_size(p.update_size))
        folders_to_create = p.folders_to_create or len(getattr(p, "dirs_to_create", []) or [])
        if folders_to_create:
            table.add_row("需要创建", f"{folders_to_create} 文件夹", "-")
        if p.folder_renames:
            table.add_row("识别重命名", f"{len(p.folder_renames)} 文件夹", "-")
        table.add_row(
            "需要删除",
            f"{p.files_to_delete} 文件 / {p.folders_to_delete} 文件夹",
            _format_size(p.delete_size),
        )
        if getattr(p, "ignored_empty_remote", None):
            table.add_row("忽略空文件", f"{len(p.ignored_empty_remote)}", "-")

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
        for rel in getattr(p, "dirs_to_create", []) or []:
            if rel:
                dirs_to_create.add(rel)

        extra_files = p.extra_local_files if mirror else []

        dirs_to_delete: list[str] = []
        if mirror:
            expected_files: set[str] = set()
            for f in p.status.files:
                folder_path = str(f.get("folder_path") or "")
                name = str(f.get("file_name") or "")
                expected_files.add(group_relative_file_path(folder_path, name))

            local_kept, local_ignored = list_group_files_rel(self.fs, group_root, self.ignore)
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

        summary_rows: list[tuple[str, int, str]] = []

        def add_summary(action: str, n: int, note: str) -> None:
            if n:
                summary_rows.append((action, int(n), note))

        add_summary("忽略空文件", len(getattr(p, "ignored_empty_remote", []) or []), "0B：不会下载/替换")
        add_summary("保留空文件", len(getattr(p, "ignored_empty_local", []) or []), "0B：不参与删除")
        add_summary("重命名文件夹", len(getattr(p, "folder_renames", []) or []), "先执行")
        add_summary("创建文件夹", len(dirs_to_create), "仅需要时创建")
        add_summary("替换文件", len(replace_files), "删除后下载")
        add_summary("下载缺失", len(download_files), "仅缺失项")
        add_summary("删除多余", len(extra_files), "仅镜像模式")
        add_summary("清理空文件夹", len(dirs_to_delete), "推测（以实际清理为准）")

        rows: list[tuple[str, str, str]] = []

        for it in sorted(getattr(p, "ignored_empty_remote", []) or []):
            rows.append(("IGNORE_EMPTY", it, "0B：自动忽略"))
        for it in sorted(getattr(p, "ignored_empty_local", []) or []):
            rows.append(("KEEP_EMPTY", it, "0B：保留，不参与删除"))

        for src, dst in getattr(p, "folder_renames", []) or []:
            rows.append(("RENAME_DIR", f"{src} -> {dst}", "重命名文件夹"))

        for it in sorted(dirs_to_create):
            rows.append(("MKDIR", it, "创建文件夹"))
        for it in replace_files:
            rows.append(("REPLACE", it, "删除后下载"))
        for it in download_files:
            rows.append(("DOWNLOAD", it, "下载缺失"))

        for it in extra_files:
            rows.append(("DELETE", it, "镜像：删除多余"))
        for it in dirs_to_delete:
            rows.append(("RMDIR", it, "镜像：清理空文件夹(推测)"))

        style_map: dict[str, str] = {
            "IGNORE_EMPTY": "yellow",
            "KEEP_EMPTY": "yellow",
            "RENAME_DIR": "cyan",
            "MKDIR": "cyan",
            "REPLACE": "magenta",
            "DOWNLOAD": "green",
            "DELETE": "red",
            "RMDIR": "red",
        }
        print_plan_panel(console, summary_rows, rows, style_map=style_map, title="操作计划", limit=200)

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
        url_semaphore = asyncio.Semaphore(self.url_workers)
        download_semaphore = asyncio.Semaphore(self.download_workers)
        expected_total = sum(int(f.file_size) for f in update_map.values() if int(f.file_size) > 0)
        phase1_total = len(update_map)
        progress = create_progress(console, description="同步")
        progress_task = progress.add_task(
            "同步",
            total=max(0, expected_total),
            phase1_total=phase1_total,
            phase1_done=0,
        )
        progress.start()
        total_lock = asyncio.Lock()
        phase1_lock = asyncio.Lock()
        total_expected = int(expected_total)
        phase1_done = 0

        async def reduce_total(sz: int) -> None:
            nonlocal total_expected
            if sz <= 0:
                return
            async with total_lock:
                total_expected = max(0, total_expected - int(sz))
                progress.update(progress_task, total=total_expected)

        async def advance_phase1() -> None:
            nonlocal phase1_done
            async with phase1_lock:
                phase1_done += 1
                progress.update(progress_task, phase1_done=phase1_done)
        ts_map = self._timestamp_cache.setdefault(group_id_str, {})

        try:
            async def resolve_url(full_rel: str, f: GroupFileInfo) -> tuple[str, GroupFileInfo, str] | None:
                async with url_semaphore:
                    safe_name = (f.file_name or "").strip()
                    if not safe_name:
                        logging.getLogger(__name__).warning(
                            "skip abnormal file record (empty file_name): group=%s folder_path=%r file_id=%s busid=%s",
                            group_id_str,
                            f.folder_path,
                            f.file_id,
                            f.busid,
                        )
                        await reduce_total(int(getattr(f, "file_size", 0) or 0))
                        await advance_phase1()
                        return None

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
                        await reduce_total(int(getattr(f, "file_size", 0) or 0))
                        await advance_phase1()
                        return None
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
                        await reduce_total(int(getattr(f, "file_size", 0) or 0))
                        await advance_phase1()
                        return None
                    await self._reset_invalid_url_count(group_id_str, rel_path)
                    await advance_phase1()
                    return (full_rel, f, url)

            url_tasks = [asyncio.create_task(resolve_url(k, v)) for k, v in update_map.items()]
            url_results = await asyncio.gather(*url_tasks)
            download_items = [r for r in url_results if r]
            if not download_items:
                return

            limits = httpx.Limits(
                max_connections=self.download_workers,
                max_keepalive_connections=self.download_workers,
            )
            async with httpx.AsyncClient(follow_redirects=True, timeout=httpx.Timeout(60.0), limits=limits) as client:

                async def download_one(full_rel: str, f: GroupFileInfo, url: str) -> None:
                    async with download_semaphore:
                        try:
                            safe_name = (f.file_name or "").strip()
                            rel_path = group_relative_file_path(f.folder_path, safe_name)

                            target = self.fs.base_path / Path(full_rel)
                            target.parent.mkdir(parents=True, exist_ok=True)

                            # 先下载到临时文件，然后进行替换
                            tmp_suffix = sanitize_component(str(f.file_id).strip("/") or "file")[:16]
                            tmp = target.with_name(f"{target.name}.part.{tmp_suffix}")

                            logging.getLogger(__name__).debug("download request: %s", url)
                            start_ts = time.monotonic()
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
                                    await reduce_total(int(getattr(f, "file_size", 0) or 0))
                                    return
                                resp.raise_for_status()
                                async with aiofiles.open(tmp, "wb") as af:
                                    async for chunk in resp.aiter_bytes():
                                        await af.write(chunk)
                                        progress.update(progress_task, advance=len(chunk))
                            elapsed_ms = int((time.monotonic() - start_ts) * 1000)
                            logging.getLogger(__name__).debug(
                                "download complete: url=%s elapsed_ms=%s",
                                url,
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

                tasks = [
                    asyncio.create_task(download_one(full_rel, f, url))
                    for full_rel, f, url in download_items
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            progress.stop()

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

        existing: set[str] = set()
        for rel in list_group_files(self.fs, group_root):
            if self.ignore and self.ignore.is_ignored(rel):
                continue
            full_rel = f"{group_root}/{rel}".replace("\\", "/")
            try:
                if self.fs.resolve(full_rel).stat().st_size == 0:
                    continue
            except Exception:
                pass
            existing.add(full_rel)

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

    @staticmethod
    def _folder_counts(paths: set[str]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for path in paths:
            parts = PurePosixPath(path).parts
            if len(parts) <= 1:
                continue
            for i in range(1, len(parts)):
                folder = "/".join(parts[:i])
                counts[folder] = counts.get(folder, 0) + 1
        return counts

    @staticmethod
    def _folder_sigs(paths: set[str], size_map: dict[str, int]) -> dict[str, set[tuple[str, int]]]:
        sigs: dict[str, set[tuple[str, int]]] = defaultdict(set)
        for path in paths:
            parts = PurePosixPath(path).parts
            if len(parts) <= 1:
                continue
            size = int(size_map.get(path, -1))
            for i in range(1, len(parts)):
                folder = "/".join(parts[:i])
                rel = "/".join(parts[i:])
                sigs[folder].add((rel, size))
        return sigs

    @staticmethod
    def _is_path_conflict(path: str, existing: list[str]) -> bool:
        for it in existing:
            if it == path:
                return True
            if it.startswith(path + "/") or path.startswith(it + "/"):
                return True
        return False

    def _detect_folder_renames(
        self,
        *,
        local_all: set[str],
        local_removed: set[str],
        local_size_map: dict[str, int],
        remote_all: set[str],
        remote_added: set[str],
        remote_size_map: dict[str, int],
        min_overlap_ratio: float = 0.5,
    ) -> list[tuple[str, str]]:
        local_total = self._folder_counts(local_all)
        local_removed_counts = self._folder_counts(local_removed)
        remote_total = self._folder_counts(remote_all)
        remote_added_counts = self._folder_counts(remote_added)

        old_candidates = [
            f for f, cnt in local_removed_counts.items()
            if cnt and cnt == local_total.get(f, 0)
        ]
        new_candidates = [
            f for f, cnt in remote_added_counts.items()
            if cnt and cnt == remote_total.get(f, 0)
        ]

        local_sigs = self._folder_sigs(local_removed, local_size_map)
        remote_sigs = self._folder_sigs(remote_added, remote_size_map)

        pairs: list[tuple[float, int, int, int, str, str]] = []
        for old in old_candidates:
            s1 = local_sigs.get(old)
            if not s1:
                continue
            for new in new_candidates:
                if old == new:
                    continue
                s2 = remote_sigs.get(new)
                if not s2:
                    continue
                overlap = len(s1 & s2)
                if overlap == 0:
                    continue
                denom = max(len(s1), len(s2))
                ratio = overlap / denom if denom else 0.0
                if ratio >= min_overlap_ratio:
                    pairs.append((ratio, overlap, len(s1), len(s2), old, new))

        pairs.sort(reverse=True)
        selected: list[tuple[str, str]] = []
        selected_old: list[str] = []
        selected_new: list[str] = []
        for _ratio, _overlap, _a, _b, old, new in pairs:
            if self._is_path_conflict(old, selected_old):
                continue
            if self._is_path_conflict(new, selected_new):
                continue
            selected.append((old, new))
            selected_old.append(old)
            selected_new.append(new)
        return selected

    @staticmethod
    def _build_local_rel_size_map(
        existing_set: set[str],
        existing_size_map: dict[str, int],
        empty_local_full: set[str],
        root_prefix: str,
    ) -> dict[str, int]:
        rel_size: dict[str, int] = {}
        for full_rel in existing_set:
            if full_rel in empty_local_full:
                continue
            rel = str(full_rel).replace("\\", "/")
            if rel.startswith(root_prefix):
                rel = rel[len(root_prefix) :]
            if not rel:
                continue
            rel_size[rel] = int(existing_size_map.get(full_rel, -1))
        return rel_size

    def _filter_local_renames(self, group_root: str, renames: list[tuple[str, str]]) -> list[tuple[str, str]]:
        valid: list[tuple[str, str]] = []
        for src, dst in renames:
            if not src or src == dst:
                continue
            src_path = self.fs.resolve(f"{group_root}/{src}")
            dst_path = self.fs.resolve(f"{group_root}/{dst}")
            if not src_path.exists():
                continue
            if dst_path.exists():
                continue
            valid.append((src, dst))
        return valid

    def _apply_folder_renames_to_prediction(
        self,
        group_root: str,
        update_map: dict[str, GroupFileInfo],
        extra_local_files: list[str],
        local_size_map: dict[str, int],
        remote_size_map: dict[str, int],
        renames: list[tuple[str, str]],
    ) -> tuple[dict[str, GroupFileInfo], list[str], int, int, dict[str, int]]:
        if not renames:
            return update_map, extra_local_files, 0, 0, {p: int(local_size_map.get(p, 0)) for p in extra_local_files}

        ordered = sorted(renames, key=lambda x: len(x[0].split("/")), reverse=True)
        extra_sizes: dict[str, int] = {}
        resolved_files = 0
        resolved_size = 0
        new_extra: list[str] = []

        for path in extra_local_files:
            new_path = path
            for old, new in ordered:
                prefix = f"{old}/"
                if path.startswith(prefix):
                    rel = path[len(prefix) :]
                    new_path = f"{new}/{rel}" if new else rel
                    break
            local_size = int(local_size_map.get(path, -1))
            remote_size = remote_size_map.get(new_path)
            if new_path != path and remote_size is not None and int(remote_size) == local_size:
                full_rel = f"{group_root}/{new_path}".replace("\\", "/")
                update_map.pop(full_rel, None)
                resolved_files += 1
                resolved_size += int(remote_size)
                continue
            new_extra.append(new_path)
            extra_sizes[new_path] = local_size if local_size >= 0 else 0

        new_extra = sorted(set(new_extra))
        return update_map, new_extra, resolved_files, resolved_size, extra_sizes

    def _apply_local_folder_renames(self, group_root: str, renames: list[tuple[str, str]]) -> None:
        cleanup_roots: list[str] = []
        for src, dst in renames:
            if not src or src == dst:
                continue
            src_path = self.fs.resolve(f"{group_root}/{src}")
            dst_path = self.fs.resolve(f"{group_root}/{dst}")
            if not src_path.exists():
                continue
            if dst_path.exists():
                logging.getLogger(__name__).warning(
                    "skip folder rename (destination exists): %s -> %s", src, dst
                )
                continue
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                src_path.rename(dst_path)
                cleanup_roots.append(src)
            except Exception:
                logging.getLogger(__name__).exception("rename folder failed: %s -> %s", src, dst)
        if cleanup_roots:
            self._remove_empty_dirs_for_paths(group_root, cleanup_roots)

    def _remove_empty_dirs_for_paths(self, group_root: str, rel_paths: list[str]) -> None:
        root = self.fs.base_path / Path(group_root)
        if not root.exists() or not root.is_dir():
            return
        for rel in rel_paths:
            if not rel:
                continue
            cur = root / Path(rel)
            # 逐级向上清理空目录
            while True:
                if not cur.exists() or not cur.is_dir():
                    break
                rel_str = cur.relative_to(root).as_posix()
                if rel_str and self.ignore and self.ignore.is_ignored(rel_str):
                    break
                try:
                    if any(cur.iterdir()):
                        break
                    cur.rmdir()
                except OSError:
                    break
                if cur == root:
                    break
                cur = cur.parent

    def _missing_remote_dirs(self, group_root: str, remote_folder_paths: list[str]) -> list[str]:
        if not remote_folder_paths:
            return []
        local_dirs = set(self._list_local_dirs_rel(group_root))
        out: list[str] = []
        for rel in sorted(set(remote_folder_paths)):
            if not rel:
                continue
            if self.ignore and self.ignore.is_ignored(rel):
                continue
            if rel not in local_dirs:
                out.append(rel)
        return out

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

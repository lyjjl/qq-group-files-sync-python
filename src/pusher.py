from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from filesystem import FileSystemManager
from group_paths import group_root_dir, sanitize_component
from onebot import OneBotWsClient, parse_group_numeric_id
from ignore_rules import IgnoreMatcher

console = Console()


@dataclass
class RemoteFolder:
    folder_id: str
    folder_name: str
    folder_path: str


@dataclass
class RemoteFile:
    file_id: str
    file_name: str
    folder_path: str
    file_size: int


def _require_ok(action: str, result) -> None:
    """当 OneBot API 返回失败状态时在 message 或 wording 字段中可能有详细的说明
    将这些信息呈现出来，以便用户确认具体原因。
    """

    rc = getattr(result, "retcode", None)
    st = getattr(result, "status", None)
    if rc != 0 or (st not in {"ok", "OK", "", None}):
        msg = getattr(result, "message", None)
        wording = getattr(result, "wording", None)
        raise RuntimeError(
            "OneBot API failed: "
            f"{action} retcode={rc} status={st} message={msg!r} wording={wording!r}"
        )


def _okish(result) -> bool:
    """如果从response看起来执行成功，则返回 True

    有些API返回空data，需要注意"""

    rc = getattr(result, "retcode", 0)
    if rc not in (0, None):
        return False
    st = getattr(result, "status", "ok")
    if st is None:
        return True
    st = str(st).strip().lower()
    return (st == "" or st == "ok")


class GroupFilePusher:
    """上传本地文件到群文件"""

    def __init__(self, fs: FileSystemManager, *, concurrency: int = 2, ignore: IgnoreMatcher | None = None):
        self.fs = fs
        self.concurrency = max(1, int(concurrency))
        self.ignore = ignore


    async def push_group(self, bot: OneBotWsClient, group_id_str: str, *, mirror: bool = False, plan: bool = False) -> None:
        """将本地文件推送到群文件。

        - mirror=False：上传远程缺失的本地文件
        - mirror=True：远程应在名称和大小上与本地保持一致
            - 上传缺失的文件
            - 替换差异文件（删除远程文件后上传本地文件）
            - 删除多余的远程文件

        文件一致性通过相对路径和文件大小判定
        """

        if mirror:
            await self.push_group_mirror(bot, group_id_str, plan=plan)
        else:
            await self.push_group_missing_only(bot, group_id_str, plan=plan)

    async def push_group_mirror(self, bot: OneBotWsClient, group_id_str: str, *, plan: bool = False) -> None:

        group_id_num = parse_group_numeric_id(group_id_str)
        group_root = group_root_dir(group_id_str)

        remote_files_all, remote_folders = await self._fetch_remote_tree(bot, group_id_num)

        # ignore
        if self.ignore:
            remote_files = {k: v for k, v in remote_files_all.items() if not self.ignore.is_ignored(k)}
        else:
            remote_files = remote_files_all

        remote_keys = set(remote_files.keys())

        local_rel_files_all = self._list_local_files_under_group(group_root)
        local_rel_files = [p for p in local_rel_files_all if not (self.ignore and self.ignore.is_ignored(p))]

        local_non_empty, local_empty, local_size = self._split_local_files_by_size(group_root, local_rel_files)
        # 注意：上传/替换时会忽略空文件（因为群文件没法上传空文件），但在 --mirror 模式下，它们也会使远程路径不被删除，有点奇怪哈
        local_keys_non_empty = set(local_non_empty)
        local_keys_protect = local_keys_non_empty | set(local_empty)

        to_upload = sorted([k for k in local_keys_non_empty if k not in remote_keys])
        to_replace = sorted([k for k in (local_keys_non_empty & remote_keys) if int(local_size.get(k, -1)) != int(remote_files[k].file_size)])
        to_delete = sorted([k for k in (remote_keys - local_keys_protect)])

        overview = Table(show_header=False, box=None)
        overview.add_column("项")
        overview.add_column("值", justify="right")
        overview.add_row("远端文件", str(len(remote_keys)))
        overview.add_row("本地文件", f"{len(local_keys_non_empty)}（空文件忽略 {len(local_empty)}）")
        overview.add_row("待上传", str(len(to_upload)))
        overview.add_row("待替换", str(len(to_replace)))
        overview.add_row("待删除远端多余", str(len(to_delete)))
        console.print(Panel(overview, title=f"镜像推送对比 {group_id_str}", expand=False))

        if plan:
            self._print_plan(remote_folders, to_upload=to_upload, to_replace=to_replace, to_delete=to_delete, ignored_empty=local_empty)
            return

        if local_empty:
            logging.getLogger(__name__).warning("ignored %d empty local file(s) (0B) for group=%s", len(local_empty), group_id_str)
            console.print(f"[yellow]WARN[/yellow]: 已忽略 {len(local_empty)} 个本地空文件(0B)，不会上传/替换，也不会触发远端删除。")

        # 确保顶级文件夹存在
        folder_path_to_id = {f.folder_path: f.folder_id for f in remote_folders.values()}
        folder_path_to_id[""] = ""

        semaphore = asyncio.Semaphore(self.concurrency)
        failures: list[str] = []

        async def delete_remote(key: str) -> None:
            rf = remote_files.get(key)
            if not rf:
                return
            res = await bot.call_api("delete_group_file", {"group_id": group_id_num, "file_id": rf.file_id})
            if not _okish(res):
                raise RuntimeError(
                    f"delete_group_file failed: retcode={getattr(res,'retcode',None)} status={getattr(res,'status',None)}"
                )

        async def upload_local(key: str) -> None:
            folder_path, file_name = self._split_key(key)
            target_folder_id = await self._ensure_folder(bot, group_id_num, folder_path_to_id, folder_path)

            abs_path = self.fs.resolve(f"{group_root}/{key}").resolve()
            if not abs_path.exists() or not abs_path.is_file():
                raise FileNotFoundError(f"local file not found for upload: {abs_path}")
            try:
                if abs_path.stat().st_size == 0:
                    logging.getLogger(__name__).warning("skip upload empty file (0B): group=%s path=%s", group_id_str, abs_path)
                    return
            except Exception:
                pass
            logging.getLogger(__name__).debug(
                "upload_group_file: group=%s file=%s name=%s", group_id_str, abs_path, file_name
            )
            res = await bot.call_api(
                "upload_group_file",
                {"group_id": group_id_num, "file": str(abs_path), "name": file_name},
            )
            _require_ok("upload_group_file", res)
            data = res.data or {}
            file_id = str(data.get("file_id") or "").strip()
            if not file_id:
                raise RuntimeError("upload_group_file returned empty file_id")

            if target_folder_id:
                await self._move_uploaded_file(bot, group_id_num, file_id, target_folder_id)

        async def handle_replace_or_upload(key: str, *, replace: bool) -> None:
            async with semaphore:
                try:
                    if replace:
                        await delete_remote(key)
                    await upload_local(key)
                except Exception:
                    logging.getLogger(__name__).exception("mirror push failed: group=%s key=%s", group_id_str, key)
                    failures.append(key)

        async def handle_delete_only(key: str) -> None:
            async with semaphore:
                try:
                    await delete_remote(key)
                except Exception:
                    logging.getLogger(__name__).exception("delete remote failed: group=%s key=%s", group_id_str, key)
                    failures.append(key)

        tasks = []
        tasks += [asyncio.create_task(handle_delete_only(k)) for k in to_delete]
        tasks += [asyncio.create_task(handle_replace_or_upload(k, replace=True)) for k in to_replace]
        tasks += [asyncio.create_task(handle_replace_or_upload(k, replace=False)) for k in to_upload]

        if tasks:
            await asyncio.gather(*tasks)

        if failures:
            console.print(
                Panel(
                    f"失败 {len(failures)} 个。\n详情请查看 error.log。",
                    title="镜像推送结果",
                    expand=False,
                )
            )
        else:
            console.print(Panel("全部成功", title="镜像推送结果", expand=False))

    async def push_group_missing_only(self, bot: OneBotWsClient, group_id_str: str, *, plan: bool = False) -> None:
        """将远端缺失的本地文件上传

        1) 获取远程文件夹/文件列表
        2) 与本地 QQ-Group_<id> 目录下的内容进行对比
        3) 对于远程缺失的项目，上传本地文件并在需要时将其移动到目标文件夹中"""

        group_id_num = parse_group_numeric_id(group_id_str)
        group_root = group_root_dir(group_id_str)

        remote_files, remote_folders = await self._fetch_remote_tree(bot, group_id_num)
        remote_file_paths = {self._remote_path_key(f.folder_path, f.file_name) for f in remote_files.values()}

        local_rel_files_all = self._list_local_files_under_group(group_root)
        local_rel_files = [p for p in local_rel_files_all if not (self.ignore and self.ignore.is_ignored(p))]

        local_non_empty, local_empty_all, _size_map = self._split_local_files_by_size(group_root, local_rel_files)
        # only consider non-empty files for upload; empty files are ignored
        missing_local = [p for p in local_non_empty if p not in remote_file_paths]
        ignored_empty = [p for p in local_empty_all if p not in remote_file_paths]

        overview = Table(show_header=False, box=None)
        overview.add_column("项")
        overview.add_column("值", justify="right")
        overview.add_row("远端文件", str(len(remote_file_paths)))
        overview.add_row("本地文件", f"{len(local_non_empty)}（空文件忽略 {len(ignored_empty)}）")
        overview.add_row("待上传", str(len(missing_local)))
        console.print(Panel(overview, title=f"推送对比 {group_id_str}", expand=False))
        if plan:
            self._print_plan(remote_folders, to_upload=sorted(missing_local), to_replace=[], to_delete=[], ignored_empty=sorted(ignored_empty))
            return

        if ignored_empty:
            logging.getLogger(__name__).warning("ignored %d empty local file(s) (0B) for group=%s", len(ignored_empty), group_id_str)
            console.print(f"[yellow]WARN[/yellow]: 已忽略 {len(ignored_empty)} 个本地空文件(0B)，不会上传。")

        if not missing_local:
            return

        folder_path_to_id = {f.folder_path: f.folder_id for f in remote_folders.values()}
        folder_path_to_id[""] = ""  # root

        semaphore = asyncio.Semaphore(self.concurrency)
        failures: list[str] = []

        async def upload_one(rel_key: str) -> None:
            async with semaphore:
                try:
                    folder_path, file_name = self._split_key(rel_key)
                    target_folder_id = await self._ensure_folder(bot, group_id_num, folder_path_to_id, folder_path)

                    abs_path = self.fs.resolve(f"{group_root}/{rel_key}").resolve()
                    if not abs_path.exists() or not abs_path.is_file():
                        raise FileNotFoundError(f"local file not found for upload: {abs_path}")
                    try:
                        if abs_path.stat().st_size == 0:
                            logging.getLogger(__name__).warning("skip upload empty file (0B): group=%s path=%s", group_id_str, abs_path)
                            return
                    except Exception:
                        pass
                    logging.getLogger(__name__).debug(
                        "upload_group_file: group=%s file=%s name=%s", group_id_str, abs_path, file_name
                    )
                    res = await bot.call_api(
                        "upload_group_file",
                        {
                            "group_id": group_id_num,
                            "file": str(abs_path),
                            "name": file_name,
                        },
                    )
                    _require_ok("upload_group_file", res)
                    data = res.data or {}
                    file_id = str(data.get("file_id") or "").strip()
                    if not file_id:
                        raise RuntimeError("upload_group_file returned empty file_id")

                    # 移动到对应的文件夹
                    if target_folder_id:
                        await self._move_uploaded_file(bot, group_id_num, file_id, target_folder_id)

                except Exception:
                    logging.getLogger(__name__).exception("push failed: group=%s file=%s", group_id_str, rel_key)
                    failures.append(rel_key)

        tasks = [asyncio.create_task(upload_one(k)) for k in missing_local]
        await asyncio.gather(*tasks)

        if failures:
            console.print(
                Panel(
                    f"失败 {len(failures)} 个。\n详情请查看 error.log。",
                    title="上传结果",
                    expand=False,
                )
            )
        else:
            console.print(Panel("全部成功", title="上传结果", expand=False))

    async def _fetch_remote_tree(self, bot: OneBotWsClient, group_id_num: int) -> tuple[dict[str, RemoteFile], dict[str, RemoteFolder]]:
        """返回 files_by_key, folders_by_path

        - files_by_key: 类似于 "a/b/file.ext"
        - folders_by_path: 类似于 "a/b"
        """

        files: dict[str, RemoteFile] = {}
        folders: dict[str, RemoteFolder] = {}
        seen_folder_id: set[str] = set()

        async def fetch(folder_id: str, folder_path: str) -> None:
            if folder_id:
                res = await bot.call_api("get_group_files_by_folder", {"group_id": group_id_num, "folder_id": folder_id})
            else:
                res = await bot.call_api("get_group_root_files", {"group_id": group_id_num})
            _require_ok("get_group_files", res)
            data = res.data or {}

            for f in data.get("files", []) or []:
                name = sanitize_component(str(f.get("file_name", "")))
                fid = str(f.get("file_id", "")).strip()
                if not name or not fid:
                    continue
                key = self._remote_path_key(folder_path, name)
                files[key] = RemoteFile(file_id=fid, file_name=name, folder_path=folder_path, file_size=int(f.get('file_size', 0) or 0))

            for d in data.get("folders", []) or []:
                did = str(d.get("folder_id", "")).strip()
                if not did or did in seen_folder_id:
                    continue
                seen_folder_id.add(did)
                dname = sanitize_component(str(d.get("folder_name", "")))
                next_path = str(PurePosixPath(folder_path) / dname) if folder_path else dname
                folders[next_path] = RemoteFolder(folder_id=did, folder_name=dname, folder_path=next_path)

            # recurse
            for d in data.get("folders", []) or []:
                did = str(d.get("folder_id", "")).strip()
                dname = sanitize_component(str(d.get("folder_name", "")))
                if not did or not dname:
                    continue
                next_path = str(PurePosixPath(folder_path) / dname) if folder_path else dname
                try:
                    await fetch(did, next_path)
                except Exception:
                    logging.getLogger(__name__).exception("failed to fetch folder: %s", next_path)

        await fetch("", "")
        return files, folders

    def _split_local_files_by_size(self, group_root: str, rel_files: list[str]) -> tuple[list[str], list[str], dict[str, int]]:
        """将群组目录下的本地文件转化为 non_empty, empty, size_map

        - empty: 大小为 0 的文件
        - size_map: 仅包含非空文件；若 stat（获取文件状态）失败，则大小记录为 -1
        """

        non_empty: list[str] = []
        empty: list[str] = []
        size_map: dict[str, int] = {}
        for k in rel_files:
            try:
                abs_path = self.fs.resolve(f"{group_root}/{k}").resolve()
                sz = int(abs_path.stat().st_size)
                if sz == 0:
                    empty.append(k)
                else:
                    non_empty.append(k)
                    size_map[k] = sz
            except Exception:
                # keep it in non-empty so failures are surfaced during upload
                non_empty.append(k)
                size_map[k] = -1
        non_empty.sort()
        empty.sort()
        return non_empty, empty, size_map

    def _list_local_files_under_group(self, group_root: str) -> list[str]:
        """返回相对于根目录的本地文件路径，例如 "a/b/file.ext" """

        full = self.fs.list_files_under(group_root)
        out: list[str] = []
        root_prefix = f"{PurePosixPath(group_root).as_posix().rstrip('/')}/"
        for p in full:
            pp = PurePosixPath(p)
            ps = pp.as_posix()
            if not ps.startswith(root_prefix):
                continue
            rel = ps[len(root_prefix) :]
            if rel:
                out.append(rel)
        out.sort()
        return out

    def _print_plan(
        self,
        remote_folders: dict[str, RemoteFolder],
        *,
        to_upload: list[str],
        to_replace: list[str],
        to_delete: list[str],
        ignored_empty: list[str] | None = None,
    ) -> None:
        """当存在 --plan 参数时，打印预计执行的操作。"""

        existing_folder_paths = set(remote_folders.keys())
        needed_folders: set[str] = set()
        for k in (to_upload + to_replace):
            folder_path, _name = self._split_key(k)
            if folder_path and folder_path not in existing_folder_paths:
                needed_folders.add(folder_path)

        folders_to_create = sorted([p for p in needed_folders if "/" not in p])
        blocked_nested = sorted([p for p in needed_folders if "/" in p])

        summary = Table(show_header=True, header_style="bold", box=None)
        summary.add_column("动作")
        summary.add_column("数量", justify="right")
        summary.add_column("说明")

        def add_row(action: str, n: int, note: str) -> None:
            if n:
                summary.add_row(action, str(n), note)

        add_row("忽略本地空文件", len(ignored_empty or []), "0B：自动忽略")
        add_row("创建远端一级文件夹", len(folders_to_create), "仅支持一级")
        add_row("缺失嵌套文件夹", len(blocked_nested), "需手动创建")
        add_row("删除远端多余文件", len(to_delete), "仅镜像模式")
        add_row("替换远端不同文件", len(to_replace), "删除后上传")
        add_row("上传远端缺失文件", len(to_upload), "仅缺失项")

        console.print(Panel(summary, title="操作计划 (--plan)", expand=False))

        rows: list[tuple[str, str, str]] = []

        for it in sorted(ignored_empty or []):
            rows.append(("IGNORE_EMPTY", it, "0B：自动忽略"))

        for it in folders_to_create:
            rows.append(("MKDIR_REMOTE", it, "创建远端一级文件夹"))

        for it in blocked_nested:
            rows.append(("BLOCKED", it, "缺失嵌套文件夹：需手动创建"))

        for it in to_delete:
            rows.append(("DELETE_REMOTE", it, "镜像：删除远端多余"))

        for it in to_replace:
            rows.append(("REPLACE_REMOTE", it, "删除后上传"))

        for it in to_upload:
            rows.append(("UPLOAD", it, "上传缺失"))

        if rows:
            self._print_plan_table(rows, title="计划明细", limit=200)

        console.print("[dim]注：上传到子文件夹会先上传到根目录，再 move 进目标文件夹；不会删除远端文件夹（即使为空）。[/dim]")

    def _print_plan_table(self, rows: list[tuple[str, str, str]], *, title: str, limit: int = 200) -> None:
        style_map: dict[str, str] = {
            "IGNORE_EMPTY": "yellow",
            "MKDIR_REMOTE": "cyan",
            "BLOCKED": "yellow",
            "DELETE_REMOTE": "red",
            "REPLACE_REMOTE": "magenta",
            "UPLOAD": "green",
        }

        table = Table(show_header=True, header_style="bold", box=None)
        table.add_column("动作", no_wrap=True)
        table.add_column("路径")
        table.add_column("说明")

        show = rows[:limit]
        for action, path, note in show:
            st = style_map.get(action, "")
            table.add_row(Text(action, style=st), path, note)

        console.print(Panel(table, title=title, expand=False))
        if len(rows) > limit:
            console.print(f"[dim]...（已截断，剩余 {len(rows) - limit} 项）[/dim]")

    def _print_list(self, items: list[str], *, prefix: str) -> None:
        limit = 200
        show = items[:limit]
        for it in show:
            console.print(prefix + it)
        if len(items) > limit:
            console.print(prefix + f"...（已截断，剩余 {len(items) - limit} 项）")

    @staticmethod
    def _remote_path_key(folder_path: str, file_name: str) -> str:
        if not folder_path:
            return sanitize_component(file_name)
        return str(PurePosixPath(folder_path) / sanitize_component(file_name))

    @staticmethod
    def _split_key(key: str) -> tuple[str, str]:
        p = PurePosixPath(key)
        folder = p.parent.as_posix()
        if folder == ".":
            folder = ""
        return folder, p.name

    async def _ensure_folder(self, bot: OneBotWsClient, group_id_num: int, folder_path_to_id: dict[str, str], folder_path: str) -> str:

        folder_path = (folder_path or "").strip("/")
        if not folder_path:
            return ""

        if folder_path in folder_path_to_id:
            return folder_path_to_id[folder_path]

        if "/" in folder_path:
            raise RuntimeError(
                f"remote folder missing: '{folder_path}'. current API cannot create nested folders safely; please create it in group file UI first."
            )

        res = await bot.call_api("create_group_file_folder", {"group_id": group_id_num, "name": folder_path})
        _require_ok("create_group_file_folder", res)
        data = res.data or {}
        folder_id = str(data.get("folder_id") or "").strip()
        if not folder_id:
            raise RuntimeError("create_group_file_folder returned empty folder_id")

        folder_path_to_id[folder_path] = folder_id
        return folder_id

    async def _move_uploaded_file(self, bot: OneBotWsClient, group_id_num: int, file_id: str, target_folder_id: str) -> None:
        for parent in ("", "/"):
            res = await bot.call_api(
                "move_group_file",
                {
                    "group_id": group_id_num,
                    "file_id": file_id,
                    "parent_directory": parent,
                    "target_directory": target_folder_id,
                },
            )
            if getattr(res, "retcode", -1) == 0:
                return
        _require_ok("move_group_file", res)

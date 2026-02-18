from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
import time
import unicodedata
import warnings
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, time as dt_time
from pathlib import PurePosixPath
from typing import Any, Callable

from config import AppConfig
from filesystem import FileSystemManager
from group_paths import group_relative_file_path, sanitize_component
from onebot_common import require_ok as _require_ok
from onebot import OneBotWsClient, parse_group_numeric_id

_SHORT_ID_ALPHABET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"
_TOKEN_RE = re.compile(r"[0-9a-z]+|[\u4e00-\u9fff]+")

try:
    warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"jieba(\.|$)")
    import jieba as _jieba  # type: ignore
except Exception:  # pragma: no cover
    _jieba = None

try:
    from rapidfuzz.distance import DamerauLevenshtein as _rf_damerau_levenshtein  # type: ignore
    from rapidfuzz.distance import Levenshtein as _rf_levenshtein  # type: ignore
except Exception:  # pragma: no cover
    _rf_damerau_levenshtein = None
    _rf_levenshtein = None

try:
    from rapidfuzz import fuzz as _rf_fuzz  # type: ignore
except Exception:  # pragma: no cover
    _rf_fuzz = None


def format_group_id(group_id_num: int) -> str:
    return f"QQ-Group:{group_id_num}"


def normalize_group_id_input(group_id: str | int) -> str:
    return format_group_id(parse_group_numeric_id(group_id))


@dataclass
class IndexedRemoteFile:
    group_id: str
    group_id_num: int
    api_group_id: int
    file_id: str
    file_name: str
    folder_id: str
    folder_path: str
    dead_time: int
    download_times: int
    uploader_id: int
    uploader_name: str
    upload_time: int
    modify_time: int
    file_size: int
    busid: int
    alias: str = ""


@dataclass
class IndexUpdateStats:
    total_remote: int = 0
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    deleted: int = 0
    failed: bool = False
    error: str = ""


@dataclass
class SearchQuery:
    raw: str
    search_by: str
    keys: list[str]
    group_id: str | None
    uploader_id: int | None


@dataclass
class SearchResult:
    row_id: int
    short_id: str
    group_id: str
    group_name: str
    file_id: str
    folder_id: str
    folder_path: str
    relative_path: str
    file_name: str
    uploader_name: str
    uploader_id: int
    upload_time: int
    dead_time: int
    modify_time: int
    download_times: int
    file_size: int
    md5: str
    alias: str
    match_tag: str  # Exact | Fuzzy
    matched_count: int


@dataclass
class IndexedFileRecord:
    row_id: int
    short_id: str
    group_id: str
    group_name: str
    group_id_num: int
    api_group_id: int
    file_id: str
    file_name: str
    folder_id: str
    folder_path: str
    uploader_name: str
    uploader_id: int
    upload_time: int
    dead_time: int
    modify_time: int
    download_times: int
    file_size: int
    busid: int
    md5: str
    alias: str


@dataclass
class IndexInfo:
    total_groups: int = 0
    total_records: int = 0
    total_file_size: int = 0
    avg_file_size: int = 0
    earliest_upload_time: int = 0
    latest_upload_time: int = 0
    earliest_modify_time: int = 0
    latest_modify_time: int = 0


@dataclass
class RemoteGroupInfo:
    group_id: str
    group_id_num: int
    group_name: str = ""


@dataclass
class IndexedGroupInfo:
    group_id: str
    group_id_num: int
    group_name: str
    file_count: int
    total_file_size: int
    updated_at: int


class GroupFileIndexer:
    def __init__(self, cfg: AppConfig, fs: FileSystemManager):
        self.cfg = cfg
        self.fs = fs
        raw_db = (cfg.search.index_db or ".index/group_files.db").strip() or ".index/group_files.db"
        db_path = (fs.base_path / raw_db).resolve()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.min_results = max(1, int(cfg.search.min_results))
        self.fuzzy_edit_distance = max(0, min(2, int(getattr(cfg.search, "fuzzy_edit_distance", 2))))
        self.fuzzy_max_terms_per_token = max(4, int(getattr(cfg.search, "fuzzy_max_terms_per_token", 24)))
        self._jieba = _jieba
        if self._jieba is not None:
            try:
                self._jieba.setLogLevel(logging.WARNING)
            except Exception:
                pass
        self._fts_available = True
        self._fts_vocab_available = False
        self._term_expand_cache: dict[tuple[str, int], tuple[str, ...]] = {}
        self._opencc_t2s = None
        self._opencc_s2t = None
        try:
            from opencc import OpenCC  # type: ignore

            self._opencc_t2s = OpenCC("t2s")
            self._opencc_s2t = OpenCC("s2t")
        except Exception:
            pass
        self._conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        self._conn.row_factory = sqlite3.Row
        self._prepare_db()

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def _prepare_db(self) -> None:
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA temp_store=MEMORY")
        self._conn.execute("PRAGMA busy_timeout=10000")

        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS group_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id TEXT NOT NULL,
                group_id_num INTEGER NOT NULL,
                api_group_id INTEGER NOT NULL DEFAULT 0,
                file_id TEXT NOT NULL,
                folder_id TEXT NOT NULL DEFAULT '',
                folder_path TEXT NOT NULL DEFAULT '',
                file_name TEXT NOT NULL,
                uploader TEXT NOT NULL DEFAULT '',
                uploader_id INTEGER NOT NULL DEFAULT 0,
                uploader_name TEXT NOT NULL DEFAULT '',
                upload_time INTEGER NOT NULL DEFAULT 0,
                dead_time INTEGER NOT NULL DEFAULT 0,
                modify_time INTEGER NOT NULL DEFAULT 0,
                download_times INTEGER NOT NULL DEFAULT 0,
                file_size INTEGER NOT NULL DEFAULT 0,
                busid INTEGER NOT NULL DEFAULT 0,
                md5 TEXT NOT NULL,
                alias TEXT NOT NULL DEFAULT '',
                updated_at INTEGER NOT NULL DEFAULT 0,
                UNIQUE(group_id, file_id)
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS group_index_groups (
                group_id TEXT PRIMARY KEY,
                group_id_num INTEGER NOT NULL DEFAULT 0,
                group_name TEXT NOT NULL DEFAULT '',
                file_count INTEGER NOT NULL DEFAULT 0,
                total_file_size INTEGER NOT NULL DEFAULT 0,
                updated_at INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        self._ensure_group_files_schema()
        self._ensure_group_index_schema()
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_group_files_gid ON group_files(group_id)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_group_files_uid ON group_files(uploader_id)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_group_files_upload ON group_files(upload_time)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_group_files_modify ON group_files(modify_time)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_group_index_gid_num ON group_index_groups(group_id_num)")
        try:
            self._conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS group_files_fts
                USING fts5(file_name, uploader, alias, tokenize='unicode61')
                """
            )
            try:
                self._conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS group_files_vocab
                    USING fts5vocab(group_files_fts, 'row')
                    """
                )
                self._fts_vocab_available = True
            except Exception:
                self._fts_vocab_available = False
        except Exception:
            self._fts_available = False
            self._fts_vocab_available = False
            logging.getLogger(__name__).warning("FTS5 unavailable, fallback to regex-only scan search.")
        self._backfill_group_index_rows_from_files()
        self._conn.commit()

    def _ensure_group_files_schema(self) -> None:
        self._ensure_column("group_files", "api_group_id", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("group_files", "uploader_id", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("group_files", "uploader_name", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column("group_files", "dead_time", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("group_files", "download_times", "INTEGER NOT NULL DEFAULT 0")

    def _ensure_group_index_schema(self) -> None:
        self._ensure_column("group_index_groups", "group_id_num", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("group_index_groups", "group_name", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column("group_index_groups", "file_count", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("group_index_groups", "total_file_size", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("group_index_groups", "updated_at", "INTEGER NOT NULL DEFAULT 0")

    def _backfill_group_index_rows_from_files(self) -> None:
        rows = self._conn.execute(
            """
            SELECT
                group_id,
                COALESCE(MAX(group_id_num), 0) AS group_id_num,
                COUNT(*) AS file_count,
                COALESCE(SUM(file_size), 0) AS total_file_size,
                COALESCE(MAX(updated_at), 0) AS updated_at
            FROM group_files
            GROUP BY group_id
            """
        ).fetchall()
        for r in rows:
            gid = str(r["group_id"] or "").strip()
            if not gid:
                continue
            self._conn.execute(
                """
                INSERT INTO group_index_groups(group_id, group_id_num, group_name, file_count, total_file_size, updated_at)
                VALUES (?, ?, '', ?, ?, ?)
                ON CONFLICT(group_id) DO UPDATE SET
                    group_id_num = excluded.group_id_num,
                    file_count = excluded.file_count,
                    total_file_size = excluded.total_file_size,
                    updated_at = excluded.updated_at
                """,
                (
                    gid,
                    int(r["group_id_num"] or 0),
                    int(r["file_count"] or 0),
                    int(r["total_file_size"] or 0),
                    int(r["updated_at"] or 0),
                ),
            )

    def _ensure_column(self, table: str, column: str, column_def: str) -> None:
        rows = self._conn.execute(f"PRAGMA table_info({table})").fetchall()
        for r in rows:
            if str(r["name"] or "").strip().lower() == column.lower():
                return
        self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_def}")

    def count_records(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS c FROM group_files").fetchone()
        return int(row["c"] if row else 0)

    @staticmethod
    def short_id_from_row_id(row_id: int) -> str:
        n = int(row_id)
        if n <= 0:
            raise ValueError("row_id must be positive")
        base = len(_SHORT_ID_ALPHABET)
        out: list[str] = []
        while n > 0:
            n, rem = divmod(n, base)
            out.append(_SHORT_ID_ALPHABET[rem])
        return "R" + "".join(reversed(out))

    @staticmethod
    def row_id_from_short_id(short_id: str) -> int:
        s = str(short_id or "").strip().upper()
        if s.startswith("R"):
            s = s[1:]
        if not s:
            raise ValueError("empty short id")
        base = len(_SHORT_ID_ALPHABET)
        n = 0
        for ch in s:
            idx = _SHORT_ID_ALPHABET.find(ch)
            if idx < 0:
                raise ValueError(f"invalid short id: {short_id}")
            n = n * base + idx
        if n <= 0:
            raise ValueError(f"invalid short id: {short_id}")
        return n

    def get_index_info(self) -> IndexInfo:
        row = self._conn.execute(
            """
            SELECT
                COUNT(*) AS total_records,
                (
                    SELECT COUNT(*)
                    FROM (
                        SELECT group_id FROM group_index_groups
                        UNION
                        SELECT DISTINCT group_id FROM group_files
                    )
                ) AS total_groups,
                COALESCE(SUM(file_size), 0) AS total_file_size,
                COALESCE(AVG(file_size), 0) AS avg_file_size,
                COALESCE(MIN(NULLIF(upload_time, 0)), 0) AS earliest_upload_time,
                COALESCE(MAX(upload_time), 0) AS latest_upload_time,
                COALESCE(MIN(NULLIF(modify_time, 0)), 0) AS earliest_modify_time,
                COALESCE(MAX(modify_time), 0) AS latest_modify_time
            FROM group_files
            """
        ).fetchone()
        if not row:
            return IndexInfo()
        return IndexInfo(
            total_groups=int(row["total_groups"] or 0),
            total_records=int(row["total_records"] or 0),
            total_file_size=int(row["total_file_size"] or 0),
            avg_file_size=int(float(row["avg_file_size"] or 0)),
            earliest_upload_time=int(row["earliest_upload_time"] or 0),
            latest_upload_time=int(row["latest_upload_time"] or 0),
            earliest_modify_time=int(row["earliest_modify_time"] or 0),
            latest_modify_time=int(row["latest_modify_time"] or 0),
        )

    @staticmethod
    def format_group_display(group_id: str, group_name: str | None) -> str:
        name = str(group_name or "").strip()
        gid = str(group_id or "").strip()
        if gid and name:
            return f"{name} ({gid})"
        if name:
            return name
        if gid:
            return f"未知群名 ({gid})"
        return gid

    def get_group_name(self, group_id: str) -> str:
        gid = normalize_group_id_input(group_id)
        row = self._conn.execute(
            "SELECT group_name FROM group_index_groups WHERE group_id = ?",
            (gid,),
        ).fetchone()
        if not row:
            return ""
        return str(row["group_name"] or "").strip()

    def list_indexed_groups_info(self) -> list[IndexedGroupInfo]:
        rows = self._conn.execute(
            """
            SELECT group_id, group_id_num, group_name, file_count, total_file_size, updated_at
            FROM group_index_groups
            ORDER BY file_count DESC, total_file_size DESC, group_id ASC
            """
        ).fetchall()
        out: list[IndexedGroupInfo] = []
        for r in rows:
            out.append(
                IndexedGroupInfo(
                    group_id=str(r["group_id"] or ""),
                    group_id_num=int(r["group_id_num"] or 0),
                    group_name=str(r["group_name"] or ""),
                    file_count=int(r["file_count"] or 0),
                    total_file_size=int(r["total_file_size"] or 0),
                    updated_at=int(r["updated_at"] or 0),
                )
            )
        return out

    async def list_remote_groups(self, bot: OneBotWsClient, *, no_cache: bool = False) -> list[RemoteGroupInfo]:
        res = await bot.call_api("get_group_list", {"no_cache": bool(no_cache)})
        _require_ok("get_group_list", res)
        out: list[RemoteGroupInfo] = []
        for g in (res.data or []):
            try:
                gid_num = int(g.get("group_id"))
            except Exception:
                continue
            out.append(
                RemoteGroupInfo(
                    group_id=format_group_id(gid_num),
                    group_id_num=gid_num,
                    group_name=str(g.get("group_name") or "").strip(),
                )
            )
        # 保序去重
        seen: set[str] = set()
        uniq: list[RemoteGroupInfo] = []
        for item in out:
            if item.group_id in seen:
                continue
            seen.add(item.group_id)
            uniq.append(item)
        return uniq

    async def update_index(
        self,
        bot: OneBotWsClient,
        target: str = "all",
        *,
        no_cache: bool = False,
        on_group_done: Callable[[int, int, str, bool], None] | None = None,
    ) -> dict[str, IndexUpdateStats]:
        t = (target or "all").strip()
        is_all_target = (not t) or (t.lower() == "all")
        if is_all_target:
            groups = await self.list_remote_groups(bot, no_cache=no_cache)
        else:
            gid = normalize_group_id_input(t)
            gid_num = parse_group_numeric_id(gid)
            gname = self.get_group_name(gid)
            if not gname:
                try:
                    for item in await self.list_remote_groups(bot, no_cache=no_cache):
                        if item.group_id == gid:
                            gid_num = int(item.group_id_num or gid_num)
                            gname = str(item.group_name or "").strip()
                            break
                except Exception:
                    pass
            groups = [
                RemoteGroupInfo(
                    group_id=gid,
                    group_id_num=gid_num,
                    group_name=gname,
                )
            ]

        result: dict[str, IndexUpdateStats] = {}
        total = len(groups)
        done = 0
        for g in groups:
            gid = g.group_id
            ok = True
            try:
                files, scan_complete = await self._fetch_group_files(bot, gid)
            except Exception as e:
                ok = False
                # 即使文件抓取失败，也尽量保留群元信息，避免群名长期缺失。
                if str(g.group_name or "").strip():
                    with self._conn:
                        self._update_group_index_row(gid, g.group_id_num, str(g.group_name or "").strip())
                logging.getLogger(__name__).exception("index refresh failed: group=%s", gid)
                result[gid] = IndexUpdateStats(failed=True, error=str(e))
            else:
                if not scan_complete:
                    logging.getLogger(__name__).warning(
                        "index refresh for group=%s is partial; stale-row deletion skipped in this run",
                        gid,
                    )
                stats = self._upsert_group_files(gid, files, allow_delete_stale=scan_complete)
                result[gid] = stats
                group_name = str(g.group_name or "").strip()
                if not group_name and files:
                    # 单群更新时若缺少群名，尽量沿用已有数据。
                    group_name = self.get_group_name(gid)
                with self._conn:
                    self._update_group_index_row(gid, g.group_id_num, group_name)
            finally:
                done += 1
                if on_group_done is not None:
                    on_group_done(done, total, gid, ok)

        # all 模式下，移除已不在远端群列表中的本地索引群，避免陈旧数据污染搜索结果。
        if is_all_target:
            remote_groups = {g.group_id for g in groups}
            indexed_groups = set(self._list_indexed_groups())
            stale_groups = sorted(indexed_groups - remote_groups)
            for gid in stale_groups:
                deleted = self._clear_group_exact(gid)
                with self._conn:
                    self._delete_group_index_row(gid)
                if deleted > 0:
                    result[gid] = IndexUpdateStats(
                        total_remote=0,
                        inserted=0,
                        updated=0,
                        unchanged=0,
                        deleted=int(deleted),
                    )
        return result

    async def _fetch_group_files(
        self,
        bot: OneBotWsClient,
        group_id_str: str,
    ) -> tuple[list[IndexedRemoteFile], bool]:
        group_id_num = parse_group_numeric_id(group_id_str)
        files: list[IndexedRemoteFile] = []
        seen_file: set[str] = set()
        seen_folder: set[str] = set()
        scan_complete = True

        async def fetch(folder_id: str, folder_path: str) -> None:
            nonlocal scan_complete
            if folder_id:
                res = await bot.call_api(
                    "get_group_files_by_folder",
                    {"group_id": group_id_num, "folder_id": folder_id},
                )
            else:
                res = await bot.call_api("get_group_root_files", {"group_id": group_id_num})
            _require_ok("get_group_files", res)
            data = res.data or {}

            for f in (data.get("files", []) or []):
                file_id = str(f.get("file_id") or "").strip()
                if not file_id or file_id in seen_file:
                    continue
                seen_file.add(file_id)
                file_name = str(f.get("file_name") or "").strip()
                uploader_id = self._safe_ts(f.get("uploader"))
                uploader_name = str(f.get("uploader_name") or "").strip()
                files.append(
                    IndexedRemoteFile(
                        group_id=group_id_str,
                        group_id_num=group_id_num,
                        api_group_id=(self._safe_ts(f.get("group_id")) or group_id_num),
                        file_id=file_id,
                        file_name=file_name,
                        folder_id=folder_id,
                        folder_path=folder_path,
                        dead_time=self._safe_ts(f.get("dead_time")),
                        download_times=self._safe_ts(f.get("download_times")),
                        uploader_id=uploader_id,
                        uploader_name=uploader_name,
                        upload_time=self._safe_ts(f.get("upload_time")),
                        modify_time=self._safe_ts(f.get("modify_time")),
                        file_size=int(f.get("file_size") or 0),
                        busid=int(f.get("busid") or 0),
                        alias="",
                    )
                )

            for d in (data.get("folders", []) or []):
                did = str(d.get("folder_id") or "").strip()
                if not did or did in seen_folder:
                    continue
                seen_folder.add(did)
                name = sanitize_component(str(d.get("folder_name") or ""))
                next_path = str(PurePosixPath(folder_path) / name) if folder_path else name
                try:
                    await fetch(did, next_path)
                except Exception:
                    scan_complete = False
                    logging.getLogger(__name__).exception("index fetch folder failed: %s", next_path)

        await fetch("", "")
        return files, scan_complete

    @staticmethod
    def _safe_ts(v: Any) -> int:
        try:
            return int(v or 0)
        except Exception:
            return 0

    @staticmethod
    def _file_md5(f: IndexedRemoteFile) -> str:
        payload = "\t".join(
            [
                f.group_id,
                str(f.api_group_id),
                f.file_id,
                f.file_name,
                str(f.file_size),
                str(f.upload_time),
                str(f.modify_time),
                f.folder_id,
                f.folder_path,
                str(f.busid),
                str(f.dead_time),
                str(f.download_times),
                str(f.uploader_id),
                f.uploader_name,
            ]
        )
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    def _group_index_stats(self, group_id: str) -> tuple[int, int]:
        row = self._conn.execute(
            """
            SELECT COUNT(*) AS c, COALESCE(SUM(file_size), 0) AS s
            FROM group_files
            WHERE group_id = ?
            """,
            (group_id,),
        ).fetchone()
        if not row:
            return (0, 0)
        return (int(row["c"] or 0), int(row["s"] or 0))

    def _update_group_index_row(self, group_id: str, group_id_num: int, group_name: str) -> None:
        file_count, total_file_size = self._group_index_stats(group_id)
        now = int(time.time())
        self._conn.execute(
            """
            INSERT INTO group_index_groups(group_id, group_id_num, group_name, file_count, total_file_size, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(group_id) DO UPDATE SET
                group_id_num=excluded.group_id_num,
                group_name=CASE
                    WHEN excluded.group_name != '' THEN excluded.group_name
                    ELSE group_index_groups.group_name
                END,
                file_count=excluded.file_count,
                total_file_size=excluded.total_file_size,
                updated_at=excluded.updated_at
            """,
            (
                group_id,
                int(group_id_num or 0),
                str(group_name or "").strip(),
                int(file_count),
                int(total_file_size),
                now,
            ),
        )

    def _delete_group_index_row(self, group_id: str) -> None:
        self._conn.execute("DELETE FROM group_index_groups WHERE group_id = ?", (group_id,))

    def _upsert_group_files(
        self,
        group_id: str,
        remote_files: list[IndexedRemoteFile],
        *,
        allow_delete_stale: bool = True,
    ) -> IndexUpdateStats:
        stats = IndexUpdateStats(total_remote=len(remote_files))
        rows = self._conn.execute(
            """
            SELECT id, file_id, modify_time, md5, api_group_id, uploader_id, uploader_name, dead_time, download_times
            FROM group_files
            WHERE group_id = ?
            """,
            (group_id,),
        ).fetchall()
        existing = {str(r["file_id"]): r for r in rows}
        seen: set[str] = set()
        now = int(time.time())

        with self._conn:
            for rf in remote_files:
                fid = rf.file_id
                seen.add(fid)
                cur = existing.get(fid)
                md5_now = self._file_md5(rf)
                if cur is None:
                    row_id = self._conn.execute(
                        """
                        INSERT INTO group_files (
                            group_id, group_id_num, api_group_id, file_id, folder_id, folder_path,
                            file_name, uploader, uploader_id, uploader_name, upload_time, dead_time,
                            modify_time, download_times, file_size, busid, md5, alias, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            rf.group_id,
                            rf.group_id_num,
                            rf.api_group_id,
                            rf.file_id,
                            rf.folder_id,
                            rf.folder_path,
                            rf.file_name,
                            rf.uploader_name,
                            int(rf.uploader_id or 0),
                            rf.uploader_name,
                            int(rf.upload_time or 0),
                            int(rf.dead_time or 0),
                            int(rf.modify_time or 0),
                            int(rf.download_times or 0),
                            int(rf.file_size or 0),
                            int(rf.busid or 0),
                            md5_now,
                            rf.alias,
                            now,
                        ),
                    ).lastrowid
                    if self._fts_available:
                        self._conn.execute(
                            "INSERT INTO group_files_fts(rowid, file_name, uploader, alias) VALUES (?, ?, ?, ?)",
                            (row_id, rf.file_name, rf.uploader_name, rf.alias),
                        )
                    stats.inserted += 1
                    continue

                old_modify = int(cur["modify_time"] or 0)
                old_md5 = str(cur["md5"] or "")
                row_id = int(cur["id"])
                old_api_group_id = int(cur["api_group_id"] or 0)
                old_uploader_id = int(cur["uploader_id"] or 0)
                old_uploader_name = str(cur["uploader_name"] or "")
                old_dead_time = int(cur["dead_time"] or 0)
                old_download_times = int(cur["download_times"] or 0)
                if (
                    old_modify == int(rf.modify_time or 0)
                    and old_md5 == md5_now
                    and old_api_group_id == int(rf.api_group_id or 0)
                    and old_uploader_id == int(rf.uploader_id or 0)
                    and old_uploader_name == str(rf.uploader_name or "")
                    and old_dead_time == int(rf.dead_time or 0)
                    and old_download_times == int(rf.download_times or 0)
                ):
                    stats.unchanged += 1
                    continue

                self._conn.execute(
                    """
                    UPDATE group_files
                    SET api_group_id = ?,
                        folder_id = ?,
                        folder_path = ?,
                        file_name = ?,
                        uploader = ?,
                        uploader_id = ?,
                        uploader_name = ?,
                        upload_time = ?,
                        dead_time = ?,
                        modify_time = ?,
                        download_times = ?,
                        file_size = ?,
                        busid = ?,
                        md5 = ?,
                        alias = ?,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        int(rf.api_group_id or 0),
                        rf.folder_id,
                        rf.folder_path,
                        rf.file_name,
                        rf.uploader_name,
                        int(rf.uploader_id or 0),
                        rf.uploader_name,
                        int(rf.upload_time or 0),
                        int(rf.dead_time or 0),
                        int(rf.modify_time or 0),
                        int(rf.download_times or 0),
                        int(rf.file_size or 0),
                        int(rf.busid or 0),
                        md5_now,
                        rf.alias,
                        now,
                        row_id,
                    ),
                )
                if self._fts_available:
                    self._conn.execute("DELETE FROM group_files_fts WHERE rowid = ?", (row_id,))
                    self._conn.execute(
                        "INSERT INTO group_files_fts(rowid, file_name, uploader, alias) VALUES (?, ?, ?, ?)",
                        (row_id, rf.file_name, rf.uploader_name, rf.alias),
                    )
                stats.updated += 1

            if allow_delete_stale:
                stale = [int(r["id"]) for fid, r in existing.items() if fid not in seen]
                for row_id in stale:
                    if self._fts_available:
                        self._conn.execute("DELETE FROM group_files_fts WHERE rowid = ?", (row_id,))
                    self._conn.execute("DELETE FROM group_files WHERE id = ?", (row_id,))
                stats.deleted = len(stale)
            else:
                stats.deleted = 0

        return stats

    @staticmethod
    def parse_query(raw: str) -> SearchQuery:
        text = (raw or "").strip()
        if not text:
            raise ValueError("空查询")

        group_id: str | None = None
        uploader_id: int | None = None
        main = text
        if "@" in text:
            left, right = text.rsplit("@", 1)
            scope = right.strip()
            if scope:
                if scope.lower().startswith("qq:"):
                    qq_text = scope[3:].strip()
                    if not qq_text:
                        raise ValueError("QQ 号不能为空")
                    try:
                        uploader_id = int(qq_text)
                    except Exception as e:
                        raise ValueError(f"无效 QQ 号: {scope}") from e
                    main = left.strip()
                else:
                    try:
                        group_id = normalize_group_id_input(scope)
                        main = left.strip()
                    except Exception:
                        main = text

        search_by = "name"
        keys_part = main
        if "::" in main:
            left, right = main.split("::", 1)
            search_by = (left or "").strip().lower()
            keys_part = right.strip()

        alias_map = {
            "name": "name",
            "uploader": "uploader",
            "time-period": "time-period",
            "tp": "time-period",
        }
        search_by = alias_map.get(search_by, search_by)
        if search_by not in {"name", "uploader", "time-period"}:
            raise ValueError(f"不支持的 searchBy: {search_by}")

        keys = [k.strip() for k in keys_part.split(",") if k.strip()]
        if not keys:
            raise ValueError("查询关键字不能为空")

        return SearchQuery(raw=raw, search_by=search_by, keys=keys, group_id=group_id, uploader_id=uploader_id)

    def search(self, query: SearchQuery, *, strict: bool = False, min_results: int | None = None) -> list[SearchResult]:
        threshold = max(1, int(self.min_results if min_results is None else min_results))

        if query.search_by == "time-period":
            periods = [self._parse_time_period(k) for k in query.keys]
            rows = self._load_rows(query.group_id, query.uploader_id, periods=periods)
            if not rows:
                return []

            scored: list[tuple[sqlite3.Row, list[bool]]] = []
            for row in rows:
                ts = int(row["upload_time"] or 0)
                flags = [self._match_period(ts, start, end) for start, end in periods]
                if any(flags):
                    scored.append((row, flags))
            return self._build_results(
                scored,
                query.keys,
                search_by=query.search_by,
                strict=strict,
                threshold=threshold,
                bm25_map={},
            )

        rows = self._load_rows(query.group_id, query.uploader_id)
        if not rows:
            return []

        text_col = "file_name" if query.search_by == "name" else "uploader_name"
        regex_patterns: list[re.Pattern[str] | None] = []
        fts_hits: list[set[int] | None] = []
        for key in query.keys:
            if self._has_regex_meta(key):
                regex_patterns.append(re.compile(key))
                fts_hits.append(None)
                continue
            regex_patterns.append(None)
            fts_hits.append(
                self._fts_key_rowids(
                    search_by=query.search_by,
                    key=key,
                    group_id=query.group_id,
                    uploader_id=query.uploader_id,
                    strict=strict,
                )
            )

        scored2: list[tuple[sqlite3.Row, list[bool]]] = []
        for row in rows:
            row_id = int(row["id"])
            text = str(row[text_col] or row["uploader"] or "")
            text_fold = text.casefold()
            flags: list[bool] = []
            for idx, key in enumerate(query.keys):
                p = regex_patterns[idx]
                if p is not None:
                    flags.append(bool(p.search(text)))
                    continue

                literal_hit = key.casefold() in text_fold
                hits = fts_hits[idx]
                fuzzy_hit = hits is not None and row_id in hits
                fuzzy_text_hit = (not strict) and self._fuzzy_text_match(key, text)
                # 兼容旧行为：子串匹配始终作为基线召回；FTS/模糊用于补充召回与排序。
                flags.append(literal_hit or fuzzy_hit or fuzzy_text_hit)
            if any(flags):
                scored2.append((row, flags))

        bm25_map = self._bm25_map(query, strict=strict)
        return self._build_results(
            scored2,
            query.keys,
            search_by=query.search_by,
            strict=strict,
            threshold=threshold,
            bm25_map=bm25_map,
        )

    def _load_rows(
        self,
        group_id: str | None,
        uploader_id: int | None,
        periods: list[tuple[int | None, int | None]] | None = None,
    ) -> list[sqlite3.Row]:
        where_parts: list[str] = []
        params: list[Any] = []

        if group_id:
            where_parts.append("gf.group_id = ?")
            params.append(group_id)
        if uploader_id is not None:
            where_parts.append("gf.uploader_id = ?")
            params.append(int(uploader_id))

        if periods:
            period_parts: list[str] = []
            for start, end in periods:
                if start is None and end is None:
                    continue
                if start is None:
                    period_parts.append("gf.upload_time <= ?")
                    params.append(int(end))
                    continue
                if end is None:
                    period_parts.append("gf.upload_time >= ?")
                    params.append(int(start))
                    continue
                period_parts.append("gf.upload_time BETWEEN ? AND ?")
                params.append(int(start))
                params.append(int(end))
            if period_parts:
                where_parts.append("(" + " OR ".join(period_parts) + ")")

        where_sql = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""
        rows = self._conn.execute(
            f"""
            SELECT
                gf.id, gf.group_id, gf.file_id, gf.folder_id, gf.folder_path, gf.file_name, gf.uploader,
                gf.uploader_id, gf.uploader_name, gf.upload_time, gf.dead_time, gf.modify_time,
                gf.download_times, gf.file_size, gf.md5, gf.alias,
                COALESCE(gg.group_name, '') AS group_name
            FROM group_files AS gf
            LEFT JOIN group_index_groups AS gg ON gg.group_id = gf.group_id
            {where_sql}
            ORDER BY gf.file_name COLLATE NOCASE ASC
            """,
            tuple(params),
        ).fetchall()
        return list(rows)

    def _build_results(
        self,
        scored: list[tuple[sqlite3.Row, list[bool]]],
        keys: list[str],
        *,
        search_by: str,
        strict: bool,
        threshold: int,
        bm25_map: dict[int, float],
    ) -> list[SearchResult]:
        if not scored:
            return []

        exact_rows: list[tuple[sqlite3.Row, list[bool]]] = []
        fuzzy_rows: list[tuple[sqlite3.Row, list[bool]]] = []
        for row, flags in scored:
            if all(flags):
                exact_rows.append((row, flags))
            elif any(flags):
                fuzzy_rows.append((row, flags))

        sim_cache: dict[int, float] = {}

        def _row_similarity(item: tuple[sqlite3.Row, list[bool]]) -> float:
            row, flags = item
            row_id = int(row["id"])
            cached = sim_cache.get(row_id)
            if cached is not None:
                return cached
            score = self._similarity_score(row, keys, flags, search_by=search_by)
            sim_cache[row_id] = score
            return score

        def _exact_sort_key(item: tuple[sqlite3.Row, list[bool]]) -> tuple[float, float, str]:
            row, _flags = item
            row_id = int(row["id"])
            bm25 = float(bm25_map.get(row_id, 1e18))
            if strict:
                return (0.0, bm25, str(row["file_name"] or "").lower())
            sim = _row_similarity(item)
            return (-sim, bm25, str(row["file_name"] or "").lower())

        def _fuzzy_sort_key(item: tuple[sqlite3.Row, list[bool]]) -> tuple[float, int, tuple[int, ...], float, str]:
            row, flags = item
            sim = _row_similarity(item)
            matched_count = sum(1 for f in flags if f)
            positions = tuple(i for i, f in enumerate(flags) if f)
            row_id = int(row["id"])
            bm25 = float(bm25_map.get(row_id, 1e18))
            return (-sim, -matched_count, positions, bm25, str(row["file_name"] or "").lower())

        exact_rows.sort(key=_exact_sort_key)
        fuzzy_rows.sort(key=_fuzzy_sort_key)

        out: list[SearchResult] = [self._row_to_result(r, f, match_tag="Exact") for r, f in exact_rows]
        if strict:
            return out

        if len(out) >= threshold:
            return out

        need = threshold - len(out)
        for row, flags in fuzzy_rows[:need]:
            out.append(self._row_to_result(row, flags, match_tag="Fuzzy"))
        return out

    def _row_to_result(self, row: sqlite3.Row, flags: list[bool], *, match_tag: str) -> SearchResult:
        folder_path = str(row["folder_path"] or "")
        file_name = str(row["file_name"] or "")
        row_id = int(row["id"])
        return SearchResult(
            row_id=row_id,
            short_id=self.short_id_from_row_id(row_id),
            group_id=str(row["group_id"] or ""),
            group_name=str(row["group_name"] or ""),
            file_id=str(row["file_id"] or ""),
            folder_id=str(row["folder_id"] or ""),
            folder_path=folder_path,
            relative_path=group_relative_file_path(folder_path, file_name),
            file_name=file_name,
            uploader_name=str(row["uploader_name"] or row["uploader"] or ""),
            uploader_id=int(row["uploader_id"] or 0),
            upload_time=int(row["upload_time"] or 0),
            dead_time=int(row["dead_time"] or 0),
            modify_time=int(row["modify_time"] or 0),
            download_times=int(row["download_times"] or 0),
            file_size=int(row["file_size"] or 0),
            md5=str(row["md5"] or ""),
            alias=str(row["alias"] or ""),
            match_tag=match_tag,
            matched_count=sum(1 for f in flags if f),
        )

    def _similarity_score(
        self,
        row: sqlite3.Row,
        keys: list[str],
        flags: list[bool],
        *,
        search_by: str,
    ) -> float:
        if not keys:
            return 0.0

        if search_by == "name":
            text = str(row["file_name"] or "")
        elif search_by == "uploader":
            text = str(row["uploader_name"] or row["uploader"] or "")
        else:
            # time-period 等非文本场景仍按命中比例给稳定分数
            return float(sum(1 for f in flags if f)) * 100.0 / float(len(flags))

        text_norm = self._normalize_text(text).replace(" ", "")
        if not text_norm:
            return 0.0

        scores: list[float] = []
        for key, matched in zip(keys, flags):
            scores.append(self._key_similarity_score(key, text_norm, matched=matched))
        if not scores:
            return 0.0
        return float(sum(scores) / len(scores))

    def _key_similarity_score(self, key: str, text_norm: str, *, matched: bool) -> float:
        if self._has_regex_meta(key):
            return 100.0 if matched else 0.0

        q = self._normalize_text(key).replace(" ", "")
        if not q:
            return 0.0
        if q in text_norm:
            return 100.0

        if re.fullmatch(r"[\u4e00-\u9fff]+", q):
            need = Counter(q)
            have = Counter(text_norm)
            overlap = sum(min(have.get(ch, 0), cnt) for ch, cnt in need.items())
            return 100.0 * float(overlap) / float(max(1, len(q)))

        if _rf_fuzz is not None:
            try:
                return max(
                    float(_rf_fuzz.ratio(q, text_norm)),
                    float(_rf_fuzz.partial_ratio(q, text_norm)),
                    float(_rf_fuzz.token_set_ratio(q, text_norm)),
                )
            except Exception:
                pass

        # 兜底：不依赖第三方时给出稳定相似度
        try:
            from difflib import SequenceMatcher

            return float(SequenceMatcher(None, q, text_norm).ratio() * 100.0)
        except Exception:
            return 0.0

    @staticmethod
    def _has_regex_meta(text: str) -> bool:
        return bool(re.search(r"[.^$*+?{}\[\]|()\\]", text))

    def _bm25_map(self, query: SearchQuery, *, strict: bool) -> dict[int, float]:
        if not self._fts_available or query.search_by not in {"name", "uploader"}:
            return {}

        key_exprs: list[str] = []
        for key in query.keys:
            if self._has_regex_meta(key):
                continue
            expr = self._build_key_match_expr(query.search_by, key, strict=strict)
            if expr:
                key_exprs.append(f"({expr})")
        if not key_exprs:
            return {}
        # 相关性打分按“任一关键词命中”聚合，便于多关键词下也能比较候选项。
        match_expr = " OR ".join(key_exprs)
        rows = self._fts_scores(match_expr, query.group_id, query.uploader_id)

        out: dict[int, float] = {}
        for row_id, score in rows:
            out[row_id] = score
        return out

    def _fts_key_rowids(
        self,
        *,
        search_by: str,
        key: str,
        group_id: str | None,
        uploader_id: int | None,
        strict: bool,
    ) -> set[int] | None:
        if not self._fts_available or search_by not in {"name", "uploader"}:
            return None
        expr = self._build_key_match_expr(search_by, key, strict=strict)
        if not expr:
            return None
        return self._fts_rowids(expr, group_id, uploader_id)

    def _build_key_match_expr(self, search_by: str, key: str, *, strict: bool) -> str:
        tokens = self._tokenize_normalized(key)
        if not tokens:
            return ""

        col = "file_name" if search_by == "name" else "uploader"
        clauses: list[str] = []
        for token in tokens:
            terms: set[str] = set(self._term_variants(token))
            if not strict and self.fuzzy_edit_distance > 0:
                for variant in list(terms):
                    for near in self._expand_term_by_distance(variant, self.fuzzy_edit_distance):
                        terms.add(near)
            if not terms:
                continue

            ordered = sorted(terms, key=lambda x: (abs(len(x) - len(token)), x))
            limited = ordered[: self.fuzzy_max_terms_per_token]
            token_terms = [f"{col}:{self._fts_quote(t)}" for t in limited]
            if len(token_terms) == 1:
                clauses.append(token_terms[0])
            else:
                clauses.append("(" + " OR ".join(token_terms) + ")")

        return " AND ".join(clauses)

    def _tokenize_normalized(self, text: str) -> list[str]:
        s = self._normalize_text(text)
        if not s:
            return []
        # 优先使用 jieba 搜索分词，提升中文关键词召回；缺失时回退到正则切分。
        if self._jieba is not None and re.search(r"[\u4e00-\u9fff]", s):
            raw = self._jieba.cut_for_search(s)
        else:
            raw = _TOKEN_RE.findall(s)

        out: list[str] = []
        seen: set[str] = set()
        for part in raw:
            norm_part = self._normalize_text(part)
            if not norm_part:
                continue
            for tok in _TOKEN_RE.findall(norm_part):
                if not tok:
                    continue
                if tok in seen:
                    continue
                seen.add(tok)
                out.append(tok)

        if out:
            return out
        return [s]

    def _normalize_text(self, text: str) -> str:
        s = unicodedata.normalize("NFKC", str(text or "")).casefold()
        s = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", s)
        return s.strip()

    def _term_variants(self, token: str) -> tuple[str, ...]:
        base = self._normalize_text(token)
        if not base:
            return ()
        out: set[str] = {base}
        if self._opencc_t2s is not None and self._opencc_s2t is not None:
            try:
                out.add(self._normalize_text(self._opencc_t2s.convert(base)))
                out.add(self._normalize_text(self._opencc_s2t.convert(base)))
            except Exception:
                pass
        out.discard("")
        return tuple(sorted(out))

    def _expand_term_by_distance(self, term: str, max_dist: int) -> tuple[str, ...]:
        if not self._fts_vocab_available:
            return ()
        token = self._normalize_text(term)
        if not token:
            return ()
        cache_key = (token, int(max_dist))
        cached = self._term_expand_cache.get(cache_key)
        if cached is not None:
            return cached

        cands = self._candidate_terms(token, max_dist)
        scored: list[tuple[int, int, str]] = []
        for cand in cands:
            d = self._levenshtein_with_cutoff(token, cand, max_dist)
            if 1 <= d <= max_dist:
                scored.append((d, abs(len(cand) - len(token)), cand))
        scored.sort(key=lambda x: (x[0], x[1], x[2]))
        expanded = tuple(c for _, _, c in scored[: self.fuzzy_max_terms_per_token])
        self._term_expand_cache[cache_key] = expanded
        return expanded

    def _fuzzy_text_match(self, key: str, text: str) -> bool:
        max_dist = int(self.fuzzy_edit_distance)
        if max_dist <= 0:
            return False

        needle = self._normalize_text(key).replace(" ", "")
        hay = self._normalize_text(text).replace(" ", "")
        if not needle or not hay:
            return False
        if needle in hay:
            return True
        if len(needle) <= 1:
            return False

        # 纯中文：按“字符覆盖”做无序模糊，避免 unicode61 对中文子串召回不足；
        # 同时避免中文窗口编辑距离带来的高开销与高噪声。
        if re.fullmatch(r"[\u4e00-\u9fff]+", needle):
            need = Counter(needle)
            have = Counter(hay)
            return all(have.get(ch, 0) >= cnt for ch, cnt in need.items())

        # 短关键字的编辑距离窗口匹配噪声很高，限定 4 字符以上再启用。
        if len(needle) < 4:
            return False

        n = len(needle)
        min_len = max(1, n - max_dist)
        max_len = min(len(hay), n + max_dist)
        if min_len > max_len:
            return False

        for win_len in range(min_len, max_len + 1):
            limit = len(hay) - win_len + 1
            for i in range(limit):
                seg = hay[i : i + win_len]
                dist = self._damerau_levenshtein_with_cutoff(needle, seg, max_dist)
                if dist <= max_dist:
                    return True
        return False

    def _candidate_terms(self, token: str, max_dist: int) -> list[str]:
        if not token or not self._fts_vocab_available:
            return []
        min_len = max(1, len(token) - max_dist)
        max_len = len(token) + max_dist
        first = token[0]
        hard_limit = max(200, self.fuzzy_max_terms_per_token * 32)
        try:
            rows = self._conn.execute(
                """
                SELECT term
                FROM group_files_vocab
                WHERE length(term) BETWEEN ? AND ? AND term GLOB ?
                LIMIT ?
                """,
                (min_len, max_len, f"{first}*", hard_limit),
            ).fetchall()
            if not rows:
                rows = self._conn.execute(
                    """
                    SELECT term
                    FROM group_files_vocab
                    WHERE length(term) BETWEEN ? AND ?
                    LIMIT ?
                    """,
                    (min_len, max_len, hard_limit),
                ).fetchall()
        except Exception:
            return []

        out: list[str] = []
        for r in rows:
            t = self._normalize_text(str(r["term"] or ""))
            if not t:
                continue
            out.append(t)
        return out

    @staticmethod
    def _levenshtein_with_cutoff(a: str, b: str, max_dist: int) -> int:
        if _rf_levenshtein is not None:
            try:
                return int(_rf_levenshtein.distance(a, b, score_cutoff=int(max_dist)))
            except Exception:
                pass
        if a == b:
            return 0
        if abs(len(a) - len(b)) > max_dist:
            return max_dist + 1
        if not a or not b:
            return max(len(a), len(b))

        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, start=1):
            cur = [i]
            row_min = i
            for j, cb in enumerate(b, start=1):
                cost = 0 if ca == cb else 1
                v = min(
                    prev[j] + 1,
                    cur[j - 1] + 1,
                    prev[j - 1] + cost,
                )
                cur.append(v)
                if v < row_min:
                    row_min = v
            if row_min > max_dist:
                return max_dist + 1
            prev = cur
        return prev[-1]

    @staticmethod
    def _damerau_levenshtein_with_cutoff(a: str, b: str, max_dist: int) -> int:
        if _rf_damerau_levenshtein is not None:
            try:
                return int(_rf_damerau_levenshtein.distance(a, b, score_cutoff=int(max_dist)))
            except Exception:
                pass
        if a == b:
            return 0
        if abs(len(a) - len(b)) > max_dist:
            return max_dist + 1
        if not a or not b:
            return max(len(a), len(b))

        n = len(a)
        m = len(b)
        inf = n + m
        d = [[0] * (m + 2) for _ in range(n + 2)]
        d[0][0] = inf
        for i in range(n + 1):
            d[i + 1][0] = inf
            d[i + 1][1] = i
        for j in range(m + 1):
            d[0][j + 1] = inf
            d[1][j + 1] = j

        da: dict[str, int] = {}
        for i in range(1, n + 1):
            db = 0
            row_min = max_dist + 1
            ai = a[i - 1]
            for j in range(1, m + 1):
                bj = b[j - 1]
                i1 = da.get(bj, 0)
                j1 = db
                cost = 0 if ai == bj else 1
                if cost == 0:
                    db = j
                d[i + 1][j + 1] = min(
                    d[i][j] + cost,
                    d[i + 1][j] + 1,
                    d[i][j + 1] + 1,
                    d[i1][j1] + (i - i1 - 1) + 1 + (j - j1 - 1),
                )
                if d[i + 1][j + 1] < row_min:
                    row_min = d[i + 1][j + 1]
            da[ai] = i
            if row_min > max_dist:
                return max_dist + 1
        return d[n + 1][m + 1]

    def _fts_rowids(self, match_expr: str, group_id: str | None, uploader_id: int | None) -> set[int]:
        rows = self._fts_query(match_expr, group_id, uploader_id, with_score=False)
        return {row_id for row_id, _score in rows}

    def _fts_scores(self, match_expr: str, group_id: str | None, uploader_id: int | None) -> list[tuple[int, float]]:
        return self._fts_query(match_expr, group_id, uploader_id, with_score=True)

    def _fts_query(
        self,
        match_expr: str,
        group_id: str | None,
        uploader_id: int | None,
        *,
        with_score: bool,
    ) -> list[tuple[int, float]]:
        if not self._fts_available:
            return []
        select_score = ", bm25(group_files_fts) AS score" if with_score else ", 0.0 AS score"
        where = ["group_files_fts MATCH ?"]
        params: list[Any] = [match_expr]
        if group_id:
            where.append("group_files.group_id = ?")
            params.append(group_id)
        if uploader_id is not None:
            where.append("group_files.uploader_id = ?")
            params.append(int(uploader_id))
        where_sql = " AND ".join(where)
        try:
            rows = self._conn.execute(
                f"""
                SELECT group_files_fts.rowid AS rowid{select_score}
                FROM group_files_fts
                JOIN group_files ON group_files.id = group_files_fts.rowid
                WHERE {where_sql}
                """,
                tuple(params),
            ).fetchall()
        except Exception:
            return []

        out: list[tuple[int, float]] = []
        for r in rows:
            try:
                out.append((int(r["rowid"]), float(r["score"])))
            except Exception:
                continue
        return out

    @staticmethod
    def _fts_quote(text: str) -> str:
        return '"' + str(text or "").replace('"', '""') + '"'

    @staticmethod
    def _parse_time_period(text: str) -> tuple[int | None, int | None]:
        s = (text or "").strip()
        if not s:
            raise ValueError("空时间条件")

        lower = s.lower()
        if lower.startswith("before:"):
            end_ts = GroupFileIndexer._parse_ts(s[len("before:") :].strip(), is_end=True)
            return None, end_ts
        if lower.startswith("after:"):
            start_ts = GroupFileIndexer._parse_ts(s[len("after:") :].strip(), is_end=False)
            return start_ts, None
        if "/" in s:
            start_raw, end_raw = s.split("/", 1)
            start_ts = GroupFileIndexer._parse_ts(start_raw.strip(), is_end=False) if start_raw.strip() else None
            end_ts = GroupFileIndexer._parse_ts(end_raw.strip(), is_end=True) if end_raw.strip() else None
            return start_ts, end_ts

        ts = GroupFileIndexer._parse_ts(s, is_end=False)
        return ts, ts

    @staticmethod
    def _parse_ts(text: str, *, is_end: bool) -> int:
        s = (text or "").strip()
        if not s:
            raise ValueError("时间不能为空")

        if re.fullmatch(r"\d{13}", s):
            return int(s) // 1000
        if re.fullmatch(r"\d{10}", s):
            return int(s)

        # 日期（本地时区）
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
            d = datetime.strptime(s, "%Y-%m-%d").date()
            t = dt_time(23, 59, 59) if is_end else dt_time(0, 0, 0)
            dt = datetime.combine(d, t).astimezone()
            return int(dt.timestamp())

        # 完整 ISO 时间
        norm = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(norm)
        if dt.tzinfo is None:
            dt = dt.astimezone()
        else:
            dt = dt.astimezone()
        return int(dt.timestamp())

    @staticmethod
    def _match_period(ts: int, start: int | None, end: int | None) -> bool:
        if start is not None and ts < start:
            return False
        if end is not None and ts > end:
            return False
        return True

    def clear_group(self, group_id: str) -> int:
        gid = normalize_group_id_input(group_id)
        return self._clear_group_exact(gid)

    def _clear_group_exact(self, group_id: str) -> int:
        rows = self._conn.execute("SELECT id FROM group_files WHERE group_id = ?", (group_id,)).fetchall()
        ids = [int(r["id"]) for r in rows]
        with self._conn:
            for row_id in ids:
                if self._fts_available:
                    self._conn.execute("DELETE FROM group_files_fts WHERE rowid = ?", (row_id,))
            self._conn.execute("DELETE FROM group_files WHERE group_id = ?", (group_id,))
            self._delete_group_index_row(group_id)
        if not ids:
            return 0
        return len(ids)

    def _list_indexed_groups(self) -> list[str]:
        rows = self._conn.execute(
            """
            SELECT group_id FROM group_index_groups
            UNION
            SELECT DISTINCT group_id FROM group_files
            """
        ).fetchall()
        out: list[str] = []
        for r in rows:
            gid = str(r["group_id"] or "").strip()
            if gid:
                out.append(gid)
        out.sort()
        return out

    def get_group_file_path(self, row_id: int) -> str:
        row = self._conn.execute(
            """
            SELECT folder_path, file_name
            FROM group_files
            WHERE id = ?
            """,
            (int(row_id),),
        ).fetchone()
        if not row:
            return ""
        return group_relative_file_path(str(row["folder_path"] or ""), str(row["file_name"] or ""))

    def get_indexed_record_by_short_id(self, short_id: str) -> IndexedFileRecord | None:
        try:
            row_id = self.row_id_from_short_id(short_id)
        except Exception:
            return None
        row = self._conn.execute(
            """
            SELECT
                gf.id, gf.group_id, gf.group_id_num, gf.api_group_id, gf.file_id, gf.file_name, gf.folder_id, gf.folder_path,
                gf.uploader, gf.uploader_id, gf.uploader_name, gf.upload_time, gf.dead_time, gf.modify_time,
                gf.download_times, gf.file_size, gf.busid, gf.md5, gf.alias,
                COALESCE(gg.group_name, '') AS group_name
            FROM group_files AS gf
            LEFT JOIN group_index_groups AS gg ON gg.group_id = gf.group_id
            WHERE gf.id = ?
            """,
            (int(row_id),),
        ).fetchone()
        return self._row_to_indexed_record(row) if row else None

    def get_indexed_record_by_group_file(self, group_id: str, file_id: str) -> IndexedFileRecord | None:
        gid = normalize_group_id_input(group_id)
        fid = str(file_id or "").strip()
        if not fid:
            return None
        row = self._conn.execute(
            """
            SELECT
                gf.id, gf.group_id, gf.group_id_num, gf.api_group_id, gf.file_id, gf.file_name, gf.folder_id, gf.folder_path,
                gf.uploader, gf.uploader_id, gf.uploader_name, gf.upload_time, gf.dead_time, gf.modify_time,
                gf.download_times, gf.file_size, gf.busid, gf.md5, gf.alias,
                COALESCE(gg.group_name, '') AS group_name
            FROM group_files AS gf
            LEFT JOIN group_index_groups AS gg ON gg.group_id = gf.group_id
            WHERE gf.group_id = ? AND gf.file_id = ?
            """,
            (gid, fid),
        ).fetchone()
        return self._row_to_indexed_record(row) if row else None

    def _row_to_indexed_record(self, row: sqlite3.Row) -> IndexedFileRecord:
        row_id = int(row["id"])
        return IndexedFileRecord(
            row_id=row_id,
            short_id=self.short_id_from_row_id(row_id),
            group_id=str(row["group_id"] or ""),
            group_name=str(row["group_name"] or ""),
            group_id_num=int(row["group_id_num"] or 0),
            api_group_id=int(row["api_group_id"] or 0),
            file_id=str(row["file_id"] or ""),
            file_name=str(row["file_name"] or ""),
            folder_id=str(row["folder_id"] or ""),
            folder_path=str(row["folder_path"] or ""),
            uploader_name=str(row["uploader_name"] or row["uploader"] or ""),
            uploader_id=int(row["uploader_id"] or 0),
            upload_time=int(row["upload_time"] or 0),
            dead_time=int(row["dead_time"] or 0),
            modify_time=int(row["modify_time"] or 0),
            download_times=int(row["download_times"] or 0),
            file_size=int(row["file_size"] or 0),
            busid=int(row["busid"] or 0),
            md5=str(row["md5"] or ""),
            alias=str(row["alias"] or ""),
        )

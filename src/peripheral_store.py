from __future__ import annotations

import sqlite3
import time
from pathlib import Path


class GroupIndexStore:
    def upsert_group(
        self,
        group_id: str,
        group_id_num: int,
        group_name: str,
        file_count: int,
        total_file_size: int,
    ) -> None:
        raise NotImplementedError

    def delete_group(self, group_id: str) -> None:
        raise NotImplementedError

    def get_group_name(self, group_id: str) -> str:
        raise NotImplementedError

    def list_group_rows(self) -> list[dict[str, int | str]]:
        raise NotImplementedError

    def close(self) -> None:
        return


class SQLiteGroupIndexStore(GroupIndexStore):
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def upsert_group(
        self,
        group_id: str,
        group_id_num: int,
        group_name: str,
        file_count: int,
        total_file_size: int,
    ) -> None:
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

    def delete_group(self, group_id: str) -> None:
        self._conn.execute("DELETE FROM group_index_groups WHERE group_id = ?", (group_id,))

    def get_group_name(self, group_id: str) -> str:
        row = self._conn.execute(
            "SELECT group_name FROM group_index_groups WHERE group_id = ?",
            (group_id,),
        ).fetchone()
        if not row:
            return ""
        return str(row["group_name"] or "").strip()

    def list_group_rows(self) -> list[dict[str, int | str]]:
        rows = self._conn.execute(
            """
            SELECT group_id, group_id_num, group_name, file_count, total_file_size, updated_at
            FROM group_index_groups
            ORDER BY file_count DESC, total_file_size DESC, group_id ASC
            """
        ).fetchall()
        out: list[dict[str, int | str]] = []
        for r in rows:
            out.append(
                {
                    "group_id": str(r["group_id"] or ""),
                    "group_id_num": int(r["group_id_num"] or 0),
                    "group_name": str(r["group_name"] or ""),
                    "file_count": int(r["file_count"] or 0),
                    "total_file_size": int(r["total_file_size"] or 0),
                    "updated_at": int(r["updated_at"] or 0),
                }
            )
        return out


class PeeweeGroupIndexStore(GroupIndexStore):
    def __init__(self, db_path: Path):
        try:
            from peewee import IntegerField, Model, TextField
            from playhouse.sqlite_ext import SqliteExtDatabase
        except Exception as e:  # pragma: no cover
            raise RuntimeError("peewee backend is unavailable; install dependency 'peewee>=3.17'") from e

        self._db = SqliteExtDatabase(
            str(db_path),
            pragmas=(
                ("journal_mode", "wal"),
                ("synchronous", "normal"),
                ("temp_store", "memory"),
                ("busy_timeout", 10000),
            ),
        )
        self._db.connect(reuse_if_open=True)

        class _BaseModel(Model):
            class Meta:
                database = self._db

        class _GroupIndexGroup(_BaseModel):
            group_id = TextField(primary_key=True)
            group_id_num = IntegerField(default=0)
            group_name = TextField(default="")
            file_count = IntegerField(default=0)
            total_file_size = IntegerField(default=0)
            updated_at = IntegerField(default=0)

            class Meta:
                table_name = "group_index_groups"

        self._model = _GroupIndexGroup
        self._db.create_tables([self._model], safe=True)

    def upsert_group(
        self,
        group_id: str,
        group_id_num: int,
        group_name: str,
        file_count: int,
        total_file_size: int,
    ) -> None:
        now = int(time.time())
        name = str(group_name or "").strip()
        with self._db.atomic():
            row = self._model.get_or_none(self._model.group_id == group_id)
            if row is None:
                self._model.create(
                    group_id=group_id,
                    group_id_num=int(group_id_num or 0),
                    group_name=name,
                    file_count=int(file_count),
                    total_file_size=int(total_file_size),
                    updated_at=now,
                )
                return
            row.group_id_num = int(group_id_num or 0)
            if name:
                row.group_name = name
            row.file_count = int(file_count)
            row.total_file_size = int(total_file_size)
            row.updated_at = now
            row.save()

    def delete_group(self, group_id: str) -> None:
        self._model.delete().where(self._model.group_id == group_id).execute()

    def get_group_name(self, group_id: str) -> str:
        row = self._model.select(self._model.group_name).where(self._model.group_id == group_id).first()
        if row is None:
            return ""
        return str(row.group_name or "").strip()

    def list_group_rows(self) -> list[dict[str, int | str]]:
        query = (
            self._model.select(
                self._model.group_id,
                self._model.group_id_num,
                self._model.group_name,
                self._model.file_count,
                self._model.total_file_size,
                self._model.updated_at,
            )
            .order_by(self._model.file_count.desc(), self._model.total_file_size.desc(), self._model.group_id.asc())
            .dicts()
        )
        out: list[dict[str, int | str]] = []
        for row in query:
            out.append(
                {
                    "group_id": str(row.get("group_id") or ""),
                    "group_id_num": int(row.get("group_id_num") or 0),
                    "group_name": str(row.get("group_name") or ""),
                    "file_count": int(row.get("file_count") or 0),
                    "total_file_size": int(row.get("total_file_size") or 0),
                    "updated_at": int(row.get("updated_at") or 0),
                }
            )
        return out

    def close(self) -> None:
        if not self._db.is_closed():
            self._db.close()


def build_group_index_store(db_path: Path, conn: sqlite3.Connection, backend: str) -> GroupIndexStore:
    normalized = str(backend or "auto").strip().lower()
    if normalized not in {"auto", "sqlite", "peewee"}:
        normalized = "auto"
    if normalized == "sqlite":
        return SQLiteGroupIndexStore(conn)
    if normalized == "peewee":
        return PeeweeGroupIndexStore(db_path)
    try:
        return PeeweeGroupIndexStore(db_path)
    except Exception:
        return SQLiteGroupIndexStore(conn)

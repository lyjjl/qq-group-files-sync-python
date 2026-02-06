from __future__ import annotations

from pathlib import PurePosixPath
from typing import Iterable

from filesystem import FileSystemManager
from ignore_rules import IgnoreMatcher


def _root_prefix(group_root: str) -> str:
    root = PurePosixPath(group_root).as_posix().rstrip("/")
    return f"{root}/" if root else ""


def list_group_files(fs: FileSystemManager, group_root: str) -> list[str]:

    full = fs.list_files_under(group_root)
    out: list[str] = []
    prefix = _root_prefix(group_root)
    for p in full:
        ps = PurePosixPath(p).as_posix()
        if prefix and not ps.startswith(prefix):
            continue
        rel = ps[len(prefix) :]
        if rel:
            out.append(rel)
    out.sort()
    return out


def filter_ignored(paths: Iterable[str], ignore: IgnoreMatcher | None) -> list[str]:
    if not ignore:
        return list(paths)
    return [p for p in paths if not ignore.is_ignored(p)]


def split_files_by_size(
    fs: FileSystemManager,
    group_root: str,
    rel_files: Iterable[str],
) -> tuple[list[str], list[str], dict[str, int]]:

    non_empty: list[str] = []
    empty: list[str] = []
    size_map: dict[str, int] = {}
    for k in rel_files:
        try:
            abs_path = fs.resolve(f"{group_root}/{k}")
            sz = int(abs_path.stat().st_size)
            if sz == 0:
                empty.append(k)
            else:
                non_empty.append(k)
                size_map[k] = sz
        except Exception:
            non_empty.append(k)
            size_map[k] = -1
    non_empty.sort()
    empty.sort()
    return non_empty, empty, size_map


def list_group_files_rel(
    fs: FileSystemManager,
    group_root: str,
    ignore: IgnoreMatcher | None = None,
) -> tuple[set[str], set[str]]:

    kept: set[str] = set()
    ignored: set[str] = set()
    for rel in list_group_files(fs, group_root):
        if ignore and ignore.is_ignored(rel):
            ignored.add(rel)
        else:
            kept.add(rel)
    return kept, ignored

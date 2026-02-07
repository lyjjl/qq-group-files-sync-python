from __future__ import annotations

import re
from pathlib import PurePosixPath

_UNSAFE = re.compile(r"[<>:\"|?*\\/]")


def sanitize_component(name: str) -> str:
    safe = _UNSAFE.sub("_", name)
    if safe in {"", ".", ".."}:
        return "_"
    return safe


def normalize_group_id(group_id: str) -> str:
    safe = sanitize_component(group_id)
    if safe.startswith("QQ-Group_"):
        safe = safe.removeprefix("QQ-Group_")
    return safe


def group_root_dir(group_id: str) -> str:
    normalized = normalize_group_id(group_id)
    if not normalized:
        return "QQ-Group"
    return f"QQ-Group_{normalized}"


def group_status_file_path(group_id: str) -> str:
    return f"{group_root_dir(group_id)}.json"


def group_timestamp_file_path(group_id: str) -> str:
    return f"{group_root_dir(group_id)}_timestamps.txt"


def group_invalid_url_count_file_path(group_id: str) -> str:
    return f"invalidFiles/{group_root_dir(group_id)}.json"


def group_relative_file_path(folder_path: str, file_name: str) -> str:
    file_name = sanitize_component(file_name)
    if not folder_path:
        return file_name
    return str(PurePosixPath(folder_path) / file_name)

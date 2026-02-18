from __future__ import annotations

from collections import defaultdict
from pathlib import PurePosixPath


def folder_counts(paths: set[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in paths:
        parts = PurePosixPath(path).parts
        if len(parts) <= 1:
            continue
        for i in range(1, len(parts)):
            folder = "/".join(parts[:i])
            counts[folder] = counts.get(folder, 0) + 1
    return counts


def folder_signatures(paths: set[str], size_map: dict[str, int]) -> dict[str, set[tuple[str, int]]]:
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


def is_path_conflict(path: str, existing: list[str]) -> bool:
    for it in existing:
        if it == path:
            return True
        if it.startswith(path + "/") or path.startswith(it + "/"):
            return True
    return False


def detect_folder_renames(
    *,
    old_all: set[str],
    old_removed: set[str],
    old_size_map: dict[str, int],
    new_all: set[str],
    new_added: set[str],
    new_size_map: dict[str, int],
    min_overlap_ratio: float = 0.5,
) -> list[tuple[str, str]]:
    old_total = folder_counts(old_all)
    old_removed_counts = folder_counts(old_removed)
    new_total = folder_counts(new_all)
    new_added_counts = folder_counts(new_added)

    old_candidates = [f for f, cnt in old_removed_counts.items() if cnt and cnt == old_total.get(f, 0)]
    new_candidates = [f for f, cnt in new_added_counts.items() if cnt and cnt == new_total.get(f, 0)]

    old_sigs = folder_signatures(old_removed, old_size_map)
    new_sigs = folder_signatures(new_added, new_size_map)

    pairs: list[tuple[float, int, int, int, str, str]] = []
    for old in old_candidates:
        s1 = old_sigs.get(old)
        if not s1:
            continue
        for new in new_candidates:
            if old == new:
                continue
            s2 = new_sigs.get(new)
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
        if is_path_conflict(old, selected_old):
            continue
        if is_path_conflict(new, selected_new):
            continue
        selected.append((old, new))
        selected_old.append(old)
        selected_new.append(new)
    return selected

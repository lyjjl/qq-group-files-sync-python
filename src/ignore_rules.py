from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath


@dataclass(frozen=True)
class IgnoreRule:
    pattern: str
    is_dir: bool
    anchored: bool


class IgnoreMatcher:
    """
    用于 pull/push 的忽略匹配器

    示例路径："docs/readme.md"

    语法：
     - '#' 开头为注释（仅限行首）
     - 忽略空行
     - 以 '/' 结尾表示目录规则（忽略该目录下的所有内容）
     - 不含 '/' 的模式匹配任意路径组件
     - 含有 '/' 的模式将从组根目录开始锚定
     - 支持 glob 通配符：'*'、'?' 和 '**'（目录通配符）
    """

    def __init__(self, rules: list[IgnoreRule]):
        self._rules = rules

    @classmethod
    def from_file(cls, path: Path) -> "IgnoreMatcher":
        rules: list[IgnoreRule] = []
        raw = path.read_text(encoding="utf-8", errors="ignore")
        for line in raw.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            s = s.replace("\\", "/")
            is_dir = s.endswith("/")
            if is_dir:
                s = s[:-1]
            s = s.lstrip("./").lstrip("/")
            if not s:
                continue
            anchored = "/" in s
            rules.append(IgnoreRule(pattern=s, is_dir=is_dir, anchored=anchored))
        return cls(rules)

    def is_ignored(self, rel_posix_path: str) -> bool:
        p = _norm_rel(rel_posix_path)
        if not p:
            return False
        parts = PurePosixPath(p).parts

        for r in self._rules:
            pat = r.pattern

            if r.anchored:
                pat_parts = tuple(PurePosixPath(pat).parts)
                if r.is_dir:
                    # pat/**
                    if _match_parts(parts, pat_parts + ("**",)):
                        return True
                else:
                    if _match_parts(parts, pat_parts):
                        return True
                continue

            if r.is_dir:
                for i, comp in enumerate(parts[:-1]):
                    if fnmatchcase(comp, pat):
                        return True
                if parts and fnmatchcase(parts[-1], pat):
                    return True
            else:
                for comp in parts:
                    if fnmatchcase(comp, pat):
                        return True

        return False


def _norm_rel(p: str) -> str:
    s = (p or "").replace("\\", "/").strip()
    while s.startswith("./"):
        s = s[2:]
    s = s.lstrip("/")
    return s


def _match_parts(path_parts: tuple[str, ...], pat_parts: tuple[str, ...]) -> bool:
    """支持 "**" 的路径部分通配符匹配"""

    i = 0
    j = 0
    stack: list[tuple[int, int]] = []

    while True:
        if j == len(pat_parts):
            return i == len(path_parts)

        if pat_parts[j] == "**":
            while j < len(pat_parts) and pat_parts[j] == "**":
                j += 1
            if j == len(pat_parts):
                return True
            stack.append((i, j))
            continue

        if i < len(path_parts) and fnmatchcase(path_parts[i], pat_parts[j]):
            i += 1
            j += 1
            continue

        if stack:
            i0, j0 = stack.pop()
            if i0 < len(path_parts):
                stack.append((i0 + 1, j0))
                i = i0 + 1
                j = j0
                continue

        return False

from __future__ import annotations

import os
import shutil
from pathlib import Path


class FileSystemManager:
    def __init__(self, base_path: str | Path):
        self.base_path = Path(base_path).expanduser().resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)

    def resolve(self, relative: str | Path) -> Path:
        rel = Path(relative)
        return (self.base_path / rel).resolve()

    def write_text(self, relative: str | Path, text: str, *, encoding: str = "utf-8") -> None:
        path = self.base_path / Path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding=encoding)

    def write_bytes(self, relative: str | Path, data: bytes) -> None:
        path = self.base_path / Path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def read_bytes(self, relative: str | Path) -> bytes:
        path = self.base_path / Path(relative)
        return path.read_bytes()

    def exists(self, relative: str | Path) -> bool:
        return (self.base_path / Path(relative)).exists()

    def mkdir_all(self, relative: str | Path) -> None:
        (self.base_path / Path(relative)).mkdir(parents=True, exist_ok=True)

    def list_status_files(self) -> list[str]:
        files = []
        for p in self.base_path.glob("QQ-Group_*.json"):
            if p.is_file() and p.parent == self.base_path:
                files.append(p.name)
        files.sort()
        return files

    def list_files_under(self, relative_dir: str | Path) -> list[str]:
        root = self.base_path / Path(relative_dir)
        if not root.exists() or not root.is_dir():
            return []
        out: list[str] = []
        for p in root.rglob("*"):
            if p.is_file():
                out.append(str(p.relative_to(self.base_path).as_posix()))
        return out

    def remove_file(self, relative_file: str | Path) -> None:
        p = self.base_path / Path(relative_file)
        if p.exists() and p.is_file():
            p.unlink()

    def remove_tree(self, relative_dir: str | Path) -> None:
        p = self.base_path / Path(relative_dir)
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)

    def remove_empty_dirs(self, relative_dir: str | Path) -> None:
        root = self.base_path / Path(relative_dir)
        if not root.exists() or not root.is_dir():
            return

        for d in sorted([p for p in root.rglob("*") if p.is_dir()], key=lambda x: len(x.parts), reverse=True):
            try:
                if not any(d.iterdir()):
                    d.rmdir()
            except OSError:
                pass

    def set_file_times(self, relative_file: str | Path, *, upload_time: int | None, modify_time: int | None) -> None:
        p = self.base_path / Path(relative_file)
        if not p.exists():
            return
        atime = float(upload_time) if upload_time is not None else p.stat().st_atime
        mtime = float(modify_time) if modify_time is not None else p.stat().st_mtime
        os.utime(p, times=(atime, mtime))

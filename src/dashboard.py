from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from config import AppConfig
from filesystem import FileSystemManager
from group_paths import group_root_dir, group_status_file_path


@dataclass
class DashboardGroupDef:
    id: str
    alias: str
    description: str
    root_dir: str
    status_path: str


def _env(template_dir: Path) -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    return env


def generate_dashboard(cfg: AppConfig, fs: FileSystemManager) -> tuple[str, str]:
    output_file = (cfg.web.dashboard_file or "list.html").strip() or "list.html"
    base_url = (cfg.web.base_url or "").strip()
    if not base_url:
        base_url = "."
    else:
        base_url = base_url.rstrip("/") or "."

    title = (cfg.web.title or "群文件导航").strip() or "群文件导航"

    group_defs: list[DashboardGroupDef] = []
    for g in cfg.groups:
        gid = (g.id or "").strip()
        if not gid:
            continue
        alias = (g.alias or "").strip() or gid
        group_defs.append(
            DashboardGroupDef(
                id=gid,
                alias=alias,
                description=(g.description or "").strip(),
                root_dir=group_root_dir(gid),
                status_path=group_status_file_path(gid),
            )
        )

    data_groups = _collect_statuses(fs)
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "groups": data_groups,
    }

    # data file name: xxx_data.json
    out_path = Path(output_file)
    if out_path.suffix:
        data_file = out_path.with_suffix("").name + "_data.json"
    else:
        data_file = out_path.name + "_data.json"

    fs.write_text(data_file, json.dumps(summary, ensure_ascii=False, indent=2))

    template_dir = Path(__file__).parent / "templates"
    env = _env(template_dir)
    tmpl = env.get_template("dashboard.html")
    html = tmpl.render(
        title=title,
        data_file_name=Path(data_file).name,
        base_url=base_url,
        config_groups_json=json.dumps([d.__dict__ for d in group_defs], ensure_ascii=False),
    )

    fs.write_text(output_file, html)
    return output_file, data_file


def _collect_statuses(fs: FileSystemManager) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for name in fs.list_status_files():
        try:
            raw = fs.read_bytes(name)
            status = json.loads(raw)
        except Exception:
            continue

        gid = str(status.get("group_id") or status.get("groupId") or "").strip() or _derive_group_id_from_filename(name)
        status_path = name
        root_dir = group_root_dir(gid)
        entry = {
            "id": gid,
            "root_dir": root_dir,
            "status_path": status_path,
            "status": status,
        }
        prev = groups.get(status_path)
        if prev:
            prev_status = prev.get("status") or {}
            if int(prev_status.get("last_update") or 0) >= int(status.get("last_update") or 0):
                continue
        groups[status_path] = entry

    result = list(groups.values())
    result.sort(key=lambda x: (x.get("id") or "", x.get("status_path") or ""))
    return result


def _derive_group_id_from_filename(name: str) -> str:
    stem = Path(name).stem
    if stem.startswith("QQ-Group_"):
        stem = stem.removeprefix("QQ-Group_")
    return stem

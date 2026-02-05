from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

from pydantic import BaseModel, Field
from pydantic.config import ConfigDict


class OneBot11Config(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)
    ws_url: str = Field(
        default="ws://127.0.0.1:3001",
        alias="wsUrl",
        description="OneBot11 正向 WS 地址",
    )
    access_token: str = Field(default="", alias="accessToken")


class FileSystemConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)
    local_path: str = Field(default="./data", alias="localPath")


class GroupConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)
    id: str = Field(alias="id")
    alias: str = Field(default="", alias="alias")
    description: str = Field(default="", alias="description")
    no_pull: bool = Field(default=False, alias="noPull")
    no_push: bool = Field(default=False, alias="noPush")


class WebConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)
    title: str = Field(default="群文件导航", alias="title")
    base_url: str = Field(default="", alias="baseUrl")
    dashboard_file: str = Field(default="list.html", alias="dashboardFile")


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)
    onebot11: OneBot11Config = Field(default_factory=OneBot11Config, alias="oneBot11")
    file_system: FileSystemConfig = Field(default_factory=FileSystemConfig, alias="fileSystem")
    log_file: str = Field(default="./logs/main.log", alias="logFile")
    log_level: str = Field(default="info", alias="logLevel")
    invalid_files_log: str = Field(default="./logs/invalidFiles.log", alias="invalidFilesLog")
    groups: list[GroupConfig] = Field(default_factory=list, alias="groups")
    web: WebConfig = Field(default_factory=WebConfig, alias="web")


def _parse_json_line(text: str) -> dict[str, Any]:
    """Parse the first non-empty, non-comment JSON line."""
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#") or line.startswith("//"):
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise ValueError("config must be a JSON object")
        return obj
    return {}


def default_config() -> AppConfig:
    return AppConfig(
        onebot11=OneBot11Config(ws_url="ws://127.0.0.1:3001", access_token=""),
        file_system=FileSystemConfig(local_path="./data"),
        log_file="./logs/main.log",
        log_level="info",
        invalid_files_log="./logs/invalidFiles.log",
        groups=[
            GroupConfig(
                id="QQ-Group:123456",
                alias="示例名称",
                description="实例介绍",
            )
        ],
        web=WebConfig(title="群文件导航", base_url="", dashboard_file="list.html"),
    )


def is_placeholder_config(cfg: AppConfig) -> bool:
    """检测未修改示例配置的情况"""
    if len(cfg.groups) != 1:
        return False
    g = cfg.groups[0]
    if g.id != "QQ-Group:123456":
        return False
    if (g.alias or "") != "示例名称":
        return False
    if (g.description or "") != "实例介绍":
        return False
    return True


def load_config(path: str | Path = "config.toml") -> AppConfig:
    p = Path(path)
    if not p.exists():
        cfg = default_config()
        save_config(cfg, p)
        return cfg

    raw = p.read_text(encoding="utf-8")
    if not raw.strip():
        cfg = default_config()
        save_config(cfg, p)
        return cfg

    data: dict[str, Any]
    suffix = p.suffix.lower()
    if suffix in {".toml", ""}:
        parsed = tomllib.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("config must be a TOML table")
        data = parsed
    elif suffix in {".json", ".jsonl"}:
        # 理论上不应该支持 jsonl 的...
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                data = parsed
            else:
                raise ValueError("config must be a JSON object")
        except Exception:
            data = _parse_json_line(raw)
    else:
        raise ValueError(
            f"unsupported config format: {p.name}. Please use TOML (config.toml), JSON (config.json) or JSONL (config.jsonl)."
        )

    cfg = AppConfig.model_validate(data)

    if not cfg.web.dashboard_file:
        cfg.web.dashboard_file = "list.html"
    if not cfg.web.title:
        cfg.web.title = "群文件导航"

    return cfg


def _toml_quote(text: str) -> str:
    s = str(text or "")
    s = s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{s}"'


def render_config_toml(cfg: AppConfig) -> str:
    """写默认 toml 配置"""

    lines: list[str] = []
    lines.append("# qq-group-files-sync 配置文件（TOML）")
    lines.append("# 以 # 开头的是注释。")
    lines.append("# 首次运行若文件不存在，会自动生成一份带示例的配置。")
    lines.append("")

    lines.append("[onebot11]")
    lines.append("# OneBot11 正向 WS 地址")
    lines.append(f"ws_url = {_toml_quote(cfg.onebot11.ws_url)}")
    lines.append("# 若协议端启用了鉴权，请填写 token")
    lines.append(f"access_token = {_toml_quote(cfg.onebot11.access_token)}")
    lines.append("")

    lines.append("[file_system]")
    lines.append("# 本地数据目录（群文件与展示页面都会放在这里）")
    lines.append(f"local_path = {_toml_quote(cfg.file_system.local_path)}")
    lines.append("")

    lines.append("# 日志：main.log（全量）与 error.log（warn/error）")
    lines.append(f"log_file = {_toml_quote(cfg.log_file)}")
    lines.append("# 控制台日志级别：none / debug / info / warn / error")
    lines.append(f"log_level = {_toml_quote(cfg.log_level)}")
    lines.append("# 失效文件日志：每次拉取失败（过期/无下载链接等）会追加到该文件")
    lines.append(f"invalid_files_log = {_toml_quote(cfg.invalid_files_log)}")
    lines.append("")

    lines.append("# 需要同步的群列表：每个群一个 [[groups]]")
    if cfg.groups:
        for g in cfg.groups:
            lines.append("[[groups]]")
            lines.append("# 群标识：支持 QQ-Group:123456 或纯数字")
            lines.append(f"id = {_toml_quote(g.id)}")
            lines.append(f"alias = {_toml_quote(g.alias)}")
            lines.append(f"description = {_toml_quote(g.description)}")
            lines.append("# pull all 时跳过该群（单群 pull 不受影响）")
            lines.append(f"no_pull = {'true' if bool(getattr(g, 'no_pull', False)) else 'false'}")
            lines.append("# push all 时跳过该群（单群 push 不受影响）")
            lines.append(f"no_push = {'true' if bool(getattr(g, 'no_push', False)) else 'false'}")
            lines.append("")
    else:
        lines.append("[[groups]]")
        lines.append("id = \"QQ-Group:123456\"")
        lines.append("alias = \"示例名称\"")
        lines.append("description = \"实例介绍\"")
        lines.append("no_pull = false")
        lines.append("no_push = false")
        lines.append("")

    lines.append("[web]")
    lines.append("# 展示页面标题")
    lines.append(f"title = {_toml_quote(cfg.web.title)}")
    lines.append("# base_url 用于 list.html 中的相对链接前缀（留空则为本地相对路径）")
    lines.append(f"base_url = {_toml_quote(cfg.web.base_url)}")
    lines.append("# 输出文件名（写入到 file_system.local_path 下）")
    lines.append(f"dashboard_file = {_toml_quote(cfg.web.dashboard_file)}")
    lines.append("")

    return "\n".join(lines)


def save_config(cfg: AppConfig, path: str | Path = "config.toml") -> None:
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".toml" or not suffix:
        p.write_text(render_config_toml(cfg), encoding="utf-8")
        return

    payload: dict[str, Any] = cfg.model_dump(by_alias=True, exclude_none=True)

    if suffix in {".json", ".jsonl"}:
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        p.write_text(text + "\n", encoding="utf-8")
        return

    raise ValueError(
        f"unsupported config format: {p.name}. Please use TOML (config.toml), JSON (config.json) or JSONL (config.jsonl)."
    )

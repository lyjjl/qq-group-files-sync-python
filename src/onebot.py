from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Optional

import inspect

import websockets


@dataclass
class ActionResult:
    status: str
    retcode: int
    data: Any
    echo: str


def _headers(access_token: str) -> dict[str, str]:
    if not access_token:
        return {}
    # Token
    return {"Authorization": f"Bearer {access_token}"}


class OneBotWsClient:
    def __init__(self, ws_url: str, access_token: str = "", *, request_timeout_s: float = 30.0):
        self.ws_url = ws_url
        self.access_token = access_token
        self.request_timeout_s = request_timeout_s
        self._ws: Any | None = None
        self._recv_task: asyncio.Task | None = None
        self._pending: Dict[str, asyncio.Future] = {}
        self._events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def __aenter__(self) -> "OneBotWsClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def connect(self) -> None:
        if self._ws is not None:
            return
        kwargs: dict[str, Any] = {
            "ping_interval": 20,
            "ping_timeout": 20,
            "max_size": 16 * 1024 * 1024,
        }
        headers = _headers(self.access_token)
        sig = inspect.signature(websockets.connect)
        if "additional_headers" in sig.parameters:
            kwargs["additional_headers"] = headers
        else:
            kwargs["extra_headers"] = headers

        self._ws = await websockets.connect(self.ws_url, **kwargs)
        self._recv_task = asyncio.create_task(self._recv_loop())

    async def close(self) -> None:
        if self._recv_task:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
            self._recv_task = None
        if self._ws:
            await self._ws.close()
            self._ws = None
        for fut in list(self._pending.values()):
            if not fut.done():
                fut.set_exception(RuntimeError("websocket closed"))
        self._pending.clear()

    async def _recv_loop(self) -> None:
        assert self._ws is not None
        async for msg in self._ws:
            try:
                payload = json.loads(msg)
            except Exception:
                continue

            if isinstance(payload, dict) and "echo" in payload:
                echo = str(payload.get("echo"))
                fut = self._pending.pop(echo, None)
                if fut is not None and not fut.done():
                    fut.set_result(payload)
                continue

            if isinstance(payload, dict):
                await self._events.put(payload)

    async def call_api(self, action: str, params: Optional[dict[str, Any]] = None) -> ActionResult:
        if self._ws is None:
            raise RuntimeError("not connected")

        echo = uuid.uuid4().hex
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[echo] = fut

        req: dict[str, Any] = {"action": action, "params": params or {}, "echo": echo}
        await self._ws.send(json.dumps(req, ensure_ascii=False))

        raw = await asyncio.wait_for(fut, timeout=self.request_timeout_s)
        return ActionResult(
            status=str(raw.get("status", "")),
            retcode=int(raw.get("retcode", -1)),
            data=raw.get("data"),
            echo=echo,
        )

    async def events(self) -> AsyncIterator[dict[str, Any]]:
        while True:
            evt = await self._events.get()
            yield evt


def parse_group_numeric_id(group_id: str | int) -> int:
    if isinstance(group_id, int):
        return group_id
    s = str(group_id).strip()
    if s.startswith("QQ-Group:"):
        s = s.removeprefix("QQ-Group:")
    return int(s)


def extract_plain_text(message: Any) -> str:
    """提取消息文本"""
    if message is None:
        return ""
    if isinstance(message, str):
        return message.strip()
    if isinstance(message, list):
        parts = []
        for seg in message:
            if isinstance(seg, dict) and seg.get("type") == "text":
                data = seg.get("data") or {}
                parts.append(str(data.get("text", "")))
        return "".join(parts).strip()
    return str(message).strip()

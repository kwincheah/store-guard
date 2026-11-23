"""Asyncio WebSocket client hub broadcasting JSON envelopes safely from worker threads."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import threading
from typing import Any, Dict, Optional, Set

from fastapi import WebSocket


class EventBroadcaster:
    """Thread-safe pub/sub fan-out for /ws/events clients."""

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._clients: Set[WebSocket] = set()
        self._lock = threading.Lock()

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        with self._lock:
            self._loop = loop

    async def register(self, ws: WebSocket) -> None:
        await ws.accept()
        with self._lock:
            self._clients.add(ws)

    def unregister(self, ws: WebSocket) -> None:
        with self._lock:
            self._clients.discard(ws)

    @staticmethod
    def build_envelope(event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "type": event_type,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "payload": payload,
        }

    def emit_sync(self, event_type: str, payload: Dict[str, Any]) -> None:
        envelope = self.build_envelope(event_type, payload)
        with self._lock:
            loop = self._loop
            if loop is None or loop.is_closed() or not self._clients:
                return
        asyncio.run_coroutine_threadsafe(self._broadcast(envelope), loop)

    async def _broadcast(self, envelope: Dict[str, Any]) -> None:
        with self._lock:
            clients = list(self._clients)
        stale = []
        for ws in clients:
            try:
                await ws.send_json(envelope)
            except Exception:
                stale.append(ws)
        if stale:
            with self._lock:
                for ws in stale:
                    self._clients.discard(ws)

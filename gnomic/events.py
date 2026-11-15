"""Event bus. The engine emits events, and they fan out to a JSONL log and live subscribers (the web UI).

Events are for spectators and analysis, so they may contain private information
(hands, hidden rules, each player's reasoning). Players never see events: an
LLM's prompt is built only from `PlayerView` (see game.py), which holds public
information plus that player's own hand.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from typing import Any


def _default(o: Any):
    if hasattr(o, "to_json"):
        return o.to_json()
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


class EventBus:
    def __init__(self, log_path: str | Path | None = None):
        self._lock = threading.Lock()
        self.history: list[dict] = []
        self._subs: list[queue.Queue] = []
        self._fh = None
        if log_path:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(log_path, "a", encoding="utf-8")

    def emit(self, type_: str, **data: Any) -> dict:
        event = json.loads(json.dumps({"type": type_, "ts": time.time(), **data}, default=_default))
        with self._lock:
            event["seq"] = len(self.history)
            self.history.append(event)
            if self._fh:
                self._fh.write(json.dumps(event) + "\n")
                self._fh.flush()
            for q in self._subs:
                q.put(event)
        return event

    __call__ = emit

    def subscribe(self) -> tuple[list[dict], queue.Queue]:
        """Return a snapshot of past events and a queue of future ones, without gaps."""
        q: queue.Queue = queue.Queue()
        with self._lock:
            snapshot = list(self.history)
            self._subs.append(q)
        return snapshot, q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None


def null_emit(type_: str, **data: Any) -> None:
    return None

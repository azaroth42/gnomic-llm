"""Watching finished matches again (`gnomic replay`).

GET /                          index of the match logs in the logs directory (static/replays.html)
GET /api/matches               the index as JSON
GET /match/<file>              the spectator UI (static/index.html) for one log
GET /match/<file>/events       SSE: the log's events. With ?spt=N (seconds per turn) they are paced
                               like a live match; without it they arrive all at once.

Pacing (`--seconds-per-turn`, or `spt=` from the index page): every turn takes N seconds, however
long the players really took, so slow local models can be watched at a steady speed. Within a turn
(turn_start to resolution) events keep their real proportions: if the player took 30 s and the
judges 10 s, the play comes at 3/4 N and the calls at N. A gap between turns (rule making, game
end, the engine's own delay) is replayed as it was, but never longer than N. Each event's `ts`
is rewritten to the moment it is sent, so the UI's "Thinking N s" timers show replay time.

Each viewer gets their own playback from the start of the match. The spectator UI shows
everything, hidden rules included, so this server is for reviewing finished matches.
"""

from __future__ import annotations

import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from .server import STATIC, _QuietServer

LOG_NAME = re.compile(r"^[\w.-]+\.jsonl$")
TURN_END = ("resolution", "no_cards")             # the last event of a turn
OUTSIDE = ("turn_start", "game_start", "game_end", "match_end")  # events that can't be inside a turn
MAX_SPT = 600.0


def load_events(path: str | Path) -> list[dict]:
    events = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    break  # a match that crashed mid-write: keep what came before
    for i, ev in enumerate(events):
        ev["seq"] = i
    return events


def schedule(events: list[dict], seconds_per_turn: float) -> list[float]:
    """When to send each event, in seconds from the start of playback (see the module docstring)."""
    out: list[float] = []
    t = 0.0
    i, n = 0, len(events)
    while i < n:
        ev = events[i]
        gap = ev.get("ts", 0) - events[i - 1].get("ts", 0) if i else 0.0
        t += min(max(gap, 0.0), seconds_per_turn)  # the time since the previous event, at most one turn
        if ev.get("type") != "turn_start":
            out.append(t)
            i += 1
            continue
        last = i  # the turn runs from its turn_start to its resolution (or no_cards)
        for j in range(i + 1, n):
            kind = events[j].get("type")
            if kind in OUTSIDE:
                break
            last = j
            if kind in TURN_END:
                break
        t0 = ev.get("ts", 0)
        span = events[last].get("ts", 0) - t0
        prev = t
        for k in range(i, last + 1):
            frac = (events[k].get("ts", 0) - t0) / span if span > 0 else (k - i) / max(1, last - i)
            prev = max(prev, t + min(max(frac, 0.0), 1.0) * seconds_per_turn)
            out.append(prev)
        t += seconds_per_turn
        i = last + 1
    return out


def match_info(path: Path) -> dict | None:
    """One row of the index, or None if the file isn't a match log."""
    try:
        events = load_events(path)
    except (OSError, UnicodeDecodeError):
        return None
    start = next((e for e in events if e.get("type") == "match_start"), None)
    if start is None:
        return None
    end = next((e for e in events if e.get("type") == "match_end"), None)
    games = [{"winner": e.get("winner"), "winner_name": e.get("winner_name"), "turns": e.get("turns"),
              "reason": e.get("reason")} for e in events if e.get("type") == "game_end"]
    cfg = start.get("config") or {}
    return {
        "file": path.name,
        "started": start.get("ts"),
        "last": events[-1].get("ts"),
        "players": [{"name": p.get("name"), "model": p.get("model"), "kind": p.get("kind")}
                    for p in start.get("players", [])],
        "num_games": start.get("num_games"),
        "games": games,
        "turns": sum(1 for e in events if e.get("type") == "turn_start"),
        "rules": sum(1 for e in events if e.get("type") == "rule_added"),
        "propose_rules": bool(cfg.get("propose_rules")),
        "status": "incomplete" if end is None else "stopped" if (end.get("summary") or {}).get("stopped")
                  else "finished",
    }


def match_index(logs_dir: str | Path) -> list[dict]:
    """Every match log in the directory, newest first."""
    rows = [match_info(p) for p in Path(logs_dir).glob("*.jsonl") if LOG_NAME.match(p.name)]
    return sorted((r for r in rows if r), key=lambda r: (r["started"] or 0, r["file"]), reverse=True)


def parse_spt(value: str | None) -> float | None:
    try:
        spt = float(value) if value not in (None, "") else None
    except ValueError:
        return None
    return None if spt is None or spt <= 0 else min(spt, MAX_SPT)


def make_replay_handler(logs_dir: Path, seconds_per_turn: float | None):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _log(self, name: str) -> Path | None:
            name = unquote(name)
            path = logs_dir / name
            return path if LOG_NAME.match(name) and path.is_file() else None

        def do_GET(self):
            url = urlsplit(self.path)
            parts = url.path.strip("/").split("/")
            if url.path in ("/", "/index.html"):
                self._send(200, (STATIC / "replays.html").read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/matches":
                body = {"logs": str(logs_dir), "seconds_per_turn": seconds_per_turn, "matches": match_index(logs_dir)}
                self._send(200, json.dumps(body).encode(), "application/json")
            elif len(parts) in (2, 3) and parts[0] == "match" and (path := self._log(parts[1])) is not None:
                if len(parts) == 2:
                    self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
                elif parts[2] == "events":
                    query = parse_qs(url.query, keep_blank_values=True)
                    spt = parse_spt((query.get("spt") or [None])[0]) if "spt" in query else seconds_per_turn
                    self._stream(path, spt)
                else:
                    self._send(404, b"not found", "text/plain")
            else:
                self._send(404, b"not found", "text/plain")

        def _stream(self, path: Path, spt: float | None) -> None:
            events = load_events(path)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            replay = {"file": path.name, "seconds_per_turn": spt, "events": len(events)}
            try:
                self._write({"type": "hello", "server_now": time.time(), "replaying": len(events), "replay": replay})
                if spt is None:
                    for ev in events:
                        self._write(ev)
                    self._write({"type": "caught_up", "server_now": time.time()})
                else:
                    self._write({"type": "caught_up", "server_now": time.time()})
                    start = time.monotonic()
                    for ev, at in zip(events, schedule(events, spt)):
                        wait = start + at - time.monotonic()
                        while wait > 0:  # keep the connection alive through long gaps
                            time.sleep(min(wait, 15))
                            wait = start + at - time.monotonic()
                            if wait > 0:
                                self.wfile.write(b": keepalive\n\n")
                                self.wfile.flush()
                        self._write({**ev, "ts": time.time()})
                self._write({"type": "replay_end"})
                while True:  # stay open, or the browser would reconnect and start the replay again
                    time.sleep(15)
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

        def _write(self, ev: dict) -> None:
            self.wfile.write(b"data: " + json.dumps(ev).encode() + b"\n\n")
            self.wfile.flush()

    return Handler


def serve_replays(logs_dir: str | Path, host: str = "127.0.0.1", port: int = 8765, *,
                  seconds_per_turn: float | None = None) -> _QuietServer:
    server = _QuietServer((host, port), make_replay_handler(Path(logs_dir), seconds_per_turn))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True, name="replay-http").start()
    return server

"""Spectator web UI: a stdlib HTTP server that streams engine events over Server-Sent Events.

GET  /              the single-page UI (gnomic/static/index.html)
GET  /events        SSE: every past event, then live events
POST /api/control   {"action": "pause"|"resume"|"stop"} or {"delay": seconds}

The spectator UI shows everything, including hidden rules, hands and each player's
private reasoning. Nothing it serves reaches the players. When human players sit at
the table (`web` seats), the spectator routes need `?key=<spectator key>`, so a player
can't open them by accident, and they are off entirely unless --ui was given.

Human players (players/web.py), one secret token each:
GET  /play/<token>          the player page (gnomic/static/play.html)
GET  /play/<token>/events   SSE: that player's own messages only (its outbox)
POST /play/<token>/reply    {"id": request id, "text": reply in the LLM reply format}
"""

from __future__ import annotations

import hmac
import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, urlsplit

from .events import EventBus
from .game import Controls

STATIC = Path(__file__).parent / "static"


def make_handler(bus: EventBus, controls: Controls | None, players: Iterable = (), *,
                 spectator: bool = True, key: str | None = None):
    seats = {p.token: p for p in players}   # web player token -> WebHumanPlayer

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # keep the console quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _spectator_ok(self, query: str) -> bool:
            if not spectator:
                return False
            if key is None:
                return True
            given = (parse_qs(query).get("key") or [""])[0]
            return hmac.compare_digest(given.encode(), key.encode())

        def _player(self, path: str):
            """(player, rest of the path) for /play/<token>[/...], or (None, None)."""
            parts = path.split("/")
            if len(parts) >= 3 and parts[1] == "play":
                p = seats.get(parts[2])
                if p is not None:
                    return p, "/".join(parts[3:])
            return None, None

        def do_GET(self):
            url = urlsplit(self.path)
            player, rest = self._player(url.path)
            if player is not None:
                if rest == "":
                    self._send(200, (STATIC / "play.html").read_bytes(), "text/html; charset=utf-8")
                elif rest == "events":
                    self._stream(player.outbox, {})
                else:
                    self._send(404, b"not found", "text/plain")
            elif url.path in ("/", "/index.html", "/events") and not self._spectator_ok(url.query):
                self._send(403, b"forbidden: the spectator view shows hidden information", "text/plain")
            elif url.path in ("/", "/index.html"):
                self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/events":
                self._stream(bus, {"controls": None if controls is None else
                                   {"paused": controls.paused, "delay": controls.delay}})
            else:
                self._send(404, b"not found", "text/plain")

        def _body(self):
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return None
            return body if isinstance(body, dict) else None

        def do_POST(self):
            url = urlsplit(self.path)
            player, rest = self._player(url.path)
            if player is not None and rest == "reply":
                body = self._body()
                if body is None:
                    self._send(400, b"bad json", "text/plain")
                    return
                try:
                    rid = int(body.get("id"))
                except (TypeError, ValueError):
                    rid = -1
                error = player.submit(rid, body.get("text"))
                self._send(200 if error is None else 409, json.dumps({"ok": error is None, "error": error}).encode(),
                           "application/json")
                return
            if url.path != "/api/control" or controls is None or not self._spectator_ok(url.query):
                self._send(404, b"not found", "text/plain")
                return
            body = self._body()
            if body is None:
                self._send(400, b"bad json", "text/plain")
                return
            action = body.get("action")
            if action == "pause":
                controls.pause()
            elif action == "resume":
                controls.resume()
            elif action == "stop":
                controls.stop()
            if "delay" in body:
                controls.delay = max(0.0, float(body["delay"]))
            state = {"paused": controls.paused, "delay": controls.delay, "stopped": controls.stop_requested}
            bus.emit("controls", **state)
            self._send(200, json.dumps(state).encode(), "application/json")

        def _stream(self, source: EventBus, hello_extra: dict):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            snapshot, q = source.subscribe()
            try:
                self._write({"type": "hello", "server_now": time.time(), "replaying": len(snapshot), **hello_extra})
                for ev in snapshot:
                    self._write(ev)
                self._write({"type": "caught_up", "server_now": time.time()})
                while True:
                    try:
                        ev = q.get(timeout=15)
                        self._write(ev)
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                source.unsubscribe(q)

        def _write(self, ev: dict) -> None:
            self.wfile.write(b"data: " + json.dumps(ev).encode() + b"\n\n")
            self.wfile.flush()

    return Handler


class _QuietServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        import sys
        if isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):
            return  # a browser closed a connection; nothing to report
        super().handle_error(request, client_address)


def serve(bus: EventBus, controls: Controls | None, host: str = "127.0.0.1", port: int = 8765, *,
          players: Iterable = (), spectator: bool = True, key: str | None = None) -> ThreadingHTTPServer:
    server = _QuietServer((host, port), make_handler(bus, controls, players, spectator=spectator, key=key))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True, name="http").start()
    return server

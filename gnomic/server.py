"""Spectator web UI: a stdlib HTTP server that streams engine events over Server-Sent Events.

GET  /              the single-page UI (gnomic/static/index.html)
GET  /events        SSE: every past event, then live events
POST /api/control   {"action": "pause"|"resume"|"stop"} or {"delay": seconds}

The UI is for human spectators and shows everything, including hidden rules,
hands and each player's private reasoning. Nothing served here reaches the players.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .events import EventBus
from .game import Controls

STATIC = Path(__file__).parent / "static"


def make_handler(bus: EventBus, controls: Controls | None):
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

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif self.path.startswith("/events"):
                self._stream()
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/api/control" or controls is None:
                self._send(404, b"not found", "text/plain")
                return
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
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

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            snapshot, q = bus.subscribe()
            try:
                hello = {"type": "hello", "server_now": time.time(), "replaying": len(snapshot),
                         "controls": None if controls is None else
                         {"paused": controls.paused, "delay": controls.delay}}
                self._write(hello)
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
                bus.unsubscribe(q)

        def _write(self, ev: dict) -> None:
            self.wfile.write(b"data: " + json.dumps(ev).encode() + b"\n\n")
            self.wfile.flush()

    return Handler


def serve(bus: EventBus, controls: Controls | None, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(bus, controls))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True, name="http").start()
    return server

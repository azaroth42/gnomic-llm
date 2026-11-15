import json
import socket
import urllib.request

from gnomic.events import EventBus
from gnomic.game import Controls
from gnomic.match import Match, MatchConfig
from gnomic.server import serve


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_server_serves_ui_and_replays_events():
    bus = EventBus()
    Match(MatchConfig(players=["oracle", "random"], num_games=1, seed=1), emit=bus).run()
    controls = Controls()
    port = free_port()
    server = serve(bus, controls, port=port)
    try:
        html = urllib.request.urlopen(f"http://127.0.0.1:{port}/").read().decode()
        assert "Gnomic Mau" in html
        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/events", timeout=5)
        types = []
        while "caught_up" not in types:
            line = resp.readline().decode()
            if line.startswith("data: "):
                types.append(json.loads(line[6:])["type"])
        assert types[0] == "hello" and "match_start" in types and "match_end" in types
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/control", data=b'{"action":"pause"}',
                                     headers={"Content-Type": "application/json"}, method="POST")
        assert json.loads(urllib.request.urlopen(req).read())["paused"] is True
        assert controls.paused
    finally:
        server.shutdown()

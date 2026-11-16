"""`gnomic replay`: pacing, the index of match logs, and the replay server."""

import json
import socket
import time
import urllib.error
import urllib.request

import pytest

from gnomic.events import EventBus
from gnomic.match import Match, MatchConfig
from gnomic.replay import load_events, match_index, schedule, serve_replays


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def ev(type_, ts):
    return {"type": type_, "ts": ts}


def test_every_turn_takes_the_same_time_with_real_proportions():
    events = [ev("match_start", 0), ev("game_start", 100),   # 100 s of rule making: capped
              ev("turn_start", 101), ev("play", 131), ev("call_phase", 131), ev("call", 136), ev("resolution", 141),
              ev("turn_start", 141.5), ev("play", 142), ev("resolution", 142.5),   # a fast turn
              ev("game_end", 142.6), ev("match_end", 500)]
    at = schedule(events, 2.0)
    assert at[:2] == [0.0, 2.0]                     # the long gap before game_start is cut to 2 s
    assert at[2] == pytest.approx(3.0)              # gap 1 s between game_start and the turn, kept
    assert at[3] == pytest.approx(4.5)              # the play came at 30/40 of the turn
    assert at[6] == pytest.approx(5.0)              # the turn lasts 2 s
    assert at[7] == pytest.approx(5.5)              # the 0.5 s between turns is kept
    assert at[9] == pytest.approx(7.5)              # 2 s again, however fast the turn really was
    assert at[10] == pytest.approx(7.6) and at[11] == pytest.approx(9.6)
    assert at == sorted(at)


def test_a_turn_with_no_resolution_and_equal_times_still_paces():
    events = [ev("turn_start", 5), ev("play", 5), ev("turn_start", 5), ev("no_cards", 5)]
    assert schedule(events, 1.0) == [0.0, 1.0, 1.0, 2.0]


@pytest.fixture
def logs(tmp_path):
    bus = EventBus(tmp_path / "match-1.jsonl")
    Match(MatchConfig(players=["oracle?name=Ora", "random?name=Ran", "stats"], num_games=2, seed=3), emit=bus).run()
    (tmp_path / "match-2.jsonl").write_text("")            # an empty log (a match that never started)
    (tmp_path / "match-1.summary.json").write_text("{}")  # not a log
    return tmp_path


def test_index_lists_match_logs(logs):
    rows = match_index(logs)
    assert [r["file"] for r in rows] == ["match-1.jsonl"]
    r = rows[0]
    assert r["status"] == "finished" and r["num_games"] == 2 and len(r["games"]) == 2
    assert [p["name"] for p in r["players"]] == ["Ora", "Ran", "stats"]
    assert r["turns"] == sum(g["turns"] for g in r["games"]) and r["rules"] >= 2


def test_server_serves_index_and_paced_replay(logs):
    port = free_port()
    serve_replays(logs, port=port, seconds_per_turn=0.01)
    base = f"http://127.0.0.1:{port}"
    assert "matches" in urllib.request.urlopen(base + "/").read().decode()
    body = json.loads(urllib.request.urlopen(base + "/api/matches").read())
    assert body["seconds_per_turn"] == 0.01 and body["matches"][0]["file"] == "match-1.jsonl"
    assert "Gnomic Mau" in urllib.request.urlopen(base + "/match/match-1.jsonl").read().decode()
    for bad in ("/match/../cli.py", "/match/match-1.summary.json", "/match/nope.jsonl/events"):
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + bad)

    def stream(query):
        resp = urllib.request.urlopen(f"{base}/match/match-1.jsonl/events{query}", timeout=10)
        out = []
        while not out or out[-1]["type"] != "replay_end":
            line = resp.readline().decode()
            if line.startswith("data: "):
                out.append(json.loads(line[6:]))
        resp.close()
        return out

    t0 = time.time()
    paced = stream("")
    original = load_events(logs / "match-1.jsonl")
    assert paced[0]["type"] == "hello" and paced[0]["replay"]["seconds_per_turn"] == 0.01
    sent = [e for e in paced if e["type"] not in ("hello", "caught_up", "replay_end")]
    assert [e["type"] for e in sent] == [e["type"] for e in original]
    assert all(e["ts"] >= t0 for e in sent)        # timestamps are replay time, for the UI's timers
    instant = stream("?spt=")                        # the index page's "all at once"
    assert instant[0]["replay"]["seconds_per_turn"] is None
    assert [e["type"] for e in instant].index("caught_up") == len(original) + 1

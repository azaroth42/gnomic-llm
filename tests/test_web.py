"""The browser player (players/web.py) and its server routes. A bot stands in for the page."""

import json
import socket
import threading
import urllib.error
import urllib.request

from gnomic.events import EventBus
from gnomic.game import Controls
from gnomic.match import Match, MatchConfig
from gnomic.players.web import WebHumanPlayer
from gnomic.server import serve


def drive(player: WebHumanPlayer, answer):
    """Answer every request on the player's page with answer(request_event, invalid_event_or_None)."""
    _, q = player.outbox.subscribe()
    seen: list[dict] = []

    def run():
        req = None
        while True:
            ev = q.get()
            seen.append(ev)
            if ev["type"] == "request":
                req = ev
                player.submit(ev["id"], answer(ev, None))
            elif ev["type"] == "invalid":
                player.submit(ev["id"], answer(req, ev))
            elif ev["type"] == "match_end":
                return

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return seen, t


def polite(req, invalid):
    """A reasonable player: first a bad card (to see INVALID), then the first card in hand."""
    v = req["view"]
    if req["kind"] == "rule":
        return "NOTES:\nmy notes\nRULE: you may not play a red card on a red card"
    if req["kind"] == "judge":
        return "NOTES:\nmy notes\nCALL: NONE"
    if invalid is None and req["id"] == 1:
        return "PLAY: 1X"
    lines = ["NOTES:", f"notes {req['id']}", f"PLAY: {v['hand'][0]}"]
    if v["mechanics"]:
        lines += ["ANGLE: 90", "DECLARE: hello"]
    if v["can_call"]:
        lines.append("CALL: NONE")
    return "\n".join(lines)


def test_web_player_plays_a_match_like_an_llm():
    events = []
    m = Match(MatchConfig(players=["web?name=Ann", "oracle?name=Zebulon", "stats?name=Quintus"], num_games=3,
                          seed=2, max_rounds=4, dealer=1),
              emit=lambda kind, **kw: events.append((kind, kw)))
    web = m.players[0]
    assert isinstance(web, WebHumanPlayer) and web.kind == "human"
    seen, t = drive(web, polite)
    s = m.run()
    t.join(5)
    kinds = [e["type"] for e in seen]
    assert kinds[0] == "intro" and "request" in kinds and kinds[-1] == "match_end"
    assert "invalid" in kinds  # the bad first card got the LLM's INVALID message
    intro = seen[0]
    assert intro["label"] == "P1" and "REPLY FORMAT" in intro["system_prompt"]
    req = next(e for e in seen if e["type"] == "request")
    assert "STATE" in req["message"] and ("YOUR TURN" in req["message"] or "JUDGE" in req["message"])
    # The page followed the table live: other players' turns, each public event exactly once.
    updates = [e for e in seen if e["type"] == "update"]
    assert any(u["current"] not in (None, 0) for u in updates)
    lines = [line for e in seen if e["type"] in ("update", "request", "match_end") for line in e["events"]]
    played = [line for line in lines if " played " in line]
    assert len(played) == len(set(played)) == sum(len(r.stats) for r in m.results)
    assert any(" P2 played " in line for line in lines)
    me = s["players"][0]
    assert me["plays"] > 0 and me["forced_plays"] == 0
    # The notes reached the spectators' play events, as an LLM's do.
    plays = [kw for k, kw in events if k == "play" and kw["seat"] == 0]
    assert all(kw["notes"].startswith("notes ") for kw in plays)
    # Game 3 has table mechanics: the options went through.
    assert any(kw["move"] and kw["move"].get("angle") == 90 for kw in plays if kw["game"] == 2)
    # Nothing on the page that the player may not know: no names, no other player's rule text.
    page = json.dumps(seen)
    assert "Zebulon" not in page and "Quintus" not in page
    for e in m.stack.entries:
        if e.owner != web.name:
            assert e.rule.description not in page


def test_unusable_replies_fall_back_like_an_llm():
    m = Match(MatchConfig(players=["web?max_retries_turn=1", "random"], num_games=1, seed=3, max_rounds=1, dealer=1))
    web = m.players[0]
    seen, t = drive(web, lambda req, invalid: "I pass")
    s = m.run()
    t.join(5)
    assert s["players"][0]["forced_plays"] == 1
    assert [e["type"] for e in seen].count("invalid") == 1  # one retry, then the engine decides
    done = next(e for e in seen if e["type"] == "done")
    assert done["accepted"] is False and "random card" in done["fallback"]


def test_stop_releases_a_waiting_web_player():
    controls = Controls()
    m = Match(MatchConfig(players=["web", "random"], num_games=2, seed=1, dealer=1), controls=controls)
    web = m.players[0]
    web.should_stop = lambda: controls.stop_requested
    _, q = web.outbox.subscribe()
    out = {}
    t = threading.Thread(target=lambda: out.update(m.run()), daemon=True)
    t.start()
    while q.get(timeout=5)["type"] != "request":
        pass
    controls.stop()
    t.join(5)
    assert not t.is_alive() and out["stopped"] is True


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def status(url, data=None):
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"},
                                 method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_server_player_routes_and_spectator_key():
    web = WebHumanPlayer("Ann")
    web.outbox.emit("intro", seat=0, label="P1", players=2, system_prompt="hi")
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    server = serve(EventBus(), Controls(), port=port, players=[web], key="sekrit")
    try:
        assert status(f"{base}/")[0] == 403
        assert status(f"{base}/events")[0] == 403
        assert status(f"{base}/api/control", b'{"action":"pause"}')[0] == 404
        assert status(f"{base}/?key=sekrit")[0] == 200
        code, body = status(f"{base}/play/{web.token}")
        assert code == 200 and b"Gnomic Mau" in body
        assert status(f"{base}/play/not-a-token")[0] == 404
        # No request is open, so a reply is refused.
        code, body = status(f"{base}/play/{web.token}/reply", json.dumps({"id": 1, "text": "PLAY: 5D"}).encode())
        assert code == 409 and json.loads(body)["ok"] is False
        resp = urllib.request.urlopen(f"{base}/play/{web.token}/events", timeout=5)
        types = []
        while "caught_up" not in types:
            line = resp.readline().decode()
            if line.startswith("data: "):
                types.append(json.loads(line[6:])["type"])
        assert types == ["hello", "intro", "caught_up"]
    finally:
        server.shutdown()
    server = serve(EventBus(), Controls(), port=(port := free_port()), players=[web], spectator=False)
    try:
        assert status(f"http://127.0.0.1:{port}/")[0] == 403
        assert status(f"http://127.0.0.1:{port}/play/{web.token}")[0] == 200
    finally:
        server.shutdown()


def test_page_marks_penalties_on_discard_from_the_public_log():
    from gnomic.cards import Card
    from gnomic.game import PlayerView, PublicEvent
    from gnomic.players.web import view_json

    c = Card.parse
    log = (
        PublicEvent("play", 0, 0, player=1, card=c("5D"), top_before=c("2C")),  # earlier game: ignored
        PublicEvent("game_start", 1, player=0, card=c("2C")),
        PublicEvent("play", 1, 0, player=0, card=c("9H"), top_before=c("2C"), calls=(2,), call_rules=(1,),
                    upheld=True, upheld_by=(2,), penalized=(0,)),
        PublicEvent("play", 1, 1, player=1, card=c("KS"), top_before=c("9H"), calls=(2,), call_rules=(1,),
                    upheld=False, penalized=(2,)),
        PublicEvent("play", 1, 2, player=2, card=c("5H"), top_before=c("KS"), move="5D 5H at 90", engine_call=True,
                    penalized=(2,)),
        PublicEvent("play", 1, 3, player=0, card=c("3C"), top_before=c("5H")),
    )
    view = PlayerView(seat=0, player_names=("a", "b", "c"), hand=(c("AH"),),
                      discard=tuple(c(x) for x in ("2C", "9H", "KS", "5D", "5H", "3C")), deck_size=30,
                      hand_sizes=(1, 5, 5), game=1, num_games=3, turn=4, num_rules=1, scores=(1, 0, 0), log=log)
    v = view_json(view)
    marks = {card: m["mark"] for card, m in v["pile_marks"].items()}
    assert marks == {"9H": "upheld", "KS": "rejected", "5D": "upheld", "5H": "upheld"}
    assert "P3 under R1 (upheld)" in v["pile_marks"]["9H"]["text"]
    assert v["last_play"]["mark"] is None and "3C" in v["last_play"]["text"]
    # After a reshuffle only the top card keeps its mark.
    view2 = PlayerView(**{**view.__dict__, "log": log + (PublicEvent("reshuffle", 1, 4, card=c("3C"), detail="5"),),
                          "discard": (c("3C"),)})
    assert view_json(view2)["pile_marks"] == {}

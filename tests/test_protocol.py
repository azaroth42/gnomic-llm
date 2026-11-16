import re

from gnomic.cards import Card
from gnomic.match import Match, MatchConfig
from gnomic.players.llm import LLMPlayer
from gnomic.protocol import parse_call, parse_play, render_state, system_prompt, turn_request
from gnomic.providers import ChatSession, Completion

HAND = tuple(Card.parse(c) for c in ["3S", "9S", "4H", "KH", "10C"])


def test_parse_play_with_notes_and_self_call():
    text = "I think red on black is bad.\nNOTES:\n- R1 maybe colour\nPLAY: 9S\nCALL: NONE"
    r = parse_play(text, HAND)
    assert r.card == Card.parse("9S") and r.penalty is False
    assert r.notes == "- R1 maybe colour" and r.reasoning.startswith("I think")


def test_parse_play_variants():
    assert parse_play("**PLAY:** 10C", HAND).card == Card.parse("10C")
    assert parse_play("PLAY: K♥\nCALL: PENALTY", HAND, own=(1,)).penalty is True
    assert parse_play("PLAY: K♥\nCALL: PENALTY", HAND).penalty is False   # owns no rules: can't call
    assert parse_play("PLAY: 2D", HAND).error == "2D is not in your hand"
    assert parse_play("I'll go with 9S", HAND).error
    # the last PLAY line wins
    assert parse_play("PLAY: 3S\nactually no\nPLAY: 4H", HAND).card == Card.parse("4H")


def test_parse_call():
    assert parse_call("hmm\nCALL: PENALTY!", own=(1,)).rule == 1
    assert parse_call("NOTES:\nx\nCALL: NONE").penalty is False
    assert parse_call("Call: no call").penalty is False
    assert parse_call("no idea").error


class FakeSession(ChatSession):
    """Plays the first card in its hand, never calls, and writes unique secret notes."""
    provider = "fake"
    transcripts: dict = {}

    def __init__(self, model, **opts):
        super().__init__(model, **opts)
        self.messages = []
        FakeSession.transcripts[model] = self.messages

    def _reset(self):
        pass

    def send(self, text):
        self.messages.append(text)
        secret = f"SECRET-{self.model}-{len(self.messages)}"
        if "YOUR TURN" in text.splitlines()[-1]:
            hand = re.search(r"YOUR HAND: (.*)", text).group(1).split()
            return Completion(f"reasoning\nNOTES:\n{secret}\nPLAY: {hand[0]}\nCALL: NONE", input_tokens=10, output_tokens=5)
        return Completion(f"NOTES:\n{secret}\nCALL: NONE", input_tokens=10, output_tokens=5)


def test_llm_players_full_match_with_private_notes(monkeypatch):
    import gnomic.players.factory as factory
    monkeypatch.setitem(factory.PROVIDERS, "fake", FakeSession)
    FakeSession.transcripts.clear()
    m = Match(MatchConfig(players=["fake:alpha", "fake:beta", "fake:gamma"], num_games=2, seed=5, max_rounds=20))
    s = m.run()
    assert len(s["games"]) == 2
    owners = {e.owner for e in m.stack.entries}
    for p in m.players:
        assert isinstance(p, LLMPlayer) and p.reasoning_log
        # Only rule owners may call, so only they are ever asked to judge.
        if p.name not in owners:
            assert not any(e["kind"] == "call" for e in p.reasoning_log)
            assert not any(msg.splitlines()[-1].startswith("JUDGE") for msg in FakeSession.transcripts[p.session.model])
    first_dealer = next(p for p in m.players if p.name == m.stack.entries[0].owner)  # owns R1
    assert any(e["kind"] == "call" for e in first_dealer.reasoning_log)
    # The first message carries the full public log; one player's notes never reach another player.
    first = FakeSession.transcripts["alpha"][0]
    assert "RULE R1 ADDED" in first and "G1 START" in first
    for me, msgs in FakeSession.transcripts.items():
        for other in FakeSession.transcripts:
            if other != me:
                assert not any(f"SECRET-{other}-" in msg for msg in msgs)
    # Only game 1's dealer is shown the text of R1.
    dealer = first_dealer.session.model
    for name, msgs in FakeSession.transcripts.items():
        shown = any("YOUR RULES (secret): R1:" in msg for msg in msgs)
        assert shown == (name == dealer)


def test_history_window_resyncs(monkeypatch):
    import gnomic.players.factory as factory
    monkeypatch.setitem(factory.PROVIDERS, "fake", FakeSession)
    FakeSession.transcripts.clear()
    m = Match(MatchConfig(players=["fake:a1?history_messages=5", "fake:b1"], num_games=1, seed=2, max_rounds=15))
    m.run()
    assert any(msg.startswith("RESYNC") for msg in FakeSession.transcripts["a1"])
    assert not any(msg.startswith("RESYNC") for msg in FakeSession.transcripts["b1"])


def test_turn_request_and_state_reflect_judging_rights():
    from gnomic.game import PlayerView
    base = dict(seat=1, player_names=("a", "b", "c"), hand=HAND, discard=(Card.parse("5S"),), deck_size=30,
                hand_sizes=(5, 5, 5), game=0, num_games=3, turn=4, num_rules=1, scores=(0, 0, 0), log=(),
                max_rounds=20, judges=(0,))
    no_rules = PlayerView(**base)
    owner = PlayerView(**{**base, "seat": 0, "my_rules": ((1, "x"),)})
    assert "CALL" in turn_request(owner) and "no CALL line" in turn_request(no_rules)
    state = render_state(no_rules)
    assert "round 2/20" in state and "JUDGES: P1" in state
    assert "after 20 rounds" in system_prompt(no_rules)


def test_penalty_calls_name_one_of_your_rules():
    assert parse_call("CALL: PENALTY R3", own=(1, 3)).rule == 3
    assert parse_call("**CALL:** PENALTY under my rule R1", own=(1, 3)).rule == 1
    assert parse_call("CALL: PENALTY (rule 3)", own=(3,)).rule == 3
    assert parse_call("CALL: PENALTY", own=(2,)).rule == 2            # only one rule: implied
    r = parse_call("CALL: PENALTY", own=(1, 3))
    assert r.penalty and r.error and "name the rule" in r.error   # several rules: must name one
    r = parse_call("CALL: PENALTY R2", own=(1, 3))
    assert r.error and "not one of your rules" in r.error
    assert parse_call("CALL: NONE", own=(1, 3)).penalty is False
    # a self-call naming someone else's rule is sent back, not silently dropped
    r = parse_play("PLAY: 9S\nCALL: PENALTY R2", HAND, own=(1,))
    assert r.card == Card.parse("9S") and r.error and "self-call" in r.error
    r = parse_play("PLAY: 9S\nCALL: PENALTY R1", HAND, own=(1,))
    assert r.error is None and r.penalty and r.rule == 1

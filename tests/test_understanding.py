"""Rule hypotheses: parsing, grading by code-and-compare, understanding metrics, and the coder cache."""

import random
import re

import pytest

from gnomic.cards import Card
from gnomic.coder import CachedRuleCoder, CoderCache, CoderFailure, RuleCoder
from gnomic.match import Match, MatchConfig
from gnomic.protocol import parse_call, parse_hypotheses, parse_play, render_hypotheses
from gnomic.providers import ChatSession, Completion
from gnomic.rules import Rule, RuleEntry, sample_contexts
from gnomic.sandbox import CodeRule
from gnomic.understanding import UNDERSTOOD, _outcomes, agreement

RED_ON_BLACK = "def applies(ctx):\n    return ctx.top is not None and ctx.card.is_red and not ctx.top.is_red\n"
RED_ON_RED = "def applies(ctx):\n    return ctx.top is not None and ctx.card.is_red and ctx.top.is_red\n"


def test_parse_hypotheses_and_notes():
    text = ("thinking\nNOTES:\nmy notes\nHYPOTHESES:\nR2: no red on red\n  unless it is a King\n**R3:** ?\n"
            "R4 - Queens are wild.\nPLAY: 8H\nCALL: NONE")
    assert parse_hypotheses(text) == {2: "no red on red unless it is a King", 3: "", 4: "Queens are wild."}
    r = parse_play(text, (Card.parse("8H"),), (1,))
    assert r.card == Card.parse("8H") and r.notes == "my notes" and r.reasoning == "thinking"
    c = parse_call("HYPOTHESES:\nR1: x\nNOTES:\nnn\nCALL: NONE")
    assert c.penalty is False and c.notes == "nn"
    assert parse_hypotheses("NOTES:\nR1: in my notes\nPLAY: 8H") == {}


def test_agreement_scores():
    samples = sample_contexts(random.Random(2), 120)
    truth = _outcomes(Rule("no_red_on_black"), "restriction", samples, [])
    same = CodeRule(RED_ON_BLACK, "restriction", "same")
    assert agreement(truth, _outcomes(same, "restriction", samples, [])) == 1.0
    other = CodeRule(RED_ON_RED, "restriction", "other")
    assert agreement(truth, _outcomes(other, "restriction", samples, [])) < 0.2
    never = CodeRule("def applies(ctx):\n    return False", "restriction", "never")
    assert agreement(truth, _outcomes(never, "restriction", samples, [])) == 0.0
    as_permission = CodeRule(RED_ON_BLACK, "permission", "wrong kind")
    assert agreement(truth, _outcomes(as_permission, "restriction", samples, [])) == 0.0


# ---------------------------------------------------------------- a match with fakes

class HypPlayer(ChatSession):
    """Plays its first card. Guesses wrong at first, then states the rule from its 4th reply on."""
    provider = "hyp"
    seen: dict = {}

    def __init__(self, model, **opts):
        super().__init__(model, **opts)
        self.n = 0
        HypPlayer.seen[model] = []

    def _reset(self):
        pass

    def send(self, text):
        HypPlayer.seen[self.model].append(text)
        self.n += 1
        guess = "No red card on a red card." if self.n < 4 else "No red card on a black card."
        hyp = f"HYPOTHESES:\nR1: {guess}\n"
        if "YOUR TURN" in text.splitlines()[-1]:
            hand = re.search(r"YOUR HAND: (.*)", text).group(1).split()
            return Completion(f"NOTES:\nn\n{hyp}PLAY: {hand[0]}\nCALL: NONE")
        return Completion(f"NOTES:\nn\n{hyp}CALL: NONE")


class HypCoder(ChatSession):
    """Codes the two known texts; CANNOT otherwise."""
    provider = "hypcoder"
    calls: list = []

    def _reset(self):
        pass

    def send(self, text):
        HypCoder.calls.append(text)
        code = {"RULE: No red card on a black card.": RED_ON_BLACK,
                "RULE: No red card on a red card.": RED_ON_RED}.get(text.strip())
        if code is None:
            return Completion("CANNOT: unclear")
        return Completion(f"KIND: restriction\nDESCRIPTION: {text[6:]}\n```python\n{code}```")


class FixedAuthor:
    def author_rule(self, stack, owner, after_game):
        return RuleEntry(Rule("no_red_on_black"), owner, after_game)


@pytest.fixture
def hyp_fakes(monkeypatch):
    import gnomic.players.factory as factory
    monkeypatch.setitem(factory.PROVIDERS, "hyp", HypPlayer)
    monkeypatch.setitem(factory.PROVIDERS, "hypcoder", HypCoder)
    HypCoder.calls.clear()


def test_hypotheses_are_graded_and_scored(hyp_fakes):
    events = []
    m = Match(MatchConfig(players=["hyp:a", "hyp:b", "hyp:c", "stats"], num_games=1, seed=5, dealer=0, max_rounds=6,
                          grade_hypotheses=True, coder="hypcoder:x"),
              emit=lambda kind, **kw: events.append((kind, kw)), author=FixedAuthor())
    s = m.run()
    und = s["understanding"]
    # The owner of R1 (the dealer, a) is never scored on it, and the stats baseline states nothing.
    assert und["players"]["a"]["rules_to_learn"] == 0 and "stats" not in und["players"]
    for name in ("b", "c"):
        u = und["players"][name]
        r1 = u["rules"]["R1"]
        assert r1["understood"] and r1["final_score"] == 1.0 and 0 < r1["turns_to_understand"] < r1["turns_in_effect"]
        assert 0 < r1["mean_score"] < 1 and u["rules_understood"] == 1
    row = next(p for p in s["players"] if p["name"] == "b")
    assert row["rules_understood"] == 1
    # Two distinct texts, stated many times by two players: the coder was asked twice in all.
    assert len(HypCoder.calls) == 2
    assert s["coder_cache"]["coder_calls"] == 2
    graded = [kw for k, kw in events if k == "hypothesis_graded"]
    assert sorted(g["score"] for g in graded)[-1] == 1.0 and sorted(g["score"] for g in graded)[0] < UNDERSTOOD
    assert all(g["error"] is None for g in graded) and len(graded) == 2
    # The owner's own hypothesis about R1 was dropped before it reached the engine.
    assert not any(kw["player"] == "a" for k, kw in events if k == "hypothesis")
    # Players see their current hypotheses in STATE; the owner has none to state.
    assert any("YOUR HYPOTHESES: R1: No red card on a black card." in msg for msg in HypPlayer.seen["b"])
    assert not any("YOUR HYPOTHESES" in msg for msg in HypPlayer.seen["a"])


def test_render_hypotheses_skips_own_rules():
    from gnomic.game import PlayerView
    v = PlayerView(seat=0, player_names=("a", "b"), hand=(), discard=(), deck_size=0, hand_sizes=(0, 0), game=0,
                   num_games=1, turn=0, num_rules=3, scores=(0, 0), log=(), my_rules=((2, "mine"),))
    assert render_hypotheses(v, {1: "x"}) == "YOUR HYPOTHESES: R1: x | R3: ?"


# ---------------------------------------------------------------- the cache

class CountingSession(ChatSession):
    provider = "count"

    def __init__(self, model="m", replies=None, **opts):
        super().__init__(model, **opts)
        self.replies = replies or {}
        self.calls = 0

    def _reset(self):
        pass

    def send(self, text):
        self.calls += 1
        reply = self.replies.get(text.strip())
        if isinstance(reply, Exception):
            raise reply
        return Completion(reply or "CANNOT: unclear")


def test_coder_cache_codes_each_text_once_and_persists(tmp_path):
    path = tmp_path / "cache.jsonl"
    samples = sample_contexts(random.Random(0), 20)
    ok = f"KIND: restriction\nDESCRIPTION: No red on black.\n```python\n{RED_ON_BLACK}```"
    session = CountingSession(replies={"RULE: No red on black.": ok})
    coder = CachedRuleCoder(RuleCoder(session), CoderCache(path))
    first = coder.implement("No red on black.", samples)
    again = coder.implement("No  red on black", samples)          # same text after normalizing
    assert session.calls == 1 and again.meta["cached"] and again.rule.code == first.rule.code
    assert again.rule.proposal == "No  red on black"                 # keeps the caller's own wording
    with pytest.raises(CoderFailure):
        coder.implement("gibberish", samples)
    with pytest.raises(CoderFailure):
        coder.implement("gibberish", samples)                      # CANNOT is remembered too
    assert session.calls == 2
    # A new match (new cache object, same file) asks the coder for nothing it has seen.
    session2 = CountingSession(replies={})
    coder2 = CachedRuleCoder(RuleCoder(session2), CoderCache(path))
    assert coder2.implement("No red on black.", samples).rule.code == first.rule.code
    with pytest.raises(CoderFailure):
        coder2.implement("gibberish", samples)
    assert session2.calls == 0
    # Different allowances are a different question.
    with pytest.raises(CoderFailure):
        coder2.implement("No red on black.", samples, mechanics=True)
    assert session2.calls == 1


def test_coder_cache_does_not_keep_api_failures(tmp_path):
    samples = sample_contexts(random.Random(0), 10)
    session = CountingSession(replies={"RULE: x": RuntimeError("overloaded")})
    coder = CachedRuleCoder(RuleCoder(session, max_fixes=0), CoderCache(tmp_path / "c.jsonl"))
    for _ in range(2):
        with pytest.raises(CoderFailure):
            coder.implement("x", samples)
    assert session.calls == 2 and not (tmp_path / "c.jsonl").exists()

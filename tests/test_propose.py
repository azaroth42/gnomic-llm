"""--propose-rules: players make up rules, a coding agent implements them in the sandbox."""

import random
import re

import pytest

from gnomic.cards import Card
from gnomic.coder import RuleCoder, parse_coder_reply
from gnomic.match import Match, MatchConfig
from gnomic.protocol import parse_rule
from gnomic.providers import ChatSession, Completion
from gnomic.rules import PlayContext, Rule, Verdict, rule_from_json, rule_problem, sample_contexts
from gnomic.sandbox import CodeRule, SandboxError, compile_rule, smoke_test

RED_ON_BLACK = "def applies(ctx):\n    return ctx.top is not None and ctx.card.is_red and not ctx.top.is_red\n"


def ctx(card, *pile):
    return PlayContext(Card.parse(card), tuple(Card.parse(c) for c in pile), 0, (5, 5), 3, 2)


def test_code_rule_matches_library_rule_and_round_trips():
    coded = CodeRule(RED_ON_BLACK, "restriction", "No red on black.", proposal="no red on black please")
    lib = Rule("no_red_on_black")
    for c, cands in sample_contexts(random.Random(1), 40):
        for card in cands:
            cc = PlayContext(card, c.pile, c.player, c.hand_sizes, c.turn, c.num_players)
            assert coded.verdict(cc) == lib.verdict(cc)
    again = rule_from_json(coded.to_json())
    assert isinstance(again, CodeRule) and again.key() == coded.key() and again.proposal == coded.proposal
    assert isinstance(rule_from_json(lib.to_json()), Rule)


@pytest.mark.parametrize("code, why", [
    ("import os\ndef applies(ctx):\n    return False", "exactly one function"),
    ("def applies(ctx):\n    return ctx.card.__class__ is None", "not allowed"),
    ("def applies(ctx):\n    return any(x for x in ctx.pile).gi_frame", "not allowed"),
    ("def applies(ctx):\n    while True:\n        pass", "While"),
    ("def applies(ctx):\n    return ctx.card.rank ** 2 > 50", "Pow"),
    ("def applies(ctx):\n    return applies(ctx)", "recursion"),
    ("def applies(ctx):\n    return open('x') is None", None),             # not a builtin here: NameError at runtime
    ("def applies(ctx):\n    f = lambda: 1\n    return False", "Lambda"),
    ("def applies(ctx):\n    return '{0.__class__}'.format(ctx)", "not allowed"),
    ("def applies(ctx, x=[]):\n    return False", "exactly 1 argument"),
    ("def helper(c):\n    return 1\ndef applies(ctx):\n    return False", "exactly one function"),
])
def test_sandbox_rejects_unsafe_code(code, why):
    with pytest.raises(SandboxError) as e:
        smoke_test(compile_rule(code), sample_contexts(random.Random(0), 3))
    if why:
        assert why in str(e.value)


def test_sandbox_runtime_limits():
    rep = compile_rule("def applies(ctx):\n    s = 'x' * 1000\n    return len(s) > 1")
    with pytest.raises(TypeError):
        rep(ctx("5H", "6S"))                   # arithmetic only on numbers: no string repetition
    big = compile_rule("def applies(ctx):\n    return len(range(1000)) > 0")
    with pytest.raises(ValueError):
        big(ctx("5H", "6S"))
    ok = compile_rule("def applies(ctx):\n    t = 0\n    for c in ctx.pile:\n        t += c.rank\n    return t + ctx.card.rank > 20")
    assert ok(ctx("KH", "9S")) is True and ok(ctx("2H", "9S")) is False
    with pytest.raises(SandboxError, match="not True/False"):
        smoke_test(compile_rule("def applies(ctx):\n    return ctx.card.rank"), sample_contexts(random.Random(0), 2))
    # a rule that crashes during play abstains instead of breaking the match
    r = CodeRule("def applies(ctx):\n    return ctx.pile[5].rank > 3", "restriction", "x")
    assert r.verdict(ctx("5H", "6S")) is Verdict.ABSTAIN and r.errors == 1


def test_parsers():
    r = parse_coder_reply("thinking...\nKIND: restriction\nDESCRIPTION: No red on black.\n```python\n" + RED_ON_BLACK + "```")
    assert r.kind == "restriction" and r.description == "No red on black." and "def applies" in r.code
    assert parse_coder_reply("CANNOT: it depends on hidden hands").cannot == "it depends on hidden hands"
    assert "KIND" in parse_coder_reply("```python\n" + RED_ON_BLACK + "```").error
    p = parse_rule("hmm\nNOTES:\nmy notes\n**RULE:** You may not play a red card\non a black card.")
    assert p.rule_text == "You may not play a red card on a black card." and p.notes == "my notes"
    assert parse_rule("no rule here").error


def test_rule_problem_reasons():
    samples = sample_contexts(random.Random(3), 100)
    perm = CodeRule("def applies(ctx):\n    return ctx.card.rank == 7", "permission", "Sevens are wild.")
    assert "first rule must be a restriction" in rule_problem([], perm, samples)
    nothing = CodeRule("def applies(ctx):\n    return False", "restriction", "Nothing.")
    assert "almost never change" in rule_problem([], nothing, samples)
    everything = CodeRule("def applies(ctx):\n    return True", "restriction", "Nothing is legal.")
    assert "too few cards legal" in rule_problem([], everything, samples)
    good = CodeRule(RED_ON_BLACK, "restriction", "No red on black.")
    assert rule_problem([], good, samples) is None
    assert "same as a rule already in effect" in rule_problem([good], CodeRule(RED_ON_BLACK, "restriction", "again"),
                                                              samples)
    same_effect = "def applies(ctx):\n    return ctx.top is not None and ctx.top.color == 'black' and ctx.card.is_red\n"
    assert "almost never change" in rule_problem([good], CodeRule(same_effect, "restriction", "again"), samples)


# ---------------------------------------------------------------- full match with fakes

class FakePlayer(ChatSession):
    """Plays its first card, never calls; proposes rules from a per-player script."""
    provider = "fakep"
    transcripts: dict = {}
    proposals = ["Players must hand me their cards.", "No red card on a black card."]

    def __init__(self, model, **opts):
        super().__init__(model, **opts)
        self.messages, self.n_rules = [], 0
        FakePlayer.transcripts[model] = self.messages

    def _reset(self):
        pass

    def send(self, text):
        self.messages.append(text)
        if "MAKE A RULE" in text:
            rule = self.proposals[min(self.n_rules, len(self.proposals) - 1)]
            self.n_rules += 1
            return Completion(f"NOTES:\nsecret-{self.model}\nRULE: {rule}", input_tokens=3, output_tokens=2)
        if "YOUR TURN" in text.splitlines()[-1]:
            hand = re.search(r"YOUR HAND: (.*)", text).group(1).split()
            return Completion(f"NOTES:\nn\nPLAY: {hand[0]}\nCALL: NONE")
        return Completion("NOTES:\nn\nCALL: NONE")


class FakeCoder(ChatSession):
    """CANNOT for hidden information; for 'red ... black' first sends broken code, then fixes it."""
    provider = "fakecoder"
    log: list = []

    def __init__(self, model, **opts):
        super().__init__(model, **opts)

    def _reset(self):
        self.turn = 0

    def send(self, text):
        FakeCoder.log.append(text)
        self.turn += 1
        if "hand me" in text:
            return Completion("CANNOT: it depends on other players' hidden cards.")
        if self.turn == 1:
            return Completion("KIND: restriction\nDESCRIPTION: x\n```python\ndef applies(ctx):\n    return ctx.card.rank\n```")
        return Completion("KIND: restriction\nDESCRIPTION: You may not play a red card on a black card.\n"
                          "```python\n" + RED_ON_BLACK + "```")


@pytest.fixture
def fakes(monkeypatch):
    import gnomic.players.factory as factory
    monkeypatch.setitem(factory.PROVIDERS, "fakep", FakePlayer)
    monkeypatch.setitem(factory.PROVIDERS, "fakecoder", FakeCoder)
    FakePlayer.transcripts.clear()
    FakeCoder.log.clear()


def test_dealer_proposes_rule_and_coder_implements_it(fakes):
    events = []
    m = Match(MatchConfig(players=["fakep:a", "fakep:b", "stats"], num_games=1, seed=4, dealer=0,
                          propose_rules=True, coder="fakecoder:c"),
              emit=lambda kind, **kw: events.append((kind, kw)))
    m.run()
    r1 = m.stack.entries[0]
    assert isinstance(r1.rule, CodeRule) and r1.owner == "a"
    assert r1.rule.proposal == "No red card on a black card."
    kinds = [k for k, _ in events]
    assert kinds.index("rule_rejected") < kinds.index("rule_coded") < kinds.index("rule_added")
    rejected = next(kw for k, kw in events if k == "rule_rejected")
    assert rejected["stage"] == "coder" and "hidden cards" in rejected["reason"]
    # the coder was shown its own error and fixed the code
    assert any(msg.startswith("ERROR:") for msg in FakeCoder.log)
    # the dealer was told why the first rule was rejected, and sees the implemented wording
    a = FakePlayer.transcripts["a"]
    assert "MAKE A RULE" in a[0] and "MAKING UP A RULE" not in a[0]  # system prompt is not in messages
    assert any("REJECTED:" in msg and "hidden cards" in msg for msg in a)
    assert any("YOUR RULES (secret): R1: You may not play a red card on a black card." in msg for msg in a)
    # nobody else sees the proposal, the rejection, or the rule text
    for msg in FakePlayer.transcripts["b"]:
        assert "red card on a black" not in msg and "hand me" not in msg and "secret-a" not in msg
    assert m.players[0].reasoning_log[0]["kind"] == "rule"


def test_falls_back_to_library_rule(fakes, monkeypatch):
    monkeypatch.setattr(FakePlayer, "proposals", ["Players must hand me their cards."])
    events = []
    m = Match(MatchConfig(players=["fakep:a", "fakep:b"], num_games=1, seed=4, dealer=0, propose_rules=True,
                          coder="fakecoder:c", rule_attempts=2),
              emit=lambda kind, **kw: events.append((kind, kw)))
    m.run()
    assert isinstance(m.stack.entries[0].rule, Rule)
    assert [k for k, _ in events].count("rule_rejected") == 2 and any(k == "rule_fallback" for k, _ in events)


def test_baseline_dealer_gets_library_rule(fakes):
    m = Match(MatchConfig(players=["stats", "fakep:a"], num_games=1, seed=4, dealer=0, propose_rules=True,
                          coder="fakecoder:c"))
    m.run()
    assert isinstance(m.stack.entries[0].rule, Rule) and not FakeCoder.log


class MechCoder(ChatSession):
    """Implements any rule; after game 2 (TABLE MECHANICS in its prompt) it writes an effect."""
    provider = "mechcoder"
    systems: list = []

    def _reset(self):
        MechCoder.systems.append(self.system)

    def send(self, text):
        if "TABLE MECHANICS (this rule may use them)" in self.system:
            return Completion("KIND: effect\nDESCRIPTION: Playing a King makes the next player draw 1.\n"
                              "```python\ndef effect(ctx, fx):\n    if ctx.card.rank == 13:\n"
                              "        fx.draw(ctx.next_player, 1)\n```")
        return Completion("KIND: restriction\nDESCRIPTION: You may not play a red card on a black card.\n"
                          "```python\n" + RED_ON_BLACK + "```")


class OptionPlayer(FakePlayer):
    """Like FakePlayer, but plays at angle 90 once table mechanics are on, and proposes many rules."""
    provider = "optp"
    proposals = ["No red card on a black card.", "Kings make the next player draw.", "Kings make the next player draw."]

    def send(self, text):
        comp = super().send(text)
        if "YOUR TURN" in text.splitlines()[-1] and "\nTOKENS:" in text:
            comp.text = comp.text.replace("\nCALL: NONE", "\nANGLE: 90\nCALL: NONE")
        return comp


def test_mechanics_with_proposed_rules(fakes, monkeypatch):
    import gnomic.players.factory as factory
    monkeypatch.setitem(factory.PROVIDERS, "optp", OptionPlayer)
    monkeypatch.setitem(factory.PROVIDERS, "mechcoder", MechCoder)
    MechCoder.systems.clear()
    events = []
    m = Match(MatchConfig(players=["optp:a", "optp:b"], num_games=4, seed=11, dealer=0, max_rounds=30,
                          propose_rules=True, coder="mechcoder:c"),
              emit=lambda kind, **kw: events.append((kind, kw)))
    m.run()
    added = [kw for k, kw in events if k == "rule_added"]
    # only rules that start in game 3 or later may use mechanics
    coded = [kw for k, kw in events if (k == "rule_coded" or (k == "rule_rejected" and kw["stage"] != "proposal"))
             and not kw["coder"].get("cached")]  # a text coded before comes from the cache, not the coder
    assert len(coded) == len(MechCoder.systems)
    for kw, system in zip(coded, MechCoder.systems):
        assert ("TABLE MECHANICS (this rule may use them)" in system) == (kw["after_game"] >= 1)
    assert any(kw["rule"]["kind"] == "effect" for kw in added if kw["after_game"] >= 1)
    # options were only played (and accepted) in games 3+
    plays = [kw for k, kw in events if k == "play"]
    assert any(kw["move"] and kw["game"] >= 2 for kw in plays)
    assert not any(kw["move"] for kw in plays if kw["game"] < 2)
    msgs = FakePlayer.transcripts["a"]
    assert any("TABLE MECHANICS" in msg and "MAKE A RULE" in msg for msg in msgs) or len(added) < 3


class WinCoder(ChatSession):
    """Writes a win condition when allowed (game 6+), else a fresh restriction each time."""
    provider = "wincoder"
    systems: list = []
    restrictions = ["ctx.card.rank == 7", "ctx.card.rank == 9", "ctx.card.rank == 3", "ctx.card.rank == 5",
                    "ctx.card.rank == 2", "ctx.card.rank == 4", "ctx.card.rank == 6", "ctx.card.rank == 8"]

    def _reset(self):
        WinCoder.systems.append(self.system)

    def send(self, text):
        if "WIN CONDITIONS (this rule may be one)" in self.system:
            return Completion("KIND: win\nDESCRIPTION: You also win if you end your turn holding one card of the "
                              "top card's suit.\n```python\ndef wins(ctx):\n    return len(ctx.hand) == 1 and "
                              "ctx.top is not None and ctx.hand[0].suit == ctx.top.suit\n```")
        cond = self.restrictions[(len(WinCoder.systems) - 1) % len(self.restrictions)]
        return Completion(f"KIND: restriction\nDESCRIPTION: No {cond}.\n```python\ndef applies(ctx):\n"
                          f"    return {cond}\n```")


def test_win_conditions_only_from_game_6_with_proposed_rules(fakes, monkeypatch):
    import gnomic.players.factory as factory
    monkeypatch.setitem(factory.PROVIDERS, "wincoder", WinCoder)
    WinCoder.systems.clear()
    events = []
    m = Match(MatchConfig(players=["fakep:a", "fakep:b"], num_games=7, seed=3, dealer=0, max_rounds=40,
                          propose_rules=True, coder="wincoder:c"),
              emit=lambda kind, **kw: events.append((kind, kw)))
    m.run()
    coded = [kw for k, kw in events if (k == "rule_coded" or (k == "rule_rejected" and kw["stage"] != "proposal"))
             and not kw["coder"].get("cached")]  # a text coded before comes from the cache, not the coder
    assert len(coded) == len(WinCoder.systems)
    for kw, system in zip(coded, WinCoder.systems):
        assert ("WIN CONDITIONS (this rule may be one)" in system) == (kw["after_game"] + 2 >= 6)
    added = [kw for k, kw in events if k == "rule_added"]
    assert all(kw["rule"]["kind"] != "win" for kw in added if kw["after_game"] + 2 < 6)
    assert any(kw["rule"]["kind"] == "win" for kw in added) or not any(kw["after_game"] + 2 >= 6 for kw in added)
    codes = [kw["rule"].get("code") for kw in added if kw["rule"].get("code")]
    assert len(codes) == len(set(codes))                      # an identical rule is rejected, not added twice
    msgs = FakePlayer.transcripts["a"]
    assert any("It may be a WIN CONDITION" in msg for msg in msgs) == any(
        kw["after_game"] + 2 >= 6 and kw["owner"] == "a" for kw in added)

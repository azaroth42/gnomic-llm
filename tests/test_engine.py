import random

import pytest

from gnomic.cards import Card
from gnomic.game import Game, PublicLog
from gnomic.match import Match, MatchConfig
from gnomic.players.base import Call, Choice, Player
from gnomic.rules import (PlayContext, Rule, RuleEntry, RuleStack, TEMPLATES, Verdict, _mean_legal,
                          generate_rule, sample_contexts)

C = Card.parse


def ctx(card, pile, player=0, sizes=(5, 5), turn=0):
    return PlayContext(C(card), tuple(C(c) for c in pile), player, sizes, turn, len(sizes))


def test_card_parsing():
    assert C("10h") == Card(10, "H")
    assert C("Q♠") == Card(12, "S")
    assert C("ace of clubs") == Card(1, "C")
    assert str(Card(11, "D")) == "JD"
    with pytest.raises(ValueError):
        C("11H")


def test_precedence_permission_overrides_older_restriction():
    stack = RuleStack([RuleEntry(Rule("no_red_on_black"), "a", -1)])
    assert not stack.is_legal(ctx("8H", ["5S"]))
    stack.add(RuleEntry(Rule("wild_rank", {"r": 8}), "b", 0))
    assert stack.is_legal(ctx("8H", ["5S"]))        # newer permission wins
    assert not stack.is_legal(ctx("9H", ["5S"]))    # older restriction still applies otherwise


def test_precedence_restriction_overrides_older_permission():
    stack = RuleStack([RuleEntry(Rule("no_same_suit"), "a", -1),
                       RuleEntry(Rule("wild_suit", {"s": "H"}), "b", 0)])
    assert stack.is_legal(ctx("8H", ["5H"]))
    stack.add(RuleEntry(Rule("not_lower"), "c", 1))
    assert not stack.is_legal(ctx("2H", ["5H"]))    # newest restriction beats the permission
    assert stack.is_legal(ctx("8H", ["5H"]))


def test_restrictions_stack():
    stack = RuleStack([RuleEntry(Rule("not_lower"), "a", -1), RuleEntry(Rule("no_same_suit"), "b", 0)])
    assert not stack.is_legal(ctx("2C", ["5H"]))
    assert not stack.is_legal(ctx("9H", ["5H"]))
    assert stack.is_legal(ctx("9C", ["5H"]))


def test_first_rule_is_always_a_restriction():
    for seed in range(40):
        rule = generate_rule([], random.Random(seed))
        assert rule.kind == "restriction", rule


def test_generated_stacks_stay_playable():
    rng = random.Random(3)
    rules = []
    for _ in range(6):
        rules.append(generate_rule(rules, rng))
    assert len({r.key() for r in rules}) == 6


def test_rule_json_roundtrip():
    r = Rule("max_distance", {"n": 3})
    assert Rule.from_json(r.to_json()).description == r.description
    assert all(t.kind in ("restriction", "permission", "win") or (t.kind == "effect" and t.mechanics)
               for t in TEMPLATES.values())


class Scripted(Player):
    """Plays cards from a script and calls according to a function. `calls` may return
    True/False or a rule number (a penalty naming that rule)."""
    def __init__(self, name, plays=(), calls=lambda view, play: False, self_call=False):
        super().__init__(name)
        self.plays = list(plays)
        self.calls = calls
        self.self_call = self_call

    def choose(self, view):
        card = self.plays.pop(0) if self.plays else view.hand[0]
        rule = self.self_call if type(self.self_call) is int else None
        return Choice(C(card) if isinstance(card, str) else card, self_call=bool(self.self_call),
                      self_call_rule=rule)

    def call(self, view, play):
        c = self.calls(view, play)
        return Call(bool(c), rule=c if type(c) is int else None)


class NullRule:
    """A rule that never decides anything; owning it makes a player a judge."""
    kind = "restriction"
    description = "(no effect)"

    def verdict(self, ctx):
        return Verdict.ABSTAIN

    def key(self):
        return ("null",)

    def to_json(self):
        return {"template": "null", "params": {}, "kind": self.kind, "description": self.description}


def make_game(players, rule="no_red_on_black", hands=None, top="5S", deck=None, judges=None, max_turns=1,
              owner=0):
    """players[owner] owns `rule`. Every other seat in `judges` (default: all) owns a rule that
    never fires, so may call (but can never be upheld)."""
    stack = RuleStack([RuleEntry(Rule(rule), players[owner].name, -1)])
    for s in (range(len(players)) if judges is None else judges):
        if s != owner:
            stack.add(RuleEntry(NullRule(), players[s].name, 0))
    events = []
    g = Game(0, 1, players, stack, PublicLog(), [0] * len(players), random.Random(0), 0,
             emit=lambda t, **d: events.append((t, d)), max_turns=max_turns)
    g._deal = lambda: None
    g.hands = [[C(c) for c in h] for h in hands]
    g.discard = [C(top)]
    g.deck = [C(c) for c in (["2C", "3C", "4C"] if deck is None else deck)]
    return g, events


def test_called_violation_makes_player_draw():
    a = Scripted("a", ["8H"])
    b = Scripted("b", calls=lambda v, p: True)
    g, _ = make_game([a, b], hands=[["8H", "9S"], ["2D"]], owner=1)
    g.play()
    assert len(g.hands[0]) == 2 and len(g.hands[1]) == 1
    ev = g.log.events[-2]
    assert ev.kind == "play" and ev.upheld is True and ev.penalized == (0,)


def test_false_call_makes_caller_draw():
    a = Scripted("a", ["9S"])
    b = Scripted("b", calls=lambda v, p: True)
    g, _ = make_game([a, b], hands=[["8H", "9S"], ["2D"]])
    g.play()
    assert len(g.hands[0]) == 1 and len(g.hands[1]) == 2
    assert g.log.events[-2].upheld is False


def test_uncalled_violation_costs_nothing_and_card_stays():
    a = Scripted("a", ["8H"])
    b = Scripted("b")
    g, _ = make_game([a, b], hands=[["8H", "9S"], ["2D"]])
    g.play()
    assert len(g.hands[0]) == 1 and g.discard[-1] == C("8H")
    ev = g.log.events[-2]
    assert ev.calls == () and ev.upheld is None   # the violation is not revealed


def test_self_call_on_forced_illegal_play():
    a = Scripted("a", ["8H"], self_call=True)
    b = Scripted("b")
    g, _ = make_game([a, b], hands=[["8H", "9H"], ["2D"]])
    g.play()
    ev = g.log.events[-2]
    assert ev.calls == (0,) and ev.upheld and len(g.hands[0]) == 2


def test_last_card_called_illegal_does_not_win():
    a = Scripted("a", ["8H"])
    b = Scripted("b", calls=lambda v, p: True)
    g, _ = make_game([a, b], hands=[["8H"], ["2D", "3D"]], owner=1)
    result = g.play()
    assert result.winner is None and len(g.hands[0]) == 1


def test_last_card_legal_wins():
    a = Scripted("a", ["9S"])
    b = Scripted("b")
    g, _ = make_game([a, b], hands=[["9S"], ["2D", "3D"]])
    assert g.play().winner == 0


def test_reshuffle_keeps_top_card():
    a = Scripted("a", ["8H"])
    b = Scripted("b", calls=lambda v, p: True)
    g, events = make_game([a, b], hands=[["8H", "9S"], ["2D"]], deck=[], owner=1)
    g.discard = [C("KD"), C("QD"), C("5S")]
    g.play()
    # 8H went on top of 5S, then the draw found DECK empty: KD QD 5S reshuffled, 8H stays.
    assert g.discard == [C("8H")]
    assert len(g.hands[0]) == 2 and len(g.deck) == 2
    assert any(t == "reshuffle" for t, _ in events)


def test_callers_see_table_before_play():
    seen = {}
    def judge(view, play):
        seen["top"] = view.top
        seen["sizes"] = view.hand_sizes
        return False
    a = Scripted("a", ["9S"])
    b = Scripted("b", calls=judge)
    g, _ = make_game([a, b], hands=[["8H", "9S"], ["2D"]])
    g.play()
    assert seen["top"] == C("5S") and seen["sizes"] == (2, 1)


def test_owner_sees_own_rule_only():
    a, b = Scripted("a"), Scripted("b")
    g, _ = make_game([a, b], hands=[["8H"], ["2D"]], judges=[0])
    assert g.view_for(0).my_rules and not g.view_for(1).my_rules


def test_baseline_match_runs_and_oracle_judges_perfectly():
    m = Match(MatchConfig(players=["oracle", "stats", "random"], num_games=4, seed=11))
    s = m.run()
    oracle = s["players"][0]
    assert oracle["avoidable_illegal"] == 0
    assert oracle["judge_fp"] == 0 and oracle["judge_fn"] == 0
    assert len(s["games"]) == 4
    assert s["rules"][0]["rule"]["kind"] == "restriction"


def test_winner_deals_next_game():
    events = []
    m = Match(MatchConfig(players=["oracle", "stats", "random"], num_games=6, seed=5, dealer=2),
              emit=lambda kind, **kw: events.append((kind, kw)))
    m.run()
    starts = [kw for k, kw in events if k == "game_start"]
    dealer = 2
    for start, result in zip(starts, m.results):
        assert start["dealer"] == dealer
        assert start["starting_player"] == (dealer + 1) % 3
        if result.winner is not None:
            dealer = result.winner
    assert m.dealer == dealer
    assert any(r.winner not in (None, 2) for r in m.results)  # the dealer actually moved

def test_gender_and_arithmetic_rules():
    assert C("JS").is_male and C("KD").is_male and C("QH").is_female and C("10H").gender is None
    same = RuleStack([RuleEntry(Rule("no_same_gender"), "a", -1)])
    assert not same.is_legal(ctx("KH", ["JS"])) and same.is_legal(ctx("QH", ["JS"]))
    assert same.is_legal(ctx("5H", ["JS"]))
    prod = RuleStack([RuleEntry(Rule("product_limit", {"n": 40}), "a", -1),
                      RuleEntry(Rule("double_or_half_ok"), "b", 0)])
    assert prod.is_legal(ctx("5H", ["8S"])) and not prod.is_legal(ctx("6H", ["8S"]))
    assert prod.is_legal(ctx("KH", ["6S"]) ) is False and prod.is_legal(ctx("QH", ["6S"]))  # 12 = 2 x 6


def test_users_example_rules_can_be_generated():
    """'No lower on higher' and similar must be reachable by the generator."""
    seen = set()
    for seed in range(300):
        seen.add(generate_rule([], random.Random(seed)).template)
    assert {"not_lower", "not_higher", "no_red_on_black", "max_distance"} <= seen


def test_players_without_rules_are_not_asked_to_judge():
    asked = []
    a = Scripted("a", ["8H"])
    b = Scripted("b", calls=lambda v, p: asked.append("b") or True)
    c = Scripted("c", calls=lambda v, p: asked.append("c") or True)
    g, events = make_game([a, b, c], hands=[["8H", "9S"], ["2D"], ["3D"]], judges=[2], owner=2)
    g.play()
    assert asked == ["c"]
    ev = g.log.events[-2]
    assert ev.calls == (2,) and ev.upheld
    phase = next(d for t, d in events if t == "call_phase")
    assert phase["callers"] == ["c"]
    assert g.view_for(1).judges == (2,) and not g.view_for(1).can_call


def test_self_call_ignored_without_rules():
    a = Scripted("a", ["2D"], self_call=True)   # b plays first here: seat 1 owns no rule
    b = Scripted("b", ["8H"], self_call=True)
    g, _ = make_game([a, b], hands=[["2D", "3D"], ["8H", "9H"]], judges=[0])
    g.current = 1
    g.play()
    ev = g.log.events[-2]
    assert ev.player == 1 and ev.calls == () and ev.upheld is None and len(g.hands[1]) == 1


def test_game_ends_after_max_rounds():
    a, b, c = Scripted("a"), Scripted("b"), Scripted("c")
    hands = [["2C", "3C", "4C", "5C"], ["2D", "3D", "4D", "5D"], ["2H", "3H", "4H", "5H"]]
    g, events = make_game([a, b, c], rule="no_same_suit", hands=hands, deck=["6C", "6D", "6H"], max_turns=None)
    g.max_rounds = 1
    g.max_turns = 3
    result = g.play()
    assert result.winner is None and result.turns == 3 and result.end_reason == "max_rounds"
    # default: 20 rounds, i.e. every player plays 20 cards
    g2, _ = make_game([Scripted("x"), Scripted("y")], hands=[["2C"], ["2D"]], max_turns=None)
    assert g2.max_turns == 40


def test_permissions_are_generated_and_forced_when_stack_is_tight():
    rng = random.Random(7)
    kinds = []
    for seed in range(20):
        rules, r = [], random.Random(seed)
        for _ in range(5):
            rules.append(generate_rule(rules, r))
        kinds += [x.kind for x in rules[1:]]
    assert 0.25 < kinds.count("permission") / len(kinds) < 0.85
    tight = [Rule("not_lower"), Rule("no_same_suit"), Rule("alternate_parity")]
    assert _mean_legal(tight, sample_contexts(rng, 200)) < 0.4
    for seed in range(10):
        assert generate_rule(tight, random.Random(seed)).kind == "permission"


def test_face_on_face_permission():
    stack = RuleStack([RuleEntry(Rule("no_face_on_face"), "a", -1),
                       RuleEntry(Rule("face_on_face_ok"), "b", 0)])
    assert stack.is_legal(ctx("QH", ["KS"]))
    assert stack.is_legal(ctx("5H", ["KS"]))


def test_call_upheld_only_for_callers_own_rule():
    """b owns the broken rule, c owns another rule: c's call is a false call even though the play is illegal."""
    a = Scripted("a", ["8H"])
    b = Scripted("b", calls=lambda v, p: True)
    c = Scripted("c", calls=lambda v, p: True)
    g, _ = make_game([a, b, c], hands=[["8H", "9S"], ["2D"], ["3D"]], owner=1, judges=[1, 2])
    g.play()
    ev = g.log.events[-2]
    assert ev.calls == (1, 2) and ev.upheld and ev.upheld_by == (1,)
    assert ev.penalized == (0, 2) and [len(h) for h in g.hands] == [2, 1, 2]


def test_non_owner_call_on_illegal_play_is_rejected():
    a = Scripted("a", ["8H"])
    b = Scripted("b")
    c = Scripted("c", calls=lambda v, p: True)
    g, events = make_game([a, b, c], hands=[["8H", "9S"], ["2D"], ["3D"]], owner=1, judges=[1, 2])
    g.play()
    ev = g.log.events[-2]
    assert ev.upheld is False and ev.penalized == (2,) and len(g.hands[0]) == 1
    res = next(d for t, d in events if t == "resolution")
    assert res["missed"] and res["violated_owners"] == [1]


def test_violated_respects_precedence():
    # R1 (a) no red on black; R2 (b) hearts always ok; R3 (c) not lower.
    stack = RuleStack([RuleEntry(Rule("no_red_on_black"), "a", -1),
                       RuleEntry(Rule("wild_suit", {"s": "H"}), "b", 0),
                       RuleEntry(Rule("not_lower"), "c", 1)])
    # 2H on 5S: R3 forbids; R2 would allow it against R1, so R1's owner can't call it.
    assert stack.violated_owners(ctx("2H", ["5S"])) == {"c"}
    # 2D on 5S: R3 forbids, R2 abstains, R1 forbids: both owners can call.
    assert stack.violated_owners(ctx("2D", ["5S"])) == {"a", "c"}
    assert stack.violated_owners(ctx("9H", ["5S"])) == set()


def test_call_must_name_the_broken_rule():
    """b owns R1 (broken) and R3 (not broken). Naming R3 is a false call; naming R1 is upheld."""
    for named, upheld in ((3, False), (1, True)):
        a = Scripted("a", ["8H"])
        b = Scripted("b", calls=lambda v, p, n=named: n)
        g, _ = make_game([a, b], hands=[["8H", "9S"], ["2D"]], owner=1)
        g.stack.add(RuleEntry(Rule("no_same_suit"), "b", 0))
        g.play()
        ev = g.log.events[-2]
        assert ev.call_rules == (named,) and ev.upheld is upheld
        assert ev.penalized == ((0,) if upheld else (1,))


def test_call_naming_someone_elses_rule_is_rejected():
    a = Scripted("a", ["8H"])
    b = Scripted("b")
    c = Scripted("c", calls=lambda v, p: 1)     # R1 is b's rule, not c's
    g, _ = make_game([a, b, c], hands=[["8H", "9S"], ["2D"], ["3D"]], owner=1, judges=[1, 2])
    g.play()
    ev = g.log.events[-2]
    assert ev.upheld is False and ev.call_rules == (1,) and ev.penalized == (2,)


def test_public_log_names_the_rule():
    from gnomic.protocol import render_event
    a = Scripted("a", ["8H"])
    b = Scripted("b", calls=lambda v, p: 1)
    g, _ = make_game([a, b], hands=[["8H", "9S"], ["2D"]], owner=1)
    g.play()
    assert "PENALTY called by P2 under R1 (upheld): P1 draws 1" in render_event(g.log.events[-2])

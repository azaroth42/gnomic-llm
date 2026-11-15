"""Game-3+ table mechanics: moves with options, R0, effects, private notices."""

import random

import pytest

from gnomic.cards import Card
from gnomic.game import Game, PendingPlay, PlayerView, PublicLog
from gnomic.match import Match, MatchConfig
from gnomic.mechanics import BASE_RULE, Effects, Move, TableState
from gnomic.players.base import Call, Choice, Player
from gnomic.protocol import judge_request, parse_play, render_event, render_state, system_prompt
from gnomic.rules import (TEMPLATES, PlayContext, Rule, RuleEntry, RuleStack, generate_rule, rule_problem,
                          sample_contexts)
from gnomic.sandbox import CodeRule, SandboxError, compile_rule

C = Card.parse


class Mover(Player):
    """Plays scripted moves (Move objects or card strings); calls via a function."""
    def __init__(self, name, moves=(), calls=lambda view, play: False):
        super().__init__(name)
        self.moves, self.calls, self.views = list(moves), calls, []

    def choose(self, view):
        self.views.append(view)
        m = self.moves.pop(0) if self.moves else view.hand[0]
        if isinstance(m, str):
            m = C(m)
        if isinstance(m, Move):
            return Choice(m.card, move=m)
        return Choice(m)

    def call(self, view, play):
        c = self.calls(view, play)
        return Call(bool(c), rule=c if type(c) is int else None)


def game(players, rules, hands, top="5S", deck=("2C", "3C", "4C", "6C", "7C"), max_turns=1, owners=None):
    stack = RuleStack([RuleEntry(r, players[(owners or [0] * len(rules))[i]].name, i)
                       for i, r in enumerate(rules)])
    events = []
    g = Game(2, 3, players, stack, PublicLog(), [0] * len(players), random.Random(0), 0,
             emit=lambda t, **d: events.append((t, d)), max_turns=max_turns, mechanics=True, mechanics_from=3)
    g._deal = lambda: None
    g.hands = [[C(c) for c in h] for h in hands]
    g.discard = [C(top)]
    g.deck = [C(c) for c in deck]
    return g, events


def test_multi_card_play_breaks_r0_and_engine_calls_it():
    a, b = Mover("a", [Move((C("8H"), C("8D")))]), Mover("b")
    g, events = game([a, b], [Rule("reverse_on_rank", {"r": 8})], [["8H", "8D", "9S"], ["2D"]])
    g.play()
    ev = g.log.events[-2]
    assert ev.engine_call and ev.upheld and ev.penalized == (0,)
    assert len(g.hands[0]) == 2 and g.direction == 1          # drew 1; the effect was cancelled
    assert "the engine under R0 (upheld)" in render_event(ev) and "8H 8D" in render_event(ev)


def test_pair_permission_allows_multi_card_play_and_effects_fire():
    a, b, c = Mover("a", [Move((C("8H"), C("8D")))]), Mover("b"), Mover("c")
    rules = [Rule("pair_ok"), Rule("reverse_on_rank", {"r": 8}), Rule("token_on_face")]
    g, _ = game([a, b, c], rules, [["8H", "8D", "9S"], ["2D"], ["3D"]])
    g.play()
    ev = g.log.events[-2]
    assert not ev.engine_call and ev.upheld is None
    assert g.hands[0] == [C("9S")] and g.discard[-2:] == [C("8H"), C("8D")]
    assert g.direction == -1 and g.current == 2                # reversed: P1 -> P3
    assert ev.effects == ("order of play reversed",)


def test_skip_extra_turn_tokens_and_spaces():
    rules = [Rule("skip_on_rank", {"r": 9}), Rule("extra_turn_on_rank", {"r": 7}), Rule("token_on_face"),
             Rule("space_on_rank", {"r": 4})]
    a, b, c = Mover("a", ["9S", "7S", "KS", "4S"]), Mover("b", ["2D"]), Mover("c", ["3D"])
    g, _ = game([a, b, c], rules, [["9S", "7S", "KS", "4S", "AS"], ["2D", "2H"], ["3D", "3H"]], max_turns=5)
    g.play()
    seats = [e.player for e in g.log.events if e.kind == "play"]
    # a plays 9 (skips b) -> c; c plays; a plays 7 (extra turn) -> a plays K -> b plays -> ...
    assert seats[:5] == [0, 2, 0, 0, 1]
    assert g.tokens[0] == 1
    assert [str(p.card) for p in g.spaces[0]] == [] and C("KS") in g.discard


def test_effects_cancelled_when_penalty_upheld_but_spend_is_kept():
    owner_calls = lambda view, play: 1  # P2 owns R1 and always calls it
    a, b = Mover("a", [Move((C("8H"),), spend=1)]), Mover("b", calls=owner_calls)
    rules = [Rule("no_red_on_black"), Rule("reverse_on_rank", {"r": 8})]
    g, _ = game([a, b], rules, [["8H", "9S"], ["2D"]], owners=[1, 0])
    g.tokens = [2, 0]
    g.play()
    ev = g.log.events[-2]
    assert ev.upheld and g.direction == 1 and g.tokens == [1, 0] and not ev.effects


def test_trade_is_private_to_the_two_players():
    a = Mover("a", [Move((C("QS"),), target=1, give=C("2S"))])
    b, c = Mover("b"), Mover("c")
    g, _ = game([a, b, c], [Rule("trade_on_target", {"r": 12})], [["QS", "2S", "9S"], ["5D", "6D"], ["7D"]],
                max_turns=2)
    g.play()
    assert C("2S") in g.hands[1] and C("2S") not in g.hands[0] and len(g.hands[0]) == 2
    assert g.private[0] and g.private[1] and not g.private[2]
    assert "gave you 2S" in g.private[1][0]
    assert "traded a card with P2" in render_event(g.log.events[-3])
    # the next player is shown their private notice; the third player isn't
    assert b.views[-1].private and not c.views


def test_peek_effect_from_code_rule_and_view_state():
    code = "def effect(ctx, fx):\n    if ctx.card.is_face and ctx.target is not None:\n        fx.peek_hand(ctx.player, ctx.target)\n"
    rule = CodeRule(code, "effect", "Face card with a target: peek at their hand.", mechanics=True)
    a, b = Mover("a", [Move((C("JS"),), target=1, declare="Hearts")]), Mover("b")
    g, _ = game([a, b], [rule], [["JS", "9S"], ["5D"]], max_turns=2)
    g.play()
    assert g.private[0] == ["G3 T1 peek: P2 holds 5D"]
    assert g.declared == ("hearts", 0)
    state = render_state(b.views[-1])
    assert 'DECLARED: "hearts" by P1' in state and "TOKENS:" in state and "ORDER:" in state


def test_sandbox_gates_mechanics():
    with pytest.raises(SandboxError):
        compile_rule("def effect(ctx, fx):\n    fx.reverse()", "effect", mechanics=False)
    with pytest.raises(SandboxError, match="attribute .tokens"):
        compile_rule("def applies(ctx):\n    return ctx.tokens[0] > 1")
    compile_rule("def applies(ctx):\n    return ctx.tokens[ctx.player] > 1", mechanics=True)
    with pytest.raises(SandboxError, match="is not allowed"):
        compile_rule("def effect(ctx, fx):\n    fx.ops.append(1)", "effect", mechanics=True)


def test_effects_api_validates_arguments():
    fx = Effects(3)
    with pytest.raises(ValueError):
        fx.draw(5, 1)
    with pytest.raises(ValueError):
        fx.gain_tokens(0, 10)
    with pytest.raises(ValueError):
        fx.trade(1, 1)
    for _ in range(6):
        fx.skip()
    with pytest.raises(ValueError):
        fx.skip()


def test_effect_rule_vetting_and_library_generation():
    samples = sample_contexts(random.Random(2), 120, mechanics=True)
    always = CodeRule("def effect(ctx, fx):\n    fx.reverse()", "effect", "x", mechanics=True)
    never = CodeRule("def effect(ctx, fx):\n    if ctx.card.rank > 20:\n        fx.reverse()", "effect", "x", mechanics=True)
    kings = CodeRule("def effect(ctx, fx):\n    if ctx.card.rank == 13:\n        fx.reverse()", "effect", "x", mechanics=True)
    stack = [Rule("no_red_on_black")]
    assert "too often" in rule_problem(stack, always, samples)
    assert "almost never" in rule_problem(stack, never, samples)
    assert rule_problem(stack, kings, samples) is None
    pair = Rule("pair_ok")
    assert rule_problem(stack, pair, samples) is None              # multi-card samples make it matter
    kinds = {generate_rule(stack, random.Random(s), mechanics=True).kind for s in range(30)}
    assert "effect" in kinds
    assert all(not TEMPLATES[generate_rule(stack, random.Random(s)).template].mechanics for s in range(30))


def test_parse_moves_and_options():
    hand = tuple(C(c) for c in ["5D", "5H", "9S", "KC"])
    base = dict(seat=0, player_names=("a", "b", "c"), hand=hand, discard=(C("5S"),), deck_size=30,
                hand_sizes=(4, 4, 4), game=2, num_games=5, turn=0, num_rules=2, scores=(0, 0, 0), log=(),
                mechanics_from=3)
    on = PlayerView(**base, table=TableState(tokens=(2, 0, 0), spaces=((), (), ()), angles=(0,), mechanics=True))
    off = PlayerView(**{**base, "game": 0})
    r = parse_play("NOTES:\nx\nPLAY: 5D 5H\nANGLE: 90\nSPEND: 1\nTARGET: P3\nDECLARE: hearts\nGIVE: KC\nCALL: NONE",
                   hand, view=on)
    assert r.error is None and r.move == Move((C("5D"), C("5H")), 90, 1, 2, "hearts", C("KC"))
    assert parse_play("PLAY: 9S\nSPEND: 3", hand, view=on).error.startswith("you have 2 token")
    assert "options are not available yet (from game 3)" in parse_play("PLAY: 5D 5H", hand, view=off).error
    assert parse_play("PLAY: 9S (safe)", hand, view=off).card == C("9S")  # commentary isn't a second card
    assert parse_play("PLAY: 9S", hand, view=on).move is None
    pend = PendingPlay(0, C("5D"), C("5S"), 0, r.move)
    assert 'plays 5D 5H at 90 (spending 1 token, targeting P3, declaring "hearts") on 5S' in judge_request(pend)
    assert "FROM GAME 3: TABLE MECHANICS" in system_prompt(off)
    assert "TABLE MECHANICS" not in system_prompt(PlayerView(**{**base, "mechanics_from": 0}))


def test_r0_in_decide_and_violated():
    stack = RuleStack([RuleEntry(Rule("no_red_on_black"), "a", -1)])
    two = PlayContext(C("8S"), (C("5S"),), 0, (5, 5), 0, 2, Move((C("8S"), C("9S"))))
    assert stack.decide(two).deciding_index == BASE_RULE and stack.violated(two) == [BASE_RULE]
    assert stack.violated_owners(two) == set()
    stack.add(RuleEntry(Rule("pair_ok"), "b", 0))
    pair = PlayContext(C("8S"), (C("5S"),), 0, (5, 5), 0, 2, Move((C("8S"), C("8C"))))
    assert stack.decide(pair).allowed


def test_baseline_match_with_mechanics_runs():
    s = Match(MatchConfig(players=["oracle", "stats", "random"], num_games=6, seed=1)).run()
    assert len(s["games"]) == 6
    assert any(r["rule"]["kind"] == "effect" for r in s["rules"])
    s0 = Match(MatchConfig(players=["oracle", "stats"], num_games=4, seed=1, mechanics_from=0)).run()
    assert all(r["rule"]["kind"] != "effect" for r in s0["rules"])


def test_mechanics_sampling_keeps_all_rule_kinds_reachable():
    # Regression: multi-card samples are illegal under R0, which used to fail every restriction and
    # permission on playability, so with mechanics on only effects (or nothing, for R1) were added.
    assert generate_rule([], random.Random(0), mechanics=True).kind == "restriction"
    stack = [Rule("no_red_on_black")]
    rules = [generate_rule(stack, random.Random(s), mechanics=True) for s in range(40)]
    assert {"restriction", "permission", "effect"} <= {r.kind for r in rules}
    assert any(r.mechanics and r.kind != "effect" for r in rules)


def test_move_cards_between_hands_spaces_and_discard():
    code = ("def effect(ctx, fx):\n"
            "    if ctx.card.rank == 13:\n"
            "        fx.move_card(fx.hand(ctx.next_player), fx.hand(ctx.player), 1, 'highest')\n"
            "        fx.move_card(fx.hand(ctx.player), fx.space(ctx.player), 1, 'chosen')\n"
            "        fx.move_card(fx.space(1), fx.discard(), 1, 'oldest')\n"
            "        fx.move_card(fx.discard(), fx.hand(2))\n")
    rule = CodeRule(code, "effect", "Kings shuffle cards around.", mechanics=True)
    a = Mover("a", [Move((C("KS"),), give=C("2S"))])
    b, c = Mover("b"), Mover("c")
    g, _ = game([a, b, c], [rule], [["KS", "2S", "9S"], ["5D", "QD"], ["7D", "8D"]], max_turns=1)
    from gnomic.mechanics import Placed
    g.spaces[1] = [Placed(C("3H"), 90), Placed(C("4H"))]
    g.play()
    assert C("QD") in g.hands[0] and g.hands[1] == [C("5D")]              # highest card taken from P2
    assert [p.card for p in g.spaces[0]] == [C("2S")]                      # GIVE card into own space
    assert [p.card for p in g.spaces[1]] == [C("4H")]                       # oldest space card ...
    # ... went onto DISCARD (at its angle), then the top DISCARD card (3H) went to P3's hand
    assert g.discard[-1] == C("KS") and C("3H") in g.hands[2]
    ev = g.log.events[-2]
    assert "a card moved from P2's hand to P1's hand" in ev.effects        # hand to hand stays private
    assert "QD" not in render_event(ev) and any("QD" in n for n in g.private[0])
    assert "2S moved from P1's hand to P1's space" in ev.effects


def test_taking_a_last_card_wins_only_at_the_end_of_the_victims_own_turn():
    code = ("def effect(ctx, fx):\n"
            "    fx.move_card(fx.hand(ctx.next_player), fx.hand(ctx.player))\n"
            "    fx.move_card(fx.discard(), fx.space(ctx.player), 3)\n")
    rule = CodeRule(code, "effect", "x", mechanics=True)
    a, b = Mover("a", ["KS"]), Mover("b")
    g, events = game([a, b], [rule], [["KS", "9S"], ["5D"]], max_turns=5)
    result = g.play()
    assert g.hands[0] == [C("9S"), C("5D")] and len(g.discard) == 1      # DISCARD keeps a card
    # P2 lost their last card on P1's turn, but wins only at the end of their own turn
    assert result.winner == 1 and result.turns == 2 and not b.views         # P2 was never asked to play
    assert render_event(g.log.events[-2]) == "G3 T2 P2 has no cards, so plays none"


def test_move_tokens_between_hands_spaces_pot_and_bank():
    code = ("def effect(ctx, fx):\n"
            "    fx.move_tokens(fx.hand(ctx.player), fx.discard(), 2)\n"
            "    fx.move_tokens(fx.bank(), fx.space(1), 3)\n"
            "    fx.move_tokens(fx.space(1), fx.hand(1), 1)\n"
            "    if ctx.card.rank == 1:\n"
            "        fx.move_tokens(fx.discard(), fx.hand(ctx.player), 'all')\n")
    rule = CodeRule(code, "effect", "x", mechanics=True)
    a, b = Mover("a", ["KS", "AS"]), Mover("b", ["5D"])
    g, _ = game([a, b], [rule], [["KS", "AS", "9S"], ["5D", "6D"]], max_turns=3)
    g.tokens = [3, 0]
    g.play()
    # T1 (P1): P1 3->1, pot 2, P2's space 3->2, P2 holds 1
    # T2 (P2): P2 1->0, pot 3, P2's space 5->4, P2 holds 1
    # T3 (P1): P1 1->0, pot 4, P2's space 7->6, P2 holds 2; the Ace takes the pot: P1 holds 4
    assert g.tokens == [4, 2] and g.pot == 0 and g.space_tokens == [0, 6]
    assert "the pot" in " ".join(g.log.events[-2].effects)
    with pytest.raises(ValueError):
        Effects(2).move_tokens(("bank",), ("hand", 0), "all")
    with pytest.raises(ValueError):
        Effects(2).move_card(("bank",), ("hand", 0))


def test_win_condition_checked_only_at_the_end_of_own_turn():
    from gnomic.rules import WinContext
    win = Rule("suit_match_win")      # one card left, same suit as the top DISCARD card
    a, b = Mover("a", ["9H"]), Mover("b", ["2C"])
    # P2 starts with one heart on a heart: they don't win out of turn (P1 plays first)
    g, events = game([a, b], [Rule("no_red_on_black"), win], [["9H", "3H", "4S"], ["2C", "5H"]], max_turns=4)
    result = g.play()
    # T1: P1 plays 9H, holds 3H 4S. T2: P2 plays 2C, holds 5H on 2C: no. T3: P1 plays 3H, holds 4S on 3H: no.
    # T4: P2 plays 5H and holds nothing: wins by emptying their hand.
    assert result.end_reason == "emptied_hand"
    # direct check of the end-of-turn context
    w = WinContext((C("2C"), C("5H")), 1, (2, 1), 3, 2, hand=(C("6H"),))
    assert win.wins(w) and not win.wins(WinContext((C("2C"),), 1, (2, 1), 3, 2, hand=(C("6H"),)))
    a, b = Mover("a", ["9S"]), Mover("b")
    g, _ = game([a, b], [win], [["9S", "4S"], ["2C", "5H"]], max_turns=3)
    result = g.play()
    assert result.winner == 0 and result.end_reason == "win_rule R1" and result.turns == 1
    assert "(win condition R1)" in render_event(g.log.events[-1])


def test_win_rules_gated_vetted_and_coded():
    samples = sample_contexts(random.Random(1), 300, mechanics=True)
    easy = CodeRule("def wins(ctx):\n    return any(c.is_face for c in ctx.hand)", "win", "x", mechanics=True)
    assert "too easy" in rule_problem([Rule("no_red_on_black")], easy, samples)
    assert rule_problem([Rule("no_red_on_black")], Rule("suit_match_win"), samples) is None
    with pytest.raises(SandboxError, match="attribute .hand"):
        compile_rule("def applies(ctx):\n    return len(ctx.hand) > 2")       # only win rules see the hand
    kinds = {generate_rule([Rule("no_red_on_black")], random.Random(s), mechanics=True, wins=True).kind
             for s in range(30)}
    assert "win" in kinds
    assert all(generate_rule([Rule("no_red_on_black")], random.Random(s), mechanics=True).kind != "win"
               for s in range(30))

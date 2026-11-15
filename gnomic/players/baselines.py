"""Players that don't use an LLM, for reference scores.

random    lower bound: plays any card and never calls
stats     a simple statistical learner over the public log, which an LLM ought to beat
oracle    upper bound: knows every hidden rule, plays legally when it can, calls perfectly

Players police only their own rules, so a baseline calls PENALTY only for a play that
breaks a rule it owns.
"""

from __future__ import annotations

import random
from collections import defaultdict

from ..cards import Card
from ..game import PendingPlay, PlayerView
from ..rules import RuleStack, Verdict
from .base import Call, Choice, Player


class RandomPlayer(Player):
    kind = "random"

    def __init__(self, name: str, seed: int | None = None):
        super().__init__(name)
        self.rng = random.Random(seed)

    def choose(self, view: PlayerView) -> Choice:
        return Choice(self.rng.choice(view.hand), reasoning="random choice")


class OraclePlayer(Player):
    kind = "oracle"

    def __init__(self, name: str, stack: RuleStack, seed: int | None = None):
        super().__init__(name)
        self.stack = stack
        self.rng = random.Random(seed)

    def _my_broken(self, view: PlayerView, card: Card, seat: int, move=None) -> int | None:
        """Number (R<n>) of the newest of my rules this play breaks, or None."""
        for i in self.stack.violated(view.context_for(card, seat=seat, move=move)):  # newest first
            if i >= 0 and self.stack.entries[i].owner == self.name:
                return i + 1
        return None

    def _breaks_mine(self, view: PlayerView, card: Card, seat: int) -> bool:
        return self._my_broken(view, card, seat) is not None

    def choose(self, view: PlayerView) -> Choice:
        legal = [c for c in view.hand if self.stack.is_legal(view.context_for(c))]
        if legal:
            return Choice(self.rng.choice(legal), reasoning=f"{len(legal)} legal option(s)")
        # No legal card: prefer one that breaks only other players' rules (they must call it).
        others_only = [c for c in view.hand if not self._breaks_mine(view, c, view.seat)]
        card = self.rng.choice(others_only or list(view.hand))
        mine = self._my_broken(view, card, view.seat)
        if mine:
            return Choice(card, self_call=True, self_call_rule=mine,
                          reasoning=f"no legal card; this breaks my rule R{mine}, so I call it")
        return Choice(card, reasoning="no legal card; this breaks only other players' rules")

    def call(self, view: PlayerView, play: PendingPlay) -> Call:
        mine = self._my_broken(view, play.card, play.seat, play.move)
        return Call(mine is not None, rule=mine, reasoning=f"breaks my R{mine}" if mine else "knows the rules")


def _features(card: Card, top: Card) -> list[tuple]:
    d = card.rank - top.rank
    return [
        ("color", card.is_red, top.is_red),
        ("suit_rel", card.suit == top.suit),
        ("rank_rel", (d > 0) - (d < 0)),
        ("parity", card.is_even, top.is_even),
        ("face", card.is_face, top.is_face),
        ("dist", min(abs(d), 6)),
        ("card_suit", card.suit),
        ("card_rank", card.rank),
        ("top_suit", top.suit),
        ("top_rank_band", top.rank // 4),
    ]


class StatsPlayer(Player):
    """Estimates how likely a play is to be illegal from feature-wise rates in the public log.

    Labels: an upheld call means illegal, a rejected call means probably legal (it may have
    broken someone else's rule), and a play nobody called counts as weak evidence of legal.

    It knows the rules it owns, as an LLM player does: it avoids cards they forbid and calls
    PENALTY only when one of them forbids a play. It can't know whether a newer rule it
    doesn't own overrides them.
    """

    kind = "stats"

    def __init__(self, name: str, stack: RuleStack, seed: int | None = None):
        super().__init__(name)
        self.stack = stack
        self.rng = random.Random(seed)

    def _mine_forbid(self, view: PlayerView, card: Card, seat: int, move=None) -> int | None:
        """Number (R<n>) of my rule that forbids this play, judging by my own rules only
        (newest first, ignoring everyone else's), or None."""
        ctx = view.context_for(card, seat=seat, move=move)
        for i in range(len(self.stack.entries) - 1, -1, -1):
            e = self.stack.entries[i]
            if e.owner != self.name:
                continue
            v = e.rule.verdict(ctx)
            if v is not Verdict.ABSTAIN:
                return i + 1 if v is Verdict.FORBID else None
        return None

    def _counts(self, view: PlayerView):
        counts: dict[tuple, list[float]] = defaultdict(lambda: [0.0, 0.0])  # feature -> [illegal, total]
        rules_now = view.num_rules
        rules_at_game: dict[int, int] = {}
        n_rules = 0
        for ev in view.log:
            if ev.kind == "rule_added":
                n_rules += 1
            elif ev.kind == "game_start":
                rules_at_game[ev.game] = n_rules
        for ev in view.log:
            if ev.kind != "play" or ev.top_before is None:
                continue
            if ev.upheld is None:
                label, w = 0.0, 0.3
            elif ev.upheld:
                label, w = 1.0, 1.0
            else:
                label, w = 0.0, 0.6
            if rules_at_game.get(ev.game, rules_now) != rules_now:
                w *= 0.5  # older games ran under fewer rules
            for f in _features(ev.card, ev.top_before):
                counts[f][0] += w * label
                counts[f][1] += w
        return counts

    @staticmethod
    def _risk(counts, card: Card, top: Card | None) -> float:
        if top is None:
            return 0.0
        rates = [(counts[f][0] + 0.2) / (counts[f][1] + 1.0) for f in _features(card, top)]
        return 0.5 * max(rates) + 0.5 * sum(rates) / len(rates)

    def choose(self, view: PlayerView) -> Choice:
        counts = self._counts(view)
        scored = sorted(view.hand, key=lambda c: (self._mine_forbid(view, c, view.seat) is not None,
                                                   self._risk(counts, c, view.top) + self.rng.random() * 1e-6))
        best = scored[0]
        mine = self._mine_forbid(view, best, view.seat)
        if mine:
            return Choice(best, self_call=True, self_call_rule=mine,
                          reasoning=f"every card breaks my rule R{mine}; calling it on myself")
        return Choice(best, reasoning=f"lowest estimated risk ({self._risk(counts, best, view.top):.2f})")

    def call(self, view: PlayerView, play: PendingPlay) -> Call:
        mine = self._mine_forbid(view, play.card, play.seat, play.move)
        return Call(mine is not None, rule=mine, reasoning=f"breaks my R{mine}" if mine else "doesn't break my rules")

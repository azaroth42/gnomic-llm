"""Where rules come from.

The engine is always the source of truth for whether a play broke the current
rule stack. Players decide whether a penalty is *called*: after every play,
each other player answers PENALTY or NONE. A justified call makes the offender
draw; a false call makes the caller draw (core rule 8). An uncalled violation
costs nothing and is never revealed.

A rule is any object with `verdict(ctx) -> Verdict`, `kind`, `description`,
`key()` and `to_json()`. Rules come from one of two authors:

  EngineRuleAuthor    a random rule from the pre-written template library.
  ProposedRuleAuthor  (`gnomic play --propose-rules`) the dealer or winner makes up the rule:
    1. the owner describes a rule in plain language (Player.propose_rule),
    2. the coding agent (coder.RuleCoder) turns it into `def applies(ctx)`, which is
       checked and smoke-tested in the sandbox (sandbox.py),
    3. `rules.rule_problem()` checks that it is playable and not redundant, with the
       same thresholds as library rules,
    4. if 2 or 3 fails, the owner is told why and asked for a different rule, up to
       `attempts` times; after that a library rule is used instead.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Callable, Protocol

from .coder import CoderFailure
from .rules import RuleEntry, RuleStack, generate_rule, rule_problem, sample_contexts

if TYPE_CHECKING:
    from .coder import RuleCoder
    from .game import PlayerView
    from .players.base import Player


class RuleAuthor(Protocol):
    def author_rule(self, stack: RuleStack, owner: str, after_game: int) -> RuleEntry: ...


def mechanics_allowed(after_game: int, mechanics_from: int) -> bool:
    """May a rule added after game `after_game` (0-based; -1 = the dealer's) use table mechanics?
    It first applies in game after_game + 2 (1-based), which must be >= mechanics_from."""
    return bool(mechanics_from) and after_game + 2 >= mechanics_from


def wins_allowed(after_game: int, wins_from: int) -> bool:
    """May a rule added after game `after_game` be a win condition? Only if it first applies in
    game >= wins_from (default 6: "only after 5 games", user decision)."""
    return bool(wins_from) and after_game + 2 >= wins_from


class EngineRuleAuthor:
    """Picks a random library rule for the dealer or winner `owner`, who is then told it privately."""

    def __init__(self, rng: random.Random, mechanics_from: int = 0, wins_from: int = 0):
        self.rng = rng
        self.mechanics_from = mechanics_from
        self.wins_from = wins_from

    def author_rule(self, stack: RuleStack, owner: str, after_game: int) -> RuleEntry:
        rule = generate_rule(stack.rules(), self.rng, mechanics=mechanics_allowed(after_game, self.mechanics_from),
                             wins=wins_allowed(after_game, self.wins_from))
        return RuleEntry(rule=rule, owner=owner, added_after_game=after_game)


class ProposedRuleAuthor:
    """The owner makes up the rule; the coding agent implements it (see module docstring).

    Players that can't make up rules (baselines) get a library rule from `fallback`,
    as does an owner whose proposals all fail. Every step is emitted for spectators
    and the log: rule_proposed, rule_rejected, rule_fallback (rule_added comes from the match).
    """

    def __init__(self, coder: "RuleCoder", fallback: RuleAuthor, players: list["Player"],
                 view: Callable[[int, int], "PlayerView"], emit: Callable[..., object],
                 rng: random.Random, attempts: int = 3, mechanics_from: int = 0, wins_from: int = 0):
        self.coder, self.fallback, self.players = coder, fallback, players
        self.mechanics_from = mechanics_from
        self.wins_from = wins_from
        self.view, self.emit, self.rng, self.attempts = view, emit, rng, attempts

    def author_rule(self, stack: RuleStack, owner: str, after_game: int) -> RuleEntry:
        seat = next(i for i, p in enumerate(self.players) if p.name == owner)
        player = self.players[seat]
        if not player.makes_rules:
            return self.fallback.author_rule(stack, owner, after_game)
        first = len(stack) == 0
        mechanics = mechanics_allowed(after_game, self.mechanics_from)
        wins = wins_allowed(after_game, self.wins_from)
        feedback: str | None = None
        reason = "no proposals"
        for attempt in range(1, self.attempts + 1):
            proposal = player.propose_rule(self.view(seat, after_game), first=first, feedback=feedback,
                                           attempt=attempt, attempts=self.attempts, mechanics=mechanics,
                                           wins=wins)
            base = dict(owner=owner, owner_seat=seat, after_game=after_game, attempt=attempt)
            if proposal is None or not proposal.text:
                reason = feedback = "no usable RULE line in your reply"
                self.emit("rule_rejected", **base, stage="proposal", reason=reason, text=None)
                continue
            self.emit("rule_proposed", **base, text=proposal.text, reasoning=proposal.reasoning,
                      notes=proposal.notes, meta=proposal.meta)
            samples = sample_contexts(self.rng, 300, mechanics=mechanics)
            try:
                result = self.coder.implement(proposal.text, samples[:60], mechanics=mechanics, wins=wins)
            except CoderFailure as e:
                reason = f"the engine could not implement it: {e}"
                self.emit("rule_rejected", **base, stage="coder", reason=reason, text=proposal.text)
                feedback = reason
                continue
            problem = rule_problem(stack.rules(), result.rule, samples, first=first)
            if problem:
                reason = f"as implemented ({result.rule.kind}: {result.rule.description}) {problem}"
                self.emit("rule_rejected", **base, stage="check", reason=reason, text=proposal.text,
                          rule=result.rule.to_json(), coder=result.meta)
                feedback = reason
                continue
            self.emit("rule_coded", **base, text=proposal.text, rule=result.rule.to_json(), coder=result.meta)
            return RuleEntry(rule=result.rule, owner=owner, added_after_game=after_game)
        self.emit("rule_fallback", owner=owner, owner_seat=seat, after_game=after_game,
                  reason=f"no usable rule after {self.attempts} proposals (last: {reason}); using a library rule")
        return self.fallback.author_rule(stack, owner, after_game)

from __future__ import annotations

from dataclasses import dataclass, field

from ..cards import Card
from ..game import PendingPlay, PlayerView
from ..mechanics import Move


@dataclass
class Choice:
    """A player's move on their own turn."""
    card: Card | None
    move: Move | None = None    # game 3+: the whole move (several cards, options); move.card == card
    self_call: bool = False     # does the player call PENALTY on their own play?
    self_call_rule: int | None = None  # ...naming which of their rules (rule number, R<n>)
    reasoning: str = ""         # shown to spectators only
    notes: str | None = None    # the player's private notes after this turn
    meta: dict = field(default_factory=dict)  # tokens, attempts, errors, thinking summary


@dataclass
class Call:
    """A player's judgement of someone else's play. penalty=None means no usable answer (counted as NONE)."""
    penalty: bool | None
    rule: int | None = None     # the caller's own rule the play broke (rule number, R<n>)
    reasoning: str = ""
    notes: str | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class Proposal:
    """A rule a player made up, in plain language (see judges.ProposedRuleAuthor)."""
    text: str
    reasoning: str = ""
    notes: str | None = None
    meta: dict = field(default_factory=dict)


class Player:
    kind = "base"
    makes_rules = False  # can this player make up its own rules (--propose-rules)? Else it gets library rules.

    def __init__(self, name: str):
        self.name = name

    @property
    def model(self) -> str | None:
        return None

    def describe(self) -> dict:
        return {"name": self.name, "kind": self.kind, "model": self.model}

    def choose(self, view: PlayerView) -> Choice:
        raise NotImplementedError

    def call(self, view: PlayerView, play: PendingPlay) -> Call:
        return Call(False)

    def propose_rule(self, view: PlayerView, *, first: bool, feedback: str | None = None,
                     attempt: int = 1, attempts: int = 1, mechanics: bool = False,
                     wins: bool = False) -> Proposal | None:
        """Make up a new rule in plain language. `feedback` says why the previous proposal was rejected."""
        return None

    def on_game_start(self, game: int) -> None:
        pass

    def on_game_end(self, game: int, winner: int | None) -> None:
        pass

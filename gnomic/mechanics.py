"""Table mechanics that rules may add from game 3 on (user decision).

From game `mechanics_from` (default 3), a new rule may use more than the card and
DISCARD. The table gains:

  tokens        a public count per player, gained and spent through rules
  spaces        face-up cards in front of each player (a "space"), each at an angle
  angles        every card on DISCARD lies at an angle: 0 (upright), 90, 180 (upside down), 270
  declared      the latest DECLARE made this game (text, by whom)
  space tokens  tokens lying in each player's space;  pot: tokens lying on DISCARD

Cards and tokens can be moved between every place (user decision): hands, spaces and
DISCARD (and, for tokens, the bank) with fx.move_card / fx.move_tokens.

and a play becomes a *move*: one or more cards plus options the player chooses
(user decision: "options + effects"):

  ANGLE: 90       the angle the card(s) are played at (default 0)
  SPEND: 2        tokens to spend (always deducted; they must be available)
  TARGET: P3      a player the move is aimed at
  DECLARE: text   a short public declaration (a suit, a colour, "last card", ...)
  GIVE: 7H        the card to hand over if a rule makes this play a trade (private)

Options do nothing by themselves: rules judge them (restrictions/permissions see
them in ctx), and *effect* rules turn plays into public effects through the
`Effects` API below. Playing several cards at once breaks the public engine rule
R0 ("play exactly one card") unless a newer permission allows it, and the engine
calls R0 itself.

Effects never touch penalties (user decision: a penalty is always "offender
draws 1, false caller draws 1"). A play whose penalty is upheld triggers no
effects (user decision); tokens spent on it stay spent.

Approved effect types (the six from the user, plus the ones the user accepted
from the proposed list): spaces, tokens, angles, rotation, trading, multi-card
plays, effect draws, declarations, stealing from spaces, peeking. Not approved:
passing and drawing instead of playing. Reversing the order of play, skipping
players and extra turns were approved, then removed (user decision): the engine
decides whose turn it is, so nobody can play out of turn, and the engine asking
someone to play announces a rule that only its owner should know.
Play always goes in seat order.
"""

from __future__ import annotations

from dataclasses import dataclass

from .cards import Card

ANGLES = (0, 90, 180, 270)
MAX_DECLARE = 40       # characters
MAX_COUNT = 3          # most tokens / cards a single effect call may move
MAX_OPS_PER_RULE = 6   # effect calls one rule may make per play
BASE_RULE = -1         # index used for R0, the engine's "play exactly one card" rule


@dataclass(frozen=True)
class Placed:
    """A card lying face up on the table at an angle."""
    card: Card
    angle: int = 0

    def __str__(self) -> str:
        return f"{self.card}@{self.angle}" if self.angle else str(self.card)


@dataclass(frozen=True)
class Move:
    """What a player does on their turn: the card(s) played, plus options (from game 3)."""
    cards: tuple[Card, ...]
    angle: int = 0
    spend: int = 0
    target: int | None = None
    declare: str | None = None
    give: Card | None = None   # private: only used if an effect makes the play a trade

    @property
    def card(self) -> Card:
        return self.cards[0]

    @property
    def has_options(self) -> bool:
        return bool(len(self.cards) > 1 or self.angle or self.spend or self.target is not None
                    or self.declare or self.give)

    def public(self) -> str:
        """How the move appears to everyone (GIVE is private)."""
        out = " ".join(str(c) for c in self.cards)
        if self.angle:
            out += f" at {self.angle}"
        extras = []
        if self.spend:
            extras.append(f"spending {self.spend} token{'s' if self.spend != 1 else ''}")
        if self.target is not None:
            extras.append(f"targeting P{self.target + 1}")
        if self.declare:
            extras.append(f'declaring "{self.declare}"')
        return out + (f" ({', '.join(extras)})" if extras else "")

    def to_json(self) -> dict:
        d: dict = {"cards": [str(c) for c in self.cards]}
        for k in ("angle", "spend", "target", "declare"):
            if getattr(self, k):
                d[k] = getattr(self, k)
        if self.target == 0:
            d["target"] = 0
        if self.give:
            d["give"] = str(self.give)
        return d


@dataclass(frozen=True)
class TableState:
    """Public table state beyond hands and DISCARD, as rules see it."""
    tokens: tuple[int, ...] = ()          # tokens each player holds (their "hand" of tokens)
    spaces: tuple[tuple[Placed, ...], ...] = ()
    space_tokens: tuple[int, ...] = ()    # tokens lying in each player's space
    pot: int = 0                          # tokens lying on the DISCARD pile
    angles: tuple[int, ...] = ()          # parallel to the DISCARD pile
    declared: str | None = None
    declared_by: int | None = None
    game: int = 0
    mechanics: bool = False               # are moves with options / effects in play this game?


def normalize_declaration(text: str) -> str:
    return " ".join(text.lower().replace('"', "").replace("'", "").split())[:MAX_DECLARE]


class Effects:
    """The `fx` object an effect rule's `effect(ctx, fx)` receives.

    Calls are only recorded; the engine applies them after the play is judged (and
    only if no penalty was upheld), oldest rule first. Seats are 0-based ints, e.g.
    ctx.player, ctx.next_player, ctx.target. Counts are 1..3."""

    def __init__(self, num_players: int):
        self.n = num_players
        self.ops: list[tuple] = []

    def _add(self, *op) -> None:
        if len(self.ops) >= MAX_OPS_PER_RULE:
            raise ValueError(f"at most {MAX_OPS_PER_RULE} effects per play")
        self.ops.append(op)

    def _seat(self, s) -> int:
        if isinstance(s, bool) or not isinstance(s, int) or not 0 <= s < self.n:
            raise ValueError(f"not a seat: {s!r}")
        return s

    @staticmethod
    def _count(k) -> int:
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= MAX_COUNT:
            raise ValueError(f"count must be 1..{MAX_COUNT}, got {k!r}")
        return k

    @staticmethod
    def _angle(a) -> int:
        if a not in ANGLES:
            raise ValueError(f"angle must be one of {ANGLES}")
        return a

    # ---- places, for move_card / move_tokens ----
    def hand(self, seat) -> tuple:
        """A player's hand (for tokens: the tokens they hold)."""
        return ("hand", self._seat(seat))

    def space(self, seat) -> tuple:
        """A player's space on the table (cards and tokens lying in front of them)."""
        return ("space", self._seat(seat))

    def discard(self) -> tuple:
        """The DISCARD pile (for tokens: the pot lying on it)."""
        return ("discard",)

    def bank(self) -> tuple:
        """The unlimited supply of tokens (tokens only)."""
        return ("bank",)

    def _place(self, loc, *, tokens: bool) -> tuple:
        ok = isinstance(loc, tuple) and (
            (len(loc) == 2 and loc[0] in ("hand", "space") and self._seat(loc[1]) is not None)
            or loc in (("discard",),) or (tokens and loc == ("bank",)))
        if not ok:
            raise ValueError("a place must be fx.hand(seat), fx.space(seat), fx.discard()"
                             + (" or fx.bank()" if tokens else ""))
        return loc

    def move_card(self, source, dest, n=1, which="default") -> None:
        """Move n cards from `source` to `dest` (fx.hand(seat), fx.space(seat), fx.discard()).

        Which card: from a hand, "random" (default), "highest", "lowest", or "chosen" (the
        GIVE card of the player who just played, if the source is their hand); from a space or
        DISCARD, "newest" (default: the newest space card / the top card), "oldest", "random",
        "highest" or "lowest". DISCARD always keeps at least one card. A player whose last card
        is taken doesn't win then: they win at the end of their own next turn (they have nothing
        to play) unless someone wins first. Cards moved between hands are seen
        only by the two players; any other move is public."""
        src, dst = self._place(source, tokens=False), self._place(dest, tokens=False)
        if src == dst:
            raise ValueError("source and destination are the same place")
        if which not in CARD_CHOICES:
            raise ValueError(f"which must be one of {sorted(CARD_CHOICES)}")
        self._add("move_card", src, dst, self._count(n), which)

    def move_tokens(self, source, dest, n=1) -> None:
        """Move n tokens (1..3, or "all") from `source` to `dest` (fx.hand(seat) = tokens a player
        holds, fx.space(seat), fx.discard() = the pot, fx.bank() = the supply). Moves as many as
        are there, up to n; "all" can't come from the bank."""
        src, dst = self._place(source, tokens=True), self._place(dest, tokens=True)
        if src == dst:
            raise ValueError("source and destination are the same place")
        if n == "all":
            if src == ("bank",):
                raise ValueError('"all" tokens can\'t come from the bank')
        else:
            self._count(n)
        self._add("move_tokens", src, dst, n)

    # ---- shorthands ----
    def gain_tokens(self, seat, n=1) -> None:
        """= move_tokens(bank(), hand(seat), n)"""
        self.move_tokens(("bank",), self.hand(seat), n)

    def lose_tokens(self, seat, n=1) -> None:
        """Lose up to n tokens (never below 0). = move_tokens(hand(seat), bank(), n)"""
        self.move_tokens(self.hand(seat), ("bank",), n)

    def draw(self, seat, n=1) -> None:
        """Make a player draw n cards from DECK. This is an effect, not a penalty."""
        self._add("draw", self._seat(seat), self._count(n))

    def to_space(self, seat) -> None:
        """The card(s) just played go into `seat`'s space instead of staying on DISCARD."""
        self._add("to_space", self._seat(seat))

    def rotate_top(self, angle) -> None:
        """Turn the top DISCARD card to `angle`."""
        self._add("rotate_top", self._angle(angle))

    def rotate_space(self, seat, angle) -> None:
        """Turn the newest card in `seat`'s space to `angle`."""
        self._add("rotate_space", self._seat(seat), self._angle(angle))

    def steal_to_space(self, from_seat, to_seat) -> None:
        """Move the newest card in from_seat's space into to_seat's space."""
        self.move_card(self.space(from_seat), self.space(to_seat))

    def steal_to_hand(self, from_seat, to_seat) -> None:
        """Move the newest card in from_seat's space into to_seat's hand."""
        self.move_card(self.space(from_seat), self.hand(to_seat))

    def trade(self, seat, other) -> None:
        """`seat` gives a card (their GIVE option if they played this move, else a random card)
        to `other` and takes a random card from `other`'s hand. Only the two of them see the cards."""
        if self._seat(seat) == self._seat(other):
            raise ValueError("a player can't trade with themselves")
        self._add("trade", seat, other)

    def peek_deck(self, seat) -> None:
        """`seat` privately sees the top card of DECK."""
        self._add("peek_deck", self._seat(seat))

    def peek_hand(self, seat, other) -> None:
        """`seat` privately sees one random card from `other`'s hand."""
        if self._seat(seat) == self._seat(other):
            raise ValueError("a player can't peek at their own hand")
        self._add("peek_hand", seat, other)


CARD_CHOICES = {"default", "random", "highest", "lowest", "chosen", "newest", "oldest"}

FX_METHODS = {"gain_tokens", "lose_tokens", "draw", "to_space", "rotate_top",
              "rotate_space", "steal_to_space", "steal_to_hand", "trade", "peek_deck", "peek_hand",
              "move_card", "move_tokens", "hand", "space", "discard", "bank"}

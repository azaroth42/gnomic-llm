"""Hidden rules and how they combine.

Each rule returns ALLOW, FORBID, or ABSTAIN for a proposed play. Rules are
evaluated newest-first and the first non-abstaining verdict decides (core rule 9,
"later rules take precedence"). If every rule abstains the play is allowed.

Restrictions only ever FORBID or ABSTAIN, and permissions only ever ALLOW or
ABSTAIN. So stacking restrictions makes them all apply together, a newer
permission overrides older restrictions, and a newer restriction overrides older
permissions.

Rules are built from named templates plus parameters, so they serialize to JSON
and can be rebuilt exactly when replaying a log.

Rules may use these card properties: value 1-13, colour, suit, face/not-face,
gender (J and K male, Q female), and odd/even. They may also use public game
state. Value arithmetic is limited to > < + - * and / (division, as in double or
half, or odd/even). No square roots, exponents, primes, modular wrap-around, etc.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Callable, Iterable

from .cards import SUITS, SUIT_NAMES, Card, full_deck
from .mechanics import ANGLES, BASE_RULE, Effects, Move, Placed, TableState


class Verdict(Enum):
    ALLOW = "allow"
    FORBID = "forbid"
    ABSTAIN = "abstain"


class _TableView:
    """Accessors shared by PlayContext and WinContext; need pile, player, hand_sizes,
    num_players and table."""

    @property
    def top(self) -> Card | None:
        return self.pile[-1] if self.pile else None

    # ---- the table ----
    @property
    def next_player(self) -> int:
        return (self.player + 1) % self.num_players

    @property
    def tokens(self) -> tuple[int, ...]:
        return self.table.tokens if self.table and self.table.tokens else (0,) * self.num_players

    @property
    def spaces(self) -> tuple[tuple[Placed, ...], ...]:
        return self.table.spaces if self.table and self.table.spaces else ((),) * self.num_players

    @property
    def space_tokens(self) -> tuple[int, ...]:
        return self.table.space_tokens if self.table and self.table.space_tokens else (0,) * self.num_players

    @property
    def pot(self) -> int:
        return self.table.pot if self.table else 0

    @property
    def pile_angles(self) -> tuple[int, ...]:
        if self.table and len(self.table.angles) == len(self.pile):
            return self.table.angles
        return (0,) * len(self.pile)

    @property
    def declared(self) -> str | None:
        return self.table.declared if self.table else None

    @property
    def declared_by(self) -> int | None:
        return self.table.declared_by if self.table else None

    @property
    def game(self) -> int:
        return self.table.game if self.table else 0

    def back(self, n: int) -> Card | None:
        """n=1 is the top card, n=2 the one under it, and so on."""
        return self.pile[-n] if len(self.pile) >= n else None

    @property
    def hand_size(self) -> int:
        return self.hand_sizes[self.player]


@dataclass(frozen=True)
class PlayContext(_TableView):
    """Everything a rule may look at. All of it is public information."""

    card: Card                    # the card being played
    pile: tuple[Card, ...]        # discard pile before this play (top = last)
    player: int                   # seat of the player
    hand_sizes: tuple[int, ...]   # every player's hand size before this play
    turn: int                     # 0-based turn number within the game
    num_players: int
    move: Move | None = None      # the whole move, if it has options or several cards (game 3+)
    table: TableState | None = None  # tokens, spaces, angles, declaration (game 3+)

    # ---- the move (see mechanics.py); defaults describe a plain one-card play ----
    @property
    def cards(self) -> tuple[Card, ...]:
        return self.move.cards if self.move else (self.card,)

    @property
    def angle(self) -> int:
        return self.move.angle if self.move else 0

    @property
    def spend(self) -> int:
        return self.move.spend if self.move else 0

    @property
    def target(self) -> int | None:
        return self.move.target if self.move else None

    @property
    def declare(self) -> str | None:
        return self.move.declare if self.move else None


@dataclass(frozen=True)
class WinContext(_TableView):
    """What a win-condition rule sees at the end of a player's turn (game 6+ by default).
    Unlike PlayContext it includes the player's own hand: the engine, not the players,
    checks win conditions, so they may depend on it."""

    pile: tuple[Card, ...]        # DISCARD now, bottom to top
    player: int                   # the player whose turn is ending
    hand_sizes: tuple[int, ...]   # every player's hand size now
    turn: int
    num_players: int
    hand: tuple[Card, ...] = ()   # the player's hand now (private; only the engine sees it)
    cards: tuple[Card, ...] = ()  # the card(s) they played this turn (empty if they had none)
    penalized: bool = False       # was a penalty upheld on their play this turn?
    table: TableState | None = None

    @property
    def card(self) -> Card | None:
        return self.cards[0] if self.cards else None


Predicate = Callable[[PlayContext], bool]


@dataclass(frozen=True)
class RuleTemplate:
    name: str
    kind: str  # "restriction" | "permission" | "effect" | "win"
    describe: Callable[..., str]
    build: Callable[..., Predicate]  # returns forbids(ctx), allows(ctx), or (effect) triggers(ctx)
    sample_params: Callable[[random.Random], dict] = lambda rng: {}
    weight: float = 1.0
    effect: Callable[..., Callable[[PlayContext, Effects], None]] | None = None  # (params) -> fn(ctx, fx)
    mechanics: bool = False  # uses game-3+ mechanics (options, tokens, effects, ...)
    # kind "win": build(**params) returns wins(WinContext) -> bool; only for rules from game 6 on


@dataclass(frozen=True)
class Rule:
    template: str
    params: dict = field(default_factory=dict, hash=False, compare=False)

    @property
    def spec(self) -> RuleTemplate:
        return TEMPLATES[self.template]

    @property
    def kind(self) -> str:
        return self.spec.kind

    @property
    def description(self) -> str:
        return self.spec.describe(**self.params)

    @property
    def mechanics(self) -> bool:
        return self.spec.mechanics

    def verdict(self, ctx: PlayContext) -> Verdict:
        if self.kind in ("effect", "win"):
            return Verdict.ABSTAIN
        pred = _predicate_cache(self)
        if self.kind == "restriction":
            return Verdict.FORBID if pred(ctx) else Verdict.ABSTAIN
        return Verdict.ALLOW if pred(ctx) else Verdict.ABSTAIN

    def effect(self, ctx: PlayContext, fx: Effects) -> None:
        if self.kind == "effect" and _predicate_cache(self)(ctx):
            self.spec.effect(**self.params)(ctx, fx)

    def wins(self, ctx: "WinContext") -> bool:
        return self.kind == "win" and bool(_predicate_cache(self)(ctx))

    def key(self) -> tuple:
        return (self.template, tuple(sorted(self.params.items())))

    def to_json(self) -> dict:
        return {"template": self.template, "params": self.params,
                "kind": self.kind, "description": self.description}

    @classmethod
    def from_json(cls, d: dict) -> "Rule":
        return cls(d["template"], dict(d.get("params", {})))


_PRED_CACHE: dict[tuple, Predicate] = {}


def _predicate_cache(rule: Rule) -> Predicate:
    k = rule.key()
    if k not in _PRED_CACHE:
        _PRED_CACHE[k] = rule.spec.build(**rule.params)
    return _PRED_CACHE[k]


@dataclass
class RuleEntry:
    rule: Rule             # a library Rule, or anything with the same interface (e.g. sandbox.CodeRule)
    owner: str             # name of the player who made it up (the dealer, or a game's winner)
    added_after_game: int  # -1 for the dealer's initial rule

    def to_json(self) -> dict:
        return {"rule": self.rule.to_json(), "owner": self.owner,
                "added_after_game": self.added_after_game}


@dataclass
class Decision:
    allowed: bool
    deciding_index: int | None  # index into the stack, or None if every rule abstained


class RuleStack:
    def __init__(self, entries: Iterable[RuleEntry] = ()):
        self.entries: list[RuleEntry] = list(entries)

    def __len__(self) -> int:
        return len(self.entries)

    def add(self, entry: RuleEntry) -> None:
        self.entries.append(entry)

    def rules(self) -> list[Rule]:
        return [e.rule for e in self.entries]

    def decide(self, ctx: PlayContext) -> Decision:
        return decide(self.rules(), ctx)

    def is_legal(self, ctx: PlayContext) -> bool:
        return self.decide(ctx).allowed

    def violated(self, ctx: PlayContext) -> list[int]:
        return violated(self.rules(), ctx)

    def violated_owners(self, ctx: PlayContext) -> set[str]:
        """Owners whose rules this play breaks. Only they can call it successfully (R0 is the engine's)."""
        return {self.entries[i].owner for i in self.violated(ctx) if i != BASE_RULE}

    def winning_rule(self, ctx: WinContext) -> int | None:
        """Index of the newest win-condition rule that the player meets at the end of their
        turn, or None. A win rule that raises counts as not met."""
        for i in range(len(self.entries) - 1, -1, -1):
            rule = self.entries[i].rule
            if rule.kind != "win":
                continue
            try:
                if rule.wins(ctx):
                    return i
            except Exception:  # noqa: BLE001
                continue
        return None

    def effects(self, ctx: PlayContext) -> list[tuple[int, tuple]]:
        """Run every rule's effect, oldest first: [(rule index, op), ...]. A rule whose effect
        raises (or makes too many calls) contributes nothing."""
        out: list[tuple[int, tuple]] = []
        for i, e in enumerate(self.entries):
            fn = getattr(e.rule, "effect", None)
            if fn is None or e.rule.kind != "effect":
                continue
            fx = Effects(ctx.num_players)
            try:
                fn(ctx, fx)
            except Exception:  # noqa: BLE001 - a broken effect does nothing
                continue
            out.extend((i, op) for op in fx.ops)
        return out

    def owner_violated(self, ctx: PlayContext, owner: str) -> bool:
        return owner in self.violated_owners(ctx)


def decide(rules: list[Rule], ctx: PlayContext) -> Decision:
    """Newest rule first; the first non-abstaining verdict decides. Under every rule lies
    R0, the engine's public rule "play exactly one card" (deciding_index BASE_RULE)."""
    for i in range(len(rules) - 1, -1, -1):
        v = rules[i].verdict(ctx)
        if v is Verdict.ALLOW:
            return Decision(True, i)
        if v is Verdict.FORBID:
            return Decision(False, i)
    if len(ctx.cards) > 1:
        return Decision(False, BASE_RULE)
    return Decision(True, None)


def violated(rules: list[Rule], ctx: PlayContext) -> list[int]:
    """Indices of the rules this play breaks: rules that forbid it with no later rule
    allowing it. Empty exactly when the play is legal. The deciding rule of an illegal
    play is always included, and so is any older restriction it isn't shielded from.
    BASE_RULE (R0) is included for an unpermitted multi-card play."""
    out: list[int] = []
    for i in range(len(rules) - 1, -1, -1):
        v = rules[i].verdict(ctx)
        if v is Verdict.ALLOW:
            return out
        if v is Verdict.FORBID:
            out.append(i)
    if len(ctx.cards) > 1:
        out.append(BASE_RULE)
    return out


# --------------------------------------------------------------------------
# Template library
# --------------------------------------------------------------------------

def _top(fn: Callable[[Card, Card], bool]) -> Predicate:
    """Lift a (card, top) predicate; the rule abstains when the pile is empty."""
    return lambda ctx: ctx.top is not None and fn(ctx.card, ctx.top)


_RANK_WORDS = {1: "Aces", 11: "Jacks", 12: "Queens", 13: "Kings"}


def _rank_word(r: int) -> str:
    return _RANK_WORDS.get(r, f"{r}s")


def _strict_fewest(ctx: PlayContext) -> bool:
    me = ctx.hand_size
    return all(me < h for i, h in enumerate(ctx.hand_sizes) if i != ctx.player)


def _strict_most(ctx: PlayContext) -> bool:
    me = ctx.hand_size
    return all(me > h for i, h in enumerate(ctx.hand_sizes) if i != ctx.player)


def _between_last_two(ctx: PlayContext) -> bool:
    a, b = ctx.back(1), ctx.back(2)
    if a is None or b is None:
        return False
    lo, hi = sorted((a.rank, b.rank))
    return lo < ctx.card.rank < hi


def _suit_in_last_two(ctx: PlayContext) -> bool:
    return any(c is not None and c.suit == ctx.card.suit for c in (ctx.back(1), ctx.back(2)))


_LIST: list[RuleTemplate] = [
    # ---- restrictions on the relation to the top card ----
    RuleTemplate("no_red_on_black", "restriction",
                 lambda: "You may not play a red card on a black card.",
                 lambda: _top(lambda c, t: c.is_red and not t.is_red)),
    RuleTemplate("no_black_on_red", "restriction",
                 lambda: "You may not play a black card on a red card.",
                 lambda: _top(lambda c, t: not c.is_red and t.is_red)),
    RuleTemplate("alternate_colors", "restriction",
                 lambda: "You must play a card of the opposite color to the top card.",
                 lambda: _top(lambda c, t: c.is_red == t.is_red)),
    RuleTemplate("match_color", "restriction",
                 lambda: "You must play a card of the same color as the top card.",
                 lambda: _top(lambda c, t: c.is_red != t.is_red)),
    RuleTemplate("not_lower", "restriction",
                 lambda: "You may not play a lower-value card on a higher-value card.",
                 lambda: _top(lambda c, t: c.rank < t.rank)),
    RuleTemplate("not_higher", "restriction",
                 lambda: "You may not play a higher-value card on a lower-value card.",
                 lambda: _top(lambda c, t: c.rank > t.rank)),
    RuleTemplate("max_distance", "restriction",
                 lambda n: f"Your card's value must be within {n} of the top card's value.",
                 lambda n: _top(lambda c, t: abs(c.rank - t.rank) > n),
                 lambda rng: {"n": rng.choice([3, 4, 5])}),
    RuleTemplate("min_distance", "restriction",
                 lambda n: f"Your card's value must differ from the top card's value by at least {n}.",
                 lambda n: _top(lambda c, t: abs(c.rank - t.rank) < n),
                 lambda rng: {"n": rng.choice([2, 3])}),
    RuleTemplate("alternate_parity", "restriction",
                 lambda: "You must play an odd card on an even card and an even card on an odd card.",
                 lambda: _top(lambda c, t: c.is_even == t.is_even)),
    RuleTemplate("match_parity", "restriction",
                 lambda: "Your card must have the same parity (odd/even) as the top card.",
                 lambda: _top(lambda c, t: c.is_even != t.is_even)),
    RuleTemplate("no_same_suit", "restriction",
                 lambda: "You may not play a card of the same suit as the top card.",
                 lambda: _top(lambda c, t: c.suit == t.suit)),
    RuleTemplate("match_suit_or_rank", "restriction",
                 lambda: "You must match the top card's suit or value.",
                 lambda: _top(lambda c, t: c.suit != t.suit and c.rank != t.rank), weight=0.7),
    RuleTemplate("face_only_on_face", "restriction",
                 lambda: "Face cards (J, Q, K) may only be played on face cards.",
                 lambda: _top(lambda c, t: c.is_face and not t.is_face)),
    RuleTemplate("no_face_on_face", "restriction",
                 lambda: "You may not play a face card on a face card.",
                 lambda: _top(lambda c, t: c.is_face and t.is_face)),
    RuleTemplate("sum_limit", "restriction",
                 lambda n: f"Your card's value plus the top card's value may not exceed {n}.",
                 lambda n: _top(lambda c, t: c.rank + t.rank > n),
                 lambda rng: {"n": rng.choice([14, 15, 16, 17])}),
    RuleTemplate("product_limit", "restriction",
                 lambda n: f"Your card's value times the top card's value may not exceed {n}.",
                 lambda n: _top(lambda c, t: c.rank * t.rank > n),
                 lambda rng: {"n": rng.choice([40, 50, 60])}, weight=0.6),
    RuleTemplate("low_after_face", "restriction",
                 lambda n: f"After a face card, you must play a card of value {n} or less.",
                 lambda n: _top(lambda c, t: t.is_face and c.rank > n),
                 lambda rng: {"n": rng.choice([6, 7, 8])}),
    RuleTemplate("red_must_climb", "restriction",
                 lambda: "A red card must be higher in value than the top card.",
                 lambda: _top(lambda c, t: c.is_red and c.rank <= t.rank)),
    RuleTemplate("suit_banned_on_suit", "restriction",
                 lambda a, b: f"You may not play {SUIT_NAMES[a]} on {SUIT_NAMES[b]}.",
                 lambda a, b: _top(lambda c, t: c.suit == a and t.suit == b),
                 lambda rng: dict(zip(("a", "b"), rng.sample(SUITS, 2)))),
    RuleTemplate("no_same_gender", "restriction",
                 lambda: "You may not play a face card on a face card of the same gender (J and K are male, Q is female).",
                 lambda: _top(lambda c, t: c.gender is not None and c.gender == t.gender), weight=0.7),
    RuleTemplate("gender_banned_on_gender", "restriction",
                 lambda a, b: f"You may not play a {a} card on a {b} card (J and K are male, Q is female).",
                 lambda a, b: _top(lambda c, t: c.gender == a and t.gender == b),
                 lambda rng: dict(zip(("a", "b"), rng.sample(["male", "female"], 2))), weight=0.7),
    # ---- restrictions that look further back or at game state ----
    RuleTemplate("not_between_last_two", "restriction",
                 lambda: "You may not play a card whose value lies strictly between the values of the top two cards on the pile.",
                 lambda: _between_last_two, weight=0.6),
    RuleTemplate("no_suit_of_last_two", "restriction",
                 lambda: "You may not play a card that shares a suit with either of the top two cards on the pile.",
                 lambda: _suit_in_last_two, weight=0.6),
    RuleTemplate("even_turn_color", "restriction",
                 lambda color: f"On even-numbered turns (turn 0, 2, 4, ...) you must play a {color} card.",
                 lambda color: (lambda ctx: ctx.turn % 2 == 0 and ctx.card.color != color),
                 lambda rng: {"color": rng.choice(["red", "black"])}, weight=0.5),
    RuleTemplate("leader_no_face", "restriction",
                 lambda: "The player with strictly the fewest cards may not play face cards.",
                 lambda: (lambda ctx: ctx.card.is_face and _strict_fewest(ctx)), weight=0.5),
    # ---- permissions ----
    RuleTemplate("wild_rank", "permission",
                 lambda r: f"{_rank_word(r)} may always be played.",
                 lambda r: (lambda ctx: ctx.card.rank == r),
                 lambda rng: {"r": rng.randint(1, 13)}),
    RuleTemplate("wild_suit", "permission",
                 lambda s: f"{SUIT_NAMES[s].capitalize()} may always be played.",
                 lambda s: (lambda ctx: ctx.card.suit == s),
                 lambda rng: {"s": rng.choice(SUITS)}, weight=0.6),
    RuleTemplate("same_rank_ok", "permission",
                 lambda: "You may always play a card of the same value as the top card.",
                 lambda: _top(lambda c, t: c.rank == t.rank)),
    RuleTemplate("same_suit_ok", "permission",
                 lambda: "You may always play a card of the same suit as the top card.",
                 lambda: _top(lambda c, t: c.suit == t.suit)),
    RuleTemplate("sum_thirteen_ok", "permission",
                 lambda: "You may always play a card whose value plus the top card's value is 13.",
                 lambda: _top(lambda c, t: c.rank + t.rank == 13)),
    RuleTemplate("adjacent_ok", "permission",
                 lambda: "You may always play a card exactly one higher or one lower than the top card.",
                 lambda: _top(lambda c, t: abs(c.rank - t.rank) == 1)),
    RuleTemplate("double_or_half_ok", "permission",
                 lambda: "You may always play a card worth exactly double or exactly half the top card's value.",
                 lambda: _top(lambda c, t: c.rank == 2 * t.rank or 2 * c.rank == t.rank), weight=0.6),
    RuleTemplate("royal_couple_ok", "permission",
                 lambda: "You may always play a face card on a face card of the opposite gender (J and K are male, Q is female).",
                 lambda: _top(lambda c, t: c.gender is not None and t.gender is not None and c.gender != t.gender),
                 weight=0.6),
    RuleTemplate("underdog_free", "permission",
                 lambda: "The player with strictly the most cards may play anything.",
                 lambda: _strict_most, weight=0.5),
    RuleTemplate("face_on_face_ok", "permission",
                 lambda: "You may always play a face card (J, Q, K) on a face card.",
                 lambda: _top(lambda c, t: c.is_face and t.is_face)),
    RuleTemplate("same_color_ok", "permission",
                 lambda: "You may always play a card of the same color as the top card.",
                 lambda: _top(lambda c, t: c.is_red == t.is_red), weight=0.6),
    RuleTemplate("color_on_color_ok", "permission",
                 lambda a, b: f"You may always play a {a} card on a {b} card.",
                 lambda a, b: _top(lambda c, t: c.color == a and t.color == b),
                 lambda rng: {"a": rng.choice(["red", "black"]), "b": rng.choice(["red", "black"])}),
    RuleTemplate("suit_on_suit_ok", "permission",
                 lambda a, b: f"You may always play {SUIT_NAMES[a]} on {SUIT_NAMES[b]}.",
                 lambda a, b: _top(lambda c, t: c.suit == a and t.suit == b),
                 lambda rng: {"a": rng.choice(SUITS), "b": rng.choice(SUITS)}, weight=0.7),
    RuleTemplate("near_ok", "permission",
                 lambda n: f"You may always play a card within {n} of the top card's value.",
                 lambda n: _top(lambda c, t: abs(c.rank - t.rank) <= n),
                 lambda rng: {"n": rng.choice([2, 3])}),
    RuleTemplate("low_on_face_ok", "permission",
                 lambda n: f"You may always play a card of value {n} or less on a face card.",
                 lambda n: _top(lambda c, t: t.is_face and c.rank <= n),
                 lambda rng: {"n": rng.choice([3, 4, 5])}),
    RuleTemplate("small_sum_ok", "permission",
                 lambda n: f"You may always play a card if its value plus the top card's value is at most {n}.",
                 lambda n: _top(lambda c, t: c.rank + t.rank <= n),
                 lambda rng: {"n": rng.choice([8, 10, 12])}),
    RuleTemplate("short_hand_free", "permission",
                 lambda n: ("A player with only 1 card left may play anything." if n == 1
                            else f"A player with {n} or fewer cards may play anything."),
                 lambda n: (lambda ctx: ctx.hand_size <= n),
                 lambda rng: {"n": rng.choice([1, 2])}, weight=0.5),
    # ---- game 3+ mechanics (see mechanics.py): options, multi-card plays, effects ----
    RuleTemplate("pair_ok", "permission",
                 lambda: "You may play two cards of the same value together.",
                 lambda: (lambda ctx: len(ctx.cards) == 2 and ctx.cards[0].rank == ctx.cards[1].rank),
                 mechanics=True),
    RuleTemplate("spend_wild", "permission",
                 lambda: "If you spend at least one token, you may play any single card.",
                 lambda: (lambda ctx: len(ctx.cards) == 1 and ctx.spend >= 1), mechanics=True, weight=0.7),
    RuleTemplate("face_sideways", "restriction",
                 lambda: "Face cards must be played at angle 90 (sideways).",
                 lambda: (lambda ctx: any(c.is_face for c in ctx.cards) and ctx.angle != 90),
                 mechanics=True, weight=0.6),
    RuleTemplate("declared_suit_after", "restriction",
                 lambda r: (f"After {_rank_word(r)} is played, the next card must be of the suit its player declared "
                            f"(DECLARE: hearts, spades, ...), unless it is also one of the {_rank_word(r)}."),
                 lambda r: (lambda ctx: ctx.top is not None and ctx.top.rank == r and ctx.declared_by is not None
                            and ctx.declared in _SUIT_WORDS and ctx.card.rank != r
                            and _SUIT_WORDS[ctx.declared] != ctx.card.suit),
                 lambda rng: {"r": rng.choice([1, 8, 11])}, mechanics=True, weight=0.6),
    RuleTemplate("draw_on_rank", "effect",
                 lambda r, n: f"Playing one of the {_rank_word(r)} makes the next player draw {n}.",
                 lambda r, n: (lambda ctx: ctx.card.rank == r),
                 lambda rng: {"r": rng.randint(1, 13), "n": rng.choice([1, 2])}, mechanics=True,
                 effect=lambda r, n: (lambda ctx, fx: fx.draw(ctx.next_player, n))),
    RuleTemplate("token_on_face", "effect",
                 lambda: "Playing a face card earns you a token.",
                 lambda: (lambda ctx: ctx.card.is_face), mechanics=True,
                 effect=lambda: (lambda ctx, fx: fx.gain_tokens(ctx.player, 1))),
    RuleTemplate("space_on_rank", "effect",
                 lambda r: f"{_rank_word(r)} you play go into your space instead of onto DISCARD.",
                 lambda r: (lambda ctx: ctx.card.rank == r),
                 lambda rng: {"r": rng.randint(2, 10)}, mechanics=True, weight=0.6,
                 effect=lambda r: (lambda ctx, fx: fx.to_space(ctx.player))),
    RuleTemplate("trade_on_target", "effect",
                 lambda r: (f"Playing one of the {_rank_word(r)} with a TARGET makes you trade a card with that "
                            "player (you give your GIVE card, or a random one, and take a random card)."),
                 lambda r: (lambda ctx: ctx.card.rank == r and ctx.target is not None and ctx.target != ctx.player),
                 lambda rng: {"r": rng.choice([11, 12, 13])}, mechanics=True, weight=0.5,
                 effect=lambda r: (lambda ctx, fx: fx.trade(ctx.player, ctx.target))),
    # ---- game 6+ win conditions (checked only at the end of the player's own turn) ----
    RuleTemplate("suit_match_win", "win",
                 lambda: ("You also win if, at the end of your turn, you hold exactly one card and it has the "
                          "same suit as the top DISCARD card."),
                 lambda: (lambda ctx: len(ctx.hand) == 1 and ctx.top is not None and ctx.hand[0].suit == ctx.top.suit)),
    RuleTemplate("pair_win", "win",
                 lambda: "You also win if, at the end of your turn, your hand is exactly two cards of the same value.",
                 lambda: (lambda ctx: len(ctx.hand) == 2 and ctx.hand[0].rank == ctx.hand[1].rank)),
    RuleTemplate("tribute_on_rank", "effect",
                 lambda r: f"Playing one of the {_rank_word(r)} makes the next player give you their highest card.",
                 lambda r: (lambda ctx: ctx.card.rank == r),
                 lambda rng: {"r": rng.choice([1, 11, 12, 13])}, mechanics=True, weight=0.5,
                 effect=lambda r: (lambda ctx, fx: fx.move_card(fx.hand(ctx.next_player), fx.hand(ctx.player),
                                                                1, "highest"))),
    RuleTemplate("pot_ante", "effect",
                 lambda: ("Playing a face card puts one of your tokens into the pot on DISCARD; "
                          "playing an Ace takes the whole pot."),
                 lambda: (lambda ctx: (ctx.card.is_face and ctx.tokens[ctx.player] > 0) or (ctx.card.rank == 1 and ctx.pot > 0)),
                 mechanics=True, weight=0.5,
                 effect=lambda: (lambda ctx, fx: fx.move_tokens(fx.hand(ctx.player), fx.discard(), 1) if ctx.card.is_face
                                 else fx.move_tokens(fx.discard(), fx.hand(ctx.player), "all"))),
]

_SUIT_WORDS = {"spades": "S", "hearts": "H", "diamonds": "D", "clubs": "C",
               "spade": "S", "heart": "H", "diamond": "D", "club": "C", "s": "S", "h": "H", "d": "D", "c": "C"}

TEMPLATES: dict[str, RuleTemplate] = {t.name: t for t in _LIST}


# --------------------------------------------------------------------------
# Generating rules that are playable and actually matter
# --------------------------------------------------------------------------

_DECLARATIONS = ("hearts", "spades", "diamonds", "clubs", "red", "black", "last card", "7", "king")


def sample_contexts(rng: random.Random, n: int = 400, mechanics: bool = False) -> list[tuple[PlayContext, list[Card]]]:
    """Random game situations: (context template, candidate cards). With `mechanics`, the
    table (tokens, spaces, angles, declaration) and the move's options and extra
    cards are randomized too; the candidate replaces the move's first card."""
    out = []
    deck = full_deck()
    for _ in range(n):
        rng.shuffle(deck)
        num_players = rng.randint(2, 6)
        pile = tuple(deck[: rng.randint(1, 6)])
        player = rng.randrange(num_players)
        sizes = tuple(rng.randint(1, 12) for _ in range(num_players))
        turn = rng.randint(0, 120)
        candidates = [c for c in deck if c not in pile]
        move = table = None
        if mechanics:
            rest = candidates[1:]
            extra: tuple[Card, ...] = ()
            if rng.random() < 0.25:
                same = [c for c in rest if c.rank == candidates[0].rank]
                pool = same if same and rng.random() < 0.5 else rest
                extra = tuple(rng.sample(pool, min(len(pool), rng.choice([1, 1, 2]))))
            tokens = tuple(rng.choice([0, 0, 1, 2, 3, 4]) for _ in range(num_players))
            declarer = rng.randrange(num_players) if rng.random() < 0.5 else None
            move = Move((candidates[0],) + extra,
                        angle=rng.choice(ANGLES) if rng.random() < 0.3 else 0,
                        spend=rng.randint(1, tokens[player]) if tokens[player] and rng.random() < 0.3 else 0,
                        target=rng.choice([s for s in range(num_players) if s != player]) if rng.random() < 0.3 else None,
                        declare=rng.choice(_DECLARATIONS) if rng.random() < 0.2 else None)
            used = set(pile) | set(move.cards)
            free = [c for c in deck if c not in used]
            spaces = tuple(tuple(Placed(free.pop(), rng.choice(ANGLES)) for _ in range(rng.choice([0, 0, 1, 2, 3])))
                           for _ in range(num_players))
            table = TableState(tokens=tokens, spaces=spaces,
                               space_tokens=tuple(rng.choice([0, 0, 0, 1, 2]) for _ in range(num_players)),
                               pot=rng.choice([0, 0, 1, 2, 3, 5]),
                               angles=tuple(rng.choice(ANGLES) if rng.random() < 0.3 else 0 for _ in pile),
                               declared=(rng.choice(_DECLARATIONS[:4] if rng.random() < 0.6 else _DECLARATIONS)
                                         if declarer is not None else None),
                               declared_by=declarer, game=rng.randint(2, 9), mechanics=True)
            candidates = [c for c in candidates if c not in extra]
        out.append((PlayContext(candidates[0], pile, player, sizes, turn, num_players, move, table), candidates))
    return out


def legal_fraction(rules: list[Rule], ctx: PlayContext, candidates: list[Card]) -> float:
    ok = sum(decide(rules, _single(_with_card(ctx, c))).allowed for c in candidates)
    return ok / len(candidates)


def _with_card(ctx: PlayContext, card: Card) -> PlayContext:
    """The same situation with `card` as the (first) card played."""
    move = ctx.move
    if move is not None:
        m = move
        if len(m.cards) > 1 and all(c.rank == m.cards[0].rank for c in m.cards):
            # keep a set a set: the extra cards become other suits of the candidate's value
            extra = tuple(Card(card.rank, su) for su in SUITS if su != card.suit)[:len(m.cards) - 1]
        else:
            extra = tuple(c for c in m.cards[1:] if c != card)
        move = Move((card,) + extra, m.angle, m.spend, m.target, m.declare, m.give)
    return PlayContext(card, ctx.pile, ctx.player, ctx.hand_sizes, ctx.turn, ctx.num_players, move, ctx.table)


def _single(ctx: PlayContext) -> PlayContext:
    """The same play with only its first card (keeping the options)."""
    m = ctx.move
    if m is None or len(m.cards) == 1:
        return ctx
    return PlayContext(ctx.card, ctx.pile, ctx.player, ctx.hand_sizes, ctx.turn, ctx.num_players,
                       Move((ctx.card,), m.angle, m.spend, m.target, m.declare, m.give), ctx.table)


def win_contexts(samples) -> list[WinContext]:
    """End-of-turn situations derived from play samples: the player holds hand_size cards
    of the (shuffled) candidates (four different hands per sample) and has just played the
    sample's move."""
    out = []
    for ctx, cands in samples:
        rest = [c for c in cands if c not in ctx.cards]
        pile = ctx.pile + ctx.cards
        angles = None
        if ctx.table is not None:
            angles = TableState(**{**ctx.table.__dict__, "angles": ctx.pile_angles + (ctx.angle,) * len(ctx.cards)})
        sizes = tuple(max(h - len(ctx.cards), 0) if i == ctx.player else h for i, h in enumerate(ctx.hand_sizes))
        k = sizes[ctx.player]
        for j in range(4 if k else 1):  # several different hands per situation, for a steadier estimate
            if (j + 1) * k <= len(rest):
                out.append(WinContext(pile, ctx.player, sizes, ctx.turn, ctx.num_players,
                                      tuple(rest[j * k:(j + 1) * k]), ctx.cards, False, angles))
    return out


def win_share(rule, samples) -> float:
    """Share of sampled end-of-turn situations in which a win-condition rule is met."""
    wcs = win_contexts(samples)
    hits = 0
    for w in wcs:
        try:
            hits += bool(rule.wins(w))
        except Exception:  # noqa: BLE001
            pass
    return hits / len(wcs)


def effect_share(rule, samples) -> float:
    """Share of sampled plays on which an effect rule does something."""
    hits = total = 0
    for ctx, cands in samples:
        for c in cands[:13]:
            fx = Effects(ctx.num_players)
            try:
                rule.effect(_with_card(ctx, c), fx)
            except Exception:  # noqa: BLE001
                pass
            hits += bool(fx.ops)
            total += 1
    return hits / total


# --------------------------------------------------------------------------
# A rule may only use table components that exist (user decision)
# --------------------------------------------------------------------------
# Every game starts with no tokens anywhere and every space empty. Tokens come into the game
# only from the bank, and cards reach a space only, through effects. So a rule may use
# tokens (held, in spaces, the pot, SPEND) or cards in spaces only if some rule in effect, or
# the rule itself, can actually bring them into the game.

COMPONENTS = ("tokens", "spaces")
MISSING_COMPONENT = {
    "tokens": ("it uses tokens, but no rule in effect ever gives anyone a token (every game starts with no "
               "tokens; they only come from the bank, through an effect)"),
    "spaces": ("it uses cards in players' spaces, but no rule in effect ever puts a card into a space (every "
               "game starts with the spaces empty; cards only get there through an effect)"),
}


def _without(ctx, components: Iterable[str]):
    """The same situation with the given components absent: no tokens anywhere (so no SPEND
    either), or every space empty. Works for PlayContext and WinContext."""
    t = ctx.table
    if t is None:
        return ctx
    changes: dict = {}
    for comp in components:
        n = ctx.num_players
        if comp == "tokens":
            t = replace(t, tokens=(0,) * n, space_tokens=(0,) * n, pot=0)
            move = getattr(ctx, "move", None)
            if move is not None and move.spend:
                changes["move"] = replace(move, spend=0)
        elif comp == "spaces":
            t = replace(t, spaces=((),) * n)
    return replace(ctx, table=t, **changes)


def _op_components(op: tuple) -> tuple[set[str], set[str]]:
    """(provides, uses) for one recorded effect call (see mechanics.Effects)."""
    kind = op[0]
    if kind == "move_tokens":
        return ({"tokens"}, set()) if op[1] == ("bank",) else (set(), {"tokens"})
    if kind == "move_card":
        if op[1][0] == "space":
            return set(), {"spaces"}
        return ({"spaces"} if op[2][0] == "space" else set()), set()
    if kind == "to_space":
        return {"spaces"}, set()
    if kind == "rotate_space":
        return set(), {"spaces"}
    return set(), set()


def _effect_ops(rule, ctx) -> tuple | None:
    fx = Effects(ctx.num_players)
    try:
        rule.effect(ctx, fx)
    except Exception:  # noqa: BLE001 - a broken effect does nothing
        return None
    return tuple(fx.ops)


def _behaviour(rule, samples, absent: Iterable[str] = ()) -> list:
    """What the rule does on every sampled play (verdicts, effect calls, or win checks)."""
    absent = tuple(absent)
    if rule.kind == "win":
        out = []
        for w in win_contexts(samples):
            try:
                out.append(bool(rule.wins(_without(w, absent))))
            except Exception:  # noqa: BLE001
                out.append(None)
        return out
    out = []
    for ctx, cands in samples:
        base = _without(ctx, absent)
        for c in cands[:13]:
            cc = _with_card(base, c)
            if rule.kind == "effect":
                out.append(_effect_ops(rule, cc))
            else:
                try:
                    out.append(rule.verdict(cc))
                except Exception:  # noqa: BLE001
                    out.append(None)
    return out


def _uses_mechanics(rule) -> bool:
    return bool(getattr(rule, "mechanics", True))


def components_used(rule, samples) -> set[str]:
    """Components the rule takes from (an effect moving tokens or space cards away) or depends
    on (it acts differently when the component is absent)."""
    if not _uses_mechanics(rule) or not any(ctx.table is not None for ctx, _ in samples):
        return set()
    used: set[str] = set()
    base = _behaviour(rule, samples)
    if rule.kind == "effect":
        for ops in base:
            for op in ops or ():
                used |= _op_components(op)[1]
    for comp in COMPONENTS:
        if comp not in used and _behaviour(rule, samples, absent=(comp,)) != base:
            used.add(comp)
    return used


def components_available(rules: list, samples) -> set[str]:
    """Components the rules can bring into a game that starts without any. A rule that only
    brings in tokens when there already are some ("if you have a token, gain another") brings
    in none: sources are found with the still-missing components absent, until nothing changes."""
    effects = [r for r in rules if r.kind == "effect" and _uses_mechanics(r)]
    have: set[str] = set()
    while effects:
        absent = [c for c in COMPONENTS if c not in have]
        found = set(have)
        for r in effects:
            for ops in _behaviour(r, samples, absent=absent):
                for op in ops or ():
                    found |= _op_components(op)[0]
        if found == have:
            break
        have = found
    return have


def missing_components(existing: list, new, samples, have: set[str] | None = None) -> list[str]:
    """Components `new` uses that neither the rules in effect nor `new` itself bring into the game."""
    used = components_used(new, samples)
    if have is None:
        have = components_available(existing, samples)
    if used <= have:
        return []
    have = components_available(list(existing) + [new], samples)
    return [c for c in COMPONENTS if c in used and c not in have]


@dataclass
class RuleCheck:
    mean_legal: float       # average fraction of cards legal after adding the rule
    min_ok_share: float     # share of situations with at least min_legal_cards legal cards
    changed_share: float    # share of (situation, card) verdicts the rule changes


def evaluate_candidate(existing: list[Rule], new: Rule, samples, min_legal_cards: int = 3) -> RuleCheck:
    """Playability is measured on single-card plays (a player can always play one card; with
    table mechanics R0 makes most multi-card samples illegal). Verdict changes are counted on
    the whole sampled move, so a rule about multi-card plays still counts as making a difference."""
    total = changed = 0
    fracs, ok_situations = [], 0
    rules_after = existing + [new]
    for ctx, cands in samples:
        legal = 0
        for c in cands:
            cc = _with_card(ctx, c)
            before = decide(existing, cc).allowed
            after = decide(rules_after, cc).allowed
            changed += before != after
            total += 1
            single = _single(cc)
            legal += after if single is cc else decide(rules_after, single).allowed
        fracs.append(legal / len(cands))
        ok_situations += legal >= min_legal_cards
    return RuleCheck(sum(fracs) / len(fracs), ok_situations / len(samples), changed / total)


def _mean_legal(rules: list[Rule], samples) -> float:
    return sum(legal_fraction(rules, ctx, cands) for ctx, cands in samples) / len(samples)


MIN_MEAN_LEGAL = 0.2   # a new rule must leave at least this share of cards legal on average,
MIN_OK_SHARE = 0.85    # ...at least 3 legal cards in this share of situations,
MIN_CHANGED = 0.01     # ...and flip at least this share of verdicts compared with the current stack.
MIN_EFFECT_SHARE = 0.02  # an effect rule must do something on at least this share of plays,
MAX_EFFECT_SHARE = 0.5   # ...and at most this share.
MIN_WIN_SHARE = 0.001    # a win condition must be met at the end of at least this share of sampled turns,
MAX_WIN_SHARE = 0.03     # ...and at most this share (so it doesn't end games almost at once).


def rule_problem(existing: list, new, samples, *, first: bool | None = None) -> str | None:
    """Why `new` can't join the stack, in words its author can act on, or None if it can.

    Used to vet rules that players make up (see judges.ProposedRuleAuthor); the
    thresholds are the ones generate_rule applies to library rules."""
    first = (not existing) if first is None else first
    if any(r.key() == new.key() for r in existing):
        return "it is the same as a rule already in effect"
    if first and new.kind != "restriction":
        return ("the first rule must be a restriction (it must forbid some plays); "
                "a permission does nothing when no other rule forbids anything")
    missing = missing_components(existing, new, samples)
    if missing:
        return MISSING_COMPONENT[missing[0]]
    if new.kind == "win":
        if first:
            return "the first rule must be a restriction"
        share = win_share(new, samples)
        if share < MIN_WIN_SHARE:
            return f"it would almost never let anyone win ({share:.2%} of turns)"
        if share > MAX_WIN_SHARE:
            return (f"it would make winning too easy (met at the end of {share:.1%} of turns; "
                    f"at most {MAX_WIN_SHARE:.0%} is allowed)")
        return None
    if new.kind == "effect":
        share = effect_share(new, samples)
        if share < MIN_EFFECT_SHARE:
            return f"its effect would almost never happen ({share:.1%} of plays)"
        if share > MAX_EFFECT_SHARE:
            return f"its effect would happen too often ({share:.0%} of plays; at most {MAX_EFFECT_SHARE:.0%})"
        return None
    check = evaluate_candidate(existing, new, samples)
    if check.mean_legal < MIN_MEAN_LEGAL:
        return (f"it would leave too few cards legal: with the rules in effect, about "
                f"{check.mean_legal:.0%} of cards would be legal on average; at least {MIN_MEAN_LEGAL:.0%} must be")
    if check.min_ok_share < MIN_OK_SHARE:
        return (f"too often fewer than 3 cards would be legal (in {1 - check.min_ok_share:.0%} of situations; "
                f"at most {1 - MIN_OK_SHARE:.0%} is allowed)")
    if check.changed_share < MIN_CHANGED:
        return ("it would almost never change whether a play is legal, given the rules already in effect "
                "(it may duplicate an existing rule, or apply too rarely)")
    return None


def rule_from_json(d: dict):
    """Rebuild a library Rule or a coded rule (sandbox.CodeRule) from its JSON."""
    if d.get("template") == "code":
        from .sandbox import CodeRule
        return CodeRule.from_json(d)
    return Rule.from_json(d)


def generate_rule(existing: list[Rule], rng: random.Random, *, min_mean_legal: float = MIN_MEAN_LEGAL,
                  min_ok_share: float = MIN_OK_SHARE, min_changed: float = MIN_CHANGED,
                  permission_share: float = 0.4, loosen_below: float = 0.4,
                  max_attempts: int = 300, mechanics: bool = False, effect_share: float = 0.3,
                  wins: bool = False, win_share: float = 0.25) -> Rule:
    """Pick a random rule that keeps the game playable and changes some verdicts.

    Playable means at least 20% of cards are legal on average, and at least 3 cards
    are legal in 85% of sampled situations. The bar is lenient on purpose: rules like
    "no lower card on a higher card" leave almost nothing legal on a King, but an
    illegal card stays on DISCARD and resets the top, so the game never stalls.
    "Changes some verdicts" means it flips at least 1% of them compared with the
    current stack.

    With `mechanics` (rules that start in game 3 or later) the game-3+ templates are in
    the mix, and a rule is an effect with probability `effect_share`. With `wins` (rules that
    start in game 6 or later) a rule is a win condition with probability `win_share`.

    The first rule is always a restriction, because a permission alone does nothing.
    After that, a rule is a permission with probability `permission_share`, so the
    stack doesn't only ever tighten. If the current stack already leaves less than
    `loosen_below` of cards legal on average, only permissions are drawn: with
    restrictions alone the game can collapse until nothing is playable.
    """
    samples = sample_contexts(rng, 300, mechanics=mechanics)
    have = components_available(existing, samples) if mechanics else set()
    used = {r.key() for r in existing}
    by_kind = {k: [t for t in _LIST if t.kind == k and (mechanics or not t.mechanics)]
               for k in ("restriction", "permission", "effect", "win")}
    if not existing:
        p_perm = 0.0
    elif _mean_legal(existing, samples) < loosen_below:
        p_perm = 1.0
    else:
        p_perm = permission_share
    for _ in range(max_attempts):
        if wins and existing and rng.random() < win_share:
            kind = "win"
        elif mechanics and existing and rng.random() < effect_share:
            kind = "effect"
        else:
            kind = "permission" if rng.random() < p_perm else "restriction"
        templates = by_kind[kind]
        t = rng.choices(templates, [t.weight for t in templates])[0]
        rule = Rule(t.name, t.sample_params(rng))
        if rule.key() in used:
            continue
        if kind in ("effect", "win"):
            if rule_problem(existing, rule, samples) is None:
                return rule
            continue
        if mechanics and missing_components(existing, rule, samples, have):
            continue
        check = evaluate_candidate(existing, rule, samples)
        if (check.mean_legal >= min_mean_legal and check.min_ok_share >= min_ok_share
                and check.changed_share >= min_changed):
            return rule
    raise RuntimeError("could not find a playable, non-redundant rule; the stack may be saturated")

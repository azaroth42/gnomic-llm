"""One game of Gnomic Mau, plus the public log and player views.

Each turn:
  1. The current player plays a card onto DISCARD. The card stays there whether or not it was legal.
  2. Every player who owns at least one rule answers PENALTY or NONE, in parallel and without
     seeing each other's answers. Players police only their own rules, so a player with no
     rules cannot call and is not asked. The player who made the play (if they own a rule)
     gives their own call in the same reply as the play.
  3. The engine knows the truth. A penalty call names one of the caller's own rules ("a
     penalty under my rule R1"). It is upheld if the play broke that rule (the rule forbids it
     and no later rule allows it). If any call is upheld, the player draws 1 card. Each caller
     whose call is not upheld draws 1 for a false call.
     A violation its owner doesn't call costs nothing and is never revealed.
  4. From game 3 (`mechanics`, see mechanics.py) a play is a *move* with options, and if no
     penalty was upheld the rules' effects happen (reverse, skip, tokens, spaces, trades...).
     A multi-card move that no rule permits breaks R0, which the engine calls itself.
  5. A player who has emptied their hand after steps 3 and 4 wins.

A game with no winner ends after `max_rounds` rounds (max_rounds x players turns; with skips
and extra turns that is no longer exactly every player's card count).
"""

from __future__ import annotations

import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from .cards import Card, shuffled_deck
from .mechanics import ANGLES, BASE_RULE, MAX_DECLARE, Move, Placed, TableState, normalize_declaration
from .rules import PlayContext, RuleStack, WinContext

if TYPE_CHECKING:
    from .players.base import Player


# --------------------------------------------------------------------------
# Public log: what every player at the table can see
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PublicEvent:
    kind: str                     # rule_added | game_start | play | reshuffle | game_end
    game: int
    turn: int | None = None
    player: int | None = None
    card: Card | None = None
    top_before: Card | None = None
    calls: tuple[int, ...] = ()           # seats that called PENALTY
    call_rules: tuple[int | None, ...] = ()  # rule number each caller named, aligned with calls
    upheld: bool | None = None            # was any call upheld (the player drew)? None if nobody called
    upheld_by: tuple[int, ...] = ()       # callers whose own rule the play broke; the rest were false calls
    penalized: tuple[int, ...] = ()       # seats that drew a card (a seat can appear twice)
    winner: int | None = None
    detail: str = ""
    move: str = ""                        # the public form of a move with options or several cards
    engine_call: bool = False             # the engine called R0 (an unpermitted multi-card play)
    effects: tuple[str, ...] = ()         # public descriptions of the effects that happened

    def to_json(self) -> dict:
        out = {}
        for k, v in self.__dict__.items():
            if v is None or v == "" or v == ():
                continue
            out[k] = str(v) if isinstance(v, Card) else v
        return out


class PublicLog:
    """Append-only log shared across the whole match."""

    def __init__(self) -> None:
        self.events: list[PublicEvent] = []

    def add(self, ev: PublicEvent) -> None:
        self.events.append(ev)


# --------------------------------------------------------------------------
# What one player is allowed to know
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PlayerView:
    seat: int
    player_names: tuple[str, ...]
    hand: tuple[Card, ...]
    discard: tuple[Card, ...]          # bottom -> top
    deck_size: int
    hand_sizes: tuple[int, ...]
    game: int                          # 0-based game index
    num_games: int
    turn: int                          # 0-based turn within the game
    num_rules: int
    scores: tuple[int, ...]            # games won per seat
    log: tuple[PublicEvent, ...]       # full public log of the match so far
    my_rules: tuple[tuple[int, str], ...] = ()  # (rule number, text) for rules this player made up
    max_rounds: int = 20               # a game with no winner ends after this many rounds
    judges: tuple[int, ...] = ()       # seats that own at least one rule, i.e. may call PENALTY
    players_make_rules: bool = False   # do the dealer and winners make up their own rules (--propose-rules)?
    table: TableState | None = None    # game 3+ table state (direction, tokens, spaces, angles, declaration)
    private: tuple[str, ...] = ()      # private notices for this player (trades, peeks), whole match
    mechanics_from: int = 0            # 1-based game from which mechanics apply; 0 = never
    wins_from: int = 0                 # 1-based game from which rules may be win conditions; 0 = never

    @property
    def mechanics(self) -> bool:
        return self.table is not None and self.table.mechanics

    @property
    def my_tokens(self) -> int:
        return self.table.tokens[self.seat] if self.table and self.table.tokens else 0

    @property
    def can_call(self) -> bool:
        return bool(self.my_rules)

    @property
    def my_rule_numbers(self) -> tuple[int, ...]:
        return tuple(n for n, _ in self.my_rules)

    @property
    def top(self) -> Card | None:
        return self.discard[-1] if self.discard else None

    def context_for(self, card: Card, seat: int | None = None, move: Move | None = None) -> PlayContext:
        return PlayContext(card, self.discard, self.seat if seat is None else seat,
                           self.hand_sizes, self.turn, len(self.player_names), move, self.table)


@dataclass(frozen=True)
class PendingPlay:
    """A play that has just been made, which the other players must judge."""
    seat: int
    card: Card
    top_before: Card | None
    turn: int
    move: Move | None = None   # set when the move has options or several cards

    def public(self) -> str:
        return self.move.public() if self.move else str(self.card)


# --------------------------------------------------------------------------
# The game
# --------------------------------------------------------------------------

@dataclass
class TurnStat:
    player: int
    legal: bool
    legal_options: int         # legal cards the player held (from the engine's knowledge)
    forced: bool               # engine picked a card because the player's choice was unusable
    elapsed: float
    calls: dict[int, bool]     # seat -> called PENALTY? (only seats that were allowed to call)
    call_forced: dict[int, bool] = field(default_factory=dict)  # seat -> no usable answer, counted as NONE
    meta: dict = field(default_factory=dict)
    can_self_call: bool = True  # did the player own a rule, so that a self-call was possible?
    violated_owners: frozenset[int] = frozenset()  # seats whose own rules the play broke
    upheld_by: frozenset[int] = frozenset()        # callers who named a rule of theirs that the play broke
    call_rules: dict[int, int | None] = field(default_factory=dict)  # caller seat -> rule number named
    engine_call: bool = False  # the engine called R0 on this play


@dataclass
class GameResult:
    index: int
    winner: int | None
    turns: int
    end_reason: str
    stats: list[TurnStat]


class Controls:
    """Pause, stop, and pacing controls, shared with the web UI."""

    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self._running = threading.Event()
        self._running.set()
        self.stop_requested = False

    def pause(self) -> None:
        self._running.clear()

    def resume(self) -> None:
        self._running.set()

    def stop(self) -> None:
        self.stop_requested = True
        self._running.set()

    @property
    def paused(self) -> bool:
        return not self._running.is_set()

    def gate(self) -> None:
        self._running.wait()
        if self.stop_requested:
            raise StopMatch()
        if self.delay:
            time.sleep(self.delay)


class StopMatch(Exception):
    pass


class Game:
    def __init__(
        self,
        index: int,
        num_games: int,
        players: list["Player"],
        stack: RuleStack,
        log: PublicLog,
        scores: list[int],
        rng: random.Random,
        starting_player: int,
        emit: Callable[..., object],
        controls: Controls | None = None,
        hand_size: int = 7,
        max_rounds: int = 20,
        max_turns: int | None = None,  # hard cap on turns; overrides max_rounds (used by tests)
        players_make_rules: bool = False,
        mechanics: bool = False,       # game-3+ table mechanics are in play this game
        mechanics_from: int = 0,
        private: list[list[str]] | None = None,  # per-seat private notices, shared across the match
        wins_from: int = 0,
    ):
        self.index, self.num_games = index, num_games
        self.players, self.stack, self.log = players, stack, log
        self.scores, self.rng, self.emit = scores, rng, emit
        self.controls = controls or Controls()
        self.current = starting_player
        self.n = len(players)
        self.hand_size, self.max_rounds = hand_size, max_rounds
        self.players_make_rules = players_make_rules
        self.mechanics, self.mechanics_from = mechanics, mechanics_from
        self.wins_from = wins_from
        self.private = private if private is not None else [[] for _ in players]
        self.direction = 1
        self.tokens = [0] * len(players)
        self.space_tokens = [0] * len(players)
        self.pot = 0
        self.spaces: list[list[Placed]] = [[] for _ in players]
        self.angles: list[int] = []           # parallel to self.discard
        self.declared: tuple[str, int] | None = None
        self.max_turns = max_turns if max_turns is not None else max_rounds * self.n
        self.deck: list[Card] = []
        self.discard: list[Card] = []
        self.hands: list[list[Card]] = [[] for _ in players]
        self.turn = 0
        self._pool = ThreadPoolExecutor(max_workers=max(1, self.n - 1), thread_name_prefix="call")

    # ---- setup and helpers -------------------------------------------------

    def _names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.players)

    def _deal(self) -> None:
        self.deck = shuffled_deck(self.rng)
        for _ in range(self.hand_size):
            for seat in range(self.n):
                self.hands[seat].append(self.deck.pop())
        self.discard = [self.deck.pop()]
        self.angles = [0]

    def _reshuffle(self) -> bool:
        """Base rule: when a player must draw from an empty DECK, DISCARD (except its
        top card, which stays put) is shuffled to form the new DECK."""
        if len(self.discard) <= 1:
            return False
        top = self.discard[-1]
        self.deck = self.discard[:-1]
        self.rng.shuffle(self.deck)
        self.discard = [top]
        self.angles = [self.angles[-1] if self.angles else 0]
        self.log.add(PublicEvent("reshuffle", self.index, self.turn, card=top, detail=f"{len(self.deck)} cards"))
        self.emit("reshuffle", game=self.index, turn=self.turn, deck_size=len(self.deck), top=top)
        return True

    def _draw(self, seat: int, reason: str) -> Card | None:
        if not self.deck and not self._reshuffle():
            self.emit("draw_failed", game=self.index, seat=seat, reason=reason)
            return None
        card = self.deck.pop()
        self.hands[seat].append(card)
        self.emit("draw", game=self.index, seat=seat, player=self.players[seat].name, card=card,
                  reason=reason, hand_size=len(self.hands[seat]), deck_size=len(self.deck))
        return card

    def _my_rules(self, seat: int) -> tuple[tuple[int, str], ...]:
        name = self.players[seat].name
        return tuple((i + 1, e.rule.description) for i, e in enumerate(self.stack.entries) if e.owner == name)

    def can_call(self, seat: int) -> bool:
        """Only a player who owns at least one rule may call a penalty."""
        name = self.players[seat].name
        return any(e.owner == name for e in self.stack.entries)

    def judges(self) -> tuple[int, ...]:
        return tuple(s for s in range(self.n) if self.can_call(s))

    def view_for(self, seat: int) -> PlayerView:
        return PlayerView(
            seat=seat,
            player_names=self._names(),
            hand=tuple(sorted(self.hands[seat], key=lambda c: (c.suit, c.rank))),
            discard=tuple(self.discard),
            deck_size=len(self.deck),
            hand_sizes=tuple(len(h) for h in self.hands),
            game=self.index,
            num_games=self.num_games,
            turn=self.turn,
            num_rules=len(self.stack),
            scores=tuple(self.scores),
            log=tuple(self.log.events),
            my_rules=self._my_rules(seat),
            max_rounds=self.max_rounds,
            judges=self.judges(),
            players_make_rules=self.players_make_rules,
            table=self._table(),
            private=tuple(self.private[seat]),
            mechanics_from=self.mechanics_from,
            wins_from=self.wins_from,
        )

    def _table(self) -> TableState | None:
        if not self.mechanics:
            return None
        return TableState(direction=self.direction, tokens=tuple(self.tokens),
                          spaces=tuple(tuple(sp) for sp in self.spaces), angles=tuple(self.angles),
                          space_tokens=tuple(self.space_tokens), pot=self.pot,
                          declared=self.declared[0] if self.declared else None,
                          declared_by=self.declared[1] if self.declared else None,
                          game=self.index, mechanics=True)

    def _ctx(self, seat: int, card: Card, move: Move | None = None) -> PlayContext:
        return PlayContext(card, tuple(self.discard), seat,
                           tuple(len(h) for h in self.hands), self.turn, self.n, move, self._table())

    def valid_move(self, seat: int, move: Move) -> str | None:
        """Why a move can't be made at all (not a rules question), or None. Players are asked again."""
        hand = self.hands[seat]
        if not move.cards or any(c not in hand for c in move.cards) or len(set(move.cards)) != len(move.cards):
            return "every card you play must be a different card from your hand"
        if move.has_options and not self.mechanics:
            return "options and multi-card plays are only available from game " + str(self.mechanics_from or "?")
        if move.angle not in ANGLES:
            return f"ANGLE must be one of {', '.join(map(str, ANGLES))}"
        if not 0 <= move.spend <= self.tokens[seat]:
            return f"you can spend at most the {self.tokens[seat]} token(s) you have"
        if move.target is not None and not (0 <= move.target < self.n and move.target != seat):
            return "TARGET must be another player"
        if move.declare is not None and not 0 < len(move.declare) <= MAX_DECLARE:
            return f"DECLARE must be 1-{MAX_DECLARE} characters"
        if move.give is not None and (move.give not in hand or move.give in move.cards):
            return "GIVE must be a card in your hand that you are not playing"
        return None

    # ---- effects (game 3+) -------------------------------------------------

    def _note(self, seat: int, text: str) -> None:
        self.private[seat].append(f"G{self.index + 1} T{self.turn + 1} {text}")

    def _apply(self, seat: int, move: Move, op: tuple) -> str | None:
        """Apply one effect; return its public description (None if it did nothing)."""
        P = lambda s: f"P{s + 1}"  # noqa: E731
        kind, args = op[0], op[1:]
        if kind == "reverse":
            self.direction = -self.direction
            return "order of play reversed"
        if kind == "skip":
            return "next player skipped"
        if kind == "extra_turn":
            return f"{P(seat)} plays again"
        if kind == "move_tokens":
            return self._move_tokens(*args)
        if kind == "move_card":
            src, dst, k, which = args
            done = [t for t in (self._move_card(seat, move, src, dst, which) for _ in range(k)) if t]
            return "; ".join(done) or None
        if kind == "draw":
            s, k = args
            got = sum(self._draw(s, "effect") is not None for _ in range(k))
            return f"{P(s)} draws {got}" if got else None
        if kind == "to_space":
            (s,) = args
            moved = []
            for c in move.cards:
                if c in self.discard and len(self.discard) > 1:
                    i = len(self.discard) - 1 - self.discard[::-1].index(c)
                    self.discard.pop(i)
                    self.spaces[s].append(Placed(c, self.angles.pop(i)))
                    moved.append(str(c))
            return f"{' '.join(moved)} moved from DISCARD to {P(s)}'s space" if moved else None
        if kind == "rotate_top":
            (a,) = args
            self.angles[-1] = a
            return f"top DISCARD card {self.discard[-1]} turned to {a}"
        if kind == "rotate_space":
            s, a = args
            if not self.spaces[s]:
                return None
            self.spaces[s][-1] = Placed(self.spaces[s][-1].card, a)
            return f"{self.spaces[s][-1].card} in {P(s)}'s space turned to {a}"
        if kind == "trade":
            a, b = args
            if not self.hands[a] or not self.hands[b]:
                return None
            give = move.give if a == seat and move.give in self.hands[a] else self.rng.choice(self.hands[a])
            take = self.rng.choice(self.hands[b])
            self.hands[a].remove(give)
            self.hands[b].remove(take)
            self.hands[a].append(take)
            self.hands[b].append(give)
            self._note(a, f"trade: you gave {give} to {P(b)} and took {take}")
            self._note(b, f"trade: {P(a)} gave you {give} and took your {take}")
            return f"{P(a)} traded a card with {P(b)}"
        if kind == "peek_deck":
            (s,) = args
            if not self.deck:
                return None
            self._note(s, f"peek: the top card of DECK is {self.deck[-1]}")
            return f"{P(s)} peeked at the top card of DECK"
        if kind == "peek_hand":
            s, o = args
            if not self.hands[o]:
                return None
            self._note(s, f"peek: {P(o)} holds {self.rng.choice(self.hands[o])}")
            return f"{P(s)} peeked at a card in {P(o)}'s hand"
        return None

    @staticmethod
    def _where(loc: tuple, tokens: bool = False) -> str:
        if loc[0] == "hand":
            return f"P{loc[1] + 1}" if tokens else f"P{loc[1] + 1}'s hand"
        if loc[0] == "space":
            return f"P{loc[1] + 1}'s space"
        if loc[0] == "discard":
            return "the pot" if tokens else "DISCARD"
        return "the bank"

    def _pick(self, cards: list, which: str, newest_default: bool) -> int:
        """Index of the card to take: by `which`, ties broken at random."""
        if which in ("highest", "lowest"):
            best = (max if which == "highest" else min)(c.rank for c in cards)
            return self.rng.choice([i for i, c in enumerate(cards) if c.rank == best])
        if which == "oldest":
            return 0
        if which == "random" or not newest_default:
            return self.rng.randrange(len(cards))
        return len(cards) - 1  # newest / top

    def _move_card(self, seat: int, move: Move, src: tuple, dst: tuple, which: str) -> str | None:
        """Move one card between hands, spaces and DISCARD; return the public description."""
        angle = 0
        if src[0] == "hand":
            hand = self.hands[src[1]]
            if not hand:
                return None
            if which == "chosen" and src[1] == seat and move.give in hand:
                i = hand.index(move.give)
            else:
                i = self._pick(hand, which, newest_default=False)
            card = hand.pop(i)
        elif src[0] == "space":
            space = self.spaces[src[1]]
            if not space:
                return None
            placed = space.pop(self._pick([p.card for p in space], which, newest_default=True))
            card, angle = placed.card, placed.angle
        else:
            if len(self.discard) <= 1:
                return None  # DISCARD always keeps a top card
            i = self._pick(self.discard, which, newest_default=True)
            card, angle = self.discard.pop(i), self.angles.pop(i)
        if dst[0] == "hand":
            self.hands[dst[1]].append(card)
        elif dst[0] == "space":
            self.spaces[dst[1]].append(Placed(card, angle))
        else:
            self.discard.append(card)
            self.angles.append(angle)
        if src[0] == "hand" and dst[0] == "hand":
            a, b = src[1], dst[1]
            self._note(a, f"move: your {card} went to P{b + 1}'s hand")
            self._note(b, f"move: you received {card} from P{a + 1}'s hand")
            return f"a card moved from {self._where(src)} to {self._where(dst)}"
        return f"{card} moved from {self._where(src)} to {self._where(dst)}"

    def _tokens_at(self, loc: tuple) -> int | None:
        if loc[0] == "hand":
            return self.tokens[loc[1]]
        if loc[0] == "space":
            return self.space_tokens[loc[1]]
        if loc[0] == "discard":
            return self.pot
        return None  # bank: unlimited

    def _add_tokens(self, loc: tuple, k: int) -> None:
        if loc[0] == "hand":
            self.tokens[loc[1]] += k
        elif loc[0] == "space":
            self.space_tokens[loc[1]] += k
        elif loc[0] == "discard":
            self.pot += k

    def _move_tokens(self, src: tuple, dst: tuple, n) -> str | None:
        have = self._tokens_at(src)
        k = (have if n == "all" else n) if have is None else (have if n == "all" else min(n, have))
        if not k:
            return None
        self._add_tokens(src, -k)
        self._add_tokens(dst, k)
        plural = f"{k} token{'s' if k != 1 else ''}"
        if src == ("bank",) and dst[0] == "hand":
            return f"P{dst[1] + 1} gains {plural}"
        if dst == ("bank",) and src[0] == "hand":
            return f"P{src[1] + 1} loses {plural}"
        return f"{plural} moved from {self._where(src, True)} to {self._where(dst, True)}"

    def _table_json(self) -> dict | None:
        if not self.mechanics:
            return None
        return {"direction": self.direction, "tokens": list(self.tokens), "space_tokens": list(self.space_tokens),
                "pot": self.pot, "spaces": [[str(p) for p in sp] for sp in self.spaces], "angles": self.angles[-12:],
                "declared": list(self.declared) if self.declared else None}

    # ---- call phase --------------------------------------------------------

    def _named_rule(self, seat: int, rule: int | None) -> int | None:
        """The rule number a call by `seat` names: as given, or their only rule if they gave none."""
        mine = [n for n, _ in self._my_rules(seat)]
        if rule is None and len(mine) == 1:
            return mine[0]
        return rule

    def _call_stands(self, seat: int, rule: int | None, broken: list[int]) -> bool:
        """Upheld iff the named rule is the caller's own and the play broke it."""
        return (rule is not None and 1 <= rule <= len(self.stack) and rule - 1 in broken
                and self.stack.entries[rule - 1].owner == self.players[seat].name)

    def _collect_calls(self, pending: PendingPlay) -> tuple[dict[int, bool], dict[int, bool], dict[int, dict],
                                                            dict[int, int | None]]:
        """Ask every other player who owns a rule, in parallel. Each sees the table as it was
        before the play. Players without rules cannot call, so they are not asked."""
        from .players.base import Call

        callers = [s for s in range(self.n) if s != pending.seat and self.can_call(s)]
        views = {s: self.view_for(s) for s in callers}
        self.emit("call_phase", game=self.index, turn=self.turn, seat=pending.seat, card=pending.card,
                  callers=[self.players[s].name for s in callers])

        def ask(s: int):
            t0 = time.monotonic()
            try:
                call = self.players[s].call(views[s], pending)
            except Exception as e:
                call = Call(None, meta={"error": f"{type(e).__name__}: {e}"})
            elapsed = time.monotonic() - t0
            self.emit("call", game=self.index, turn=self.turn, seat=s, player=self.players[s].name,
                      penalty=bool(call.penalty), rule=call.rule if call.penalty else None,
                      forced=call.penalty is None, elapsed=round(elapsed, 3),
                      reasoning=call.reasoning, notes=call.notes, meta=call.meta)
            return s, call, elapsed

        calls, forced, metas, rules = {}, {}, {}, {}
        if not callers:
            return calls, forced, metas, rules
        for s, call, elapsed in self._pool.map(ask, callers):
            calls[s] = bool(call.penalty)
            forced[s] = call.penalty is None
            metas[s] = {"elapsed": round(elapsed, 3), **call.meta}
            if call.penalty:
                rules[s] = call.rule
        return calls, forced, metas, rules

    # ---- main loop ---------------------------------------------------------

    def play(self) -> GameResult:
        from .players.base import Choice

        self._deal()
        if len(self.angles) != len(self.discard):
            self.angles = [0] * len(self.discard)
        self.log.add(PublicEvent("game_start", self.index, 0, player=self.current,
                                 card=self.discard[-1], detail=str(len(self.stack))))
        self.emit("game_start", game=self.index, starting_player=self.current,
                  top=self.discard[-1], deck_size=len(self.deck),
                  hands={p.name: [str(c) for c in h] for p, h in zip(self.players, self.hands)},
                  num_rules=len(self.stack), mechanics=self.mechanics, table=self._table_json())
        for p in self.players:
            p.on_game_start(self.index)

        stats: list[TurnStat] = []
        winner: int | None = None
        end_reason = "max_rounds"  # unless someone wins first
        try:
            while self.turn < self.max_turns:
                self.controls.gate()
                seat = self.current
                player = self.players[seat]
                hand = self.hands[seat]
                view = self.view_for(seat)
                self.emit("turn_start", game=self.index, turn=self.turn, seat=seat, player=player.name,
                          hand=[str(c) for c in view.hand], top=view.top,
                          round=self.turn // self.n, max_rounds=self.max_rounds)

                if not hand:
                    # "Play a card if you have any" (user decision). A hand emptied during someone
                    # else's turn (an effect took its last card) has nothing to play, and an empty
                    # hand at the end of your own turn wins.
                    self.log.add(PublicEvent("no_cards", self.index, self.turn, player=seat))
                    self.emit("no_cards", game=self.index, turn=self.turn, seat=seat, player=player.name)
                    self.turn += 1
                    winner, end_reason = seat, "emptied_hand"
                    break

                t0 = time.monotonic()
                try:
                    choice = player.choose(view)
                except Exception as e:  # a crashing player must not kill the match
                    choice = Choice(card=None, meta={"error": f"{type(e).__name__}: {e}"})
                elapsed = time.monotonic() - t0

                card, forced = choice.card, False
                move = choice.move if choice.move is not None and choice.move.has_options else None
                if move is not None and (move.card != card or self.valid_move(seat, move)):
                    move = None
                if card is None or card not in hand:
                    forced, move = True, None
                    card = self.rng.choice(hand)
                cards = move.cards if move else (card,)

                can_self_call = self.can_call(seat)
                self_call = bool(choice.self_call) and not forced and can_self_call
                legal_options = sum(self.stack.is_legal(self._ctx(seat, c)) for c in hand)
                ctx = self._ctx(seat, card, move)
                decision = self.stack.decide(ctx)
                broken = self.stack.violated(ctx)
                engine_call = BASE_RULE in broken
                owners = {self.stack.entries[i].owner for i in broken if i != BASE_RULE}
                violated_seats = frozenset(s for s in range(self.n) if self.players[s].name in owners)
                top_before = self.discard[-1] if self.discard else None
                pending = PendingPlay(seat, card, top_before, self.turn, move)
                if decision.deciding_index == BASE_RULE:
                    deciding = "R0 (engine): play exactly one card."
                elif decision.deciding_index is not None:
                    deciding = self.stack.entries[decision.deciding_index].rule.description
                else:
                    deciding = None
                self.emit("play", game=self.index, turn=self.turn, seat=seat, player=player.name,
                          card=card, move=move.to_json() if move else None, move_text=pending.public(),
                          top_before=top_before, legal=decision.allowed,
                          deciding_rule_index=decision.deciding_index, deciding_rule=deciding,
                          broken_rules=broken, violated_owners=sorted(violated_seats),
                          legal_options=legal_options, forced=forced, elapsed=round(elapsed, 3),
                          self_call=self_call, can_self_call=can_self_call,
                          self_call_rule=choice.self_call_rule if self_call else None,
                          reasoning=choice.reasoning, notes=choice.notes, meta=choice.meta)

                # The other players who own rules judge the play, seeing the table as it was
                # before the card moved. The player's own call came with their PLAY reply, and
                # counts only if they own a rule.
                calls, call_forced, call_meta, named = self._collect_calls(pending)
                if can_self_call:
                    calls[seat] = self_call
                    call_forced[seat] = forced
                    if self_call:
                        named[seat] = choice.self_call_rule
                calls = dict(sorted(calls.items()))
                for c in cards:
                    hand.remove(c)
                    self.discard.append(c)
                    self.angles.append(move.angle if move else 0)
                if move and move.spend:
                    self.tokens[seat] -= move.spend
                if move and move.declare:
                    self.declared = (normalize_declaration(move.declare), seat)

                # Players police their own rules: a call names one of the caller's rules and
                # stands only if the play broke that rule. The player draws once however many
                # calls stand; every other caller made a false call and draws.
                callers = tuple(s for s, c in calls.items() if c)
                call_rules = {s: self._named_rule(s, named.get(s)) for s in callers}
                upheld_by = tuple(s for s in callers if self._call_stands(s, call_rules[s], broken))
                rejected = [s for s in callers if s not in upheld_by]
                # R0 (one card per play) belongs to the engine, which calls it itself.
                penalty = bool(upheld_by) or engine_call
                penalized: list[int] = ([seat] if penalty else []) + rejected
                if penalty:
                    self._draw(seat, "penalty")
                for s in rejected:
                    self._draw(s, "false_call")
                upheld = penalty if (callers or engine_call) else None

                # Effects happen only if no penalty was upheld (user decision); they never touch penalties.
                effects: list[str] = []
                skips, extra_turn = 0, False
                if self.mechanics and not penalty:
                    for _, op in self.stack.effects(ctx):
                        text = self._apply(seat, move or Move((card,)), op)
                        skips += op[0] == "skip"
                        extra_turn |= op[0] == "extra_turn"
                        if text:
                            effects.append(text)
                    if effects:
                        self.emit("effects", game=self.index, turn=self.turn, seat=seat, effects=effects,
                                  table=self._table_json(),
                                  hands={p.name: [str(c) for c in h] for p, h in zip(self.players, self.hands)},
                                  private={self.players[i].name: self.private[i][-3:] for i in range(self.n)})

                self.log.add(PublicEvent("play", self.index, self.turn, player=seat, card=card,
                                         top_before=top_before, calls=callers,
                                         call_rules=tuple(call_rules[s] for s in callers), upheld=upheld,
                                         upheld_by=upheld_by, penalized=tuple(penalized),
                                         move=move.public() if move else "", engine_call=engine_call,
                                         effects=tuple(effects)))
                stats.append(TurnStat(seat, decision.allowed, legal_options, forced, elapsed,
                                      calls, call_forced,
                                      {"play": dict(choice.meta), "calls": call_meta},
                                      can_self_call, violated_seats, frozenset(upheld_by), call_rules,
                                      engine_call))
                self.emit("resolution", game=self.index, turn=self.turn, seat=seat, card=card,
                          legal=decision.allowed, callers=list(callers), upheld=upheld,
                          call_rules={str(s): call_rules[s] for s in callers},
                          upheld_by=list(upheld_by), rejected=rejected,
                          violated_owners=sorted(violated_seats),
                          missed=(not decision.allowed and not penalty), penalized=penalized,
                          engine_call=engine_call, effects=effects, table=self._table_json(),
                          hand_sizes=[len(h) for h in self.hands], deck_size=len(self.deck),
                          discard=[str(c) for c in self.discard[-12:]])

                # Wins are checked only at the end of the player's own turn (user decision): an
                # empty hand, or (game 6+) a win-condition rule. Nobody wins out of turn.
                won_by = None
                if hand:
                    wctx = WinContext(tuple(self.discard), seat, tuple(len(h) for h in self.hands), self.turn,
                                      self.n, tuple(hand), cards, penalty, self._table())
                    won_by = self.stack.winning_rule(wctx)
                self.turn += 1
                if not hand:
                    winner = seat
                    end_reason = "emptied_hand"
                    break
                if won_by is not None:
                    winner = seat
                    end_reason = f"win_rule R{won_by + 1}"
                    self.emit("win_rule", game=self.index, seat=seat, rule_index=won_by,
                              rule=self.stack.entries[won_by].rule.description,
                              hand=[str(c) for c in hand])
                    break
                if extra_turn:
                    self.current = seat
                else:
                    self.current = (seat + self.direction * (1 + skips)) % self.n
        finally:
            self._pool.shutdown(wait=False)

        self.log.add(PublicEvent("game_end", self.index, self.turn, winner=winner, detail=end_reason))
        self.emit("game_end", game=self.index, winner=winner,
                  winner_name=self.players[winner].name if winner is not None else None,
                  turns=self.turn, reason=end_reason)
        for p in self.players:
            p.on_game_end(self.index, winner)
        return GameResult(self.index, winner, self.turn, end_reason, stats)

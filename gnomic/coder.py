"""The coding agent: turns a player's plain-language rule into sandboxed code.

The engine (never a player) talks to the coder. The coder sees only the rule's
text, never the other rules in effect or anything about the players, so nothing
secret leaks through it. Each rule gets a fresh conversation:

  engine -> coder   RULE: <the player's text>
  coder  -> engine  KIND / DESCRIPTION / ```python def applies(ctx): ...```   or   CANNOT: <reason>

If the code fails the sandbox checks or the smoke test (crashes, returns a
non-bool, too slow), the engine sends the error back and the coder tries again,
up to `max_fixes` times. If it still fails, or the coder answers CANNOT, the rule
can't be implemented and the engine asks the rule's author for a different rule
(see judges.ProposedRuleAuthor).

Caching (user decision): the coder is never asked to code the same text twice. `CoderCache`
keys each result by (coder model, coder prompt, mechanics/wins allowances, normalized text),
whether the text is a proposed rule or a player's hypothesis, and for any rule number. It stores
code and also CANNOT / no-working-code outcomes (but not API failures, which are retried). Two
threads asking for the same text at once share one coder call. With a path, the cache persists
across matches as JSONL.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

from .providers import ChatSession
from .mechanics import ANGLES, MAX_COUNT
from .sandbox import RANGE_CAP, CodeRule, SandboxError, compile_rule, smoke_test

CODER_PROMPT = f"""\
You turn plain-language rules for a card game into Python code for the game engine.

THE GAME
Players take turns playing one card from their hand onto a face-up DISCARD pile. Hidden rules,
each made up by a player, decide whether a play is legal. You are given one such rule, written by
its author in plain language. Implement it faithfully.

Every rule is exactly one of two kinds:
  restriction  it forbids some plays and says nothing about the others.
  permission   it allows some plays (even if an older rule forbids them) and says nothing about the others.
"You must X" is a restriction that forbids not-X. A rule that both forbids and allows (e.g. "you must
play red, but Kings may always be played") is really two rules and cannot be implemented.

WHAT TO WRITE
Exactly one function, nothing else (no imports, no helpers, no constants outside it):
    def applies(ctx):
        ...
It returns True when the rule applies to this play: for a restriction, True means the play is
FORBIDDEN; for a permission, True means the play is ALLOWED. Otherwise it returns False. It must
always return a bool. If DISCARD is empty or too short for the rule, return False.

ctx, all of it public and read-only:
  ctx.card          the card being played
  ctx.pile          tuple of DISCARD cards before this play, bottom to top; ctx.pile[-1] is the top card
  ctx.top           the top DISCARD card, or None if DISCARD is empty
  ctx.back(n)       n=1 the top card, n=2 the card under it, ...; None if DISCARD is shorter than n
  ctx.player        seat number (0-based) of the player making the play
  ctx.hand_sizes    tuple of every player's hand size before this play, by seat
  ctx.hand_size     the playing player's hand size before this play (it includes the card being played)
  ctx.turn          turn number within the current game, 0-based (the first play of a game is turn 0)
  ctx.num_players   number of players
A card has:
  .rank       int 1..13 (A=1, 2..10, J=11, Q=12, K=13)
  .suit       "S", "H", "D" or "C" (spades, hearts, diamonds, clubs); suits have no order
  .is_red     hearts and diamonds;  .color is "red" or "black"
  .is_face    J, Q and K
  .gender     "male" for J and K, "female" for Q, None for other cards;  .is_male, .is_female
  .is_even    rank is even
  .label      "A", "2", ..., "10", "J", "Q" or "K"

A checker enforces these limits on the code:
  - only `def applies(ctx):`; no imports, other functions, lambdas, classes, while loops, try, recursion
  - names and attributes may not start with an underscore; the only attributes are the ones listed
    above, plus .count and .index on tuples and lists
  - builtins: abs min max len sum any all sorted reversed enumerate zip int bool set tuple list range
    (no keyword arguments; range is capped at {RANGE_CAP} items)
  - arithmetic (+ - * / // %) only on numbers, and no **
  - it must be deterministic

WHAT A RULE MAY USE
Card value, colour, suit, face or not, gender, odd/even; the cards on DISCARD; the turn number; hand
sizes; who is playing. Value arithmetic is limited to comparisons and + - * / (sums, differences,
products, doubling, halving, odd/even). No primes, squares, square roots, powers or wrap-around (K
followed by A). A rule may not depend on anything secret, such as the cards in anyone's hand, or on
anything random.

{{mechanics}}{{wins}}HOW TO REPLY
If the rule can't be implemented within these limits, or is ambiguous beyond a natural reading, or
mixes forbidding and allowing, or isn't about which plays are legal, reply only with
CANNOT: <one sentence, addressed to the rule's author, saying why>

Otherwise reply with exactly:
KIND: restriction   (or: KIND: permission; where the sections above allow it: KIND: effect, KIND: win)
DESCRIPTION: <the rule exactly as your code implements it, in one or two precise plain-English sentences, addressed to players>
```python
def applies(ctx):        # or, for an effect: def effect(ctx, fx):   for a win condition: def wins(ctx):
    ...
```
"""

MECHANICS_SECTION = f"""\
TABLE MECHANICS (this rule may use them)
From now on a rule may also use the table and the options of a move, and there is a third kind:
  effect       it makes something happen when a play is made, and never forbids or allows anything.
For an effect, write `def effect(ctx, fx):` INSTEAD of applies. It checks its own conditions and calls
fx methods (nothing happens if it calls none). The engine applies the calls after the play is judged,
and only if no penalty was upheld; effects can never change what a penalty does.

A move is one or more cards plus options the player chose. Extra ctx fields:
  ctx.cards         tuple of all cards in the move, in order (ctx.card is ctx.cards[0]); playing more than
                    one card breaks the engine's public rule R0 unless a permission allows that move
  ctx.angle         the angle the cards are played at: one of {ANGLES} (0 upright, 180 upside down)
  ctx.spend         tokens the player spends with this move (0 if none); spent tokens are always lost
  ctx.target        the seat the move targets, or None
  ctx.declare       the text the player declares with this move (lowercase), or None
  ctx.direction     1 = seat order, -1 = reversed;  ctx.next_player is the seat that plays next by default
  ctx.tokens        tuple of the tokens each player holds, by seat
  ctx.space_tokens  tuple of the tokens lying in each player's space, by seat;  ctx.pot  tokens lying on DISCARD
  ctx.spaces        tuple, by seat, of the cards in each player's space; each has .card and .angle
  ctx.pile_angles   tuple of the angles of the DISCARD cards, aligned with ctx.pile
  ctx.declared, ctx.declared_by   the latest declaration made this game (before this move) and its seat, or None
  ctx.game          the game number, 0-based
fx methods (seats are 0-based ints such as ctx.player, ctx.next_player, ctx.target; n is 1..{MAX_COUNT}):
  fx.reverse()                     reverse the order of play
  fx.skip()                        the next player loses their turn
  fx.extra_turn()                  the player who just played plays again
  Places: fx.hand(seat), fx.space(seat), fx.discard(), and for tokens also fx.bank() (the unlimited supply).
  fx.move_card(source, dest, n, which)   move n cards between places, e.g. fx.move_card(fx.hand(ctx.target), fx.hand(ctx.player), 1, "highest")
                    which, from a hand: "random" (default), "highest", "lowest", "chosen" (the GIVE card, only from the
                    hand of the player who just played); from a space or DISCARD: "newest" (default; DISCARD's top card),
                    "oldest", "random", "highest", "lowest". DISCARD always keeps 1 card.
  fx.move_tokens(source, dest, n)        move n tokens (1..3, or "all") between places; fx.discard() is the pot
  fx.gain_tokens(seat, n)          fx.lose_tokens(seat, n)       (= moves from / to the bank)
  fx.draw(seat, n)                 seat draws n cards from DECK (an effect, not a penalty)
  fx.to_space(seat)                the cards just played go into seat's space instead of staying on DISCARD
  fx.rotate_top(angle)             fx.rotate_space(seat, angle)   (turns the newest card in that space)
  fx.trade(seat, other)            seat gives a card (their GIVE option, or a random card) and takes a random card from other
  fx.peek_deck(seat)               seat privately sees the top card of DECK
  fx.peek_hand(seat, other)        seat privately sees one random card from other's hand
A rule that makes a move legal or illegal AND has an effect is two rules and cannot be implemented.

"""


WINS_SECTION = """\
WIN CONDITIONS (this rule may be one)
A fourth kind of rule is possible now:
  win          another way to win. The engine checks it only at the end of a player's own turn (after
               their play has been judged and its effects have happened), and only for that player.
               It adds a way to win; emptying your hand always still wins. It never forbids or allows a play.
For a win condition, write `def wins(ctx):` INSTEAD of applies, returning True if the player wins now.
Its ctx describes the end of the turn and, unlike the others, includes the player's own hand:
  ctx.hand          tuple of the cards the player holds now (the engine checks this; other players can't see it)
  ctx.cards         the card(s) they played this turn (empty if they had no cards);  ctx.card is the first or None
  ctx.penalized     True if a penalty was upheld on their play this turn
  ctx.pile, ctx.top, ctx.back(n)   DISCARD now (after this turn's play)
  ctx.player, ctx.hand_sizes (now), ctx.turn, ctx.num_players, and the table fields above if table mechanics are on
"I win when ..." / "a player wins if ..." with no timing means: at the end of that player's own turn.

"""


def coder_prompt(mechanics: bool = False, wins: bool = False) -> str:
    return (CODER_PROMPT.replace("{mechanics}", MECHANICS_SECTION if mechanics else "")
            .replace("{wins}", WINS_SECTION if wins else ""))


class CoderFailure(Exception):
    """The coding agent couldn't produce a working implementation of the rule."""
    meta: dict = {}


def _failure(reason: str, meta: dict) -> CoderFailure:
    e = CoderFailure(reason)
    e.meta = meta
    return e


@dataclass
class CoderResult:
    rule: CodeRule
    meta: dict = field(default_factory=dict)


_CANNOT_RE = re.compile(r"(?im)^[\W_]*CANNOT[\W_]*:\s*(.+)")
_KIND_RE = re.compile(r"(?im)^[\W_]*KIND[\W_]*:\s*\**\s*(restriction|permission|effect|win)\b")
_DESC_RE = re.compile(r"(?im)^[\W_]*DESCRIPTION[\W_]*:\s*(.+)")
_CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


@dataclass
class CoderReply:
    kind: str | None = None
    description: str | None = None
    code: str | None = None
    cannot: str | None = None
    error: str | None = None


def parse_coder_reply(text: str) -> CoderReply:
    codes = _CODE_RE.findall(text)
    cannot = _CANNOT_RE.search(text)
    if cannot and not codes:
        return CoderReply(cannot=cannot.group(1).strip().strip("*"))
    kind, desc = _KIND_RE.search(text), _DESC_RE.search(text)
    code = codes[-1].strip() if codes else None
    for fname in ("def applies", "def effect", "def wins"):
        if code is None and fname in text:
            code = text[text.index(fname):].strip()
    missing = [n for n, v in (("KIND", kind), ("DESCRIPTION", desc), ("a ```python code block", code)) if not v]
    if missing:
        return CoderReply(error="your reply is missing " + ", ".join(missing))
    return CoderReply(kind.group(1).lower(), desc.group(1).strip().strip("*").strip(), code)


class RuleCoder:
    """Drives one coding-model session. `implement()` returns a working CodeRule or raises CoderFailure."""

    def __init__(self, session: ChatSession, *, max_fixes: int = 2):
        self.session = session
        self.max_fixes = max_fixes

    @property
    def model(self) -> str:
        return f"{self.session.provider}:{self.session.model}"

    def implement(self, proposal: str, samples, mechanics: bool = False, wins: bool = False) -> CoderResult:
        self.session.reset(coder_prompt(mechanics, wins))
        message = f"RULE: {proposal}"
        meta: dict = {"model": self.model, "attempts": [], "input_tokens": 0, "output_tokens": 0}
        last_error = "no reply"
        for _ in range(self.max_fixes + 1):
            try:
                comp = self.session.send(message)
            except Exception as e:  # noqa: BLE001 - API trouble; retry with the same message
                last_error = f"{type(e).__name__}: {e}"
                meta["attempts"].append({"error": last_error})
                continue
            meta["input_tokens"] += comp.input_tokens
            meta["output_tokens"] += comp.output_tokens
            reply = parse_coder_reply(comp.text)
            attempt = {"raw": comp.text, "thinking": comp.thinking}
            meta["attempts"].append(attempt)
            if reply.cannot:
                attempt["error"] = f"CANNOT: {reply.cannot}"
                raise _failure(reply.cannot, meta)
            if reply.error is None and reply.kind == "win" and not wins:
                raise _failure("win conditions aren't allowed yet", meta)
            if reply.error is None and reply.kind == "effect" and not mechanics:
                raise _failure("effects (table mechanics) aren't allowed yet", meta)
            if reply.error is None:
                try:
                    smoke_test(compile_rule(reply.code, reply.kind, mechanics), samples, kind=reply.kind)
                    rule = CodeRule(reply.code, reply.kind, reply.description, proposal, mechanics)
                    return CoderResult(rule, meta)
                except SandboxError as e:
                    reply.error = f"the code was rejected: {e}"
            attempt["error"] = last_error = reply.error
            message = (f"ERROR: {reply.error}\nFix it and reply again in the same format "
                       "(KIND, DESCRIPTION, ```python code block), or reply CANNOT: <reason>.")
        raise _failure(f"the engine could not produce working code for it ({last_error})", meta)


def default_cache_path() -> Path:
    """cache/coder-cache.jsonl in the repository (it is committed, so every checkout shares the
    results); in the current directory if gnomic isn't running from a checkout."""
    root = Path(__file__).resolve().parent.parent
    base = root if (root / "pyproject.toml").exists() else Path.cwd()
    return base / "cache" / "coder-cache.jsonl"


def normalize_rule_text(text: str) -> str:
    """The cache key form of a rule text: whitespace collapsed, trailing full stop dropped. Case is kept
    (an "A" may be an Ace)."""
    return " ".join(text.split()).rstrip(". ")


class CoderCache:
    """Coder results by text, shared by every coding request in a match (and across matches with a path)."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self._entries: dict[str, dict] = {}
        self._inflight: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                    self._entries[rec["key"]] = rec["value"]
                except (ValueError, KeyError):
                    continue

    @staticmethod
    def key(model: str, text: str, mechanics: bool, wins: bool) -> str:
        prompt = hashlib.sha256(coder_prompt(mechanics, wins).encode()).hexdigest()[:16]
        return json.dumps([model, prompt, mechanics, wins, normalize_rule_text(text)])

    def get_or_compute(self, key: str, compute):
        """The cached value for key, or compute() (whose result, if not None, is cached). Concurrent
        callers with the same key wait for the first one instead of calling the coder again."""
        while True:
            with self._lock:
                if key in self._entries:
                    self.hits += 1
                    return self._entries[key], True
                ev = self._inflight.get(key)
                if ev is None:
                    ev = self._inflight[key] = threading.Event()
                    break
            ev.wait()  # someone else is coding this text; use their result (or try ourselves if it failed)
        try:
            with self._lock:
                self.misses += 1
            value = compute()
            if value is not None:
                with self._lock:
                    self._entries[key] = value
                    if self.path:
                        self.path.parent.mkdir(parents=True, exist_ok=True)
                        with open(self.path, "a", encoding="utf-8") as fh:
                            fh.write(json.dumps({"key": key, "value": value}) + "\n")
            return value, False
        finally:
            with self._lock:
                self._inflight.pop(key, None)
            ev.set()

    def stats(self) -> dict:
        return {"coder_calls": self.misses, "cache_hits": self.hits, "path": str(self.path) if self.path else None}


class CachedRuleCoder:
    """A RuleCoder whose results come from a CoderCache when the same text was coded before."""

    def __init__(self, coder: RuleCoder, cache: CoderCache):
        self.coder = coder
        self.cache = cache

    @property
    def model(self) -> str:
        return self.coder.model

    def implement(self, proposal: str, samples, mechanics: bool = False, wins: bool = False) -> CoderResult:
        key = CoderCache.key(self.model, proposal, mechanics, wins)
        fresh: dict = {}

        def compute():
            try:
                result = self.coder.implement(proposal, samples, mechanics=mechanics, wins=wins)
            except CoderFailure as e:
                answered = any("raw" in a for a in getattr(e, "meta", {}).get("attempts", []))
                fresh["failure"] = e
                # A coder that answered (CANNOT, or no working code) will answer the same again;
                # API trouble is not cached.
                return {"failure": str(e)} if answered else None
            fresh["result"] = result
            r = result.rule
            return {"kind": r.kind, "description": r.description, "code": r.code}

        value, cached = self.cache.get_or_compute(key, compute)
        if not cached:
            if "failure" in fresh:
                raise fresh["failure"]
            if "result" in fresh:
                return fresh["result"]
        if value is None:  # the other caller failed on API trouble; try ourselves
            return self.coder.implement(proposal, samples, mechanics=mechanics, wins=wins)
        meta = {"model": self.model, "cached": True, "attempts": [], "input_tokens": 0, "output_tokens": 0}
        if "failure" in value:
            err = CoderFailure(value["failure"])
            err.meta = meta
            raise err
        try:
            smoke_test(compile_rule(value["code"], value["kind"], mechanics), samples, kind=value["kind"])
        except SandboxError:
            return self.coder.implement(proposal, samples, mechanics=mechanics, wins=wins)
        return CoderResult(CodeRule(value["code"], value["kind"], value["description"], proposal, mechanics), meta)

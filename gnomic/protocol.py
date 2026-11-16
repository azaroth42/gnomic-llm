"""The text protocol between the engine and an LLM player.

Players are labelled P1..Pn: seats, not model names, so models can't tell who
their opponents are. Games and turns are numbered from 1 in messages. Rules are
numbered R1, R2, ... in the order they were added.

The engine sends a player one message per request. Each message is EVENTS
since the player's last message, then a STATE snapshot, then a request:
  YOUR TURN         -> reply ends with  PLAY: <card>  (then CALL: ... if the player owns a rule)
  JUDGE: Pk plays X -> reply ends with  CALL: PENALTY R<n>  or  CALL: NONE
A penalty call names the caller's own rule that the play broke ("a penalty under my rule R1").
Players police only their own rules: only rule owners get JUDGE requests, and a call is upheld
only if the play broke one of the caller's own rules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .cards import Card
from .game import PendingPlay, PlayerView, PublicEvent
from .mechanics import ANGLES, MAX_DECLARE, Move, Placed

SYSTEM_PROMPT = """\
You are a player in a match of Gnomic Mau, a card game run by a game engine. You only talk to the engine.

CARDS
A standard 52-card deck, no jokers. A card is written as rank then suit:
  ranks A 2 3 4 5 6 7 8 9 10 J Q K; suits S H D C (spades, hearts, diamonds, clubs).
  8H = Eight of Hearts, 10S = Ten of Spades, QD = Queen of Diamonds, AC = Ace of Clubs.
Values: A=1, 2-10 face value, J=11, Q=12, K=13. Hearts and diamonds are red; spades and clubs are black. Suits have no order.
J, Q and K are face cards. Jacks and Kings are male; Queens are female.

PLACES
  DECK     face-down draw pile. Only its size is public.
  DISCARD  face-up pile, listed bottom to top; the last card listed is the top card. Its full order is public.
  HAND     a player's cards. You see only your own; everyone's hand size is public.

PLAYERS
Players are P1..P{n}; you are {me}. Turns go in seat order (P1, P2, ..., P{n}, P1, ...). Each game starts with a different player.

RULES OF THE MATCH
 1. Each game, every player is dealt 7 cards, then one card from DECK starts DISCARD.
 2. On your turn you must play exactly one card from your HAND onto DISCARD if you have any cards. You cannot pass.
 3. Hidden rules decide which plays are legal. They are numbered R1, R2, ...
{rule4} 5. Each player polices only their OWN rules. After every play, every player who owns at least one rule judges it, including the player who made it if they own a rule. A penalty call names the rule broken, which must be one of your own: CALL: PENALTY R1 means "that's a penalty under my rule R1". Otherwise answer CALL: NONE. Players who own no rules cannot call and are not asked to judge. Calls are made at the same time; nobody sees the others' calls first. If your own play breaks one of your own rules, you must call PENALTY on yourself, naming that rule.
 6. A play breaks a rule if that rule forbids it and no later rule allows it (see 8). Each PENALTY call is judged separately:
      - upheld if the play broke the rule the caller named. If at least one call is upheld, the player who made the play draws 1 card from DECK (only 1, however many calls are upheld);
      - rejected otherwise, even if the play broke a different rule. Each player whose call is rejected draws 1 card.
    Everyone sees each call, the rule it named, and whether it was upheld. A play that breaks a rule costs nothing if that rule's owner doesn't call it, and nobody is told.
 7. The played card stays on DISCARD in every case.
 8. When rules conflict, later rules take precedence over earlier ones. A newer rule can forbid something an older rule allowed, or allow something an older rule forbade. Even the owner of a rule must allow for newer rules.
 9. A rule may depend on any public information: the card played, the cards on DISCARD (not only the top card), the turn number, hand sizes, who is playing.
10. If a player must draw and DECK is empty, every DISCARD card except the top card is shuffled to form the new DECK. The top card stays on DISCARD.
11. You win the game when your HAND is empty at the end of your own turn (after your play has been judged). Nobody wins out of turn: if your last card is taken from you during someone else's turn, you play nothing on your next turn and win at its end, unless someone wins first.
12. If nobody has won after {rounds} rounds (every player has played {rounds} cards), the game ends with no winner and no new rule is added.

{mechanics}{wins}HOW THE ENGINE TALKS TO YOU
While you own a rule you judge every other player's play, so you are told about each play as it happens. While you own no rules you are only messaged on your turn, and EVENTS then catches you up on everything since. Each message from the engine contains:
  EVENTS: everything public that happened since the engine's last message to you, one per line, for example
    RULE R1 ADDED, owned by P3 (the dealer)
    G1 START first player P1; DISCARD starts with 7C; rules in effect: 1
    G1 T1 P1 played 8H on 7C; no calls
    G1 T2 P2 played QS on 8H; PENALTY called by P3 under R1 (upheld): P2 draws 1
    G1 T3 P3 played 2S on QS; PENALTY called by P3 (self) under R1 (upheld): P3 draws 1
    G1 T4 P1 played 9D on 2S; PENALTY called by P3 under R1 (rejected): P3 draws 1 for a false call
    G1 T9 DECK empty: 31 DISCARD cards shuffled into a new DECK; 4C stays on DISCARD
    G1 END winner P2
    RULE R2 ADDED, owned by P2 (winner of G1)
    G2 START first player P2; DISCARD starts with 9H; rules in effect: 2
    G2 T1 P2 played 4C on 9H; PENALTY called by P3 under R1 (upheld), P2 (self) under R2 (rejected): P2 draws 1; P2 draws 1 for a false call
  STATE: the table as it is now (before the play being judged, if any), for example
    STATE G2 T7 round 3/{rounds} you=P1 rules=2 wins: P1=0 P2=1 P3=0
    DISCARD (bottom->top): 7C 9C 2H JH
    DECK: 21
    HANDS: P1=6 (you) P2=5 P3=8
    YOUR HAND: 3S 9S 4H KH 2D 10C
    JUDGES: P2 P3                          (the players who own rules and so may call PENALTY)
    YOUR RULES (secret): R1: <text>        (only if you own rules)
  and then one request:
    YOUR TURN                 play a card, then (if you own a rule) judge your own play
    JUDGE: P2 plays 5D on JH  decide whether another player's play broke one of your rules

REPLY FORMAT
Reason as much as you like, then end your reply with:
NOTES:
<your updated private notes>
and then, for YOUR TURN, the card you play and, if you own a rule, your call on your own play:
PLAY: <card>
CALL: PENALTY R<n>      or      CALL: NONE      (leave this line out if you own no rules)
or, for JUDGE, just your call:
CALL: PENALTY R<n>      or      CALL: NONE
where R<n> is the number of one of YOUR RULES that the play broke, e.g. CALL: PENALTY R1.

NOTES is your private reasoning log: your current hypotheses about the hidden rules, with the evidence for and against each. Keep it current and concise (under about 300 words). Other players never see it. If you leave NOTES out, your previous notes are kept. To save space the engine sometimes drops older messages; when it does, it re-sends the full public log and your latest NOTES.
PLAY must be a card in YOUR HAND, e.g. PLAY: 8H. If it isn't, the engine asks you again.

Your goal is to win games. Bad plays cost you a card when someone calls them, and false calls cost you a card too, so work out the hidden rules as fast as you can.
"""


RULE4_ENGINE = (
    " 4. One player is the dealer. Before game 1 the dealer is secretly told rule R1. After each game the winner is "
    "secretly told one new rule and becomes the dealer of the next game. The rule is added to the rules in effect. Everyone is told when a rule is added and who "
    "owns it, but only its owner is told what it says. Rules carry over from game to game.\n")

RULE4_PLAYERS = (
    " 4. One player is the dealer. Before game 1 the dealer makes up rule R1. After each game the winner makes up one "
    "new rule and becomes the dealer of the next game. The rule is added to the rules in effect. The engine asks with MAKE A RULE (see below), turns the rule into "
    "code and checks that the game stays playable; if it can't use your rule it tells you why and asks for another, "
    "and if that keeps failing it gives you a rule instead. Everyone is told when a rule is added and who owns it, but "
    "only its owner knows what it says. Rules carry over from game to game.\n")

MAKE_RULE_SECTION = """
MAKING UP A RULE
When you are the dealer (before game 1) or win a game, the engine sends MAKE A RULE. Reply with your
reasoning, then NOTES: as usual, then
RULE: <your rule in one or two precise sentences>
Your rule is enforced exactly as the engine implements it, and that wording is what you will see in YOUR RULES.
"""


MECHANICS_PROMPT = """\
FROM GAME {m}: TABLE MECHANICS
From game {m} on, the table has more parts, all public, and rules added from then on may use them:
  ORDER     the order of play (seat order, or reversed)
  TOKENS    each player holds tokens; tokens can also lie in a space or on DISCARD (the pot)
  SPACES    face-up cards in front of each player
  ANGLES    a card on the table lies at 0 (upright), 90, 180 (upside down) or 270, written 9C@90
  DECLARED  the latest declaration made this game
Rules may also make things happen when a card is played (effects): reverse the order of play, skip a
player, give an extra turn, make a player draw, move cards and tokens between hands, spaces and DISCARD,
turn cards, trade cards between hands, or let a player peek at a hidden card. A card moved from one
hand to another is seen only by those two players. A play's effects happen
only if no penalty is upheld for it. Rules never change penalties.
From game {m}, after PLAY you may add any of these lines (each is optional, in any order):
  PLAY: 5D 5H        several cards at once, played in that order (the last one ends on top). R0, a public
                     rule enforced by the engine: you play exactly one card unless a rule allows more. The
                     engine calls R0 itself, and the usual penalty applies.
  ANGLE: 90          the angle you play at: 0, 90, 180 or 270 (default 0)
  SPEND: 2           spend tokens you have; they are gone even if the play is penalized
  TARGET: P3         aim your play at another player
  DECLARE: hearts    a short public declaration (up to 40 characters)
  GIVE: 7H           the card you hand over if your play makes you trade (only the receiver sees it)
These options do nothing unless a rule makes them matter. Anything only you may know (cards you see in
a trade or a peek) appears under PRIVATE in your messages.

"""


WINS_PROMPT = """\
FROM GAME {w}: WIN CONDITIONS
Rules added from game {w} on may be win conditions: another way to win, besides emptying your hand
(which always still wins). The engine checks them only at the end of a player's own turn, after the
play has been judged and its effects have happened, and only for that player. A win condition may
depend on that player's hand, which only the engine sees. When someone wins by one, the engine says
which rule it was (G6 END winner P2 (win condition R7)), but not what it says.

"""


def system_prompt(view: PlayerView) -> str:
    text = SYSTEM_PROMPT.replace("{rule4}", RULE4_PLAYERS if view.players_make_rules else RULE4_ENGINE)
    text = text.replace("{mechanics}", MECHANICS_PROMPT.replace("{m}", str(view.mechanics_from))
                        if view.mechanics_from else "")
    text = text.replace("{wins}", WINS_PROMPT.replace("{w}", str(view.wins_from)) if view.wins_from else "")
    if view.players_make_rules:
        text = text.replace("\nREPLY FORMAT\n", MAKE_RULE_SECTION + "\nREPLY FORMAT\n", 1)
    return (text.replace("{n}", str(len(view.player_names))).replace("{me}", seat_label(view.seat))
            .replace("{rounds}", str(view.max_rounds)))


def seat_label(seat: int) -> str:
    return f"P{seat + 1}"


def seats(ss) -> str:
    return ",".join(seat_label(s) for s in ss)


def cards(cs) -> str:
    return " ".join(str(c) for c in cs) if cs else "(empty)"


def _on(top: Card | None) -> str:
    return f"on {top}" if top else "on empty DISCARD"


def render_event(ev: PublicEvent) -> str:
    g = f"G{ev.game + 1}"
    if ev.kind == "play":
        line = f"{g} T{ev.turn + 1} {seat_label(ev.player)} played {ev.move or ev.card} {_on(ev.top_before)}; "
        effects = f"; effects: {'; '.join(ev.effects)}" if ev.effects else ""
        if not ev.calls and not ev.engine_call:
            return line + "no calls" + effects
        rules = dict(zip(ev.calls, ev.call_rules))
        by = [f"{seat_label(s)}{' (self)' if s == ev.player else ''}"
              f"{f' under R{rules[s]}' if rules.get(s) else ''}"
              f" ({'upheld' if s in ev.upheld_by else 'rejected'})" for s in ev.calls]
        if ev.engine_call:
            by.insert(0, "the engine under R0 (upheld)")
        out = []
        if ev.upheld_by or ev.engine_call:
            out.append(f"{seat_label(ev.player)} draws 1")
        rejected = [s for s in ev.calls if s not in ev.upheld_by]
        if rejected:
            out.append(f"{seats(rejected)} draw{'s' if len(rejected) == 1 else ''} 1 for a false call")
        return line + f"PENALTY called by {', '.join(by)}: " + "; ".join(out) + effects
    if ev.kind == "game_start":
        return (f"{g} START first player {seat_label(ev.player)}; DISCARD starts with {ev.card}; "
                f"rules in effect: {ev.detail}")
    if ev.kind == "reshuffle":
        return (f"{g} T{ev.turn + 1} DECK empty: {ev.detail.split()[0]} DISCARD cards shuffled into a new DECK; "
                f"{ev.card} stays on DISCARD")
    if ev.kind == "game_end":
        if ev.winner is None:
            return f"{g} END no winner (round limit reached)"
        if ev.detail.startswith("win_rule "):
            return f"{g} END winner {seat_label(ev.winner)} (win condition {ev.detail.split()[1]})"
        return f"{g} END winner {seat_label(ev.winner)}"
    if ev.kind == "no_cards":
        return f"{g} T{ev.turn + 1} {seat_label(ev.player)} has no cards, so plays none"
    if ev.kind == "rule_added":
        how = "the dealer" if ev.turn == -1 else f"winner of G{ev.game + 1}"
        return f"RULE R{ev.detail} ADDED, owned by {seat_label(ev.player)} ({how})"
    return f"{g} {ev.kind} {ev.detail}".strip()


def render_state(view: PlayerView) -> str:
    n = len(view.player_names)
    wins = " ".join(f"{seat_label(i)}={view.scores[i]}" for i in range(n))
    hands = " ".join(f"{seat_label(i)}={view.hand_sizes[i]}" + (" (you)" if i == view.seat else "")
                     for i in range(n))
    lines = [
        f"STATE G{view.game + 1} T{view.turn + 1} round {view.turn // n + 1}/{view.max_rounds} "
        f"you={seat_label(view.seat)} rules={view.num_rules} wins: {wins}",
        f"DISCARD (bottom->top): {_discard(view)}",
        f"DECK: {view.deck_size}",
        f"HANDS: {hands}",
        f"YOUR HAND: {cards(view.hand)}",
        "JUDGES: " + (" ".join(seat_label(s) for s in view.judges) or "(nobody)"),
    ]
    t = view.table
    if t is not None:
        order = " -> ".join(seat_label((view.seat + t.direction * k) % n) for k in range(n))
        lines.append(f"ORDER: {order} ({'seat order' if t.direction == 1 else 'reversed'})")
        tok = "TOKENS: " + " ".join(f"{seat_label(i)}={t.tokens[i]}" + (" (you)" if i == view.seat else "")
                                    for i in range(n))
        if any(t.space_tokens):
            tok += " | in spaces: " + " ".join(f"{seat_label(i)}={k}" for i, k in enumerate(t.space_tokens) if k)
        lines.append(tok + f" | pot on DISCARD: {t.pot}")
        lines.append("SPACES: " + " | ".join(f"{seat_label(i)}: {' '.join(map(str, t.spaces[i])) or '(empty)'}"
                                          for i in range(n)))
        if t.declared is not None:
            lines.append(f'DECLARED: "{t.declared}" by {seat_label(t.declared_by)}')
    if view.my_rules:
        lines.append("YOUR RULES (secret): " + " | ".join(f"R{i}: {t}" for i, t in view.my_rules))
    return "\n".join(lines)


def _discard(view: PlayerView) -> str:
    if view.table is None or len(view.table.angles) != len(view.discard):
        return cards(view.discard)
    return " ".join(str(Placed(c, a)) for c, a in zip(view.discard, view.table.angles)) or "(empty)"


def render_message(view: PlayerView, events: list[PublicEvent], request: str,
                   notes: str | None = None, resync: bool = False, state: str | None = None,
                   private: list[str] | None = None) -> str:
    parts = []
    if resync:
        parts.append("RESYNC: older messages were dropped. EVENTS below is the full public log of the match.")
        if notes:
            parts.append(f"YOUR NOTES (latest):\n{notes}")
    parts.append("EVENTS:\n" + ("\n".join(render_event(e) for e in events) if events else "(none)"))
    if private:
        parts.append("PRIVATE (only you know this):\n" + "\n".join(private))
    parts.append(state if state is not None else render_state(view))
    parts.append(request)
    return "\n\n".join(parts)


def render_rule_state(view: PlayerView) -> str:
    """STATE between games (when a rule is being made up): no table, just scores and your rules."""
    n = len(view.player_names)
    wins = " ".join(f"{seat_label(i)}={view.scores[i]}" for i in range(n))
    when = "before G1" if view.num_rules == 0 else f"after G{view.game}"
    lines = [f"STATE {when} you={seat_label(view.seat)} rules={view.num_rules} wins: {wins}"]
    if view.my_rules:
        lines.append("YOUR RULES (secret): " + " | ".join(f"R{i}: {t}" for i, t in view.my_rules))
    return "\n".join(lines)


def rule_request(view: PlayerView, *, first: bool, feedback: str | None = None,
                 attempt: int = 1, attempts: int = 1, mechanics: bool = False, wins: bool = False) -> str:
    who = "You are the dealer" if first else f"You won G{view.game}"
    lines = []
    if feedback:
        lines.append(f"REJECTED: your last rule can't be used: {feedback}")
    lines.append(f"MAKE A RULE: {who}. Make up rule R{view.num_rules + 1}"
                 f" (attempt {attempt} of {attempts}; after that the engine gives you a rule).")
    lines.append(
        "- It is either a restriction (it forbids some plays) or a permission (it allows some plays even if an "
        "older rule forbids them), not both" + (", or an effect" if mechanics else "")
        + (", or a win condition" if wins else "") + (" (see below)." if mechanics or wins else ".")
        + (" R1 must be a restriction." if first else ""))
    lines.append(
        "- It may use only public information: the card played (value A=1 ... K=13, colour, suit, face card or not, "
        "gender: J and K male, Q female; odd/even), the cards on DISCARD, the turn number, hand sizes and who is "
        "playing. Values may only be compared, added, subtracted, multiplied or divided: no primes, squares or "
        "wrapping from K to A." + (" (A win condition may also use the player's own hand.)" if wins else ""))
    if mechanics:
        lines.append(
            f"- This rule starts in game {view.game + 1}, so it may use the TABLE MECHANICS: it may be an effect "
            "(something that happens when a card is played: reverse the order, skip, extra turn, draws, moving "
            "cards or tokens between hands, spaces and DISCARD, turning cards, trades, peeks), or a restriction or permission about several cards at once, "
            "ANGLE, SPEND, TARGET, DECLARE, tokens, spaces, angles, the order of play or the latest declaration. "
            "A rule is exactly one kind. No rule may change what a penalty does.")
    if wins:
        lines.append(
            "- It may be a WIN CONDITION: another way to win, checked only at the end of a player's own turn. "
            "It may depend on that player's own hand (the engine checks it). It must be hard to meet: the "
            "engine rejects one that would be met at the end of more than 3% of turns.")
    lines.append(
        "- It must leave plenty of cards legal, and it must make a difference given the rules already in effect, "
        "which you only partly know.")
    lines.append("- Other players will try to work it out from the penalties you call, so make it learnable but not obvious.")
    lines.append("End your reply with NOTES: <your updated notes> and then RULE: <your rule>.")
    return "\n".join(lines)


def turn_request(view: PlayerView) -> str:
    if view.mechanics:
        return _turn_request(view).replace("PLAY: <card>", "PLAY: <card(s)>, any option lines", 1)
    return _turn_request(view)


def _turn_request(view: PlayerView) -> str:
    if view.can_call:
        return ("YOUR TURN (end with PLAY: <card> and then your call on your own play: "
                f"CALL: PENALTY R<n> naming {_yours(view)}, or CALL: NONE)")
    return "YOUR TURN (end with PLAY: <card>; you own no rules, so you cannot call and need no CALL line)"


def _yours(view: PlayerView) -> str:
    nums = [f"R{n}" for n, _ in view.my_rules]
    return f"your rule {nums[0]}" if len(nums) == 1 else "one of your rules (" + ", ".join(nums) + ")"


def judge_request(pending: PendingPlay, view: PlayerView | None = None) -> str:
    line = f"JUDGE: {seat_label(pending.seat)} plays {pending.public()} {_on(pending.top_before)}"
    if view is not None and view.my_rules:
        line += f" (end with CALL: PENALTY R<n> naming {_yours(view)} that it broke, or CALL: NONE)"
    return line


def render_play_retry(view: PlayerView, error: str) -> str:
    then = f", then CALL: PENALTY R<n> naming {_yours(view)}, or CALL: NONE" if view.can_call else ""
    return (f"INVALID: {error}\nYOUR HAND: {cards(view.hand)}\n"
            f"Reply again, ending with PLAY: <card> using a card from YOUR HAND{then}.")


def render_call_retry(error: str, view: PlayerView | None = None) -> str:
    named = f"naming {_yours(view)}" if view is not None and view.my_rules else "naming one of your rules"
    return f"INVALID: {error}\nReply again, ending with CALL: PENALTY R<n> {named}, or CALL: NONE."


@dataclass
class ParsedReply:
    card: Card | None = None
    penalty: bool | None = None
    rule: int | None = None        # rule number named by a PENALTY call (R<n>)
    notes: str | None = None
    reasoning: str = ""
    error: str | None = None
    rule_text: str | None = None   # a made-up rule (MAKE A RULE)
    move: Move | None = None       # game 3+: several cards and/or options


_PLAY_RE = re.compile(r"(?im)^[\W_]*PLAY[\W_]*:\s*\**\s*([^\s*`]+(?:\s+of\s+\w+)?)")
_CALL_RE = re.compile(r"(?im)^[\W_]*CALL[\W_]*:\s*\**\s*([^\n]+)")
_RULE_RE = re.compile(r"(?i)\bR(?:ULE)?\s*#?\s*(\d+)\b")
_NOTES_RE = re.compile(r"(?im)^[\W_]*NOTES[\W_]*:")


def _split(text: str, final: re.Match) -> tuple[str | None, str]:
    """Return (notes, reasoning) given the final PLAY/CALL line."""
    notes_m = [m for m in _NOTES_RE.finditer(text) if m.start() < final.start()]
    if notes_m:
        n = notes_m[-1]
        return (text[n.end():final.start()].strip() or None), text[:n.start()].strip()
    return None, text[:final.start()].strip()


def parse_play(text: str, hand: tuple[Card, ...], own: tuple[int, ...] = (),
               view: PlayerView | None = None) -> ParsedReply:
    """Parse a YOUR TURN reply: PLAY: <card>, optionally followed by a self-call naming one
    of the player's own rules `own`. No self-call, or an unreadable one, counts as NONE; a
    PENALTY that doesn't name one of their rules is an error (the player is asked again).
    A player who owns no rules cannot call, so any CALL line is ignored."""
    self_call = None
    calls = list(_CALL_RE.finditer(text))
    plays_all = list(_PLAY_RE.finditer(text))
    if calls and plays_all and calls[-1].start() > plays_all[-1].start():
        if own:
            self_call = parse_call(text[plays_all[-1].end():], own)
        text = text[:calls[-1].start()]
    r = _parse_play_only(text, hand)
    if r.error is None:
        _parse_move(text, hand, r, view)
    if self_call is not None and self_call.penalty:
        if self_call.error and r.error is None:
            r.error = f"your self-call: {self_call.error}"
        r.rule = self_call.rule
    r.penalty = bool(self_call and self_call.penalty and not self_call.error)
    return r


_PLAY_LINE_RE = re.compile(r"(?im)^[\W_]*PLAY[\W_]*:[ \t*]*([^\n]*)")
_OPTION_RE = re.compile(r"(?im)^[\W_]*(ANGLE|SPEND|TARGET|DECLARE|GIVE)[\W_]*:[ \t*]*([^\n]*)")


def _parse_move(text: str, hand: tuple[Card, ...], r: ParsedReply, view: PlayerView | None) -> None:
    """Read extra cards on the PLAY line and the option lines after it into r.move (or r.error)."""
    line = list(_PLAY_LINE_RE.finditer(text))[-1]
    tokens = [t for t in re.split(r"[\s,]+", line.group(1).replace("*", "").replace("`", "").strip()) if t]
    played = [r.card]
    if len(tokens) > 1:
        try:
            parsed = [Card.parse(t.rstrip(".;)")) for t in tokens]
        except ValueError:
            parsed = None
        if parsed:
            played = parsed
    opts = {m.group(1).upper(): m.group(2).replace("*", "").replace("`", "").strip()
            for m in _OPTION_RE.finditer(text, line.end())}
    if len(played) == 1 and not opts:
        return
    mechanics = view is not None and view.mechanics
    if not mechanics:
        when = f" (from game {view.mechanics_from})" if view is not None and view.mechanics_from else ""
        r.error = f"play exactly one card with no option lines; options are not available yet{when}"
        return
    missing = [c for c in played if c not in hand]
    if missing or len(set(played)) != len(played):
        r.error = f"{missing[0]} is not in your hand" if missing else "you listed the same card twice"
        return
    try:
        angle = int(opts.get("ANGLE", "0").split()[0]) if opts.get("ANGLE") else 0
        spend = int(opts.get("SPEND", "0").split()[0]) if opts.get("SPEND") else 0
    except ValueError:
        r.error = "ANGLE and SPEND must be numbers"
        return
    if angle not in ANGLES:
        r.error = f"ANGLE must be one of {', '.join(map(str, ANGLES))}"
        return
    if not 0 <= spend <= view.my_tokens:
        r.error = f"you have {view.my_tokens} token(s), so SPEND must be 0..{view.my_tokens}"
        return
    target = None
    if opts.get("TARGET") and opts["TARGET"].upper() not in ("NONE", "-"):
        m = re.match(r"(?i)P?(\d+)", opts["TARGET"])
        target = int(m.group(1)) - 1 if m else -1
        if not (0 <= target < len(view.player_names)) or target == view.seat:
            r.error = "TARGET must be another player, e.g. TARGET: P2"
            return
    declare = opts.get("DECLARE") or None
    if declare and declare.upper() == "NONE":
        declare = None
    if declare and len(declare) > MAX_DECLARE:
        r.error = f"DECLARE must be at most {MAX_DECLARE} characters"
        return
    give = None
    if opts.get("GIVE") and opts["GIVE"].upper() != "NONE":
        try:
            give = Card.parse(opts["GIVE"].split()[0].rstrip(".;"))
        except ValueError:
            r.error = f"could not read GIVE card {opts['GIVE']!r}"
            return
        if give not in hand or give in played:
            r.error = "GIVE must be a card in your hand that you are not playing"
            return
    move = Move(tuple(played), angle, spend, target, declare, give)
    r.card = move.card
    r.move = move if move.has_options else None


def _parse_play_only(text: str, hand: tuple[Card, ...]) -> ParsedReply:
    plays = list(_PLAY_RE.finditer(text))
    if not plays:
        notes_m = list(_NOTES_RE.finditer(text))
        notes = text[notes_m[-1].end():].strip() if notes_m else None
        return ParsedReply(notes=notes, reasoning=text.strip(), error="no 'PLAY: <card>' line found")
    play = plays[-1]
    notes, reasoning = _split(text, play)
    token = play.group(1).strip().rstrip(".,;)")
    try:
        card = Card.parse(token)
    except ValueError:
        return ParsedReply(notes=notes, reasoning=reasoning, error=f"could not read card {token!r}")
    if card not in hand:
        return ParsedReply(notes=notes, reasoning=reasoning, error=f"{card} is not in your hand")
    return ParsedReply(card=card, notes=notes, reasoning=reasoning)


_RULE_LINE_RE = re.compile(r"(?im)^[\W_]*RULE[\W_]*:[ \t*]*")


def parse_rule(text: str) -> ParsedReply:
    """Parse a MAKE A RULE reply: NOTES: ... then RULE: <text>. The rule is everything after the
    last RULE: marker (it may run over several lines), stripped of markdown emphasis."""
    found = list(_RULE_LINE_RE.finditer(text))
    if not found:
        return ParsedReply(reasoning=text.strip(), error="no 'RULE: <your rule>' line found")
    m = found[-1]
    notes, reasoning = _split(text, m)
    rule = " ".join(text[m.end():].replace("**", "").replace("`", "").split()).strip()
    if not rule:
        return ParsedReply(notes=notes, reasoning=reasoning, error="the RULE line is empty")
    if len(rule) > 600:
        return ParsedReply(notes=notes, reasoning=reasoning, error="the rule is too long; state it in one or two sentences")
    return ParsedReply(notes=notes, reasoning=reasoning, rule_text=rule)


def parse_call(text: str, own: tuple[int, ...] = ()) -> ParsedReply:
    """Parse CALL: PENALTY R<n> or CALL: NONE. R<n> must be one of the caller's rules `own`;
    if they own exactly one rule it may be left out. An invalid rule gives penalty=True plus
    an error, so the caller is asked again."""
    found = list(_CALL_RE.finditer(text))
    if not found:
        return ParsedReply(reasoning=text.strip(), error="no 'CALL: PENALTY R<n>' or 'CALL: NONE' line found")
    m = found[-1]
    notes, reasoning = _split(text, m)
    line = m.group(1).replace("*", "").replace("`", "").strip()
    word = line.upper().rstrip("!. ")
    if word.startswith("PENALTY"):
        named = _RULE_RE.search(line[len("PENALTY"):])
        yours = ", ".join(f"R{n}" for n in own) or "none"
        if named:
            n = int(named.group(1))
            if own and n not in own:
                return ParsedReply(penalty=True, rule=n, notes=notes, reasoning=reasoning,
                                   error=f"R{n} is not one of your rules (yours: {yours}); you can only call your own rules")
        elif len(own) == 1:
            n = own[0]
        else:
            return ParsedReply(penalty=True, notes=notes, reasoning=reasoning,
                               error=f"a penalty call must name the rule broken, e.g. CALL: PENALTY R{own[0] if own else 1} (yours: {yours})")
        return ParsedReply(penalty=True, rule=n, notes=notes, reasoning=reasoning)
    if word.startswith(("NONE", "NO CALL", "NO")):
        return ParsedReply(penalty=False, notes=notes, reasoning=reasoning)
    return ParsedReply(notes=notes, reasoning=reasoning, error=f"could not read call {word!r}")

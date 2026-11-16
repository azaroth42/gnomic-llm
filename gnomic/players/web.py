"""Play in a browser (`web` player spec). The human sees what an LLM player sees, as it happens.

Each request the engine would send an LLM (YOUR TURN, JUDGE, MAKE A RULE) goes to this
player's own page as a `request` event. The event holds the same message text (EVENTS,
PRIVATE, STATE and the request, built by the same protocol functions), the rules briefing
(the LLM system prompt), and a structured copy of the same `PlayerView` so the page can draw
cards and buttons. The page answers with reply text in the LLM reply format (`NOTES:` /
`PLAY:` / option lines / `CALL:` / `RULE:`), and that text goes through the same parsers. A bad
reply gets the same `INVALID:` message and the same number of retries as an LLM. After that the
engine plays a random card or counts the call as NONE.

Unlike an LLM, which hears from the engine only when it must act, the page also follows the
table live (user decision): before every turn (so right after the previous play) and at the end
of a game the engine calls
`on_table_update` with the player's own view, and the page shows the new public events and
state. This is public information the LLM gets in its next message anyway. The `message`
text of each request is still exactly what an LLM would get (EVENTS since the last request).

Information isolation: this player's outbox only ever carries what was built from its own
`PlayerView`. Spectator events never go there. The page lives at a secret URL
(`/play/<token>`, see server.py).
"""

from __future__ import annotations

import itertools
import queue
import secrets
import threading
import time
from typing import Callable

from ..events import EventBus
from ..game import PendingPlay, PlayerView, PublicEvent, StopMatch
from ..mechanics import ANGLES, MAX_DECLARE
from ..protocol import (ParsedReply, judge_request, parse_call, parse_play, parse_rule, render_call_retry,
                        render_event, render_message, render_play_retry, render_rule_state, rule_request,
                        seat_label, system_prompt, turn_request)
from .base import Call, Choice, Player, Proposal

MAX_REPLY = 20_000  # characters


def view_json(view: PlayerView, *, table: bool = True) -> dict:
    """The structured form of what render_state shows. Built only from the player's own view."""
    n = len(view.player_names)
    out = {
        "seat": view.seat, "players": n, "game": view.game, "num_games": view.num_games,
        "turn": view.turn, "round": view.turn // n + 1, "max_rounds": view.max_rounds,
        "num_rules": view.num_rules, "scores": list(view.scores), "judges": list(view.judges),
        "my_rules": [[num, text] for num, text in view.my_rules],
        "can_call": view.can_call, "players_make_rules": view.players_make_rules,
        "mechanics_from": view.mechanics_from, "wins_from": view.wins_from, "mechanics": view.mechanics,
        "table": table,
    }
    if not table:
        return out
    angles = list(view.table.angles) if view.table is not None and len(view.table.angles) == len(view.discard) \
        else [0] * len(view.discard)
    out.update({
        "hand": [str(c) for c in view.hand],
        "discard": [[str(c), a] for c, a in zip(view.discard, angles)],
        "deck_size": view.deck_size,
        "hand_sizes": list(view.hand_sizes),
        "my_tokens": view.my_tokens,
        "angles": list(ANGLES), "max_declare": MAX_DECLARE,
    })
    t = view.table
    if t is not None:
        out["mech"] = {
            "direction": t.direction,
            "order": [(view.seat + t.direction * k) % n for k in range(n)],
            "tokens": list(t.tokens), "space_tokens": list(t.space_tokens), "pot": t.pot,
            "spaces": [[[str(p.card), p.angle] for p in sp] for sp in t.spaces],
            "declared": t.declared, "declared_by": t.declared_by,
        }
    return out


class WebHumanPlayer(Player):
    kind = "human"
    makes_rules = True
    live_updates = True

    def __init__(self, name: str, *, max_retries: int = 2):
        super().__init__(name)
        self.token = secrets.token_urlsafe(12)   # the secret part of this player's page URL
        self.outbox = EventBus()                 # everything this player's page shows, and nothing else
        self.max_retries = max_retries
        self.should_stop: Callable[[], bool] = lambda: False  # set to the match's stop flag
        self.notes: str | None = None
        self.cursor = 0           # public log shown in request messages (as an LLM would see it)
        self.private_cursor = 0
        self.shown = 0            # public log lines the page has been sent (live updates included)
        self.shown_private = 0
        self.started = False
        self._shown_lock = threading.Lock()
        self.reasoning_log: list[dict] = []
        self._replies: queue.Queue = queue.Queue()
        self._open_id: int | None = None
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    @property
    def model(self) -> str:
        return "web"

    # ---- talking to the page -------------------------------------------------

    def submit(self, request_id: int, text: str) -> str | None:
        """Called by the web server with the page's reply. Returns an error, or None if it was handed on."""
        if not isinstance(text, str) or not text.strip():
            return "empty reply"
        if len(text) > MAX_REPLY:
            return f"reply too long (over {MAX_REPLY} characters)"
        if request_id != self._open_id:
            return "that request is no longer open"
        self._replies.put((request_id, text))
        return None

    def _wait(self, request_id: int) -> str:
        while True:
            if self.should_stop():
                raise StopMatch()
            try:
                rid, text = self._replies.get(timeout=0.25)
            except queue.Empty:
                continue
            if rid == request_id:
                return text

    def _intro(self, view: PlayerView) -> None:
        if not self.started:
            self.started = True
            self.outbox.emit("intro", seat=view.seat, label=seat_label(view.seat), players=len(view.player_names),
                             system_prompt=system_prompt(view))

    def _new_lines(self, log, private) -> tuple[list[str], list[str]]:
        """Public log lines and private notices the page hasn't been sent yet."""
        with self._shown_lock:
            events = [render_event(e) for e in log[self.shown:]]
            notes = list(private[self.shown_private:])
            self.shown, self.shown_private = max(self.shown, len(log)), max(self.shown_private, len(private))
        return events, notes

    def on_table_update(self, view: PlayerView, current: int | None) -> None:
        self._intro(view)
        events, private = self._new_lines(view.log, view.private)
        self.outbox.emit("update", events=events, private=private, view=view_json(view), current=current,
                         current_label=None if current is None else seat_label(current))

    def _open(self, view: PlayerView, kind: str, request: str, state: str | None,
              pending: PendingPlay | None) -> int:
        self._intro(view)
        events = list(view.log[self.cursor:])
        private = list(view.private[self.private_cursor:])
        self.cursor, self.private_cursor = len(view.log), len(view.private)
        new_events, new_private = self._new_lines(view.log, view.private)
        while not self._replies.empty():  # drop anything sent for an older request
            self._replies.get_nowait()
        rid = next(self._ids)
        self._open_id = rid
        self.outbox.emit(
            "request", id=rid, kind=kind, request=request,
            message=render_message(view, events, request, state=state, private=private),
            events=new_events, private=new_private,
            view=view_json(view, table=state is None), notes=self.notes,
            pending=None if pending is None else {
                "seat": pending.seat, "label": seat_label(pending.seat), "public": pending.public(),
                "cards": [str(c) for c in (pending.move.cards if pending.move else (pending.card,))],
                "top_before": str(pending.top_before) if pending.top_before else None})
        return rid

    def _exchange(self, view: PlayerView, kind: str, request: str, parse: Callable[[str], ParsedReply],
                  retry: Callable[[str], str], fallback: str, state: str | None = None,
                  pending: PendingPlay | None = None) -> tuple[ParsedReply | None, dict]:
        rid = self._open(view, kind, request, state, pending)
        attempts: list[dict] = []
        parsed: ParsedReply | None = None
        raw = ""
        try:
            for i in range(self.max_retries + 1):
                t0 = time.monotonic()
                raw = self._wait(rid)
                parsed = parse(raw)
                attempts.append({"error": parsed.error, "seconds": round(time.monotonic() - t0, 2)})
                if parsed.notes:
                    self.notes = parsed.notes
                if parsed.error is None:
                    break
                left = self.max_retries - i
                if left:
                    self.outbox.emit("invalid", id=rid, error=parsed.error, message=retry(parsed.error),
                                     retries_left=left)
        finally:
            self._open_id = None
        ok = parsed if parsed is not None and parsed.error is None else None
        self.outbox.emit("done", id=rid, accepted=ok is not None,
                         error=None if ok else (parsed.error if parsed else None),
                         fallback=None if ok else fallback, notes=self.notes, reply=raw)
        self.reasoning_log.append({
            "kind": kind, "game": view.game, "turn": view.turn, "request": request,
            "reasoning": parsed.reasoning if parsed else "", "notes": self.notes,
            "card": str(ok.card) if ok and ok.card else None, "penalty": ok.penalty if ok else None,
            "rule": ok.rule if ok else None, "rule_text": ok.rule_text if ok else None, "attempts": attempts})
        return ok, {"attempts": attempts, "raw": raw, "model": self.model}

    # ---- the Player interface ---------------------------------------------------

    def choose(self, view: PlayerView) -> Choice:
        with self._lock:
            parsed, meta = self._exchange(view, "play", turn_request(view),
                                          lambda t: parse_play(t, view.hand, view.my_rule_numbers, view),
                                          lambda err: render_play_retry(view, err),
                                          "the engine played a random card from your hand for you")
        if parsed is None:
            return Choice(None, meta=meta)
        return Choice(parsed.card, move=parsed.move, self_call=bool(parsed.penalty), self_call_rule=parsed.rule,
                      reasoning=parsed.reasoning, notes=self.notes, meta=meta)

    def call(self, view: PlayerView, play: PendingPlay) -> Call:
        with self._lock:
            parsed, meta = self._exchange(view, "judge", judge_request(play, view),
                                          lambda t: parse_call(t, view.my_rule_numbers),
                                          lambda err: render_call_retry(err, view),
                                          "your call counts as NONE", pending=play)
        if parsed is None:
            return Call(None, meta=meta)
        return Call(parsed.penalty, rule=parsed.rule, reasoning=parsed.reasoning, notes=self.notes, meta=meta)

    def propose_rule(self, view: PlayerView, *, first: bool, feedback: str | None = None,
                     attempt: int = 1, attempts: int = 1, mechanics: bool = False,
                     wins: bool = False) -> Proposal | None:
        with self._lock:
            parsed, meta = self._exchange(
                view, "rule", rule_request(view, first=first, feedback=feedback, attempt=attempt, attempts=attempts,
                                           mechanics=mechanics, wins=wins),
                parse_rule,
                lambda err: f"INVALID: {err}\nReply again, ending with NOTES: <notes> and then RULE: <your rule>.",
                "this attempt counts as no rule", state=render_rule_state(view))
        if parsed is None:
            return None
        return Proposal(parsed.rule_text, reasoning=parsed.reasoning, notes=self.notes, meta=meta)

    def on_match_end(self, log: tuple[PublicEvent, ...], scores: list[int], stopped: bool) -> None:
        events, _ = self._new_lines(log, ())
        self.outbox.emit("match_end", events=events, scores=list(scores),
                         stopped=stopped)

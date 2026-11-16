"""An LLM player: one ongoing conversation per match, driven by the text protocol.

Every request (YOUR TURN or JUDGE) is one exchange in the same conversation,
so the model sees each play as it happens. The player keeps a private reasoning
log (one entry per exchange: its reasoning, its thinking summary if the provider
gives one, and its updated NOTES). The spectator UI shows that log; no other
player ever sees it.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

from ..game import PendingPlay, PlayerView
from ..protocol import (ParsedReply, judge_request, parse_call, parse_hypotheses, parse_play, parse_rule, render_call_retry,
                        render_message, render_play_retry, render_rule_state, rule_request, system_prompt,
                        turn_request)
from ..providers import ChatSession
from .base import Call, Choice, Player, Proposal


class LLMPlayer(Player):
    kind = "llm"
    makes_rules = True
    states_hypotheses = True

    def __init__(self, name: str, session: ChatSession, *, max_retries: int = 2,
                 history_messages: int = 120, context_fill: float = 0.6):
        super().__init__(name)
        self.session = session
        self.max_retries = max_retries
        self.history_messages = history_messages  # start a fresh conversation (with a resync) after this many
        # ...or once the transcript uses this share of the model's context window (if known),
        # leaving room for the next message and a long reasoning reply.
        self.context_fill = context_fill
        self.transcript_tokens = 0  # prompt + reply tokens of the last exchange, i.e. the transcript size
        self.notes: str | None = None
        self.cursor = 0          # how much of the public log this player has been shown
        self.private_cursor = 0  # ...and of their private notices (trades, peeks)
        self.session_msgs = 0
        self.started = False
        self.reasoning_log: list[dict] = []
        self._lock = threading.Lock()

    @property
    def model(self) -> str:
        return f"{self.session.provider}:{self.session.model}"

    def _open(self, view: PlayerView, request: str, state: str | None = None) -> str:
        resync = False
        window = self.session.context_window
        too_long = bool(window) and self.transcript_tokens >= self.context_fill * window
        if not self.started or self.session_msgs >= self.history_messages or too_long:
            resync = self.started
            self.session.reset(system_prompt(view))
            self.session_msgs = 0
            self.transcript_tokens = 0
            self.started = True
        events = list(view.log) if resync else list(view.log[self.cursor:])
        private = list(view.private) if resync else list(view.private[self.private_cursor:])
        self.cursor = len(view.log)
        self.private_cursor = len(view.private)
        self.session_msgs += 1
        return render_message(view, events, request, self.notes, resync=resync, state=state, private=private,
                              hypotheses=self.hypotheses)

    def _exchange(self, view: PlayerView, request: str, kind: str,
                  parse: Callable[[str], ParsedReply], retry: Callable[[str], str],
                  state: str | None = None) -> tuple[ParsedReply | None, dict]:
        message = self._open(view, request, state)
        attempts: list[dict] = []
        totals = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}
        thinking: list[str] = []
        raw = ""
        parsed: ParsedReply | None = None
        for _ in range(self.max_retries + 1):
            t0 = time.monotonic()
            try:
                comp = self.session.send(message)
            except Exception as e:
                attempts.append({"error": f"{type(e).__name__}: {e}", "seconds": round(time.monotonic() - t0, 2)})
                continue  # resend; the transcript is unchanged after a failed call
            for k in totals:
                totals[k] += getattr(comp, k)
            if comp.input_tokens:
                self.transcript_tokens = comp.input_tokens + comp.output_tokens
            if comp.thinking:
                thinking.append(comp.thinking)
            raw = comp.text
            parsed = parse(comp.text)
            attempts.append({"error": parsed.error, "seconds": round(time.monotonic() - t0, 2),
                             "served_by": comp.served_by, "stop_reason": comp.stop_reason})
            if parsed.notes:
                self.notes = parsed.notes
            changed = self.record_hypotheses(view, parse_hypotheses(comp.text))
            if changed:
                attempts[-1]["hypotheses"] = changed
            if parsed.error is None:
                break
            message = retry(parsed.error)
        meta = {"attempts": attempts, "thinking": "\n\n".join(thinking), "raw": raw,
                "model": self.model, **totals}
        ok = parsed if parsed is not None and parsed.error is None else None
        self.reasoning_log.append({
            "kind": kind, "game": view.game, "turn": view.turn, "request": request,
            "reasoning": parsed.reasoning if parsed else "", "notes": self.notes,
            "thinking": meta["thinking"], "card": str(ok.card) if ok and ok.card else None,
            "penalty": ok.penalty if ok else None, "rule": ok.rule if ok else None,
            "rule_text": ok.rule_text if ok else None, "hypotheses": dict(self.hypotheses),
            "attempts": attempts, **totals,
        })
        return ok, meta

    def choose(self, view: PlayerView) -> Choice:
        with self._lock:
            parsed, meta = self._exchange(view, turn_request(view), "play",
                                          lambda t: parse_play(t, view.hand, view.my_rule_numbers, view),
                                          lambda err: render_play_retry(view, err))
        if parsed is None:
            return Choice(None, meta=meta)
        return Choice(parsed.card, move=parsed.move, self_call=bool(parsed.penalty), self_call_rule=parsed.rule,
                      reasoning=parsed.reasoning,
                      notes=self.notes, meta=meta)

    def call(self, view: PlayerView, play: PendingPlay) -> Call:
        with self._lock:
            parsed, meta = self._exchange(view, judge_request(play, view), "call",
                                          lambda t: parse_call(t, view.my_rule_numbers),
                                          lambda err: render_call_retry(err, view))
        if parsed is None:
            return Call(None, meta=meta)
        return Call(parsed.penalty, rule=parsed.rule, reasoning=parsed.reasoning, notes=self.notes, meta=meta)

    def propose_rule(self, view: PlayerView, *, first: bool, feedback: str | None = None,
                     attempt: int = 1, attempts: int = 1, mechanics: bool = False,
                     wins: bool = False) -> Proposal | None:
        with self._lock:
            parsed, meta = self._exchange(
                view, rule_request(view, first=first, feedback=feedback, attempt=attempt, attempts=attempts,
                                     mechanics=mechanics, wins=wins),
                "rule", parse_rule,
                lambda err: f"INVALID: {err}\nReply again, ending with NOTES: <notes> and then RULE: <your rule>.",
                state=render_rule_state(view))
        if parsed is None:
            return None
        return Proposal(parsed.rule_text, reasoning=parsed.reasoning, notes=self.notes, meta=meta)

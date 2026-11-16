"""Play from the terminal. You see exactly the messages an LLM player gets."""

from __future__ import annotations

from ..game import PendingPlay, PlayerView
from ..protocol import (judge_request, parse_call, parse_hypotheses, parse_play, render_message, render_rule_state,
                        rule_request, system_prompt, turn_request)
from .base import Call, Choice, Player, Proposal


class HumanPlayer(Player):
    kind = "human"
    makes_rules = True
    states_hypotheses = True

    def __init__(self, name: str):
        super().__init__(name)
        self.cursor = 0
        self.private_cursor = 0
        self.shown_intro = False

    def _show(self, view: PlayerView, request: str, state: str | None = None) -> None:
        if not self.shown_intro:
            print(system_prompt(view))
            self.shown_intro = True
        print("\n" + render_message(view, list(view.log[self.cursor:]), request, state=state,
                                     private=list(view.private[self.private_cursor:]), hypotheses=self.hypotheses))
        self.cursor = len(view.log)
        self.private_cursor = len(view.private)
        if any(n not in view.my_rule_numbers for n in range(1, view.num_rules + 1)):
            raw = input("HYPOTHESES that changed, separated by ';' (e.g. R2: no red on red; R3: ?), or blank: ").strip()
            if raw:
                self.record_hypotheses(view, parse_hypotheses("HYPOTHESES:\n" + "\n".join(raw.split(";"))))

    def choose(self, view: PlayerView) -> Choice:
        self._show(view, turn_request(view))
        while True:
            card = input("PLAY: ").strip()
            options = ""
            if view.mechanics:
                raw = input("options, separated by ';' (e.g. ANGLE: 90; SPEND: 1), or blank: ").strip()
                options = "".join(f"\n{o.strip()}" for o in raw.split(";") if o.strip())
            call = (input("CALL (PENALTY R<n> / NONE) on your own play: ").strip() or "NONE") if view.can_call else "NONE"
            parsed = parse_play(f"PLAY: {card}{options}\nCALL: {call}", view.hand, view.my_rule_numbers, view)
            if parsed.card and not parsed.error:
                return Choice(parsed.card, move=parsed.move, self_call=bool(parsed.penalty),
                              self_call_rule=parsed.rule)
            print(f"INVALID: {parsed.error}")

    def call(self, view: PlayerView, play: PendingPlay) -> Call:
        self._show(view, judge_request(play, view))
        while True:
            parsed = parse_call("CALL: " + (input("CALL (PENALTY R<n> / NONE): ").strip() or "NONE"),
                                view.my_rule_numbers)
            if parsed.penalty is not None and not parsed.error:
                return Call(parsed.penalty, rule=parsed.rule)
            print(f"INVALID: {parsed.error}")

    def propose_rule(self, view: PlayerView, *, first: bool, feedback: str | None = None,
                     attempt: int = 1, attempts: int = 1, mechanics: bool = False,
                     wins: bool = False) -> Proposal | None:
        self._show(view, rule_request(view, first=first, feedback=feedback, attempt=attempt, attempts=attempts,
                                     mechanics=mechanics, wins=wins),
                   state=render_rule_state(view))
        text = ""
        while not text:
            text = input("RULE: ").strip()
        return Proposal(text)

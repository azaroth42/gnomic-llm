"""Scoring how well, and how quickly, players work out the hidden rules.

With every reply a player may state a hypothesis for each rule it doesn't own (`HYPOTHESES:`
block, see protocol.parse_hypotheses). Players are never scored on their own rules, which they know.

Grading (user decision: code and compare, live):
  1. The coding agent turns each distinct hypothesis text into code, exactly as it implements a
     proposed rule. It sees only the hypothesis, never the real rule.
  2. The engine compares the coded hypothesis with the real rule over a fixed sample of situations,
     the same sample for every player:
       restriction / permission  verdict per sampled play (FORBID / ALLOW / ABSTAIN)
       effect                    the effects it would cause per sampled play
       win                       whether it is met per sampled end of turn
     The score is the share of situations where they agree, counting only situations where at
     least one of them does something. So a hypothesis that never applies scores 0, and 1 means
     identical on every sampled situation. A hypothesis that can't be coded, or "?", scores 0.
  3. A rule counts as understood from the turn the player's hypothesis reaches UNDERSTOOD and
     stays there until the end of the match.

Grading runs in background workers with their own coder sessions, so play isn't slowed. Results
are cached by (rule, normalized hypothesis text), so a hypothesis shared or repeated is coded once.
"""

from __future__ import annotations

import queue
import random
import threading
from dataclasses import dataclass, field
from typing import Callable

from .mechanics import Effects
from .rules import RuleEntry, Verdict, _with_card, sample_contexts, win_contexts

UNDERSTOOD = 0.95      # a hypothesis at least this close to the real rule counts as understanding it
SAMPLES = 300          # sampled situations per rule
CARDS_PER_SAMPLE = 13  # candidate cards tried per situation (plays); win rules use win_contexts


def normalize(text: str) -> str:
    """Same key as the coder cache, so equal keys always mean the same code."""
    from .coder import normalize_rule_text
    return normalize_rule_text(text)


def _outcomes(rule, kind: str, samples, wins) -> list:
    """What a rule does in each sampled situation, in a form that can be compared."""
    out = []
    if kind == "win":
        for w in wins:
            try:
                out.append(bool(rule.wins(w)) if getattr(rule, "kind", "") == "win" else False)
            except Exception:  # noqa: BLE001
                out.append(False)
        return out
    for ctx, cands in samples:
        for c in cands[:CARDS_PER_SAMPLE]:
            cc = _with_card(ctx, c)
            if kind == "effect":
                fx = Effects(ctx.num_players)
                try:
                    if getattr(rule, "kind", "") == "effect":
                        rule.effect(cc, fx)
                except Exception:  # noqa: BLE001
                    pass
                out.append(tuple(fx.ops))
            else:
                try:
                    v = rule.verdict(cc)
                except Exception:  # noqa: BLE001
                    v = Verdict.ABSTAIN
                out.append(v)
    return out


def _idle(x) -> bool:
    return x is False or x == () or x is Verdict.ABSTAIN


def agreement(truth: list, guess: list) -> float:
    """Share of situations where they agree, among those where at least one does something."""
    active = [(a, b) for a, b in zip(truth, guess) if not (_idle(a) and _idle(b))]
    if not active:
        return 1.0
    return sum(a == b for a, b in active) / len(active)


@dataclass
class RuleTarget:
    """A real rule and its precomputed behaviour on that rule's fixed sample."""
    number: int
    entry: RuleEntry
    mechanics: bool
    wins: bool
    samples: list
    win_samples: list
    truth: list

    @property
    def kind(self) -> str:
        return self.entry.rule.kind


@dataclass
class Grade:
    rule: int
    text: str
    score: float | None = None     # None while pending, or if grading itself failed (API trouble)
    kind: str | None = None        # the kind the coder gave the hypothesis
    description: str | None = None
    error: str | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class Update:
    player: str
    seat: int
    rule: int
    text: str          # "" = no hypothesis ("?")
    match_turn: int    # turns played in the match before this statement (counts from this turn on)
    game: int
    turn: int


class HypothesisGrader:
    """Collects hypothesis updates during a match and grades them in the background."""

    def __init__(self, coder_factory: Callable[[], object], emit: Callable[..., object], *,
                 workers: int = 4, seed: int = 0, mechanics_from: int = 0, wins_from: int = 0):
        from .judges import mechanics_allowed, wins_allowed
        self._mech = lambda after: mechanics_allowed(after, mechanics_from)
        self._wins = lambda after: wins_allowed(after, wins_from)
        self.coder_factory = coder_factory
        self.emit = emit
        self.seed = seed
        self.targets: dict[int, RuleTarget] = {}
        self.rule_start: dict[int, int] = {}    # rule number -> match turn from which it is in effect
        self.grades: dict[tuple[int, str], Grade] = {}
        self.updates: list[Update] = []
        self._lock = threading.Lock()
        self._jobs: queue.Queue = queue.Queue()
        self._workers = [threading.Thread(target=self._work, daemon=True, name=f"grader-{i}")
                         for i in range(max(1, workers))]
        self._started = False
        self._cancel = False

    # ---- inputs ---------------------------------------------------------------

    def add_rule(self, number: int, entry: RuleEntry, start_turn: int) -> None:
        mech, wins = self._mech(entry.added_after_game), self._wins(entry.added_after_game)
        samples = sample_contexts(random.Random(self.seed * 1009 + number), SAMPLES, mechanics=mech)
        win_samples = win_contexts(samples) if entry.rule.kind == "win" else []
        target = RuleTarget(number, entry, mech, wins, samples, win_samples, [])
        target.truth = _outcomes(entry.rule, entry.rule.kind, samples, win_samples)
        with self._lock:
            self.targets[number] = target
            self.rule_start[number] = start_turn

    def record(self, player: str, seat: int, changes: dict[int, str], match_turn: int, game: int,
               turn: int) -> None:
        """A player stated new hypotheses. Rules they own were already left out by the player."""
        for rule, text in sorted(changes.items()):
            if rule not in self.targets or self.targets[rule].entry.owner == player:
                continue
            up = Update(player, seat, rule, text, match_turn, game, turn)
            key = (rule, normalize(text))
            with self._lock:
                self.updates.append(up)
                grade = self.grades.get(key)
                new = grade is None
                if new:
                    grade = self.grades[key] = Grade(rule, text)
                    if not text:
                        grade.score, grade.error = 0.0, "no hypothesis"
            self.emit("hypothesis", player=player, seat=seat, rule=rule, text=text, match_turn=match_turn,
                      game=game, turn=turn, score=grade.score, graded=grade.score is not None or grade.error is not None)
            if new and text:
                self._start()
                self._jobs.put(key)

    # ---- grading ------------------------------------------------------------------

    def _start(self) -> None:
        if not self._started:
            self._started = True
            for w in self._workers:
                w.start()

    def _work(self) -> None:
        coder = None
        while True:
            key = self._jobs.get()
            try:
                if self._cancel:
                    continue
                if coder is None:
                    coder = self.coder_factory()
                self._grade(coder, key)
            except Exception as e:  # noqa: BLE001 - never let a worker die
                with self._lock:
                    g = self.grades[key]
                    g.error = g.error or f"{type(e).__name__}: {e}"
            finally:
                self._jobs.task_done()

    def _grade(self, coder, key: tuple[int, str]) -> None:
        from .coder import CoderFailure
        rule, _ = key
        target = self.targets[rule]
        grade = self.grades[key]
        try:
            result = coder.implement(grade.text, target.samples[:60], mechanics=target.mechanics, wins=target.wins)
        except CoderFailure as e:
            score, kind, desc, err, meta = 0.0, None, None, f"could not be coded: {e}", {}
        else:
            hyp = result.rule
            kind, desc, meta, err = hyp.kind, hyp.description, result.meta, None
            same_family = (kind == target.kind or {kind, target.kind} <= {"restriction", "permission"})
            if same_family:
                score = round(agreement(target.truth, _outcomes(hyp, target.kind, target.samples,
                                                                  target.win_samples)), 4)
            else:
                score = 0.0
        with self._lock:
            grade.score, grade.kind, grade.description, grade.error, grade.meta = score, kind, desc, err, meta
            holders = sorted({u.player for u in self.updates if (u.rule, normalize(u.text)) == key})
        self.emit("hypothesis_graded", rule=rule, text=grade.text, score=score, coded_kind=kind, description=desc,
                  error=err, players=holders, understood=score >= UNDERSTOOD,
                  coder={k: meta.get(k) for k in ("model", "input_tokens", "output_tokens")})

    def wait(self, cancel: bool = False) -> None:
        """Wait for grading to finish. With cancel, drop what hasn't started."""
        if cancel:
            self._cancel = True
        if self._started:
            self._jobs.join()

    # ---- metrics --------------------------------------------------------------------

    def score_of(self, text: str, rule: int) -> float | None:
        g = self.grades.get((rule, normalize(text)))
        return None if g is None else g.score

    def summary(self, players: list, total_turns: int) -> dict:
        """Per player and rule: final score, turns to understand, and the mean score over the turns
        the rule was in effect (rewards being right early). Rules a player owns are left out, as are
        players that never state hypotheses (baselines)."""
        with self._lock:
            updates = list(self.updates)
            starts = dict(self.rule_start)
            targets = dict(self.targets)
        per_player = {}
        for seat, p in enumerate(players):
            if not getattr(p, "states_hypotheses", False):
                continue
            rules = {}
            ungraded = 0
            for number, target in sorted(targets.items()):
                if target.entry.owner == p.name:
                    continue
                start = starts[number]
                if total_turns <= start:
                    continue
                mine = sorted((u for u in updates if u.player == p.name and u.rule == number),
                              key=lambda u: u.match_turn)
                timeline = [0.0] * (total_turns - start)
                score, k = 0.0, 0
                for t in range(start, total_turns):
                    while k < len(mine) and mine[k].match_turn <= t:
                        s = self.score_of(mine[k].text, number)
                        if s is None:
                            ungraded += 1
                        else:
                            score = s
                        k += 1
                    timeline[t - start] = score
                ok_from = None
                for i in range(len(timeline) - 1, -1, -1):
                    if timeline[i] >= UNDERSTOOD:
                        ok_from = i
                    else:
                        break
                last = mine[-1].text if mine else ""
                rules[f"R{number}"] = {
                    "final_score": round(timeline[-1], 4),
                    "mean_score": round(sum(timeline) / len(timeline), 4),
                    "understood": ok_from is not None,
                    "turns_to_understand": ok_from,
                    "turns_in_effect": len(timeline),
                    "hypotheses": len(mine),
                    "final_hypothesis": last,
                }
            n = len(rules)
            understood = [r for r in rules.values() if r["understood"]]
            per_player[p.name] = {
                "rules": rules,
                "rules_to_learn": n,
                "rules_understood": len(understood),
                "understanding_score": round(sum(r["mean_score"] for r in rules.values()) / n, 4) if n else None,
                "final_score": round(sum(r["final_score"] for r in rules.values()) / n, 4) if n else None,
                "mean_turns_to_understand": (round(sum(r["turns_to_understand"] for r in understood) / len(understood), 2)
                                             if understood else None),
                "ungraded_hypotheses": ungraded,
            }
        with self._lock:
            coded = [g for g in self.grades.values() if g.text]
        return {"threshold": UNDERSTOOD, "samples_per_rule": SAMPLES, "players": per_player,
                "hypotheses_graded": sum(g.score is not None for g in coded),
                "grading_errors": sum(g.score is None for g in coded)}

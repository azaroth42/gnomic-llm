"""A match: a series of games under a growing stack of hidden rules.

Before game 1 a randomly chosen dealer (a seated player) is secretly given rule
R1. After each game the winner is secretly given one more rule. A game with no
winner (round limit reached: every player has played `max_rounds` cards) adds no rule.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Callable

from .events import null_emit
from .game import Controls, Game, GameResult, PlayerView, PublicEvent, PublicLog, StopMatch
from .judges import EngineRuleAuthor, ProposedRuleAuthor, RuleAuthor
from .players.base import Player
from .players.factory import make_player, make_session
from .rules import RuleStack


@dataclass
class MatchConfig:
    players: list[str]                  # player specs, see players/factory.py
    num_games: int = 5
    seed: int | None = None
    hand_size: int = 7
    max_rounds: int = 20                # a game with no winner ends after every player has played this many cards
    delay: float = 0.0                  # pause between turns (for watching non-LLM games)
    dealer: int | None = None           # seat; random if None
    names: list[str] = field(default_factory=list)  # optional display names, one per player
    propose_rules: bool = False         # the dealer and winners make up their rules; a coding model implements them
    coder: str = "anthropic:claude-opus-5-5"  # coding-model spec (with propose_rules)
    rule_attempts: int = 3              # proposals per rule before falling back to a library rule
    mechanics_from: int = 3             # 1-based game from which rules may add table mechanics; 0 = never
    wins_from: int = 6                  # 1-based game from which rules may be win conditions; 0 = never


class Match:
    def __init__(self, config: MatchConfig, emit: Callable[..., object] = null_emit,
                 controls: Controls | None = None, author: RuleAuthor | None = None):
        self.config = config
        self.emit = emit
        self.controls = controls or Controls(config.delay)
        self.rng = random.Random(config.seed)
        self.stack = RuleStack()
        self.log = PublicLog()
        self.author = author or EngineRuleAuthor(random.Random(self.rng.random()), config.mechanics_from,
                                                     config.wins_from)
        self.players: list[Player] = self._build_players()
        self.scores = [0] * len(self.players)
        self.results: list[GameResult] = []
        self.private: list[list[str]] = [[] for _ in self.players]  # trade / peek notices, per seat
        self.dealer = config.dealer if config.dealer is not None else self.rng.randrange(len(self.players))
        if config.propose_rules and author is None:
            from .coder import RuleCoder
            self.author = ProposedRuleAuthor(RuleCoder(make_session(config.coder)), fallback=self.author,
                                             players=self.players, view=self._rule_view, emit=self.emit,
                                             rng=random.Random(self.rng.random()), attempts=config.rule_attempts,
                                             mechanics_from=config.mechanics_from, wins_from=config.wins_from)

    def _mechanics_from(self) -> int:
        f = self.config.mechanics_from
        return f if f and f <= self.config.num_games else 0

    def _wins_from(self) -> int:
        f = self.config.wins_from
        return f if f and f <= self.config.num_games else 0

    def mechanics_in(self, game: int) -> bool:
        """Are table mechanics in play in `game` (0-based)? From game `mechanics_from` (1-based) on."""
        return bool(self._mechanics_from()) and game + 1 >= self._mechanics_from()

    def _rule_view(self, seat: int, after_game: int) -> PlayerView:
        """What a player sees while making up a rule: no table, just the log, scores and their rules."""
        n = len(self.players)
        name = self.players[seat].name
        return PlayerView(
            seat=seat, player_names=tuple(p.name for p in self.players), hand=(), discard=(), deck_size=0,
            hand_sizes=(0,) * n, game=after_game + 1, num_games=self.config.num_games, turn=0,
            num_rules=len(self.stack), scores=tuple(self.scores), log=tuple(self.log.events),
            my_rules=tuple((i + 1, e.rule.description) for i, e in enumerate(self.stack.entries) if e.owner == name),
            max_rounds=self.config.max_rounds,
            judges=tuple(i for i, p in enumerate(self.players) if any(e.owner == p.name for e in self.stack.entries)),
            players_make_rules=self.config.propose_rules, private=tuple(self.private[seat]),
            mechanics_from=self._mechanics_from(), wins_from=self._wins_from())

    def _build_players(self) -> list[Player]:
        players, seen = [], set()
        for i, spec in enumerate(self.config.players):
            default = self.config.names[i] if i < len(self.config.names) else None
            p = make_player(spec, self.stack, seed=self.rng.randrange(2**31), default_name=default)
            base, k = p.name, 2
            while p.name in seen:  # names must be unique; they identify rule owners
                p.name = f"{base}#{k}"
                k += 1
            seen.add(p.name)
            players.append(p)
        if len(players) < 2:
            raise ValueError("need at least 2 players")
        return players

    def _add_rule(self, owner: int, after_game: int) -> None:
        try:
            entry = self.author.author_rule(self.stack, self.players[owner].name, after_game)
        except RuntimeError as e:
            self.emit("rule_skipped", owner=self.players[owner].name, reason=str(e))
            return
        self.stack.add(entry)
        number = len(self.stack)
        self.log.add(PublicEvent("rule_added", game=max(after_game, 0),
                                 turn=-1 if after_game < 0 else None, player=owner, detail=str(number)))
        self.emit("rule_added", number=number, owner=entry.owner, owner_seat=owner,
                  after_game=after_game, rule=entry.rule.to_json(),
                  stack=[e.to_json() for e in self.stack.entries])

    def run(self) -> dict:
        cfg = self.config
        n = len(self.players)
        self.emit("match_start", players=[p.describe() for p in self.players], dealer=self.dealer,
                  coder=cfg.coder if cfg.propose_rules else None,
                  dealer_name=self.players[self.dealer].name, num_games=cfg.num_games, seed=cfg.seed,
                  config={k: v for k, v in asdict(cfg).items() if k != "players"})
        self._add_rule(self.dealer, -1)
        stopped = False
        try:
            for g in range(cfg.num_games):
                game = Game(g, cfg.num_games, self.players, self.stack, self.log, self.scores, self.rng,
                            starting_player=(self.dealer + 1 + g) % n, emit=self.emit,
                            controls=self.controls, hand_size=cfg.hand_size, max_rounds=cfg.max_rounds,
                            players_make_rules=cfg.propose_rules,
                            mechanics=self.mechanics_in(g), mechanics_from=self._mechanics_from(),
                            wins_from=self._wins_from(),
                            private=self.private)
                result = game.play()
                self.results.append(result)
                if result.winner is not None:
                    self.scores[result.winner] += 1
                    if g < cfg.num_games - 1:
                        self._add_rule(result.winner, g)
        except StopMatch:
            stopped = True
        summary = summarize(self.players, self.results, self.scores)
        summary["stopped"] = stopped
        summary["rules"] = [e.to_json() for e in self.stack.entries]
        self.emit("match_end", summary=summary)
        return summary


def _ratio(a: float, b: float) -> float | None:
    return round(a / b, 4) if b else None


def summarize(players: list[Player], results: list[GameResult], scores: list[int]) -> dict:
    """Per-player benchmark metrics.

    Playing:  illegal_rate (illegal plays / plays), avoidable_illegal_rate (illegal when a legal card was in hand)
    Judging other players' plays: precision, recall, accuracy of PENALTY calls. Players police
              only their own rules: a call is correct iff it was upheld (it named a rule of the
              judge's that the play broke); a miss is a play that broke the judge's rule uncalled.
    Honesty:  self_call_rate (plays that broke the player's own rule which they called on themselves)
    """
    n = len(players)
    s = [dict(name=p.name, kind=p.kind, model=p.model, wins=scores[i], plays=0, illegal=0,
              avoidable_illegal=0, forced_plays=0, caught=0, uncaught=0,
              own_rule_breaks=0, self_call_on_own_break=0, false_self_calls=0,
              judge_tp=0, judge_fp=0, judge_fn=0, judge_tn=0, judge_forced=0,
              penalty_draws=0, false_call_draws=0, input_tokens=0, output_tokens=0,
              decision_seconds=0.0, decisions=0, illegal_by_game=[])
         for i, p in enumerate(players)]
    for r in results:
        per_game = [[0, 0] for _ in range(n)]  # [illegal, plays]
        for t in r.stats:
            me = s[t.player]
            me["plays"] += 1
            per_game[t.player][1] += 1
            me["forced_plays"] += t.forced
            self_called = t.calls.get(t.player, False)
            upheld = bool(t.upheld_by) or t.engine_call
            if not t.legal:
                me["illegal"] += 1
                per_game[t.player][0] += 1
                me["avoidable_illegal"] += t.legal_options > 0
                if upheld:
                    me["caught"] += 1
                    me["penalty_draws"] += 1
                else:
                    me["uncaught"] += 1
            if t.player in t.violated_owners:
                me["own_rule_breaks"] += 1
                me["self_call_on_own_break"] += t.player in t.upheld_by
            if self_called and t.player not in t.upheld_by:
                me["false_self_calls"] += 1
                me["false_call_draws"] += 1
            play_meta = t.meta.get("play", {})
            me["input_tokens"] += play_meta.get("input_tokens", 0)
            me["output_tokens"] += play_meta.get("output_tokens", 0)
            me["decision_seconds"] += t.elapsed
            me["decisions"] += 1
            for seat, called in t.calls.items():
                if seat == t.player:
                    continue
                j = s[seat]
                j["judge_forced"] += t.call_forced.get(seat, False)
                mine = seat in t.violated_owners  # did the play break this judge's own rule?
                if called and seat in t.upheld_by:
                    j["judge_tp"] += 1
                elif called:
                    j["judge_fp"] += 1
                    j["false_call_draws"] += 1
                elif mine:
                    j["judge_fn"] += 1
                else:
                    j["judge_tn"] += 1
                cm = t.meta.get("calls", {}).get(seat, {})
                j["input_tokens"] += cm.get("input_tokens", 0)
                j["output_tokens"] += cm.get("output_tokens", 0)
                j["decision_seconds"] += cm.get("elapsed", 0.0)
                j["decisions"] += 1
        for i in range(n):
            s[i]["illegal_by_game"].append(_ratio(*per_game[i]))
    for me in s:
        me["illegal_rate"] = _ratio(me["illegal"], me["plays"])
        me["avoidable_illegal_rate"] = _ratio(me["avoidable_illegal"], me["plays"])
        me["self_call_rate"] = _ratio(me["self_call_on_own_break"], me["own_rule_breaks"])
        tp, fp, fn, tn = me["judge_tp"], me["judge_fp"], me["judge_fn"], me["judge_tn"]
        me["judge_precision"] = _ratio(tp, tp + fp)
        me["judge_recall"] = _ratio(tp, tp + fn)
        me["judge_accuracy"] = _ratio(tp + tn, tp + fp + fn + tn)
        me["avg_decision_seconds"] = _ratio(me["decision_seconds"], me["decisions"])
        me["decision_seconds"] = round(me["decision_seconds"], 2)
    return {"games": [{"index": r.index, "winner": players[r.winner].name if r.winner is not None else None,
                       "turns": r.turns, "end_reason": r.end_reason} for r in results],
            "players": s}

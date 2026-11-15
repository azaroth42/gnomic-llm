"""Command line interface.

  gnomic play -p anthropic:claude-opus-5-5 -p openai:gpt-5 -p stats --games 5 --ui
  gnomic replay logs/match-....jsonl          # watch a finished match in the UI
  gnomic rules                                # list the rule library
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import webbrowser
from datetime import datetime
from pathlib import Path

from .events import EventBus
from .game import Controls
from .match import Match, MatchConfig
from .rules import TEMPLATES


def _print_summary(summary: dict) -> None:
    print("\nGames:")
    for g in summary["games"]:
        print(f"  G{g['index'] + 1}: winner={g['winner']} turns={g['turns']} ({g['end_reason']})")
    print("\nRules:")
    for i, r in enumerate(summary["rules"]):
        print(f"  R{i + 1} [{r['rule']['kind']}] owner={r['owner']}: {r['rule']['description']}")
        if r["rule"].get("proposal"):
            print(f"      proposed as: {r['rule']['proposal']}")
    pct = lambda x: "  -  " if x is None else f"{x * 100:4.0f}%"
    print("\n  player                 wins  plays illegal avoidable caught self-call  prec recall   acc")
    for p in summary["players"]:
        print(f"  {p['name'][:22]:22} {p['wins']:4} {p['plays']:6} {pct(p['illegal_rate'])}   {pct(p['avoidable_illegal_rate'])}"
              f"  {p['caught']:3}/{p['illegal']:<3} {pct(p['self_call_rate'])}   {pct(p['judge_precision'])} {pct(p['judge_recall'])} {pct(p['judge_accuracy'])}")


def cmd_play(args: argparse.Namespace) -> int:
    if len(args.player) < 2:
        print("need at least two --player specs", file=sys.stderr)
        return 2
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = Path(args.log or f"logs/match-{stamp}.jsonl")
    bus = EventBus(log_path)
    delay = args.delay if args.delay is not None else (0.5 if args.ui else 0.0)
    controls = Controls(delay)
    config = MatchConfig(players=args.player, num_games=args.games, seed=args.seed, hand_size=args.hand_size,
                         max_rounds=args.max_rounds, delay=delay, dealer=args.dealer,
                         propose_rules=args.propose_rules, coder=args.coder, rule_attempts=args.rule_attempts,
                         mechanics_from=args.mechanics_from, wins_from=args.wins_from)
    try:
        match = Match(config, emit=bus, controls=controls)
    except (ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.ui:
        from .server import serve
        serve(bus, controls, args.host, args.port)
        url = f"http://{args.host}:{args.port}/"
        print(f"Spectator UI: {url}")
        if args.paused:
            controls.pause()
            print("Match is paused; press Resume in the UI to start.")
        if args.open:
            webbrowser.open(url)

    summary: dict = {}

    def run():
        nonlocal summary
        summary = match.run()

    t = threading.Thread(target=run, name="match", daemon=True)
    t.start()
    try:
        while t.is_alive():
            t.join(0.5)
    except KeyboardInterrupt:
        print("\nstopping…")
        controls.stop()
        t.join(30)
    if summary:
        summary_path = log_path.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(summary, indent=2))
        _print_summary(summary)
        print(f"\nEvent log: {log_path}\nSummary:   {summary_path}")
    if args.ui:
        print("UI still serving; Ctrl-C to exit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    bus.close()
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    from .server import serve

    bus = EventBus()
    with open(args.log, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                ev = json.loads(line)
                ev.pop("seq", None)
                with bus._lock:
                    ev["seq"] = len(bus.history)
                    bus.history.append(ev)
    serve(bus, None, args.host, args.port)
    url = f"http://{args.host}:{args.port}/"
    print(f"Replaying {len(bus.history)} events at {url} (Ctrl-C to exit)")
    if args.open:
        webbrowser.open(url)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    return 0


def cmd_rules(args: argparse.Namespace) -> int:
    import random
    rng = random.Random(0)
    for t in TEMPLATES.values():
        params = t.sample_params(rng)
        print(f"{t.name:22} {t.kind:12} {t.describe(**params)}" + (f"   params e.g. {params}" if params else ""))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="gnomic", description="Gnomic Mau LLM benchmark")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("play", help="run a match")
    p.add_argument("-p", "--player", action="append", default=[], metavar="SPEC",
                   help="player spec, repeat per seat: random | stats | oracle | human | "
                        "anthropic:MODEL | openai:MODEL | gemini:MODEL | litellm:MODEL  (options: ?effort=high&name=X)")
    p.add_argument("--games", type=int, default=5)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--hand-size", type=int, default=7)
    p.add_argument("--max-rounds", type=int, default=20,
                   help="a game with no winner ends after every player has played this many cards")
    p.add_argument("--dealer", type=int, default=None, help="dealer seat (0-based); random by default")
    p.add_argument("--propose-rules", action="store_true",
                   help="the dealer (and each game's winner) makes up their rule in plain language; a coding "
                        "model implements it and the engine checks it. Players that can't (baselines) get library rules")
    p.add_argument("--coder", default="anthropic:claude-opus-5-5", metavar="SPEC",
                   help="coding model for --propose-rules, as a provider spec (default: %(default)s)")
    p.add_argument("--rule-attempts", type=int, default=3, metavar="N",
                   help="with --propose-rules, how many rules to ask for before using a library rule (default: %(default)s)")
    p.add_argument("--mechanics-from", type=int, default=3, metavar="N",
                   help="from game N on, rules may add table mechanics (tokens, spaces, angles, trades, reversing, "
                        "skips, effects, multi-card plays...); 0 disables (default: %(default)s)")
    p.add_argument("--wins-from", type=int, default=6, metavar="N",
                   help="from game N on, rules may be win conditions (checked only at the end of the player's own "
                        "turn); 0 disables (default: %(default)s)")
    p.add_argument("--log", default=None, help="event log path (JSONL)")
    p.add_argument("--ui", action="store_true", help="serve the spectator web UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--delay", type=float, default=None, help="seconds between turns (default 0.5 with --ui)")
    p.add_argument("--paused", action="store_true", help="start paused (with --ui)")
    p.add_argument("--open", action="store_true", help="open the UI in a browser")
    p.set_defaults(fn=cmd_play)

    r = sub.add_parser("replay", help="view a finished match log in the UI")
    r.add_argument("log")
    r.add_argument("--host", default="127.0.0.1")
    r.add_argument("--port", type=int, default=8765)
    r.add_argument("--open", action="store_true")
    r.set_defaults(fn=cmd_replay)

    lr = sub.add_parser("rules", help="list the rule library")
    lr.set_defaults(fn=cmd_rules)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())

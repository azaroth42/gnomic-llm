"""Command line interface.

  gnomic play -p anthropic:claude-opus-5-5 -p openai:gpt-5 -p stats --games 5 --ui
  gnomic play -p web?name=Ann -p anthropic:claude-opus-5-5 -p stats --open   # play in the browser
  gnomic replay --seconds-per-turn 2 --open   # pick a finished match from an index, paced
  gnomic replay logs/match-....jsonl          # watch one finished match in the UI
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
from .coder import default_cache_path
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
    und = summary.get("understanding")
    if und:
        print(f"\nUnderstanding (hypotheses coded and compared with each rule; understood at >= {und['threshold']:.0%}):")
        print("  player                 rules understood  score  final  turns-to-understand")
        for name, u in und["players"].items():
            mt = u["mean_turns_to_understand"]
            print(f"  {name[:22]:22} {u['rules_to_learn']:5} {u['rules_understood']:10}  {pct(u['understanding_score'])}  "
                  f"{pct(u['final_score'])}  {'-' if mt is None else mt}")
    cc = summary.get("coder_cache")
    if cc and (cc["coder_calls"] or cc["cache_hits"]):
        print(f"\nCoder: {cc['coder_calls']} texts coded, {cc['cache_hits']} answered from the cache")
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
                         mechanics_from=args.mechanics_from, wins_from=args.wins_from,
                         grade_hypotheses=not args.no_grade, grade_workers=args.grade_workers,
                         coder_cache=None if args.coder_cache.lower() == "none" else args.coder_cache)
    try:
        match = Match(config, emit=bus, controls=controls)
    except (ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    from .players.web import WebHumanPlayer
    humans = [(seat, p) for seat, p in enumerate(match.players) if isinstance(p, WebHumanPlayer)]
    for _, p in humans:
        p.should_stop = lambda: controls.stop_requested
    if args.ui or humans:
        import secrets
        import socket
        from .server import serve
        # With human players at the table, the spectator view (which shows every hidden rule
        # and hand) is served only with --ui, and only behind a key.
        key = secrets.token_urlsafe(9) if humans and args.ui else None
        serve(bus, controls, args.host, args.port, players=[p for _, p in humans], spectator=args.ui, key=key)
        host = socket.gethostname() if args.host in ("0.0.0.0", "::") else args.host
        base = f"http://{host}:{args.port}"
        urls = []
        for seat, p in humans:
            url = f"{base}/play/{p.token}"
            urls.append(url)
            print(f"Player page for P{seat + 1} ({p.name}): {url}")
        if args.ui:
            url = f"{base}/" + (f"?key={key}" if key else "")
            urls.insert(0, url)
            print(f"Spectator UI{' (shows every hidden rule and hand: keep it from the players)' if key else ''}: {url}")
        if args.paused and args.ui:
            controls.pause()
            print("Match is paused; press Resume in the UI to start.")
        if args.open:
            for url in urls:
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
    if args.ui or humans:
        print("UI still serving; Ctrl-C to exit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    bus.close()
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    from .replay import parse_spt, serve_replays

    spt = None
    if args.seconds_per_turn is not None:
        spt = parse_spt(str(args.seconds_per_turn))
        if spt is None:
            print("--seconds-per-turn must be a positive number", file=sys.stderr)
            return 2
    logs = Path(args.logs)
    page = "/"
    if args.log:
        log = Path(args.log)
        if not log.is_file():
            print(f"no such log: {log}", file=sys.stderr)
            return 2
        logs, page = log.parent, f"/match/{log.name}"
    serve_replays(logs, args.host, args.port, seconds_per_turn=spt)
    url = f"http://{args.host}:{args.port}{page}"
    pace = f", {spt:g} s per turn" if spt else ", all at once"
    print(f"Replaying {'matches in ' + str(logs) if page == '/' else args.log}{pace}: {url} (Ctrl-C to exit)")
    if page != "/":
        print(f"Index of the matches in {logs}: http://{args.host}:{args.port}/")
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
                   help="player spec, repeat per seat: random | stats | oracle | human (terminal) | web (browser) | "
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
                   help="coding model, as a provider spec: implements --propose-rules rules and grades players' "
                        "rule hypotheses (default: %(default)s)")
    p.add_argument("--no-grade", action="store_true",
                   help="don't grade players' rule hypotheses (grading uses the --coder model)")
    p.add_argument("--grade-workers", type=int, default=4, metavar="N",
                   help="parallel coder sessions for grading hypotheses (default: %(default)s)")
    p.add_argument("--coder-cache", default=str(default_cache_path()), metavar="PATH",
                   help="JSONL cache of coder results, so no text is coded twice; committed with the repo "
                        "(default: %(default)s); 'none' keeps the cache for this match only")
    p.add_argument("--rule-attempts", type=int, default=3, metavar="N",
                   help="with --propose-rules, how many rules to ask for before using a library rule (default: %(default)s)")
    p.add_argument("--mechanics-from", type=int, default=3, metavar="N",
                   help="from game N on, rules may add table mechanics (tokens, spaces, angles, trades, "
                        "effects, multi-card plays...); 0 disables (default: %(default)s)")
    p.add_argument("--wins-from", type=int, default=6, metavar="N",
                   help="from game N on, rules may be win conditions (checked only at the end of the player's own "
                        "turn); 0 disables (default: %(default)s)")
    p.add_argument("--log", default=None, help="event log path (JSONL)")
    p.add_argument("--ui", action="store_true", help="serve the spectator web UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--delay", type=float, default=None, help="seconds between turns (default 0.5 with --ui)")
    p.add_argument("--paused", action="store_true", help="start paused (with --ui)")
    p.add_argument("--open", action="store_true", help="open the UI (and any web player pages) in a browser")
    p.set_defaults(fn=cmd_play)

    r = sub.add_parser("replay", help="watch finished matches in the UI (an index of the logs, or one log)")
    r.add_argument("log", nargs="?", help="a match log to open directly (default: show the index of --logs)")
    r.add_argument("--logs", default="logs", help="directory of match logs for the index (default: %(default)s)")
    r.add_argument("--seconds-per-turn", type=float, default=None, metavar="N",
                   help="pace the replay: every turn takes N seconds, however long the players took "
                        "(default: show the whole match at once; the index page can change it)")
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

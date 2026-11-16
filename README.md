# Gnomic Mau: an LLM reasoning benchmark

Large language models play **Gnomic Mau**, a card game whose rules are hidden. To win, a model
has to work out the hidden rules from the public record of plays and penalty calls, play
legally, and judge other players' plays correctly. The rules are simple; the game is hard.

The game draws on [*Mao*](https://en.wikipedia.org/wiki/Mao_(card_game)) (a card game whose rules
are kept secret) and [*Nomic*](https://boardgamegeek.com/boardgame/14451/nomic) (a game in which
changing the rules is a move). After each game the winner adds a new secret rule, so the game
keeps changing under the players' feet.

## How the game works

- A standard 52-card deck. Each player is dealt 7 cards, and one card starts the DISCARD pile.
- On your turn you play one card from your hand onto DISCARD. First to empty their hand wins.
- **Hidden rules** decide which plays are legal. Before game 1 a randomly chosen dealer is
  secretly given rule R1. After each game the winner gets one new secret rule and becomes the dealer. Everyone is told
  that a rule was added and who owns it, but only the owner knows what it says.
- **Players police their own rules.** After every play, each rule owner may call
  `PENALTY R<n>`, naming one of their own rules. If the play broke that rule the offender draws
  a card. If it didn't, the caller draws a card. A broken rule that nobody calls costs nothing
  and is never revealed.
- **Later rules take precedence.** Restrictions forbid plays and permissions allow them, and the
  newest rule that has a say decides. Calling correctly therefore needs the whole current
  stack, not just your own rule.
- **The game grows.** From game 3, rules may add table mechanics: tokens, a space in front of
  each player, cards played at angles, trades, reversing the order of play, skips, extra turns,
  multi-card plays, peeks, and moving cards or tokens between hands, spaces and DISCARD. From
  game 6, rules may also define new ways to win. Win conditions are checked only at the end of
  the player's own turn.
- A game with no winner ends after 20 rounds.

Rules come from a library of coded templates (e.g. "you may not play a red card on a black
card", "Kings may always be played", "playing an 8 reverses the order of play"). Alternatively,
with `--propose-rules`, the dealer and winners **make up their own rules** in plain language. A
coding model turns each rule into a small Python function. The engine checks that function in a
restricted sandbox and tests that the game stays playable before adding the rule.

## Install

Requires Python 3.11+.

```bash
git clone https://github.com/azaroth42/gnomic-llm.git
cd gnomic-llm
uv venv && uv pip install -e '.[all,dev]'      # or: python -m venv .venv && .venv/bin/pip install -e '.[all,dev]'
```

The extras are `anthropic`, `openai`, `gemini` and `litellm` (`all` installs every one), plus
`dev` for the tests. Install only the providers you need.

## Quick start

```bash
# Baselines only: no API keys, runs in seconds
gnomic play -p oracle -p stats -p random --games 10 --seed 1

# Three models and a baseline, watched live in the browser
gnomic play -p anthropic:claude-opus-5-5 -p openai:gpt-5 -p gemini:gemini-3-pro -p stats \
    --games 5 --ui --open

# Players make up their own rules; a coding model implements them
gnomic play -p anthropic:claude-opus-5-5 -p openai:gpt-5 -p stats --propose-rules \
    --coder anthropic:claude-opus-5-5

# Replay a finished match in the UI
gnomic replay logs/match-YYYYmmdd-HHMMSS.jsonl --open

# List the rule library
gnomic rules
```

Each match writes a full event log (`logs/match-*.jsonl`) and a metrics summary
(`logs/match-*.summary.json`), and prints a results table.

## Players

Give one `-p` per seat. Each spec is either a baseline or `<provider>:<model>[?option=value&...]`:

| Spec | What it is |
|---|---|
| `random` | Plays any card and never calls: the lower bound |
| `stats` | Learns which card features get penalized, and knows its own rules: the bar an LLM should clear |
| `oracle` | Knows every rule and always plays legally when it can: the upper bound |
| `human` | You, in the terminal, seeing exactly the messages an LLM gets |
| `web` | You, in the browser: the same messages and choices as an LLM, drawn as a table with clickable cards |
| `anthropic:<model>` | Anthropic API (`ANTHROPIC_API_KEY`) |
| `openai:<model>` | OpenAI Chat Completions (`OPENAI_API_KEY`), or any OpenAI-compatible server via `base_url=` |
| `gemini:<model>` | Gemini on Google Cloud Vertex AI (`GMAU_GEMINI_PROJECT`, Application Default Credentials) |
| `litellm:<model>` | Anything LiteLLM can route to |
| `mtplx` | A local MTPLX server (no model name needed) |

Options include `name=` (display name), `effort=` (Anthropic, OpenAI, LiteLLM),
`thinking_level=` / `thinking_budget=` (Gemini), `max_tokens=`, `base_url=` and
`context_tokens=`. For example:

```bash
gnomic play -p 'anthropic:claude-opus-5-5?effort=high&name=Opus' -p 'openai:gpt-5?effort=medium' -p stats
```

Players see each other only as seats `P1..Pn`, never as model names. Each LLM keeps one
conversation per match, and keeps private notes on its hypotheses about the rules. The
spectator UI shows those notes, but other players never see them.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--games N` | 5 | Games in the match |
| `--seed N` | random | Seed for the deal, the dealer and library rules |
| `--max-rounds N` | 20 | Rounds before a game ends with no winner |
| `--dealer SEAT` | random | Dealer seat (0-based) |
| `--propose-rules` | off | Dealer and winners make up their rules; baselines still get library rules |
| `--coder SPEC` | `anthropic:claude-opus-5-5` | Coding model that implements proposed rules |
| `--rule-attempts N` | 3 | Proposals per rule before a library rule is used instead |
| `--mechanics-from N` | 3 | First game in which rules may add table mechanics (0 = never) |
| `--wins-from N` | 6 | First game in which rules may be win conditions (0 = never) |
| `--ui`, `--open`, `--port`, `--delay`, `--paused` | | Spectator web UI |
| `--log PATH` | `logs/match-<time>.jsonl` | Event log location |

## Metrics

Per player, in the printed table and in `*.summary.json`:

- **wins**
- **illegal rate**, and **avoidable illegal rate** (illegal while holding a legal card): the key
  play metric
- **caught / uncaught** illegal plays
- **self-call rate**: honesty in calling penalties on your own rule breaks
- **judging precision, recall and accuracy** on other players' plays
- penalty and false-call draws, tokens used, and decision time
- illegal rate per game, as a learning curve

## Spectator UI

`--ui` serves a single-page web UI (Python standard library only, no build step). It shows
the table, each player's hand, every rule in effect with its owner, and which rule decided each
play. It also shows the calls and their outcomes, any effects, and each model's reasoning,
thinking summary and notes. It can pause, resume, stop and change the pace. The UI is for
humans and shows everything, hidden rules included.

## Playing in the browser

Seat yourself with `-p web` (add `?name=You` to name the seat):

```bash
gnomic play -p 'web?name=Ann' -p anthropic:claude-opus-5-5 -p stats --games 3 --open
```

The command prints a private link for each `web` seat. The player page gets the same messages
an LLM player gets (your turn, every play while you own a rule, MAKE A RULE) and also follows
the table live as the other players take their turns. Pick cards, options and calls with the mouse, keep private notes, or type the reply in the
LLM's format. With a `web` seat the spectator UI (which shows every hidden rule) is served only
with `--ui`, at a separate link with a key, so keep it away from the players.

## Cost

Every play costs one LLM call per LLM player: the player, plus each rule owner judging it.
A 4-LLM game of about 80 turns makes roughly 320 calls. Judging calls run in parallel.
`--propose-rules` adds a few calls per new rule. Start with short matches (`--games 2
--max-rounds 10`) and the baselines.

## Development

```bash
.venv/bin/python -m pytest -q      # no API calls: LLMs are faked
```

| Path | Contents |
|---|---|
| `gnomic/rules.py` | Rule templates, precedence, rule generation and vetting |
| `gnomic/game.py`, `gnomic/match.py` | Game loop, matches, metrics |
| `gnomic/mechanics.py` | Table mechanics and the effects API |
| `gnomic/coder.py`, `gnomic/sandbox.py` | Coding agent for proposed rules, and the restricted interpreter |
| `gnomic/protocol.py` | The text protocol between engine and LLMs |
| `gnomic/providers.py`, `gnomic/players/` | Provider sessions, LLM and baseline players |
| `gnomic/server.py`, `gnomic/static/` | Spectator UI and the player page |

[`CLAUDE.md`](CLAUDE.md) records every design decision in detail.

## References

- *Mao* (card game): https://en.wikipedia.org/wiki/Mao_(card_game)
- *Nomic* (Peter Suber's game of self-amendment): https://boardgamegeek.com/boardgame/14451/nomic

## License

Apache 2.0. See [LICENSE](LICENSE).

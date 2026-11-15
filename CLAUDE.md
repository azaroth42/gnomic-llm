# Gnomic Mau: an LLM reasoning benchmark

LLMs play Gnomic Mau, a card game whose rules are hidden. To win, a player has to
induce the hidden rules from the public record of plays and penalty calls, play
legally, and judge other players' plays correctly. The rules are simple; the game is hard.

This file records the design and every decision made so far. Update it when a decision changes.

## Commands

```bash
uv venv && uv pip install -e '.[all,dev]'       # extras: anthropic, openai, gemini, litellm, dev
.venv/bin/python -m pytest -q                    # tests (no API calls; LLMs are faked)
gnomic play -p anthropic:claude-opus-5-5 -p openai:gpt-5 -p gemini:gemini-3-pro -p stats --games 5 --ui --open
gnomic play -p oracle -p stats -p random --games 10 --seed 1     # baselines only, no API cost
gnomic play -p mtplx -p oracle -p stats --games 3 --ui --open    # local MTPLX model, no API cost
gnomic replay logs/match-YYYYmmdd-HHMMSS.jsonl --open           # watch a finished match
gnomic play -p anthropic:claude-opus-5-5 -p stats --propose-rules --coder anthropic:claude-opus-5-5
gnomic rules                                                    # list the rule library
```

Player specs: `random | stats | oracle | human | <provider>:<model>[?opt=val&...]`, where
provider is `anthropic`, `openai`, `gemini`, `litellm` or `mtplx`. `mtplx` needs no model:
it uses whatever the local server has loaded. Every spec accepts `name=`.
Provider options: `effort` (Anthropic/OpenAI/LiteLLM), `thinking=off` (Anthropic, for
models without adaptive thinking), `thinking_level` / `thinking_budget` (Gemini),
`max_tokens`, `base_url` + `api_key_env` (OpenAI-compatible servers), `api_base` (LiteLLM),
`fallbacks=default` (Anthropic), `context_tokens` (any provider). Player options: `history_messages`,
`context_fill`, `max_retries_turn`.

## Game rules as implemented

Core rules given by the user:
1. A 52-card deck, no jokers. A=1, J=11, Q=12, K=13. Suits are unordered.
2. The dealer deals 7 cards to every player.
3. Each turn you play a card.
4. The first player to discard their last card wins.
5. The dealer makes up a rule before seeing their cards and tells nobody.
6. When someone breaks a rule, a penalty is called.
7. A justified penalty: the offender draws a card.
8. An unjustified penalty call: the caller draws a card.
9. Later rules take precedence over earlier ones.

Clarifications and decisions (all made by the user unless marked *default*):
- **Start of a game:** deal 7 each, then flip one card from DECK to start DISCARD (*default*).
- **On your turn** you must play exactly one card **if you have any cards in your hand** (user
  decision). There is no passing (*default*). From game 3 this is R0, a public engine rule that a
  permission may override (see Table mechanics). A player whose hand is empty at the start of
  their turn (an effect took their last card) plays nothing (`no_cards` event) and wins at the
  end of that turn.
- **Players police their own rules (user decision):** a penalty call names the caller's rule:
  `CALL: PENALTY R1` = "that's a penalty under my rule R1". It is upheld only if R1 is the
  caller's own rule and the play broke it, meaning R1 forbids it and no later rule allows it
  (`RuleStack.violated`). Naming the wrong one of your rules, or calling a play that broke
  only someone else's rule, is a false call. A player who owns exactly one rule may leave out
  the number. Naming a rule you don't own, or leaving it out while owning several, gets an
  `INVALID:` retry (self-calls too). The public log shows each call's rule
  (`P3 under R1 (upheld)`). So only a player who owns ≥1 rule may call. Players with no rules are never sent `JUDGE`
  requests, and their self-call is ignored (their `YOUR TURN` request says no `CALL` line is
  needed). In game 1 only the dealer polices plays; each game's winner becomes a judge when
  given a rule. STATE lists the current `JUDGES`.
- **Calls:** after every play, every player who may call judges it with `PENALTY` or `NONE`,
  including the player who made it. Others judge in parallel and simultaneously, seeing the
  table as it was *before* the play. The player's own call comes in the same reply as their play.
- **The engine is the source of truth** on legality, but a penalty happens **only if a broken
  rule's owner calls it**. Each call is judged separately:
  - upheld (the play broke the rule the caller named): the offender draws 1 (once, however many are upheld);
  - rejected (otherwise, even if the play broke a different rule): **that** caller draws 1 (rule 8);
  - a mixed round does both. The public log shows each caller's verdict (`P3 (upheld), P1 (rejected)`).
  - a violation no owner calls → nothing happens, and it is **never revealed** to players.
- **Self-calls:** a player whose play breaks their own rule must call PENALTY on themselves,
  naming that rule. The prompt says so; nothing enforces it, since an uncalled violation is free.
  The engine measures it as `self_call_rate` (an honesty metric): own-rule breaks they
  self-called correctly / own-rule breaks.
- **Illegal cards stay on DISCARD** in every case.
- **Win:** your hand is empty **at the end of your own turn** (after your play has been judged and
  its effects happened). **Nobody wins out of turn** (user decision): if an effect takes your last
  card during someone else's turn, you win at the end of your next turn, unless someone wins
  first. If a last card is called and upheld, you draw and play on. From game 6, win-condition
  rules add other ways to win, also checked only at the end of your own turn (see below).
- **Reshuffle:** when a player must draw and DECK is empty, every DISCARD card except the top
  card is shuffled into a new DECK. **The top card stays on DISCARD.** (The user corrected
  an earlier version that shuffled the whole pile.)
- **Round limit (user decision):** a game ends after 20 rounds (every player has played 20 cards;
  `--max-rounds`). Hitting it means no winner and no new rule. Players are told the limit and
  STATE shows `round r/20`.
- **Dealer:** a seated player chosen at random (or `--dealer`). The dealer plays like anyone else.
- **Rule growth:** before game 1 the dealer is secretly given R1. After each game the
  **winner** is secretly given one new rule (classic Mao). Rules carry over between games.
  Everyone is told when a rule is added and who owns it; only the owner is told its text,
  which is shown in their STATE as `YOUR RULES (secret)`.
- **The first rule is always a restriction** (it makes some plays penalties). A permission as
  R1 would do nothing, and the dealer would win game 1 trivially.
- Turn order is seat order. Game *g* starts with seat `(dealer + 1 + g) % n` (*default*).

## Hidden rules (`gnomic/rules.py`)

- A rule returns `ALLOW`, `FORBID` or `ABSTAIN` for a `PlayContext`. The context holds the
  card, the discard pile before the play, the player's seat, all hand sizes, the turn number
  and the number of players. All of it is public, which matches what the prompt tells players
  a rule may depend on.
- **Precedence:** evaluate newest-first; the first non-abstaining verdict decides; if all abstain
  the play is legal. **Restrictions** only FORBID or ABSTAIN, and **permissions** only ALLOW or
  ABSTAIN. So restrictions accumulate, a newer permission overrides older restrictions, and a
  newer restriction overrides older permissions. This is what "later rules take precedence"
  means here, and it's why calling correctly needs the *whole current* stack, not just your own rule.
- Rules are **pre-written, coded templates** with parameters (42 templates: 25 restrictions,
  17 permissions). They serialize as
  `{template, params}` and rebuild exactly.
- **Card properties a rule may use** (user decision): value 1–13, colour (red/black), suit,
  face/not-face, **gender** (J and K male, Q female; number cards have none), odd/even. Values
  can be required to ascend (`not_lower`), descend (`not_higher`), or stay within a range
  (`max_distance`, e.g. "within ±3").
- **Value arithmetic is limited to `> < + - * /`** (user decision). That allows sums, products,
  differences, double/half and odd/even. No square roots, exponents, primes, or modular
  wrap-around. (Earlier prime and K→A wrap rules were removed for this reason.)
- Rules may also use public game state: the last two DISCARD cards, the turn number, hand sizes.
- The briefing defines the vocabulary (values, colours, face cards, gender) but does not tell
  players which operators rules use.
- **Permissions in the mix (user decision):** after R1, each new rule is a permission
  ("enabling" rule, e.g. "you may always play a face card on a face card") with probability 0.4.
  If the current stack leaves <40% of cards legal on average, only permissions are drawn, because
  restrictions alone can collapse the game until nothing is playable.
- **Generation checks** (`generate_rule`): a candidate must keep the game playable (≥20% of
  cards legal on average, and ≥3 legal cards in ≥85% of sampled situations) and must change
  ≥1% of verdicts compared with the current stack (no redundant rules). The playability bar is
  deliberately lenient: "no lower on higher" leaves almost nothing legal on a King, but an illegal
  card stays on DISCARD and resets the top, so the game can't stall. A test ensures the user's
  example rules stay reachable. If no candidate passes, no rule is added and a `rule_skipped`
  event is emitted.
- Spectators always see every rule in effect, its owner, and which rule decided each play.

### Player-made rules (`--propose-rules`; `judges.py`, `coder.py`, `sandbox.py`)
- With `--propose-rules` the dealer (R1) and each winner make up their own rule (user decision).
  The engine sends `MAKE A RULE` in the owner's own conversation (STATE then shows only scores
  and their rules). The reply ends `NOTES: …` / `RULE: <text>`. The request tells them what a rule
  may use (public info, `+ - * /` and comparisons only), that it's a restriction or a permission
  but not both, that R1 must be a restriction, and that it must leave plenty legal and make a
  difference. The system prompt's rule 4 and a `MAKING UP A RULE` section change to match.
- The engine hands the text to a **coding agent** (`--coder SPEC`, default
  `anthropic:claude-opus-5-5`; any provider spec). The coder gets a fresh conversation per rule and
  sees only the rule text, never other rules or players. It replies `KIND` / `DESCRIPTION` /
  a ```python `def applies(ctx)` block (True = forbidden for a restriction, allowed for a
  permission), or `CANNOT: <reason>`. Sandbox or smoke-test errors go back to the coder, which
  gets up to 2 fixes.
- **Sandbox** (`sandbox.py`): an in-process restricted interpreter, not an OS sandbox. Exactly one
  `def applies(ctx)` is allowed. It uses an AST node whitelist (no imports, while, lambda, try,
  classes, `**` or recursion), a whitelist of ctx/card attributes plus `.count`/`.index`, and no
  `_` names. Builtins are minimal, `range` is capped at 64, and arithmetic is rewritten into checked
  calls that accept numbers only. `smoke_test` requires a bool on every sample, no exceptions, and
  ≤2 ms per call. During play an exception counts as ABSTAIN (`CodeRule.errors`).
- The coded rule is vetted by `rules.rule_problem` with the library thresholds (≥20% legal, ≥3
  legal cards in ≥85% of situations, ≥1% of verdicts changed; R1 a restriction). The
  `loosen_below` permissions-only bias isn't applied to proposals.
- **If the engine can't use the rule** (coder says CANNOT, code never works, or vetting fails),
  the owner gets `REJECTED: <reason>` and is asked for another rule, up to `--rule-attempts`
  (default 3) times. After that a library rule is used (`rule_fallback` event) (user decision:
  ask for another rule; the final library fallback is *default*). The reason can hint at the
  other rules (e.g. "too few cards legal"); this is accepted.
- The rule's text, in its owner's `YOUR RULES` and for spectators, is the coder's
  `DESCRIPTION` (what is actually enforced). The owner's wording is kept as `proposal`. Rules
  serialize as `{template: "code", code, proposal, kind, description}`, and `rules.rule_from_json`
  rebuilds either kind.
- Players that can't make up rules (`random`, `stats`, `oracle`) always get library rules;
  `human` types one. Events: `rule_proposed`, `rule_rejected` (stage proposal|coder|check),
  `rule_coded`, `rule_fallback`.

### Table mechanics from game 3 (`mechanics.py`; `--mechanics-from N`, default 3, 0 = off)
- **User decision:** from game 3, rules may add new things to the game: a **space** in front of
  each player, **tokens** collected and spent, cards played at **angles**, **rotating** cards,
  **trading** cards, **reversing** the order of play. From the proposed list the user also accepted:
  **skip** the next player, **extra turn**, **multi-card plays**, **effect draws** (not penalties),
  **declarations**, **stealing from spaces**, **peeking** at hidden cards. Rejected: passing and
  drawing instead of playing. Any other new effect must be proposed to the user first.
- **Rules can never change what a penalty does** (user decision): offender draws 1 (once), false
  caller draws 1. The effects API has no access to penalties.
- **Gating:** a rule may use mechanics only if it first applies in game ≥ `mechanics_from`
  (`judges.mechanics_allowed`: rules added after game 2 or later). Options and multi-card plays
  are accepted only in games ≥ `mechanics_from` (earlier they get an `INVALID:` retry). The
  system prompt describes the mechanics from the start, under `FROM GAME 3: TABLE MECHANICS`.
- **Player choices = options + effects (user decision).** A move is one or more cards plus
  optional lines after `PLAY:` (`PLAY: 5D 5H`, `ANGLE: 0|90|180|270`, `SPEND: n`, `TARGET: Pk`,
  `DECLARE: text` up to 40 chars, `GIVE: card`). Options do nothing unless a rule uses them.
  Spent tokens are always deducted, even when the play is penalized. `GIVE` is private: it's the
  card handed over if an effect makes the play a trade.
- **R0** is a public engine rule at the bottom of the stack: "play exactly one card". A newer
  permission can allow multi-card moves (e.g. `pair_ok`). The engine calls R0 itself
  (`BASE_RULE = -1` in `decide`/`violated`; `PublicEvent.engine_call`). It costs the usual draw 1,
  once, together with any upheld player calls.
- **Effects:** a third rule kind, `effect`, never judges legality. It defines `effect(ctx, fx)`,
  which records calls on `mechanics.Effects`: reverse, skip, extra_turn, draw, to_space,
  rotate_top/space, trade, peek_deck/hand, plus the general moves below. Counts are 1..3,
  with ≤6 calls per rule per play. The engine applies them after resolution, oldest rule first,
  **only if no penalty was upheld** (user decision), and logs each publicly. Peeks and trades send
  `PRIVATE` notices to the players involved only (match-long, shown under `PRIVATE` in
  their messages, re-sent on resync).
- **Moving cards and tokens between every place** (user decision):
  `fx.move_card(source, dest, n, which)` and `fx.move_tokens(source, dest, n)`. Places are
  `fx.hand(seat)`, `fx.space(seat)` and `fx.discard()`, plus `fx.bank()` (the unlimited supply)
  for tokens. Tokens can lie in a hand (held), in a space, or on DISCARD (the **pot**). `which`
  card: from a hand `random` (default), `highest`, `lowest` or `chosen` (the mover's GIVE card);
  from a space or DISCARD `newest`/top (default), `oldest`, `random`, `highest` or `lowest`.
  Token `n` may be `"all"` (but not from the bank). gain/lose_tokens and steal_to_space/hand are
  shorthands. DISCARD always keeps one card. **Any player's last card may be taken** (user
  decision; see Win). A card moved hand to hand is shown publicly as "a card moved"; the two
  players get PRIVATE notices naming it. Any other move names the card publicly. Cards keep
  their angle when moved between table places.
- **Turn order:** next = seat + direction × (1 + skips), or the same seat after an extra turn. The
  round limit is still `max_rounds × players` turns (*default*).
- **Table state** (public, reset each game): direction, tokens held, tokens in spaces, the pot, spaces (cards with angles),
  DISCARD angles (`9C@90`), latest declaration (lowercased). STATE adds `ORDER`, `TOKENS`,
  `SPACES` and `DECLARED` lines (TOKENS also shows space tokens and the pot). `PlayContext` gains `move` and `table`, exposed as `ctx.cards`,
  `angle`, `spend`, `target`, `declare`, `direction`, `next_player`, `tokens`, `spaces`,
  `space_tokens`, `pot`, `pile_angles`, `declared`, `declared_by` and `game`. Without mechanics they hold neutral defaults.
- **Library:** mechanics templates (`mechanics=True`) are drawn only for game-3+ rules. They are
  pair_ok, spend_wild, face_sideways and declared_suit_after (Crazy-8s style), plus effects
  reverse/skip/extra_turn/draw on a rank, token_on_face, space_on_rank, trade_on_target,
  tribute_on_rank (the next player gives you their highest card) and pot_ante (face cards ante a
  token into the pot; an Ace takes the pot).
  With mechanics on, a new rule is an effect with probability 0.35. Effect rules must fire on
  2–50% of sampled plays (`rule_problem`). `sample_contexts(mechanics=True)` randomizes the
  table and the move's options, and 25% of samples are multi-card moves.
  `evaluate_candidate` measures playability on the single-card version of each sampled move
  (a player can always play one card, and R0 makes most multi-card samples illegal). It counts
  verdict changes on the whole move, so a rule about multi-card plays (e.g. pairs) still counts.
  Swapping a candidate into a sampled same-value set keeps it a set.
- **Coder/sandbox:** for game-3+ rules the coder prompt adds a TABLE MECHANICS section (KIND may be
  `effect`, and the code is `def effect(ctx, fx)`). The sandbox only whitelists the mechanics
  ctx attributes and fx methods for those rules.
- **Baselines don't use options** (they play single cards). Oracle judges the whole move.
- `rule_problem` rejects a rule identical to one already in effect (same `key()`, e.g. the same
  code). Effect and win rules don't get the "changes verdicts" check, so they need this one.

### Win conditions from game 6 (`--wins-from N`, default 6, 0 = off)
- **User decision:** rules may define new win conditions, but only after 5 games ("to allow the
  base game to stabilize, and new effects to be added"). So only rules first applying in game ≥ 6
  qualify (`judges.wins_allowed`: rules added after game 5 or later). A fourth rule kind, `win`.
- **Checked only at the end of the current player's own turn** (user decision), after the play is
  judged and its effects have happened. It's checked whether or not the play was penalized
  (*default*), and only for that player: nobody wins out of turn. Win rules **add** ways to win;
  an empty hand always still wins (*default*, following the user's example "out of cards OR ...").
- A win rule sees a `WinContext`: DISCARD now, the player, hand sizes now, the table, the cards
  played this turn, whether a penalty was upheld, and **the player's own hand** (private). The
  engine, not the players, checks win conditions, so they may use hidden information. Code is
  `def wins(ctx)`; only win rules may read `ctx.hand`.
- The newest rule that's met wins it (`RuleStack.winning_rule`). The game ends with
  `end_reason = "win_rule R<n>"`, and the public log says `G6 END winner P2 (win condition R7)`
  (the number is revealed, not the text; *default*).
- **Vetting:** a win rule must be met at the end of 0.1–3% of sampled turns (`win_share`, over
  four hands per sample). A rule that's too easy, or never met, is rejected.
- **Library:** suit_match_win (the user's example: one card left, of the top DISCARD card's
  suit) and pair_win (two cards of the same value). With wins allowed, a new library rule is a
  win condition with probability 0.25.
- Prompts: system prompt section `FROM GAME 6: WIN CONDITIONS`; `MAKE A RULE` and the coder prompt
  describe the `win` kind only when it's allowed.

## Players and the LLM protocol (`gnomic/protocol.py`, `gnomic/players/`)

- **Information isolation:** a player's prompt is built only from `PlayerView` (public log,
  table, their own hand, the text of their own rules). Engine events carry private data for
  spectators and logs, and are never shown to players. Players appear as seats `P1..Pn`, never
  model names. A test checks that no player's notes reach another player.
- **One conversation per LLM per match.** The system prompt explains the engine protocol up
  front: card notation (`8H`, `10S`, `QD`), the places DECK / DISCARD / HAND, every rule above,
  the message format, and the reply format.
- **Each engine message** = `EVENTS:` (everything public since the engine last messaged this
  player) + `STATE` (compact table snapshot: DISCARD bottom→top, DECK count, hand sizes,
  your hand, your secret rules) + a request:
  - `YOUR TURN` → reply ends `NOTES: …` / `PLAY: <card>` / `CALL: PENALTY R<n>|NONE` (self-call;
    omitted by players who own no rules);
  - `JUDGE: P2 plays 5D on JH` → reply ends `NOTES: …` / `CALL: PENALTY R<n>|NONE`.
  So players are told about every play (they judge each one) and when it's their turn.
- **Reasoning log:** `NOTES` is each player's private, self-maintained hypothesis log. It's kept
  if omitted, and re-sent after a resync. The UI shows each exchange's reasoning, the provider's
  thinking summary where available, and the notes. Other players never see any of it.
- **Bad replies:** up to 2 retries with an `INVALID:` message. After that the engine plays a
  random card for them (`forced`), or counts a judge's answer as NONE. An unreadable self-call
  counts as NONE. API errors are retried with the same message, and the transcript is only
  appended on success.
- **Context limit:** after `history_messages` (default 120) exchanges, or once the last
  exchange's prompt plus reply tokens reach `context_fill` (default 0.6) of the model's context
  window (when known: MTPLX reports it, or set `context_tokens=`), the conversation restarts
  with a `RESYNC` message containing the full public log plus the latest notes. Between resets
  the transcript is append-only, which suits prompt caching and Anthropic preserved thinking.
- **Providers** (`providers.py`): each session keeps a provider-native transcript.
  - Anthropic: official SDK, streaming + `get_final_message()`, adaptive thinking with
    `display: "summarized"` (shown in the UI), optional `effort`, top-level automatic prompt
    caching, refusal → error. Server-side refusal **fallbacks are off by default**, because a
    fallback means a different model answered and the result would be credited to the wrong
    model. Opt in with `fallbacks=default`. Default model in examples: `claude-opus-5-5`.
  - OpenAI: Chat Completions, so `base_url` works with OpenAI-compatible servers; `reasoning_effort`.
  - Gemini: **Google Cloud / Vertex AI** via `google-genai` (`Client(vertexai=True, ...)`; user
    decision). The project comes from `$GMAU_GEMINI_PROJECT` (required), the location from
    `$GMAU_GEMINI_LOCATION`, then `$GOOGLE_CLOUD_LOCATION`, then `global`; `project=` / `location=`
    spec options override both. Auth is Application Default Credentials. `include_thoughts=True`,
    and native content is kept (thought signatures).
  - LiteLLM: `litellm.completion`, for anything else.
  - MTPLX: a local MTP speculative-decoding server on this Mac (the MTPLX.app / `~/.mtplx/bin/mtplx`).
    It's the OpenAI adapter pointed at `http://127.0.0.1:8888/v1` (override with `base_url=` or
    `$MTPLX_BASE_URL`). No auth by default (`$MTPLX_API_KEY` if enabled). It reads the model id and
    context length (32K for the Qwen3.8-27B build) from `/v1/models`, uses `max_tokens=8192` by
    default, and takes reasoning from `reasoning_content`. Inline `<think>` blocks are stripped from
    replies for any OpenAI-compatible server. If it isn't running: `mtplx quickstart --port 8888`.
    Verified live: an 8-turn game parsed every reply first time, at 9–47 s per reply. The server
    runs requests one at a time, so several MTPLX seats queue behind each other.
- **Baselines:** `random` (plays anything, never calls; lower bound), `stats` (feature-wise
  illegality rates learned from upheld/rejected calls, plus knowledge of its own rules, which it
  avoids and calls, ignoring newer rules it doesn't own; the bar an LLM should clear), `oracle`
  (knows every rule, plays legally when possible, calls exactly the breaks of its own rules,
  self-calls; upper bound),
  `human` (terminal, sees exactly the LLM messages).

## Metrics (`match.summarize`, also in `<log>.summary.json`)

Per player: wins; `illegal_rate`; `avoidable_illegal_rate` (illegal while holding a legal card:
the key play metric); caught/uncaught illegal plays; `self_call_rate`; judging precision,
recall and accuracy on other players' plays (a call is correct iff upheld; a miss is a play that
broke the judge's rule with no call); penalty and false-call draws; tokens; average
decision time; illegal rate per game (learning curve).

## Spectator web UI (`gnomic/server.py`, `gnomic/static/index.html`)

- Python stdlib `ThreadingHTTPServer` + Server-Sent Events; one static page, no build step.
  On connect it replays all past events, then streams live. `POST /api/control` handles pause,
  resume, stop and the turn delay.
- Shows: the current player (TURN badge, blue border, live "Thinking N s" timer), players
  judging (dashed amber, live "Judging N s" timers, then PENALTY! / no call), a judge / can't-call
  badge per player, the round counter, hands (toggle), DECK and DISCARD (each card on DISCARD is
  marked by what its play caused: red `!` = penalty upheld, amber `✗` = false call, the callers
  drew; dashed `?` = illegal but uncalled; hover for details), the last play's outcome (legal/illegal, which rule decided it, callers,
  upheld/rejected, or MISSED when nobody called a violation), the rules in effect with owners,
  the play log, a per-player reasoning-log tab, and the match summary.
- The UI is for humans only; it shows everything, hidden rules included.

## Code map

```
gnomic/cards.py       Card, parsing ("10H", "Q♠", "queen of spades"), deck
gnomic/rules.py       Verdict, PlayContext, Rule templates, RuleStack precedence, generate_rule
gnomic/judges.py      RuleAuthor protocol, EngineRuleAuthor (library rules), ProposedRuleAuthor (--propose-rules)
gnomic/coder.py       Coding agent: plain-language rule -> `def applies(ctx)`
gnomic/sandbox.py     Restricted interpreter for coded rules, CodeRule
gnomic/mechanics.py   Game-3+ table: Move (options), TableState, Placed, Effects (fx API), R0
gnomic/game.py        Game loop (play → calls → resolution), PublicLog, PlayerView, Controls
gnomic/match.py       Match (dealer, rule growth, scoring), summarize()
gnomic/protocol.py    System prompt, event/state rendering, reply parsing
gnomic/providers.py   Anthropic / OpenAI / Gemini / LiteLLM chat sessions
gnomic/players/       base (Choice, Call, Player), llm, baselines, human, factory (spec strings)
gnomic/events.py      EventBus → JSONL log + live subscribers
gnomic/server.py      Spectator HTTP/SSE server
gnomic/cli.py         `gnomic play | replay | rules`
tests/                engine, protocol (fake LLM sessions), server, propose (sandbox, coder, fakes), mechanics
```

## Cost and known limits

- Every play costs one LLM call per LLM player (the player, plus every judge). A 4-LLM game of
  ~80 turns is ~320 calls. Calls within a judging round run in parallel.
- Hosted providers (Anthropic, OpenAI, Gemini, LiteLLM) have not been exercised live (no API
  keys in the dev environment); the protocol is tested with fake sessions. MTPLX has been run live.
- Ideas not built yet: a cheaper effort level for JUDGE requests;
  running several matches concurrently; a human seat in the web UI; OpenAI Responses API for
  reasoning summaries.

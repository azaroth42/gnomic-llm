"""Build players from spec strings.

  random | stats | oracle | human | web
  <provider>:<model>[?opt=val&opt=val]   provider in anthropic, openai, gemini, litellm

Any spec can take name=<display name>. Examples:
  anthropic:claude-opus-5-5?effort=high
  openai:gpt-5?effort=medium&name=GPT
  gemini:gemini-3-pro?thinking_level=high
  litellm:openrouter/qwen/qwen3-235b
  mtplx                                   whatever model the local MTPLX server has loaded
  mtplx:mtplx-qwen38-27b-optimized-speed?name=Qwen
  openai:my-model?base_url=http://localhost:8000/v1&api_key_env=LOCAL_KEY
"""

from __future__ import annotations

from urllib.parse import parse_qsl

from ..providers import PROVIDERS, ChatSession
from ..rules import RuleStack
from .base import Player
from .baselines import OraclePlayer, RandomPlayer, StatsPlayer
from .human import HumanPlayer
from .llm import LLMPlayer
from .web import WebHumanPlayer

_PLAYER_OPTS = {"name", "max_retries_turn", "history_messages", "context_fill"}
_MODEL_OPTIONAL = {"mtplx"}


def parse_spec(spec: str) -> tuple[str, str | None, dict]:
    base, _, query = spec.partition("?")
    opts = dict(parse_qsl(query))
    provider, _, model = base.partition(":")
    return provider.strip().lower(), (model.strip() or None), opts


def make_session(spec: str) -> ChatSession:
    """A bare chat session from a `<provider>:<model>[?opts]` spec (used for the coding agent)."""
    provider, model, opts = parse_spec(spec)
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider in {spec!r}; use one of {', '.join(PROVIDERS)}")
    if not model and provider not in _MODEL_OPTIONAL:
        raise ValueError(f"spec {spec!r} needs a model, e.g. {provider}:<model>")
    try:
        return PROVIDERS[provider](model, **{k: v for k, v in opts.items() if k not in _PLAYER_OPTS})
    except ImportError as e:
        raise SystemExit(f"{provider} support is not installed: pip install 'gnomic[{provider}]' ({e})")


def make_player(spec: str, stack: RuleStack, seed: int | None = None, default_name: str | None = None) -> Player:
    provider, model, opts = parse_spec(spec)
    name = opts.get("name") or default_name or (model or provider)
    if provider == "random":
        return RandomPlayer(name, seed)
    if provider == "stats":
        return StatsPlayer(name, stack, seed)
    if provider == "oracle":
        return OraclePlayer(name, stack, seed)
    if provider == "human":
        return HumanPlayer(name)
    if provider == "web":
        return WebHumanPlayer(name, max_retries=int(opts.get("max_retries_turn", 2)))
    if provider in PROVIDERS:
        if not model and provider not in _MODEL_OPTIONAL:
            raise ValueError(f"spec {spec!r} needs a model, e.g. {provider}:<model>")
        session_opts = {k: v for k, v in opts.items() if k not in _PLAYER_OPTS}
        try:
            session = PROVIDERS[provider](model, **session_opts)
        except ImportError as e:
            raise SystemExit(f"{provider} support is not installed: pip install 'gnomic[{provider}]' ({e})")
        if "name" not in opts and not default_name:
            name = session.model
        return LLMPlayer(name, session,
                         max_retries=int(opts.get("max_retries_turn", 2)),
                         history_messages=int(opts.get("history_messages", 120)),
                         context_fill=float(opts.get("context_fill", 0.6)))
    raise ValueError(f"unknown player spec {spec!r}")

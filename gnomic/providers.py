"""Chat sessions for each LLM provider.

Each session keeps its own transcript in the provider's native format, so
provider-specific content goes back unchanged (Anthropic thinking blocks, Gemini
thought signatures), and the transcript is only ever appended to. `send()` adds
the user turn and the reply only if the call succeeds, so a failed call leaves
the transcript untouched.

Credentials come from each SDK's usual environment variables:
  anthropic  ANTHROPIC_API_KEY (or an `ant auth login` profile)
  openai     OPENAI_API_KEY (pass base_url / api_key_env for OpenAI-compatible servers)
  gemini     Google Cloud (Vertex AI): GMAU_GEMINI_PROJECT, optional GMAU_GEMINI_LOCATION
             (default "global"), and Application Default Credentials
  litellm    whatever the routed provider needs
  mtplx      none by default (local server); MTPLX_BASE_URL / MTPLX_API_KEY to override
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass


class LLMRefusal(Exception):
    pass


@dataclass
class Completion:
    text: str
    thinking: str = ""          # reasoning summary, when the provider exposes one
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    stop_reason: str | None = None
    served_by: str | None = None  # the model that actually answered, if reported


class ChatSession:
    provider = "base"
    context_window: int | None = None  # model context size in tokens, if known (players use it to resync)

    def __init__(self, model: str, **opts):
        self.model = model
        self.opts = opts
        self.system = ""
        if opts.get("context_tokens"):
            self.context_window = int(opts["context_tokens"])

    def reset(self, system: str) -> None:
        self.system = system
        self._reset()

    def _reset(self) -> None:
        raise NotImplementedError

    def send(self, text: str) -> Completion:
        raise NotImplementedError


def _int(v) -> int:
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0


class AnthropicSession(ChatSession):
    """Options: effort=low|medium|high|xhigh|max, thinking=adaptive|off,
    max_tokens=N, fallbacks=default (off by default; see README)."""

    provider = "anthropic"

    def __init__(self, model: str, **opts):
        super().__init__(model, **opts)
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic(max_retries=int(opts.get("max_retries", 5)))
        self.history: list[dict] = []

    def _reset(self) -> None:
        self.history = []

    def send(self, text: str) -> Completion:
        messages = self.history + [{"role": "user", "content": text}]
        kwargs: dict = dict(
            model=self.model,
            max_tokens=int(self.opts.get("max_tokens", 32000)),
            system=self.system,
            messages=messages,
            cache_control={"type": "ephemeral"},  # the transcript is append-only, so the prefix caches
        )
        if self.opts.get("thinking", "adaptive") == "adaptive":
            kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
        if self.opts.get("effort"):
            kwargs["output_config"] = {"effort": self.opts["effort"]}
        fallbacks = self.opts.get("fallbacks")
        if fallbacks:
            stream = self.client.beta.messages.stream(
                betas=["server-side-fallback-2026-07-01"], fallbacks=fallbacks, **kwargs)
        else:
            stream = self.client.messages.stream(**kwargs)
        with stream as s:
            msg = s.get_final_message()
        if msg.stop_reason == "refusal":
            details = getattr(msg, "stop_details", None)
            raise LLMRefusal(f"refusal ({getattr(details, 'category', None)})")
        out_text = "".join(b.text for b in msg.content if b.type == "text")
        thinking = "\n".join(b.thinking for b in msg.content if b.type == "thinking" and b.thinking)
        self.history = messages + [{"role": "assistant", "content": msg.content}]
        u = msg.usage
        return Completion(out_text, thinking, _int(u.input_tokens), _int(u.output_tokens),
                          _int(getattr(u, "cache_read_input_tokens", 0)), msg.stop_reason, msg.model)


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)


def _split_think(text: str) -> tuple[str, str]:
    """Some local servers put reasoning inline as <think>...</think>; move it out of the reply."""
    thoughts = "\n".join(m.strip() for m in _THINK_RE.findall(text))
    return _THINK_RE.sub("", text).strip(), thoughts


class OpenAISession(ChatSession):
    """Chat Completions, so it also works with OpenAI-compatible servers.
    Options: effort (reasoning_effort), base_url, api_key_env, max_tokens, context_tokens."""

    provider = "openai"
    max_tokens_field = "max_completion_tokens"

    def __init__(self, model: str, **opts):
        super().__init__(model, **opts)
        import openai

        key_env = opts.get("api_key_env")
        self.client = openai.OpenAI(
            base_url=opts.get("base_url"),
            api_key=self._api_key(key_env),
            max_retries=int(opts.get("max_retries", 5)),
            timeout=float(opts.get("timeout", 600)),
        )
        self.history: list[dict] = []

    def _api_key(self, key_env: str | None) -> str | None:
        return os.environ.get(key_env) if key_env else None

    def _reset(self) -> None:
        self.history = []

    def send(self, text: str) -> Completion:
        messages = self.history + [{"role": "user", "content": text}]
        kwargs: dict = dict(model=self.model,
                            messages=[{"role": "system", "content": self.system}] + messages)
        if self.opts.get("effort"):
            kwargs["reasoning_effort"] = self.opts["effort"]
        if self.opts.get("max_tokens"):
            kwargs[self.max_tokens_field] = int(self.opts["max_tokens"])
        resp = self.client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        out_text, inline_thoughts = _split_think(choice.message.content or "")
        if choice.finish_reason == "content_filter":
            raise LLMRefusal("content_filter")
        self.history = messages + [{"role": "assistant", "content": out_text}]
        u = resp.usage
        cached = _int(getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0)) if u else 0
        reasoning = getattr(choice.message, "reasoning_content", None) or getattr(choice.message, "reasoning", None) or ""
        return Completion(out_text, "\n".join(x for x in (reasoning, inline_thoughts) if x),
                          _int(getattr(u, "prompt_tokens", 0)), _int(getattr(u, "completion_tokens", 0)),
                          cached, choice.finish_reason, resp.model)


class GeminiSession(ChatSession):
    """Gemini on Google Cloud (Vertex AI) through the google-genai SDK.

    Project: `project=` option, else $GMAU_GEMINI_PROJECT (required).
    Location: `location=` option, else $GMAU_GEMINI_LOCATION, else $GOOGLE_CLOUD_LOCATION, else "global".
    Auth: Application Default Credentials (`gcloud auth application-default login`).
    Other options: thinking_level=low|high, thinking_budget=N, max_tokens.
    """

    provider = "gemini"

    def __init__(self, model: str, **opts):
        super().__init__(model, **opts)
        from google import genai
        from google.genai import types

        project = opts.get("project") or os.environ.get("GMAU_GEMINI_PROJECT")
        if not project:
            raise RuntimeError("Gemini runs on Google Cloud: set GMAU_GEMINI_PROJECT to your GCP project id")
        location = (opts.get("location") or os.environ.get("GMAU_GEMINI_LOCATION")
                    or os.environ.get("GOOGLE_CLOUD_LOCATION") or "global")
        self.types = types
        self.client = genai.Client(vertexai=True, project=project, location=location)
        self.history: list = []

    def _reset(self) -> None:
        self.history = []

    def send(self, text: str) -> Completion:
        t = self.types
        contents = self.history + [t.Content(role="user", parts=[t.Part(text=text)])]
        thinking = t.ThinkingConfig(include_thoughts=True)
        if self.opts.get("thinking_level"):
            thinking.thinking_level = self.opts["thinking_level"]
        if self.opts.get("thinking_budget"):
            thinking.thinking_budget = int(self.opts["thinking_budget"])
        config = t.GenerateContentConfig(system_instruction=self.system, thinking_config=thinking)
        if self.opts.get("max_tokens"):
            config.max_output_tokens = int(self.opts["max_tokens"])
        resp = self.client.models.generate_content(model=self.model, contents=contents, config=config)
        cand = resp.candidates[0] if resp.candidates else None
        if cand is None or cand.content is None:
            reason = getattr(cand, "finish_reason", None) or getattr(resp, "prompt_feedback", None)
            raise LLMRefusal(f"no content ({reason})")
        parts = cand.content.parts or []
        out_text = "".join(p.text for p in parts if p.text and not p.thought)
        thoughts = "\n".join(p.text for p in parts if p.text and p.thought)
        self.history = contents + [cand.content]
        u = resp.usage_metadata
        return Completion(out_text, thoughts, _int(getattr(u, "prompt_token_count", 0)),
                          _int(getattr(u, "candidates_token_count", 0)) + _int(getattr(u, "thoughts_token_count", 0)),
                          _int(getattr(u, "cached_content_token_count", 0)),
                          str(getattr(cand, "finish_reason", "")), getattr(resp, "model_version", None))


class LiteLLMSession(ChatSession):
    """Any model LiteLLM can route to, e.g. litellm:openrouter/meta-llama/llama-4-maverick.
    Options: effort (reasoning_effort), max_tokens, api_base."""

    provider = "litellm"

    def __init__(self, model: str, **opts):
        super().__init__(model, **opts)
        import litellm

        self.litellm = litellm
        self.history: list[dict] = []

    def _reset(self) -> None:
        self.history = []

    def send(self, text: str) -> Completion:
        messages = self.history + [{"role": "user", "content": text}]
        kwargs: dict = dict(model=self.model,
                            messages=[{"role": "system", "content": self.system}] + messages,
                            num_retries=int(self.opts.get("max_retries", 3)))
        if self.opts.get("effort"):
            kwargs["reasoning_effort"] = self.opts["effort"]
        if self.opts.get("max_tokens"):
            kwargs["max_tokens"] = int(self.opts["max_tokens"])
        if self.opts.get("api_base"):
            kwargs["api_base"] = self.opts["api_base"]
        resp = self.litellm.completion(**kwargs)
        choice = resp.choices[0]
        out_text = choice.message.content or ""
        self.history = messages + [{"role": "assistant", "content": out_text}]
        u = getattr(resp, "usage", None)
        return Completion(out_text, getattr(choice.message, "reasoning_content", "") or "",
                          _int(getattr(u, "prompt_tokens", 0)), _int(getattr(u, "completion_tokens", 0)),
                          0, choice.finish_reason, getattr(resp, "model", None))


class MTPLXSession(OpenAISession):
    """A local MTPLX server (native-MTP speculative decoding on Apple Silicon), via its
    OpenAI-compatible API.

    Spec: `mtplx` (whatever model the server has loaded) or `mtplx:<model-id>`.
    Options: base_url (default $MTPLX_BASE_URL or http://127.0.0.1:8888/v1), max_tokens
    (default 8192), context_tokens (default: the server's reported context length).
    The context window is read from /v1/models so players can resync before overflowing it.
    """

    provider = "mtplx"
    max_tokens_field = "max_tokens"
    DEFAULT_BASE_URL = "http://127.0.0.1:8888/v1"

    def __init__(self, model: str | None, **opts):
        opts.setdefault("base_url", os.environ.get("MTPLX_BASE_URL", self.DEFAULT_BASE_URL))
        opts.setdefault("max_tokens", 8192)
        opts.setdefault("max_retries", 1)
        info = self._models(opts["base_url"])
        if not model or model == "auto":
            if not info:
                raise RuntimeError(f"MTPLX server at {opts['base_url']} reports no models; is it running?")
            model = info[0]["id"]
        super().__init__(model, **opts)
        if not opts.get("context_tokens"):
            match = next((m for m in info if m.get("id") == model), info[0] if info else {})
            ctx = match.get("context_length") or match.get("max_model_len")
            self.context_window = int(ctx) if ctx else None

    def _api_key(self, key_env: str | None) -> str:
        # The local server runs without auth by default, but the SDK insists on a key.
        return os.environ.get(key_env or "MTPLX_API_KEY") or "mtplx-local"

    @staticmethod
    def _models(base_url: str) -> list[dict]:
        req = urllib.request.Request(base_url.rstrip("/") + "/models")
        key = os.environ.get("MTPLX_API_KEY")
        if key:
            req.add_header("Authorization", f"Bearer {key}")
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return json.load(r).get("data", [])
        except OSError as e:
            raise RuntimeError(f"cannot reach MTPLX at {base_url} ({e}); start it with `mtplx quickstart`") from e


PROVIDERS: dict[str, type[ChatSession]] = {
    "mtplx": MTPLXSession,
    "anthropic": AnthropicSession,
    "openai": OpenAISession,
    "gemini": GeminiSession,
    "litellm": LiteLLMSession,
}

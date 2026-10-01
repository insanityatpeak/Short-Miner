"""Single choke point for all LLM calls in the pipeline.

pipeline/scorer.py and pipeline/metadata.py call call_llm() and never touch a
provider SDK directly. That keeps the provider swappable in one place.

Currently backed by Gemini (free tier), a temporary substitute while Anthropic
credits aren't available, with free-tier fallbacks behind it: Groq, then
OpenRouter, each used only if its key is set. To swap back to Claude, add an
Anthropic Messages API provider to _providers() using utils.config.CLAUDE_MODEL
and utils.config.require_anthropic_key(); no changes needed in scorer.py or
metadata.py.
"""
import json
import logging
from typing import Callable

from utils.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GROQ_API_KEY,
    GROQ_LLM_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
    require_gemini_key,
)

logger = logging.getLogger(__name__)

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


class LLMError(Exception):
    """Raised when the underlying LLM call fails."""


class LLMQuotaExceededError(LLMError):
    """Raised when the LLM provider's rate limit/daily quota is exhausted.

    Distinct from LLMError so app.py can show visitors of the public demo a
    "come back tomorrow" message instead of a generic failure — this is
    expected/recoverable, not a bug.
    """


class LLMAccessDeniedError(LLMError):
    """Raised when the provider rejects the API key/project outright (401/403).

    Distinct so app.py doesn't mistake Gemini's "403 PERMISSION_DENIED" for a
    YouTube 403 — the fix is a new/valid key, not retrying.
    """


def require_llm_key() -> str:
    """Fail-fast check that at least one LLM provider is configured.

    app.py calls this once at startup. Gemini is the primary provider, but a
    deployment with only GROQ_API_KEY or OPENROUTER_API_KEY set still works,
    since call_llm() falls through to whichever providers have keys.
    """
    if not GEMINI_API_KEY and (GROQ_API_KEY or OPENROUTER_API_KEY):
        return GROQ_API_KEY or OPENROUTER_API_KEY
    return require_gemini_key()


def _call_gemini(prompt: str) -> str:
    from google import genai
    from google.genai import errors as genai_errors

    client = genai.Client(api_key=require_gemini_key())

    try:
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        text = response.text
    except genai_errors.APIError as exc:
        if exc.code == 429:
            raise LLMQuotaExceededError(
                "This demo's free-tier AI quota is used up for today. "
                "Please check back tomorrow — this isn't a bug, just a rate "
                "limit on the shared free key."
            ) from exc
        if exc.code in (401, 403):
            raise LLMAccessDeniedError(f"Gemini API call failed: {exc}") from exc
        raise LLMError(f"Gemini API call failed: {exc}") from exc
    except Exception as exc:
        raise LLMError(f"Gemini API call failed: {exc}") from exc

    if not text:
        raise LLMError("Gemini returned an empty response.")

    return text


def _call_openai_compatible(name: str, url: str, api_key: str, model: str, prompt: str) -> str:
    """Chat-completions call for Groq/OpenRouter, both of which speak the
    OpenAI wire format, so plain httpx is enough and no extra SDK is needed."""
    import httpx

    try:
        resp = httpx.post(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}]},
            timeout=120,
        )
    except httpx.HTTPError as exc:
        raise LLMError(f"{name} API call failed: {exc}") from exc

    if resp.status_code == 429:
        raise LLMQuotaExceededError(f"{name} free-tier quota exhausted: {resp.text[:300]}")
    if resp.status_code in (401, 403):
        raise LLMAccessDeniedError(f"{name} rejected the API key: {resp.text[:300]}")
    if resp.status_code >= 400:
        raise LLMError(f"{name} API call failed ({resp.status_code}): {resp.text[:300]}")

    try:
        text = resp.json()["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"{name} returned an unexpected response: {resp.text[:300]}") from exc
    if not text:
        raise LLMError(f"{name} returned an empty response.")
    return text


def _providers() -> list[tuple[str, Callable[[str], str]]]:
    """Configured providers in priority order: Gemini, then Groq, then OpenRouter.

    Gemini stays in the chain when nothing else is configured, so a missing
    key still surfaces as Gemini's own ConfigError rather than "no providers".
    """
    providers: list[tuple[str, Callable[[str], str]]] = []
    if GEMINI_API_KEY or not (GROQ_API_KEY or OPENROUTER_API_KEY):
        providers.append(("Gemini", _call_gemini))
    if GROQ_API_KEY:
        providers.append(("Groq", lambda p: _call_openai_compatible(
            "Groq", GROQ_CHAT_URL, GROQ_API_KEY, GROQ_LLM_MODEL, p)))
    if OPENROUTER_API_KEY:
        providers.append(("OpenRouter", lambda p: _call_openai_compatible(
            "OpenRouter", OPENROUTER_CHAT_URL, OPENROUTER_API_KEY, OPENROUTER_MODEL, p)))
    return providers


def call_llm(prompt: str) -> str:
    """Send a single-turn prompt to the first LLM provider that answers.

    Each free tier fails in its own way (daily quota, a model pulled from the
    free tier, a project denied access), so any LLMError moves on to the next
    configured provider. If every provider fails, the first provider's error
    is raised, since that's the one app.py's friendly messages describe.
    """
    first_error: LLMError | None = None
    for name, call in _providers():
        try:
            return call(prompt)
        except LLMError as exc:
            logger.warning("%s failed (%s); trying the next LLM provider.", name, exc)
            first_error = first_error or exc
    assert first_error is not None
    raise first_error


def parse_json_list(raw: str, error_cls: type[Exception]) -> list:
    """Parse an LLM's raw text response into a JSON list.

    Strips a leading/trailing markdown code fence (with or without a `json`
    language tag) if present, since LLMs routinely wrap JSON in one even when
    told not to. Raises error_cls on invalid JSON or a non-list top level, so
    callers (scorer.py, metadata.py) can pass their own exception type and
    get consistent messages without duplicating this parsing.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise error_cls(
            f"LLM did not return valid JSON: {exc}\nRaw response: {raw[:500]}"
        ) from exc

    if not isinstance(data, list):
        raise error_cls(
            f"Expected a JSON list from the LLM, got {type(data).__name__}"
        )
    return data

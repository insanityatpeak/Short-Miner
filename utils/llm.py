"""Single choke point for all LLM calls in the pipeline.

pipeline/scorer.py and pipeline/metadata.py call call_llm() and never touch a
provider SDK directly. That keeps the provider swappable in one place.

Currently backed by Gemini (free tier) per CLAUDE.md's temporary override.
To swap back to Claude (the CLAUDE.md section 4 spec — model claude-sonnet-4-6),
replace the body of call_llm() below with an Anthropic Messages API call using
utils.config.CLAUDE_MODEL and utils.config.require_anthropic_key(); no changes
needed in scorer.py or metadata.py.
"""
import logging

from utils.config import GEMINI_MODEL, require_gemini_key

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when the underlying LLM call fails."""


def require_llm_key() -> str:
    """Fail-fast check for whichever key call_llm() actually needs right now.

    app.py calls this once at startup (per CLAUDE.md section 8) instead of
    hardcoding require_anthropic_key() — that key isn't load-bearing while
    the Gemini override in call_llm() below is active. Swap this call's body
    to require_anthropic_key() when call_llm() is swapped back to Claude.
    """
    return require_gemini_key()


def call_llm(prompt: str) -> str:
    """Send a single-turn prompt to the configured LLM and return its raw text response."""
    from google import genai

    client = genai.Client(api_key=require_gemini_key())

    try:
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        text = response.text
    except Exception as exc:
        raise LLMError(f"Gemini API call failed: {exc}") from exc

    if not text:
        raise LLMError("Gemini returned an empty response.")

    return text

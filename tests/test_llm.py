from unittest.mock import MagicMock, patch

import pytest
from google.genai import errors as genai_errors

import utils.llm as llm
from utils.llm import LLMAccessDeniedError, LLMError, LLMQuotaExceededError, call_llm


@pytest.fixture(autouse=True)
def only_gemini_configured(monkeypatch):
    # Keep tests independent of whatever keys are in the developer's .env.
    monkeypatch.setattr(llm, "GEMINI_API_KEY", "k")
    monkeypatch.setattr(llm, "GROQ_API_KEY", None)
    monkeypatch.setattr(llm, "OPENROUTER_API_KEY", None)


def _call_with_api_error(code: int):
    err = genai_errors.APIError(code, {"error": {"code": code, "message": "x", "status": "S"}})
    client = MagicMock()
    client.models.generate_content.side_effect = err
    with patch("utils.llm.require_gemini_key", return_value="k"), patch("google.genai.Client", return_value=client):
        call_llm("hi")


@pytest.mark.parametrize("code", [401, 403])
def test_denied_access_is_distinct_error(code):
    with pytest.raises(LLMAccessDeniedError):
        _call_with_api_error(code)


def test_quota_and_generic_errors_unchanged():
    with pytest.raises(LLMQuotaExceededError):
        _call_with_api_error(429)
    with pytest.raises(LLMError) as ei:
        _call_with_api_error(500)
    assert not isinstance(ei.value, (LLMAccessDeniedError, LLMQuotaExceededError))


def test_falls_back_to_groq_when_gemini_quota_is_exhausted(monkeypatch):
    monkeypatch.setattr(llm, "GROQ_API_KEY", "g")
    monkeypatch.setattr(llm, "_call_gemini", MagicMock(side_effect=LLMQuotaExceededError("q")))
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"choices": [{"message": {"content": "from groq"}}]}
    with patch("httpx.post", return_value=resp) as post:
        assert call_llm("hi") == "from groq"
    assert post.call_args.args[0] == llm.GROQ_CHAT_URL


def test_all_providers_failing_raises_the_first_error(monkeypatch):
    monkeypatch.setattr(llm, "GROQ_API_KEY", "g")
    monkeypatch.setattr(llm, "OPENROUTER_API_KEY", "o")
    monkeypatch.setattr(llm, "_call_gemini", MagicMock(side_effect=LLMQuotaExceededError("gemini")))
    with patch("httpx.post", return_value=MagicMock(status_code=500, text="boom")) as post:
        with pytest.raises(LLMQuotaExceededError, match="gemini"):
            call_llm("hi")
    assert post.call_count == 2


def test_gemini_skipped_when_only_fallback_keys_set(monkeypatch):
    monkeypatch.setattr(llm, "GEMINI_API_KEY", None)
    monkeypatch.setattr(llm, "OPENROUTER_API_KEY", "o")
    assert [name for name, _ in llm._providers()] == ["OpenRouter"]
    assert llm.require_llm_key() == "o"

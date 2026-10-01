from unittest.mock import MagicMock, patch

import pytest
from google.genai import errors as genai_errors

from utils.llm import LLMAccessDeniedError, LLMError, LLMQuotaExceededError, call_llm


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

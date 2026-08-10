from unittest import mock

import pytest

from dohasecuritiesstockai.llm_clients.openai_client import (
    NormalizedChatOpenAI,
    OpenAIClient,
    OpenRouterChatOpenAI,
    _is_transient_openrouter_body_error,
)


@pytest.mark.unit
def test_openrouter_body_error_detection_is_narrow() -> None:
    assert _is_transient_openrouter_body_error(
        ValueError({"message": "Provider timed out", "code": 504})
    )
    assert _is_transient_openrouter_body_error(
        ValueError({"message": "Rate limited", "code": 429})
    )
    assert not _is_transient_openrouter_body_error(
        ValueError({"message": "Bad request", "code": 400})
    )
    assert not _is_transient_openrouter_body_error(ValueError("ordinary parsing error"))


@pytest.mark.unit
def test_openrouter_retries_embedded_transient_error() -> None:
    client = OpenRouterChatOpenAI(model="provider/model", api_key="test", max_retries=2)
    expected = mock.Mock()

    with (
        mock.patch.object(
            NormalizedChatOpenAI,
            "invoke",
            side_effect=[
                ValueError({"message": "Provider timed out", "code": 504}),
                expected,
            ],
        ) as invoke,
        mock.patch("dohasecuritiesstockai.llm_clients.openai_client.time.sleep") as sleep,
    ):
        result = client.invoke([])

    assert result is expected
    assert invoke.call_count == 2
    sleep.assert_called_once_with(1)


@pytest.mark.unit
def test_openrouter_does_not_retry_nontransient_error() -> None:
    client = OpenRouterChatOpenAI(model="provider/model", api_key="test", max_retries=6)
    error = ValueError({"message": "Bad request", "code": 400})

    with (
        mock.patch.object(NormalizedChatOpenAI, "invoke", side_effect=error) as invoke,
        mock.patch("dohasecuritiesstockai.llm_clients.openai_client.time.sleep") as sleep,
        pytest.raises(ValueError),
    ):
        client.invoke([])

    invoke.assert_called_once()
    sleep.assert_not_called()


@pytest.mark.unit
def test_openrouter_gateway_fallbacks_are_forwarded(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv(
        "TRADINGAGENTS_OPENROUTER_FALLBACK_MODELS",
        " openrouter/free, google/gemma-4-26b-a4b-it:free ",
    )

    llm = OpenAIClient(
        model="google/gemma-4-26b-a4b-it:free",
        provider="openrouter",
    ).get_llm()

    assert llm.extra_body["models"] == ["openrouter/free"]

from unittest.mock import MagicMock, patch

import pytest

from app.llm.openrouter import OpenRouterSummarizer


def test_openrouter_summarizer_requires_api_key() -> None:
    """The summarizer should reject a missing API key."""

    with patch.dict(
        "os.environ",
        {},
        clear=True,
    ):
        with pytest.raises(
            ValueError,
            match="OPENROUTER_API_KEY is required",
        ):
            OpenRouterSummarizer()


def test_openrouter_summarizer_reads_api_key_from_environment() -> None:
    """The summarizer should read the API key from the environment."""

    with patch.dict(
        "os.environ",
        {
            "OPENROUTER_API_KEY": "test-env-key",
        },
        clear=True,
    ):
        with patch(
            "app.llm.openrouter.OpenAI"
        ) as mock_openai:

            OpenRouterSummarizer()

            mock_openai.assert_called_once_with(
                api_key="test-env-key",
                base_url="https://openrouter.ai/api/v1",
            )


def test_openrouter_summarizer_explicit_api_key_takes_priority() -> None:
    """An explicitly provided API key should override the environment key."""

    with patch.dict(
        "os.environ",
        {
            "OPENROUTER_API_KEY": "environment-key",
        },
        clear=True,
    ):
        with patch(
            "app.llm.openrouter.OpenAI"
        ) as mock_openai:

            OpenRouterSummarizer(
                api_key="explicit-key"
            )

            mock_openai.assert_called_once_with(
                api_key="explicit-key",
                base_url="https://openrouter.ai/api/v1",
            )


def test_openrouter_summarizer_uses_api_response() -> None:
    """The summarizer should return the LLM response content."""

    mock_response = MagicMock()

    mock_response.choices[0].message.content = (
        "This is a generated summary."
    )

    with patch(
        "app.llm.openrouter.OpenAI"
    ) as mock_openai:

        mock_client = mock_openai.return_value

        mock_client.chat.completions.create.return_value = (
            mock_response
        )

        summarizer = OpenRouterSummarizer(
            api_key="test-key"
        )

        result = summarizer.summarize(
            "Artificial intelligence is used in many industries."
        )

    assert result == "This is a generated summary."

    mock_client.chat.completions.create.assert_called_once()


def test_openrouter_summarizer_sends_expected_request() -> None:
    """The summarizer should send the expected request parameters."""

    mock_response = MagicMock()

    mock_response.choices[0].message.content = (
        "This is a generated summary."
    )

    with patch(
        "app.llm.openrouter.OpenAI"
    ) as mock_openai:

        mock_client = mock_openai.return_value

        mock_client.chat.completions.create.return_value = (
            mock_response
        )

        summarizer = OpenRouterSummarizer(
            api_key="test-key",
            model="test-model",
        )

        summarizer.summarize(
            "Artificial intelligence is used in many industries."
        )

    mock_client.chat.completions.create.assert_called_once_with(
        model="test-model",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a concise summarization assistant. "
                    "Summarize the user's text while preserving "
                    "the most important information."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Artificial intelligence is used in many industries."
                ),
            },
        ],
        temperature=0.2,
        max_tokens=500,
    )


def test_openrouter_summarizer_rejects_empty_response() -> None:
    """The summarizer should reject an empty LLM response."""

    mock_response = MagicMock()

    mock_response.choices[0].message.content = None

    with patch(
        "app.llm.openrouter.OpenAI"
    ) as mock_openai:

        mock_client = mock_openai.return_value

        mock_client.chat.completions.create.return_value = (
            mock_response
        )

        summarizer = OpenRouterSummarizer(
            api_key="test-key"
        )

        with pytest.raises(
            ValueError,
            match="empty response",
        ):
            summarizer.summarize(
                "Some long text."
            )


def test_openrouter_summarizer_handles_empty_input() -> None:
    """Empty input should not make an API request."""

    with patch(
        "app.llm.openrouter.OpenAI"
    ) as mock_openai:

        summarizer = OpenRouterSummarizer(
            api_key="test-key"
        )

        result = summarizer.summarize("   ")

    assert result == ""

    mock_client = mock_openai.return_value

    mock_client.chat.completions.create.assert_not_called()
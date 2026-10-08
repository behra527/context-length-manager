from unittest.mock import MagicMock, patch

import pytest

from app.config import ModelConfig
from app.llm.openrouter import OpenRouterSummarizer
from app.manager import ContextManager, ContextStrategy
from app.strategies.summarization import SummarizationStrategy


class FakeSummarizer:
    """Deterministic summarizer for tests."""

    def summarize(self, text: str) -> str:
        return "This is a compact summary."


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(
        model_name="test-model",
        encoding_name="cl100k_base",
        context_window=100,
        max_output_tokens=20,
        safety_margin=10,
    )


@pytest.fixture
def manager(
    model_config: ModelConfig,
) -> ContextManager:
    summarizer = SummarizationStrategy(
        FakeSummarizer()
    )

    return ContextManager(
        model_config=model_config,
        summarization_strategy=summarizer,
    )


def test_context_within_budget_is_unchanged(
    manager: ContextManager,
) -> None:
    text = "Hello world."

    result = manager.process(text)

    assert result.strategy == "none"
    assert result.text == text
    assert result.was_modified is False

    assert result.metrics is not None
    assert result.metrics.fits is True


def test_oversized_context_can_be_truncated(
    manager: ContextManager,
) -> None:
    text = "Artificial intelligence " * 100

    result = manager.process(
        text,
        strategy=ContextStrategy.TRUNCATE,
    )

    assert result.strategy == "truncate"
    assert result.was_modified is True
    assert result.final_token_count <= (
        manager.budget.max_input_tokens
    )

    assert result.metrics is not None
    assert result.metrics.fits is False


def test_oversized_context_can_be_chunked(
    manager: ContextManager,
) -> None:
    text = "Artificial intelligence " * 100

    result = manager.process(
        text,
        strategy=ContextStrategy.CHUNK,
        chunk_size=20,
        overlap=5,
    )

    assert result.strategy == "chunk"
    assert result.was_modified is True
    assert result.chunks is not None
    assert len(result.chunks) > 1

    for chunk in result.chunks:
        assert chunk.token_count <= 20

    assert result.metrics is not None
    assert result.metrics.fits is False


def test_oversized_context_can_be_summarized(
    manager: ContextManager,
) -> None:
    text = "Artificial intelligence " * 100

    result = manager.process(
        text,
        strategy=ContextStrategy.SUMMARIZE,
    )

    assert result.strategy == "summarize"
    assert result.was_modified is True
    assert result.text == "This is a compact summary."
    assert result.final_token_count <= (
        manager.budget.max_input_tokens
    )

    assert result.metrics is not None
    assert result.metrics.fits is False


def test_chunk_size_required(
    manager: ContextManager,
) -> None:
    text = "Artificial intelligence " * 100

    with pytest.raises(
        ValueError,
        match="chunk_size is required",
    ):
        manager.process(
            text,
            strategy=ContextStrategy.CHUNK,
        )


def test_summarizer_required(
    model_config: ModelConfig,
) -> None:
    manager = ContextManager(
        model_config=model_config,
    )

    text = "Artificial intelligence " * 100

    with pytest.raises(
        ValueError,
        match="summarization_strategy is required",
    ):
        manager.process(
            text,
            strategy=ContextStrategy.SUMMARIZE,
        )


def test_manager_can_be_created_from_registered_model() -> None:
    manager = ContextManager.from_model("gpt-4")

    assert manager.model_config.model_name == "gpt-4"
    assert manager.model_config.context_window == 8192
    assert manager.budget.max_input_tokens == 6992


def test_manager_rejects_unknown_model() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported model",
    ):
        ContextManager.from_model("unknown-model")


def test_manager_reports_metrics_for_fitting_context(
    manager: ContextManager,
) -> None:
    text = "Hello world."

    result = manager.process(text)

    assert result.metrics is not None
    assert result.metrics.input_tokens == (
        result.original_token_count
    )
    assert result.metrics.max_input_tokens == (
        manager.budget.max_input_tokens
    )
    assert result.metrics.fits is True
    assert result.metrics.overflow_tokens == 0
    assert result.metrics.remaining_tokens > 0


def test_manager_reports_overflow_for_oversized_context(
    manager: ContextManager,
) -> None:
    text = "Artificial intelligence " * 100

    result = manager.process(
        text,
        strategy=ContextStrategy.TRUNCATE,
    )

    assert result.metrics is not None
    assert result.metrics.fits is False
    assert result.metrics.overflow_tokens > 0
    assert result.metrics.utilization_percent > 100


def test_chunked_result_uses_unique_token_count(
    manager: ContextManager,
) -> None:
    text = "Artificial intelligence " * 100

    result = manager.process(
        text,
        strategy=ContextStrategy.CHUNK,
        chunk_size=20,
        overlap=5,
    )

    assert result.chunks is not None

    total_chunk_tokens = sum(
        chunk.token_count
        for chunk in result.chunks
    )

    unique_token_count = (
        manager.chunking_strategy.unique_token_count(
            result.chunks
        )
    )

    assert result.final_token_count == unique_token_count
    assert result.final_token_count < total_chunk_tokens


def test_manager_can_use_openrouter_summarizer(
    model_config: ModelConfig,
) -> None:
    """Verify OpenRouter adapter works with ContextManager."""

    mock_response = MagicMock()

    mock_response.choices[0].message.content = (
        "This is a compact summary."
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

        summarization_strategy = SummarizationStrategy(
            summarizer=summarizer
        )

        manager = ContextManager(
            model_config=model_config,
            summarization_strategy=summarization_strategy,
        )

        text = "Artificial intelligence " * 100

        result = manager.process(
            text,
            strategy=ContextStrategy.SUMMARIZE,
        )

    assert result.strategy == "summarize"
    assert result.was_modified is True
    assert result.text == "This is a compact summary."
    assert result.final_token_count <= (
        manager.budget.max_input_tokens
    )

    mock_client.chat.completions.create.assert_called_once()
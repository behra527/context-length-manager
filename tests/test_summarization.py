import pytest

from app.strategies.summarization import SummarizationStrategy


class FakeSummarizer:
    """Deterministic summarizer used for unit tests."""

    def __init__(self, summary: str) -> None:
        self.summary = summary

    def summarize(self, text: str) -> str:
        return self.summary


def test_text_within_budget_is_unchanged() -> None:
    summarizer = FakeSummarizer("This is a summary.")

    strategy = SummarizationStrategy(summarizer)

    text = "Hello world."

    result = strategy.summarize(
        text,
        max_tokens=100,
    )

    assert result.text == text
    assert result.summarized is False


def test_large_text_is_summarized() -> None:
    summarizer = FakeSummarizer(
        "This is a concise summary."
    )

    strategy = SummarizationStrategy(summarizer)

    text = "Artificial intelligence " * 100

    result = strategy.summarize(
        text,
        max_tokens=20,
    )

    assert result.summarized is True
    assert result.original_token_count > result.summary_token_count
    assert result.summary_token_count <= 20
    assert result.text == "This is a concise summary."


def test_summary_exceeding_budget_is_rejected() -> None:
    summarizer = FakeSummarizer(
        "Artificial intelligence " * 100
    )

    strategy = SummarizationStrategy(summarizer)

    text = "Machine learning " * 100

    with pytest.raises(ValueError):
        strategy.summarize(
            text,
            max_tokens=20,
        )


def test_empty_text_within_budget() -> None:
    summarizer = FakeSummarizer("Summary")

    strategy = SummarizationStrategy(summarizer)

    result = strategy.summarize(
        "",
        max_tokens=10,
    )

    assert result.text == ""
    assert result.summarized is False


def test_invalid_max_tokens() -> None:
    summarizer = FakeSummarizer("Summary")

    strategy = SummarizationStrategy(summarizer)

    with pytest.raises(ValueError):
        strategy.summarize(
            "Some text",
            max_tokens=0,
        )
from dataclasses import dataclass
from typing import Protocol

from app.tokenizer import TokenCounter


class Summarizer(Protocol):
    """Interface for any LLM-backed summarization provider."""

    def summarize(self, text: str) -> str:
        """Summarize the supplied text."""
        ...


@dataclass(frozen=True)
class SummarizationResult:
    """Result produced by the summarization strategy."""

    original_token_count: int
    summary_token_count: int
    text: str
    summarized: bool


class SummarizationStrategy:
    """Compress large context using an injected summarizer."""

    def __init__(
        self,
        summarizer: Summarizer,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self.summarizer = summarizer
        self.token_counter = token_counter or TokenCounter()

    def summarize(
        self,
        text: str,
        max_tokens: int,
    ) -> SummarizationResult:
        """
        Summarize text and verify that the result fits the token budget.
        """

        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0")

        original_token_count = self.token_counter.count(text)

        if original_token_count <= max_tokens:
            return SummarizationResult(
                original_token_count=original_token_count,
                summary_token_count=original_token_count,
                text=text,
                summarized=False,
            )

        summary = self.summarizer.summarize(text)

        summary_token_count = self.token_counter.count(summary)

        if summary_token_count > max_tokens:
            raise ValueError(
                "Summarizer returned output exceeding the requested "
                "token budget"
            )

        return SummarizationResult(
            original_token_count=original_token_count,
            summary_token_count=summary_token_count,
            text=summary,
            summarized=True,
        )
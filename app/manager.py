from dataclasses import dataclass
from enum import Enum

from app.budget import ContextBudget
from app.config import ModelConfig
from app.metrics import ContextMetrics
from app.models import get_model_config
from app.strategies.chunking import ChunkingStrategy, TextChunk
from app.strategies.summarization import SummarizationStrategy
from app.strategies.truncation import TruncationStrategy
from app.tokenizer import TokenCounter


class ContextStrategy(str, Enum):
    """Available strategies for oversized context."""

    TRUNCATE = "truncate"
    CHUNK = "chunk"
    SUMMARIZE = "summarize"


@dataclass(frozen=True)
class ContextResult:
    """Final result returned by the context manager."""

    original_token_count: int
    final_token_count: int
    strategy: str
    text: str
    chunks: list[TextChunk] | None = None
    was_modified: bool = False
    metrics: ContextMetrics | None = None


class ContextManager:
    """
    Orchestrates token counting, budget validation,
    and context reduction strategies.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        summarization_strategy: SummarizationStrategy | None = None,
    ) -> None:
        self.model_config = model_config

        self.budget = ContextBudget(
            context_window=model_config.context_window,
            max_output_tokens=model_config.max_output_tokens,
            safety_margin=model_config.safety_margin,
        )

        self.token_counter = TokenCounter(
            model_config=model_config
        )

        self.truncation_strategy = TruncationStrategy(
            token_counter=self.token_counter
        )

        self.chunking_strategy = ChunkingStrategy(
            token_counter=self.token_counter
        )

        self.summarization_strategy = summarization_strategy

    @classmethod
    def from_model(
        cls,
        model_name: str,
        summarization_strategy: SummarizationStrategy | None = None,
    ) -> "ContextManager":
        """Create a context manager from a registered model."""

        model_config = get_model_config(model_name)

        return cls(
            model_config=model_config,
            summarization_strategy=summarization_strategy,
        )

    def process(
        self,
        text: str,
        strategy: ContextStrategy = ContextStrategy.TRUNCATE,
        chunk_size: int | None = None,
        overlap: int = 0,
    ) -> ContextResult:
        """Process context according to the selected strategy."""

        original_token_count = self.token_counter.count(text)

        metrics = ContextMetrics.calculate(
            input_tokens=original_token_count,
            max_input_tokens=self.budget.max_input_tokens,
        )

        # Context already fits within the available input budget.
        if metrics.fits:
            return ContextResult(
                original_token_count=original_token_count,
                final_token_count=original_token_count,
                strategy="none",
                text=text,
                chunks=None,
                was_modified=False,
                metrics=metrics,
            )

        # Handle oversized context using the selected strategy.
        if strategy == ContextStrategy.TRUNCATE:
            return self._process_truncation(
                text=text,
                metrics=metrics,
            )

        if strategy == ContextStrategy.CHUNK:
            return self._process_chunking(
                text=text,
                metrics=metrics,
                chunk_size=chunk_size,
                overlap=overlap,
            )

        if strategy == ContextStrategy.SUMMARIZE:
            return self._process_summarization(
                text=text,
                metrics=metrics,
            )

        raise ValueError(
            f"Unsupported strategy: {strategy}"
        )

    def _process_truncation(
        self,
        text: str,
        metrics: ContextMetrics,
    ) -> ContextResult:
        """Process oversized context using truncation."""

        result = self.truncation_strategy.truncate(
            text=text,
            max_tokens=self.budget.max_input_tokens,
        )

        return ContextResult(
            original_token_count=result.original_token_count,
            final_token_count=result.final_token_count,
            strategy=ContextStrategy.TRUNCATE.value,
            text=result.text,
            chunks=None,
            was_modified=True,
            metrics=metrics,
        )

    def _process_chunking(
        self,
        text: str,
        metrics: ContextMetrics,
        chunk_size: int | None,
        overlap: int,
    ) -> ContextResult:
        """Process oversized context using chunking."""

        if chunk_size is None:
            raise ValueError(
                "chunk_size is required when using chunk strategy"
            )

        chunks = self.chunking_strategy.chunk(
            text=text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        unique_token_count = (
            self.chunking_strategy.unique_token_count(chunks)
        )

        return ContextResult(
            original_token_count=metrics.input_tokens,
            final_token_count=unique_token_count,
            strategy=ContextStrategy.CHUNK.value,
            text="",
            chunks=chunks,
            was_modified=True,
            metrics=metrics,
        )

    def _process_summarization(
        self,
        text: str,
        metrics: ContextMetrics,
    ) -> ContextResult:
        """Process oversized context using summarization."""

        if self.summarization_strategy is None:
            raise ValueError(
                "summarization_strategy is required "
                "when using summarize strategy"
            )

        result = self.summarization_strategy.summarize(
            text=text,
            max_tokens=self.budget.max_input_tokens,
        )

        return ContextResult(
            original_token_count=result.original_token_count,
            final_token_count=result.summary_token_count,
            strategy=ContextStrategy.SUMMARIZE.value,
            text=result.text,
            chunks=None,
            was_modified=True,
            metrics=metrics,
        )
from dataclasses import dataclass

from app.tokenizer import TokenCounter


@dataclass(frozen=True)
class TruncationResult:
    """Result produced by the truncation strategy."""

    original_token_count: int
    final_token_count: int
    truncated: bool
    text: str


class TruncationStrategy:
    """Truncate text to a maximum token budget."""

    def __init__(self, token_counter: TokenCounter | None = None) -> None:
        self.token_counter = token_counter or TokenCounter()

    def truncate(
        self,
        text: str,
        max_tokens: int,
    ) -> TruncationResult:
        """
        Truncate text so the returned text contains at most max_tokens.
        """

        if max_tokens < 0:
            raise ValueError("max_tokens cannot be negative")

        original_tokens = self.token_counter.encode(text)
        original_count = len(original_tokens)

        if original_count <= max_tokens:
            return TruncationResult(
                original_token_count=original_count,
                final_token_count=original_count,
                truncated=False,
                text=text,
            )

        truncated_tokens = original_tokens[:max_tokens]
        truncated_text = self.token_counter.decode(truncated_tokens)

        return TruncationResult(
            original_token_count=original_count,
            final_token_count=len(truncated_tokens),
            truncated=True,
            text=truncated_text,
        )
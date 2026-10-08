
from dataclasses import dataclass

from app.tokenizer import TokenCounter


@dataclass(frozen=True)
class TextChunk:
    """Represents a token-based text chunk."""

    index: int
    text: str
    token_count: int
    start_token: int
    end_token: int


class ChunkingStrategy:
    """Split text into token-based chunks with optional overlap."""

    def __init__(self, token_counter: TokenCounter | None = None) -> None:
        self.token_counter = token_counter or TokenCounter()

    def chunk(
        self,
        text: str,
        chunk_size: int,
        overlap: int = 0,
    ) -> list[TextChunk]:
        """
        Split text into token-based chunks.

        Args:
            text: Input text.
            chunk_size: Maximum tokens per chunk.
            overlap: Number of tokens shared between consecutive chunks.

        Returns:
            A list of TextChunk objects.
        """

        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        if overlap < 0:
            raise ValueError("overlap cannot be negative")

        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        tokens = self.token_counter.encode(text)

        if not tokens:
            return []

        chunks: list[TextChunk] = []

        step = chunk_size - overlap
        start = 0
        index = 0

        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))

            chunk_tokens = tokens[start:end]
            chunk_text = self.token_counter.decode(chunk_tokens)

            chunks.append(
                TextChunk(
                    index=index,
                    text=chunk_text,
                    token_count=len(chunk_tokens),
                    start_token=start,
                    end_token=end,
                )
            )

            index += 1

            if end >= len(tokens):
                break

            start += step

        return chunks

    @staticmethod
    def unique_token_count(
        chunks: list[TextChunk],
    ) -> int:
        """Return the number of unique source tokens represented by chunks."""

        if not chunks:
            return 0

        return max(
            chunk.end_token
            for chunk in chunks
        )


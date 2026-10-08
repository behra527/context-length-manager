from dataclasses import dataclass

import tiktoken

from app.config import ModelConfig


@dataclass(frozen=True)
class TokenUsage:
    """Represents token usage for a piece of text."""

    text: str
    token_count: int


class TokenCounter:
    """Counts tokens using a model-compatible tokenizer."""

    def __init__(
        self,
        encoding_name: str | None = None,
        model_config: ModelConfig | None = None,
    ) -> None:
        if model_config is not None and encoding_name is not None:
            raise ValueError(
                "Provide either encoding_name or model_config, not both"
            )

        if model_config is not None:
            encoding_name = model_config.encoding_name

        encoding_name = encoding_name or "cl100k_base"

        self.encoding = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        """Return the number of tokens in the given text."""

        if not text:
            return 0

        return len(self.encoding.encode(text))

    def analyze(self, text: str) -> TokenUsage:
        """Return text together with its token count."""

        return TokenUsage(
            text=text,
            token_count=self.count(text),
        )

    def encode(self, text: str) -> list[int]:
        """Convert text into token IDs."""

        return self.encoding.encode(text)

    def decode(self, tokens: list[int]) -> str:
        """Convert token IDs back into text."""

        return self.encoding.decode(tokens)
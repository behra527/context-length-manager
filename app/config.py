from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for an LLM model's context limits."""

    model_name: str
    encoding_name: str
    context_window: int
    max_output_tokens: int
    safety_margin: int = 200

    def __post_init__(self) -> None:
        """Validate model configuration."""

        if not self.model_name.strip():
            raise ValueError("model_name cannot be empty")

        if not self.encoding_name.strip():
            raise ValueError("encoding_name cannot be empty")

        if self.context_window <= 0:
            raise ValueError("context_window must be greater than 0")

        if self.max_output_tokens < 0:
            raise ValueError(
                "max_output_tokens cannot be negative"
            )

        if self.safety_margin < 0:
            raise ValueError(
                "safety_margin cannot be negative"
            )

        if self.max_output_tokens + self.safety_margin >= self.context_window:
            raise ValueError(
                "max_output_tokens + safety_margin must be "
                "smaller than context_window"
            )

    @property
    def max_input_tokens(self) -> int:
        """Return the available input-token budget."""

        return (
            self.context_window
            - self.max_output_tokens
            - self.safety_margin
        )
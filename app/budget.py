from dataclasses import dataclass


@dataclass(frozen=True)
class ContextBudget:
    """Defines the token budget available for model input."""

    context_window: int
    max_output_tokens: int
    safety_margin: int = 200

    def __post_init__(self) -> None:
        """Validate budget configuration."""
        if self.context_window <= 0:
            raise ValueError("context_window must be greater than 0")

        if self.max_output_tokens < 0:
            raise ValueError("max_output_tokens cannot be negative")

        if self.safety_margin < 0:
            raise ValueError("safety_margin cannot be negative")

        if self.max_output_tokens + self.safety_margin >= self.context_window:
            raise ValueError(
                "max_output_tokens + safety_margin must be smaller "
                "than context_window"
            )

    @property
    def max_input_tokens(self) -> int:
        """Return the maximum number of tokens allowed for input."""
        return (
            self.context_window
            - self.max_output_tokens
            - self.safety_margin
        )

    def fits(self, input_tokens: int) -> bool:
        """Return True when input fits within the available budget."""
        if input_tokens < 0:
            raise ValueError("input_tokens cannot be negative")

        return input_tokens <= self.max_input_tokens

    def remaining(self, input_tokens: int) -> int:
        """Return remaining input-token capacity."""
        if input_tokens < 0:
            raise ValueError("input_tokens cannot be negative")

        return max(0, self.max_input_tokens - input_tokens)
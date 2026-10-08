from dataclasses import dataclass


@dataclass(frozen=True)
class ContextMetrics:
    """Describes token usage against the available context budget."""

    input_tokens: int
    max_input_tokens: int
    remaining_tokens: int
    overflow_tokens: int
    utilization_percent: float
    fits: bool

    @classmethod
    def calculate(
        cls,
        input_tokens: int,
        max_input_tokens: int,
    ) -> "ContextMetrics":
        """Calculate context usage metrics."""

        if input_tokens < 0:
            raise ValueError("input_tokens cannot be negative")

        if max_input_tokens <= 0:
            raise ValueError(
                "max_input_tokens must be greater than 0"
            )

        remaining_tokens = max(
            0,
            max_input_tokens - input_tokens,
        )

        overflow_tokens = max(
            0,
            input_tokens - max_input_tokens,
        )

        utilization_percent = (
            input_tokens / max_input_tokens
        ) * 100

        return cls(
            input_tokens=input_tokens,
            max_input_tokens=max_input_tokens,
            remaining_tokens=remaining_tokens,
            overflow_tokens=overflow_tokens,
            utilization_percent=utilization_percent,
            fits=input_tokens <= max_input_tokens,
        )
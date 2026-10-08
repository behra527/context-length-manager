import pytest

from app.strategies.truncation import TruncationStrategy


def test_text_within_limit_is_unchanged() -> None:
    strategy = TruncationStrategy()

    text = "Hello world"

    result = strategy.truncate(text, max_tokens=100)

    assert result.text == text
    assert result.truncated is False
    assert result.original_token_count == result.final_token_count


def test_long_text_is_truncated() -> None:
    strategy = TruncationStrategy()

    text = "Artificial intelligence " * 100

    result = strategy.truncate(text, max_tokens=20)

    assert result.truncated is True
    assert result.final_token_count <= 20
    assert result.original_token_count > result.final_token_count


def test_empty_text() -> None:
    strategy = TruncationStrategy()

    result = strategy.truncate("", max_tokens=10)

    assert result.text == ""
    assert result.original_token_count == 0
    assert result.final_token_count == 0
    assert result.truncated is False


def test_zero_token_limit() -> None:
    strategy = TruncationStrategy()

    result = strategy.truncate("Hello world", max_tokens=0)

    assert result.text == ""
    assert result.final_token_count == 0
    assert result.truncated is True


def test_negative_limit_is_rejected() -> None:
    strategy = TruncationStrategy()

    with pytest.raises(ValueError):
        strategy.truncate("Hello world", max_tokens=-1)
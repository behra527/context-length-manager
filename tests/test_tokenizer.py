import pytest

from app.config import ModelConfig
from app.tokenizer import TokenCounter


def test_token_counter_can_use_model_config() -> None:
    config = ModelConfig(
        model_name="test-model",
        encoding_name="cl100k_base",
        context_window=100,
        max_output_tokens=20,
        safety_margin=10,
    )

    counter = TokenCounter(model_config=config)

    assert counter.count("Hello world") > 0


def test_encoding_name_and_model_config_cannot_both_be_provided() -> None:
    config = ModelConfig(
        model_name="test-model",
        encoding_name="cl100k_base",
        context_window=100,
        max_output_tokens=20,
        safety_margin=10,
    )

    with pytest.raises(ValueError):
        TokenCounter(
            encoding_name="cl100k_base",
            model_config=config,
        )
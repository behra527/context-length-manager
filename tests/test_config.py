import pytest

from app.config import ModelConfig


def test_model_config_calculates_input_budget() -> None:
    config = ModelConfig(
        model_name="test-model",
        encoding_name="cl100k_base",
        context_window=100,
        max_output_tokens=20,
        safety_margin=10,
    )

    assert config.max_input_tokens == 70


def test_model_config_accepts_valid_configuration() -> None:
    config = ModelConfig(
        model_name="test-model",
        encoding_name="cl100k_base",
        context_window=100,
        max_output_tokens=20,
        safety_margin=10,
    )

    assert config.model_name == "test-model"
    assert config.encoding_name == "cl100k_base"


def test_empty_model_name_is_rejected() -> None:
    with pytest.raises(ValueError):
        ModelConfig(
            model_name="",
            encoding_name="cl100k_base",
            context_window=100,
            max_output_tokens=20,
        )


def test_empty_encoding_name_is_rejected() -> None:
    with pytest.raises(ValueError):
        ModelConfig(
            model_name="test-model",
            encoding_name="",
            context_window=100,
            max_output_tokens=20,
        )


def test_invalid_context_window_is_rejected() -> None:
    with pytest.raises(ValueError):
        ModelConfig(
            model_name="test-model",
            encoding_name="cl100k_base",
            context_window=0,
            max_output_tokens=20,
        )


def test_invalid_output_budget_is_rejected() -> None:
    with pytest.raises(ValueError):
        ModelConfig(
            model_name="test-model",
            encoding_name="cl100k_base",
            context_window=100,
            max_output_tokens=100,
        )


def test_negative_safety_margin_is_rejected() -> None:
    with pytest.raises(ValueError):
        ModelConfig(
            model_name="test-model",
            encoding_name="cl100k_base",
            context_window=100,
            max_output_tokens=20,
            safety_margin=-1,
        )
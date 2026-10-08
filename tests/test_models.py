import pytest

from app.config import ModelConfig
from app.models import MODEL_REGISTRY, get_model_config


def test_model_registry_contains_models() -> None:
    assert len(MODEL_REGISTRY) > 0


def test_get_model_config_returns_config() -> None:
    config = get_model_config("gpt-4")

    assert isinstance(config, ModelConfig)
    assert config.model_name == "gpt-4"
    assert config.context_window == 8192


def test_get_model_config_returns_gpt4o() -> None:
    config = get_model_config("gpt-4o")

    assert config.model_name == "gpt-4o"
    assert config.encoding_name == "o200k_base"
    assert config.context_window == 128000


def test_unknown_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported model"):
        get_model_config("unknown-model")
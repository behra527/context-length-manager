from app.config import ModelConfig


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "gpt-4": ModelConfig(
        model_name="gpt-4",
        encoding_name="cl100k_base",
        context_window=8192,
        max_output_tokens=1000,
        safety_margin=200,
    ),
    "gpt-4o": ModelConfig(
        model_name="gpt-4o",
        encoding_name="o200k_base",
        context_window=128000,
        max_output_tokens=2000,
        safety_margin=500,
    ),
}


def get_model_config(model_name: str) -> ModelConfig:
    """Return configuration for a registered model."""

    try:
        return MODEL_REGISTRY[model_name]
    except KeyError as exc:
        available_models = ", ".join(MODEL_REGISTRY)

        raise ValueError(
            f"Unsupported model: {model_name}. "
            f"Available models: {available_models}"
        ) from exc
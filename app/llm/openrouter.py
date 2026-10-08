import os

from dotenv import load_dotenv
from openai import OpenAI

from app.strategies.summarization import Summarizer


load_dotenv()


class OpenRouterSummarizer(Summarizer):
    """Summarizer implementation using OpenRouter."""

    def __init__(
        self,
        model: str = "openai/gpt-oss-20b",
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
    ) -> None:
        self.model = model

        resolved_api_key = (
            api_key
            or os.getenv("OPENROUTER_API_KEY")
        )

        if not resolved_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is required"
            )

        self.client = OpenAI(
            api_key=resolved_api_key,
            base_url=base_url,
        )

    def summarize(self, text: str) -> str:
        """Generate a concise summary using the configured LLM."""

        if not text.strip():
            return ""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a concise summarization assistant. "
                        "Summarize the user's text while preserving "
                        "the most important information."
                    ),
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
            temperature=0.2,
            max_tokens=500,
        )

        content = response.choices[0].message.content

        if not content:
            raise ValueError(
                "The summarization model returned an empty response"
            )

        return content.strip()
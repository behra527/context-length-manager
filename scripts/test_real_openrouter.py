from app.llm.openrouter import OpenRouterSummarizer


def main() -> None:
    summarizer = OpenRouterSummarizer()

    text = """
    Artificial intelligence is increasingly used in healthcare,
    finance, education, and software development. Large language
    models can process large amounts of information and generate
    useful summaries. However, applications must manage context
    windows carefully because models have limits on how much input
    they can process at once.
    """

    print("Calling OpenRouter...")

    summary = summarizer.summarize(text)

    print("\nOriginal text:")
    print(text.strip())

    print("\nGenerated summary:")
    print(summary)


if __name__ == "__main__":
    main()
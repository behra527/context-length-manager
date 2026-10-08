
import pytest

from app.strategies.chunking import ChunkingStrategy


def test_empty_text_returns_no_chunks() -> None:
    strategy = ChunkingStrategy()

    result = strategy.chunk("", chunk_size=100)

    assert result == []


def test_text_smaller_than_chunk_size() -> None:
    strategy = ChunkingStrategy()

    text = "Artificial intelligence is useful."

    chunks = strategy.chunk(text, chunk_size=100)

    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].token_count <= 100


def test_large_text_is_split_into_chunks() -> None:
    strategy = ChunkingStrategy()

    text = "Artificial intelligence " * 100

    chunks = strategy.chunk(
        text,
        chunk_size=20,
    )

    assert len(chunks) > 1

    for chunk in chunks:
        assert chunk.token_count <= 20


def test_chunk_overlap() -> None:
    strategy = ChunkingStrategy()

    text = "Artificial intelligence " * 100

    chunks = strategy.chunk(
        text,
        chunk_size=20,
        overlap=5,
    )

    assert len(chunks) > 1

    assert chunks[1].start_token == chunks[0].end_token - 5


def test_invalid_chunk_size() -> None:
    strategy = ChunkingStrategy()

    with pytest.raises(ValueError):
        strategy.chunk("Hello world", chunk_size=0)


def test_invalid_overlap() -> None:
    strategy = ChunkingStrategy()

    with pytest.raises(ValueError):
        strategy.chunk(
            "Hello world",
            chunk_size=10,
            overlap=10,
        )


def test_negative_overlap() -> None:
    strategy = ChunkingStrategy()

    with pytest.raises(ValueError):
        strategy.chunk(
            "Hello world",
            chunk_size=10,
            overlap=-1,
        )


def test_unique_token_count_without_overlap() -> None:
    strategy = ChunkingStrategy()

    text = "Artificial intelligence " * 20

    chunks = strategy.chunk(
        text,
        chunk_size=20,
        overlap=0,
    )

    unique_count = strategy.unique_token_count(chunks)

    assert unique_count == chunks[-1].end_token


def test_unique_token_count_with_overlap() -> None:
    strategy = ChunkingStrategy()

    text = "Artificial intelligence " * 20

    chunks = strategy.chunk(
        text,
        chunk_size=20,
        overlap=5,
    )

    total_chunk_tokens = sum(
        chunk.token_count
        for chunk in chunks
    )

    unique_count = strategy.unique_token_count(chunks)

    assert unique_count == chunks[-1].end_token
    assert unique_count < total_chunk_tokens


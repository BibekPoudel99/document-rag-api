import pytest

from app.services.chunking import (
    ChunkingStrategy,
    chunk_fixed,
    chunk_recursive,
    chunk_text,
)


def _digits(length: int) -> str:
    """Text with no whitespace, so stripping never changes it."""
    return "".join(str(i % 10) for i in range(length))


# ---------- fixed-size ----------

def test_fixed_empty_and_whitespace_input() -> None:
    assert chunk_fixed("", 10, 2) == []
    assert chunk_fixed("   \n  ", 10, 2) == []


def test_fixed_short_document_is_one_chunk() -> None:
    assert chunk_fixed("hello", 100, 10) == ["hello"]


def test_fixed_text_exactly_chunk_size() -> None:
    assert chunk_fixed("abcdefghij", 10, 2) == ["abcdefghij"]


def test_fixed_overlap_behaviour() -> None:
    assert chunk_fixed("abcdefghij", 5, 2) == ["abcde", "defgh", "ghij"]


def test_fixed_without_overlap() -> None:
    assert chunk_fixed("abcdefghij", 5, 0) == ["abcde", "fghij"]


def test_fixed_final_short_chunk_is_kept() -> None:
    assert chunk_fixed("abcdefghijk", 5, 0) == ["abcde", "fghij", "k"]


def test_fixed_no_text_is_lost_on_large_document() -> None:
    text = _digits(100_000)
    chunks = chunk_fixed(text, 800, 100)
    assert all(len(c) <= 800 for c in chunks)
    # Rebuild: first chunk + each later chunk minus its overlapping prefix.
    rebuilt = chunks[0] + "".join(c[100:] for c in chunks[1:])
    assert rebuilt == text


@pytest.mark.parametrize(
    ("size", "overlap"), [(0, 0), (-1, 0), (10, -1), (10, 10), (10, 11)]
)
def test_fixed_rejects_invalid_parameters(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_fixed("some text", size, overlap)


# ---------- recursive ----------

def test_recursive_empty_input() -> None:
    assert chunk_recursive("", 100, 10) == []
    assert chunk_recursive("  \n\n ", 100, 10) == []


def test_recursive_short_document_is_one_chunk() -> None:
    assert chunk_recursive("A short note.", 100, 10) == ["A short note."]


def test_recursive_prefers_paragraph_boundaries() -> None:
    text = "A" * 30 + "\n\n" + "B" * 30 + "\n\n" + "C" * 30
    assert chunk_recursive(text, 70, 0) == ["A" * 30 + "\n\n" + "B" * 30, "C" * 30]


def test_recursive_falls_back_to_word_boundaries() -> None:
    text = " ".join(["word"] * 100)  # one paragraph, no newlines
    chunks = chunk_recursive(text, 100, 0)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)
    assert sum(len(c.split()) for c in chunks) == 100  # no word lost or duplicated


def test_recursive_hard_splits_a_token_with_no_separators() -> None:
    assert chunk_recursive("x" * 250, 100, 0) == ["x" * 100, "x" * 100, "x" * 50]


def test_recursive_overlap_respects_max_size_and_repeats_boundary_text() -> None:
    text = " ".join(f"w{i}" for i in range(200))
    chunks = chunk_recursive(text, 100, 20)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)
    for current, following in zip(chunks, chunks[1:]):
        assert current.split()[-1] in following.split()  # boundary word is repeated
    assert set(" ".join(chunks).split()) == set(text.split())  # nothing lost


def test_recursive_produces_no_empty_chunks() -> None:
    assert chunk_recursive("a\n\n\n\n\nb", 3, 0) == ["a", "b"]


def test_recursive_large_document_stays_within_limit() -> None:
    paragraph = " ".join(["word"] * 150)
    text = "\n\n".join([paragraph] * 200)
    chunks = chunk_recursive(text, 800, 100)
    assert all(0 < len(c) <= 800 for c in chunks)


def test_recursive_rejects_overlap_that_leaves_no_room() -> None:
    with pytest.raises(ValueError):
        chunk_recursive("x" * 50, 10, 9)


# ---------- dispatcher ----------

def test_chunk_text_dispatches_by_strategy() -> None:
    text = "abcdefghij"
    assert chunk_text(text, ChunkingStrategy.FIXED, 5, 2) == chunk_fixed(text, 5, 2)
    assert chunk_text(text, ChunkingStrategy.RECURSIVE, 5, 0) == chunk_recursive(text, 5, 0)


def test_strategy_enum_accepts_api_string_values() -> None:
    assert ChunkingStrategy("fixed") is ChunkingStrategy.FIXED
    with pytest.raises(ValueError):
        ChunkingStrategy("semantic")
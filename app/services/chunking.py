"""Text chunking strategies.

Pure functions with no I/O, so they are easy to unit test.
All sizes are measured in CHARACTERS, not tokens.
"""

from collections.abc import Callable, Sequence
from enum import StrEnum


class ChunkingStrategy(StrEnum):
    FIXED = "fixed"
    RECURSIVE = "recursive"


# Tried in order: paragraph break, line break, space.
_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", " ")


def _validate(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must not be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")


def chunk_fixed(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Slide a window of `chunk_size` characters, advancing by size - overlap."""
    _validate(chunk_size, chunk_overlap)
    text = text.strip()
    if not text:
        return []

    step = chunk_size - chunk_overlap
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        # If this window already reached the end, stop. Otherwise the next
        # window would be a tiny tail that is fully contained in this chunk.
        if start + chunk_size >= len(text):
            break
        start += step
    return chunks


def _hard_split(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def _merge(units: list[str], separator: str, size: int) -> list[str]:
    """Greedily join adjacent units (with `separator`) while they fit in `size`."""
    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = unit if not current else current + separator + unit
        if len(candidate) <= size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = unit  # safe: every unit is already <= size
    if current:
        chunks.append(current)
    return chunks


def _split_recursive(text: str, separators: Sequence[str], size: int) -> list[str]:
    """Split `text` into pieces of at most `size` chars, preferring earlier separators."""
    if len(text) <= size:
        return [text]
    if not separators:
        return _hard_split(text, size)  # last resort: no separator left to use

    separator, remaining = separators[0], separators[1:]
    if separator not in text:
        return _split_recursive(text, remaining, size)

    units: list[str] = []
    for part in text.split(separator):
        if part.strip():
            units.extend(_split_recursive(part, remaining, size))
    return _merge(units, separator, size)


def _overlap_tail(previous: str, chunk_overlap: int) -> str:
    """Last <= chunk_overlap characters of `previous`, starting at a word boundary."""
    if len(previous) <= chunk_overlap:
        return previous
    tail = previous[-chunk_overlap:]
    starts_mid_word = not previous[-chunk_overlap - 1].isspace()
    if starts_mid_word:
        for index, char in enumerate(tail):
            if char.isspace():
                return tail[index + 1 :]
    return tail


def chunk_recursive(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split on natural boundaries (paragraphs, lines, words), then add overlap."""
    _validate(chunk_size, chunk_overlap)

    # Overlap (plus one joining space) is prepended afterwards, so we split into
    # smaller pieces now to keep every final chunk within chunk_size.
    budget = chunk_size - chunk_overlap - (1 if chunk_overlap else 0)
    if budget < 1:
        raise ValueError("chunk_overlap leaves no room for new text in each chunk")

    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    pieces = [p.strip() for p in _split_recursive(text, _SEPARATORS, budget)]
    pieces = [p for p in pieces if p]
    if chunk_overlap == 0 or len(pieces) < 2:
        return pieces

    chunks = [pieces[0]]
    for previous, current in zip(pieces, pieces[1:]):
        tail = _overlap_tail(previous, chunk_overlap)
        chunks.append(f"{tail} {current}".strip())
    return chunks


_CHUNKERS: dict[ChunkingStrategy, Callable[[str, int, int], list[str]]] = {
    ChunkingStrategy.FIXED: chunk_fixed,
    ChunkingStrategy.RECURSIVE: chunk_recursive,
}


def chunk_text(
    text: str, strategy: ChunkingStrategy, chunk_size: int, chunk_overlap: int
) -> list[str]:
    """Dispatch to the selected strategy."""
    return _CHUNKERS[strategy](text, chunk_size, chunk_overlap)
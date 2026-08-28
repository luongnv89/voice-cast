"""Text chunking utilities for VoiceCast.

Splits long text into chunks that respect sentence boundaries and a
maximum character limit, so downstream TTS engines can synthesize text
that would otherwise exceed their input length.
"""

import re

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_into_chunks(text: str, max_chars: int) -> list[str]:
    """
    Split text into chunks that respect sentence boundaries.

    Sentences are greedily packed into chunks so that each chunk's length
    stays within ``max_chars`` where possible. A single sentence that by
    itself exceeds ``max_chars`` is emitted whole as its own chunk rather
    than being split mid-sentence.

    Args:
        text: The input text to split.
        max_chars: The maximum number of characters allowed per chunk.

    Returns:
        A list of chunk strings. Returns an empty list for empty or
        whitespace-only input.

    Raises:
        ValueError: If ``max_chars`` is not a positive integer.
    """
    if max_chars <= 0:
        raise ValueError(f"max_chars must be a positive integer, got {max_chars}")

    stripped = text.strip()
    if not stripped:
        return []

    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(stripped) if s.strip()]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        candidate_len = len(sentence) if not current else current_len + 1 + len(sentence)

        if current and candidate_len > max_chars:
            chunks.append(" ".join(current))
            current = [sentence]
            current_len = len(sentence)
        else:
            current.append(sentence)
            current_len = candidate_len

    if current:
        chunks.append(" ".join(current))

    return chunks

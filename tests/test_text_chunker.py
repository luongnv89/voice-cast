"""Tests for utils.text_chunker."""

import pytest

from utils.text_chunker import split_into_chunks


class TestEmptyAndWhitespaceInput:
    """Empty or whitespace-only input returns an empty list."""

    @pytest.mark.parametrize("text", ["", "   ", "\n\t  \n"])
    def test_returns_empty_list(self, text):
        assert split_into_chunks(text, max_chars=100) == []


class TestShortText:
    """Text under max_chars returns a single-element list."""

    def test_short_text_single_chunk(self):
        result = split_into_chunks("Hello world.", max_chars=100)
        assert result == ["Hello world."]

    def test_short_text_no_terminator(self):
        result = split_into_chunks("Hello world", max_chars=100)
        assert result == ["Hello world"]


class TestBoundary:
    """Behavior around the max_chars boundary."""

    def test_exact_boundary_single_chunk(self):
        text = "a" * 50 + "."
        result = split_into_chunks(text, max_chars=len(text))
        assert result == [text]

    def test_one_over_boundary_splits(self):
        sentence_a = "A" * 30 + "."
        sentence_b = "B" * 30 + "."
        text = f"{sentence_a} {sentence_b}"
        max_chars = len(sentence_a) + 1 + len(sentence_b) - 1
        result = split_into_chunks(text, max_chars=max_chars)
        assert result == [sentence_a, sentence_b]

    def test_one_under_boundary_single_chunk(self):
        sentence_a = "A" * 30 + "."
        sentence_b = "B" * 30 + "."
        text = f"{sentence_a} {sentence_b}"
        max_chars = len(sentence_a) + 1 + len(sentence_b)
        result = split_into_chunks(text, max_chars=max_chars)
        assert result == [text]


class TestLongTextManySentences:
    """Long text with many sentences is packed into multiple chunks."""

    def test_multiple_chunks_respect_max_chars(self):
        sentences = [f"Sentence number {i} is here." for i in range(20)]
        text = " ".join(sentences)
        max_chars = 60
        result = split_into_chunks(text, max_chars=max_chars)

        assert len(result) > 1
        for chunk in result:
            assert len(chunk) <= max_chars

        # No sentence content is lost or duplicated.
        rejoined = " ".join(result)
        for sentence in sentences:
            assert sentence in rejoined

    def test_no_empty_chunks(self):
        text = "One. Two. Three. Four. Five."
        result = split_into_chunks(text, max_chars=10)
        assert all(chunk.strip() == chunk and chunk for chunk in result)


class TestOversizedSentence:
    """A single sentence longer than max_chars is emitted whole."""

    def test_oversized_sentence_emitted_whole(self):
        long_sentence = "A" * 200 + "."
        result = split_into_chunks(long_sentence, max_chars=50)
        assert result == [long_sentence]

    def test_oversized_sentence_among_others(self):
        long_sentence = "B" * 200 + "."
        text = f"Short one. {long_sentence} Short two."
        result = split_into_chunks(text, max_chars=50)
        assert long_sentence in result
        for chunk in result:
            if chunk != long_sentence:
                assert len(chunk) <= 50


class TestPunctuationVariants:
    """Sentence splitting handles ., !, ? and unterminated text."""

    def test_exclamation_and_question_marks(self):
        text = "Is this real? Yes it is! Great."
        result = split_into_chunks(text, max_chars=15)
        assert result == ["Is this real?", "Yes it is!", "Great."]

    def test_unterminated_sentence_kept_as_is(self):
        text = "This has no terminal punctuation at all"
        result = split_into_chunks(text, max_chars=100)
        assert result == [text]

    def test_mixed_terminated_and_unterminated(self):
        text = "First sentence. Second sentence without terminator"
        result = split_into_chunks(text, max_chars=100)
        assert result == [text]


class TestWhitespaceHandling:
    """Non-empty input is trimmed and sentence separators are normalized."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            pytest.param(
                "  First sentence.   Second sentence. \n",
                ["First sentence. Second sentence."],
                id="trim-and-collapse-spaces",
            ),
            pytest.param(
                "First sentence.\n\tSecond sentence.",
                ["First sentence. Second sentence."],
                id="collapse-newline-and-tab",
            ),
            pytest.param(
                "\tFirst sentence.\n\nSecond sentence.  ",
                ["First sentence. Second sentence."],
                id="trim-and-collapse-newlines",
            ),
        ],
    )
    def test_trims_and_normalizes_sentence_whitespace(self, text, expected):
        assert split_into_chunks(text, max_chars=100) == expected


class TestGreedyPacking:
    """Sentences are packed into each chunk until the next one no longer fits."""

    @pytest.mark.parametrize(
        ("text", "max_chars", "expected"),
        [
            pytest.param(
                "One. Two. Three.",
                9,
                ["One. Two.", "Three."],
                id="exact-first-chunk-then-remainder",
            ),
            pytest.param(
                "One. Two. Three. Four.",
                12,
                ["One. Two.", "Three. Four."],
                id="greedy-packing-resumes-after-split",
            ),
            pytest.param(
                "One. Two. Three. Four.",
                9,
                ["One. Two.", "Three.", "Four."],
                id="remainder-sentences-respect-limit",
            ),
        ],
    )
    def test_packs_sentences_greedily(self, text, max_chars, expected):
        assert split_into_chunks(text, max_chars=max_chars) == expected


class TestOversizedSentencePositions:
    """An oversized sentence remains whole wherever it appears in the input."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            pytest.param(
                "Oversized sentence. One. Two.",
                ["Oversized sentence.", "One. Two."],
                id="at-start",
            ),
            pytest.param(
                "One. Oversized sentence. Two.",
                ["One.", "Oversized sentence.", "Two."],
                id="in-middle",
            ),
            pytest.param(
                "One. Two. Oversized sentence.",
                ["One. Two.", "Oversized sentence."],
                id="at-end",
            ),
        ],
    )
    def test_oversized_sentence_isolated_at_any_position(self, text, expected):
        assert split_into_chunks(text, max_chars=10) == expected


class TestInvalidMaxChars:
    """max_chars <= 0 raises ValueError."""

    @pytest.mark.parametrize("max_chars", [0, -1, -100])
    def test_raises_value_error(self, max_chars):
        with pytest.raises(ValueError, match="max_chars"):
            split_into_chunks("Some text.", max_chars=max_chars)

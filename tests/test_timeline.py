"""Milestone 4 (PROJECT_SPEC.md §12): pure tests for the Stage 2/3/4 merge
logic in app/pipeline/timeline.py — fake WordTiming/Token objects only,
mirroring how tests/test_render.py tests Stage 5's pure logic.
"""

import pytest

from app.pipeline import timeline
from app.pipeline.align import WordTiming
from app.pipeline.map_images import ImageResult
from app.pipeline.timeline import TimelineMismatchError, build_render_words
from app.pipeline.tokens import Token


def _timing(text: str, start: float, end: float) -> WordTiming:
    return WordTiming(text=text, start=start, end=end, score=0.0)


@pytest.fixture(autouse=True)
def _stub_map_word_to_image(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake(lemma, *, context_lemma=None, cache=None, client=None):
        return ImageResult(lemma=lemma, query=lemma, source="placeholder", location=f"/imgs/{lemma}.jpg")

    monkeypatch.setattr(timeline, "map_word_to_image", _fake)


def test_build_render_words_matches_content_words_across_attached_punctuation() -> None:
    script_text = "giant cat, rides."
    timings = [_timing("giant", 0.0, 0.3), _timing("cat,", 0.3, 0.6), _timing("rides.", 0.6, 0.9)]
    tokens = [
        Token(text="giant", lemma="giant", pos="ADJ", is_content=True),
        Token(text="cat", lemma="cat", pos="NOUN", is_content=True),
        Token(text=",", lemma=",", pos="PUNCT", is_content=False),
        Token(text="rides", lemma="ride", pos="VERB", is_content=True),
        Token(text=".", lemma=".", pos="PUNCT", is_content=False),
    ]

    render_words = build_render_words(script_text, timings, tokens)

    assert [w.text for w in render_words] == ["giant", "cat,", "rides."]
    assert render_words[0].image_location == "/imgs/giant.jpg"
    assert render_words[1].image_location == "/imgs/cat.jpg"
    assert render_words[2].image_location == "/imgs/ride.jpg"


def test_build_render_words_leaves_non_content_words_without_an_image() -> None:
    script_text = "A giant cat"
    timings = [_timing("A", 0.0, 0.1), _timing("giant", 0.1, 0.3), _timing("cat", 0.3, 0.5)]
    tokens = [
        Token(text="A", lemma="a", pos="DET", is_content=False),
        Token(text="giant", lemma="giant", pos="ADJ", is_content=True),
        Token(text="cat", lemma="cat", pos="NOUN", is_content=True),
    ]

    render_words = build_render_words(script_text, timings, tokens)

    assert render_words[0].image_location is None
    assert render_words[1].image_location == "/imgs/giant.jpg"
    assert render_words[2].image_location == "/imgs/cat.jpg"


def test_build_render_words_uses_a_neighboring_content_word_as_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Doesn't assert on which lemma "wins" beyond confirming context is
    passed at all -- the retry-with-context behavior itself is already
    covered by tests/test_map_images.py; this only checks the wiring.
    """
    calls: list[tuple[str, str | None]] = []

    def _capturing_fake(lemma, *, context_lemma=None, cache=None, client=None):
        calls.append((lemma, context_lemma))
        return ImageResult(lemma=lemma, query=lemma, source="placeholder", location=f"/imgs/{lemma}.jpg")

    monkeypatch.setattr(timeline, "map_word_to_image", _capturing_fake)

    script_text = "cat dog"
    timings = [_timing("cat", 0.0, 0.3), _timing("dog", 0.3, 0.6)]
    tokens = [
        Token(text="cat", lemma="cat", pos="NOUN", is_content=True),
        Token(text="dog", lemma="dog", pos="NOUN", is_content=True),
    ]

    build_render_words(script_text, timings, tokens)

    assert calls == [("cat", "dog"), ("dog", "cat")]


def test_build_render_words_rejects_word_count_mismatch() -> None:
    script_text = "one two three"
    timings = [_timing("one", 0.0, 0.1), _timing("two", 0.1, 0.2)]  # missing "three"
    tokens = [
        Token(text="one", lemma="one", pos="NUM", is_content=False),
        Token(text="two", lemma="two", pos="NUM", is_content=False),
        Token(text="three", lemma="three", pos="NUM", is_content=False),
    ]

    with pytest.raises(TimelineMismatchError):
        build_render_words(script_text, timings, tokens)


def test_build_render_words_rejects_empty_timings() -> None:
    with pytest.raises(ValueError):
        build_render_words("", [], [])
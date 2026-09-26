"""Milestone 4: tag.py's unified dispatch across en/hi/mixed."""

import pytest

from app.pipeline.tag import tag_text

pytestmark = pytest.mark.slow  # loads real spaCy + Stanza models


def test_tag_text_routes_english() -> None:
    tokens = tag_text("The giant cat runs.", "en")
    assert any(t.text == "cat" and t.is_content for t in tokens)


def test_tag_text_routes_hindi() -> None:
    tokens = tag_text("सूरज चमकता है।", "hi")
    assert any(t.text == "सूरज" and t.is_content for t in tokens)


def test_tag_text_routes_mixed_by_script() -> None:
    tokens = tag_text("giant सूरज cat", "mixed")
    texts = [t.text for t in tokens]
    assert "giant" in texts and "सूरज" in texts and "cat" in texts


def test_tag_text_rejects_unsupported_language() -> None:
    with pytest.raises(ValueError):
        tag_text("hello", "fr")
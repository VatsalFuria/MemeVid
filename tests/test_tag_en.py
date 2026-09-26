"""Milestone 4: unit test for the English tagger (spaCy), mirroring
tests/test_tag_hi.py's shape for the Hindi backend.
"""

import pytest

from app.pipeline.tag_en import tag_english

pytestmark = pytest.mark.slow  # loads a real spaCy model


def test_tag_english_produces_expected_shape_and_tags() -> None:
    text = "The giant cat runs quickly."

    tokens = tag_english(text)
    by_text = {t.text: t for t in tokens}

    assert by_text["giant"].pos == "ADJ" and by_text["giant"].is_content is True
    assert by_text["cat"].pos == "NOUN" and by_text["cat"].is_content is True
    assert by_text["runs"].pos == "VERB" and by_text["runs"].lemma == "run"
    assert by_text["quickly"].pos == "ADV" and by_text["quickly"].is_content is True
    assert by_text["The"].pos == "DET" and by_text["The"].is_content is False


def test_tag_english_rejects_empty_text() -> None:
    with pytest.raises(ValueError):
        tag_english("   ")
"""Milestone 3c (PROJECT_SPEC.md §12): unit test comparing Stanza's Hindi
tags against a small hand-checked sample.

Expected tags follow standard Universal Dependencies conventions for Hindi
(UD_Hindi-HDTB): habitual verb forms split into a lexical participle
(VERB) + an auxiliary (AUX) carrying tense/agreement; sentence-final
purna viram (।) is PUNCT. See the module docstring in app/pipeline/tag_hi.py
for why these should be double-checked against real output rather than
assumed correct.
"""

import pytest

from app.pipeline.tag_hi import tag_hindi

pytestmark = pytest.mark.slow  # downloads + runs a real Stanza Hindi model


def test_tag_hindi_produces_expected_shape_and_tags() -> None:
    text = "सूरज चमकता है।"  # "The sun shines."

    tokens = tag_hindi(text)

    by_text = {t.text: t for t in tokens}
    assert set(by_text) == {"सूरज", "चमकता", "है", "।"}

    assert by_text["सूरज"].pos == "NOUN"
    assert by_text["सूरज"].is_content is True

    assert by_text["चमकता"].pos == "VERB"
    assert by_text["चमकता"].is_content is True

    assert by_text["है"].pos == "AUX"
    assert by_text["है"].is_content is False

    assert by_text["।"].pos == "PUNCT"
    assert by_text["।"].is_content is False


def test_tag_hindi_rejects_empty_text() -> None:
    with pytest.raises(ValueError):
        tag_hindi("   ")
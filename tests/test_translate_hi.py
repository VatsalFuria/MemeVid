"""Milestone 3d (PROJECT_SPEC.md §12): unit test translating a handful of
common Hindi nouns/verbs into English image-search queries.

These are deliberately common, unambiguous words (cat, dog, tree, book,
run) precisely so a small MT model should still get them right -- if one
of these fails, that's worth investigating, not just loosening the
assertion. Checks look for the expected English word inside the
(lowercased) output rather than an exact string match, since MT phrasing
can vary (e.g. "to run" vs "running") without being wrong.
"""

import pytest

from app.pipeline.translate_hi import translate_hindi_lemma

pytestmark = pytest.mark.slow  # downloads + runs a real MarianMT checkpoint


@pytest.mark.parametrize(
    "hindi_lemma,expected_english",
    [
        ("बिल्ली", "cat"),
        ("कुत्ता", "dog"),
        ("पेड़", "tree"),
        ("किताब", "book"),
        ("दौड़ना", "run"),
    ],
)
def test_translate_hindi_lemma_common_words(hindi_lemma: str, expected_english: str) -> None:
    result = translate_hindi_lemma(hindi_lemma)
    assert expected_english in result.lower()


def test_translate_hindi_lemma_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        translate_hindi_lemma("   ")


def test_translate_hindi_lemma_is_cached() -> None:
    """Cheap sanity check on the lru_cache itself, since nothing else in
    this test file would catch a caching regression.
    """
    translate_hindi_lemma.cache_clear()
    translate_hindi_lemma("बिल्ली")
    translate_hindi_lemma("बिल्ली")
    assert translate_hindi_lemma.cache_info().hits >= 1
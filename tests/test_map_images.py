"""Milestone/Stage 4 (PROJECT_SPEC.md §5): word -> image mapping.

Cache, translation-routing, retry, and placeholder-fallback logic are
tested with a stubbed Pexels call (no network, no real API key needed).
The two `network`-marked tests hit the real Pexels API and need a genuine
PEXELS_API_KEY in .env to pass -- skip them with `-m "not network"`.
"""

from pathlib import Path

import pytest

from app.pipeline import map_images
from app.pipeline.map_images import (
    ImageResult,
    InMemoryImageCache,
    map_word_to_image,
)


@pytest.fixture(autouse=True)
def _reset_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test gets a fresh rate-limit tracker so one test's mocked
    response headers can't leak into the next test's budget check.
    """
    monkeypatch.setattr(map_images, "_budget", map_images._PexelsBudget())


@pytest.fixture()
def placeholder_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = tmp_path / "placeholders"
    directory.mkdir()
    (directory / "neutral_1.jpg").write_bytes(b"fake-image-bytes")
    (directory / "neutral_2.jpg").write_bytes(b"fake-image-bytes")
    settings = map_images.get_settings()
    monkeypatch.setattr(settings, "placeholder_images_dir", directory, raising=False)
    map_images.get_settings.cache_clear()  # type: ignore[attr-defined]
    monkeypatch.setattr(
        map_images, "get_settings", lambda: settings.model_copy(update={"placeholder_images_dir": directory})
    )
    return directory


def test_rejects_empty_lemma() -> None:
    with pytest.raises(ValueError):
        map_word_to_image("   ")


def test_uses_cache_and_never_calls_pexels(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = InMemoryImageCache()
    cache.set("cat", "https://images.pexels.com/cached-cat.jpg")

    def _unexpected_call(*args, **kwargs):
        raise AssertionError("Pexels should not be called on a cache hit")

    monkeypatch.setattr(map_images, "_try_pexels", _unexpected_call)

    result = map_word_to_image("cat", cache=cache)

    assert result == ImageResult(
        lemma="cat", query="cat", source="cache", location="https://images.pexels.com/cached-cat.jpg"
    )


def test_pexels_hit_is_cached_on_original_lemma(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = InMemoryImageCache()
    calls: list[str] = []

    def _fake_try_pexels(query: str, *, client):
        calls.append(query)
        return map_images._PexelsResponse(photo_urls=["https://images.pexels.com/dog.jpg"], rate_limit_remaining=100)

    monkeypatch.setattr(map_images, "_try_pexels", _fake_try_pexels)

    result = map_word_to_image("dog", cache=cache)

    assert result.source == "pexels"
    assert result.location == "https://images.pexels.com/dog.jpg"
    assert cache.get("dog") == "https://images.pexels.com/dog.jpg"
    assert calls == ["dog"]


def test_hindi_lemma_is_translated_before_querying_and_cached_on_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = InMemoryImageCache()
    queries: list[str] = []

    monkeypatch.setattr(map_images, "translate_hindi_lemma", lambda lemma: "cat")

    def _fake_try_pexels(query: str, *, client):
        queries.append(query)
        return map_images._PexelsResponse(photo_urls=["https://images.pexels.com/cat.jpg"], rate_limit_remaining=100)

    monkeypatch.setattr(map_images, "_try_pexels", _fake_try_pexels)

    result = map_word_to_image("बिल्ली", cache=cache)

    assert queries == ["cat"]  # queried with the translated English word...
    assert cache.get("बिल्ली") == "https://images.pexels.com/cat.jpg"  # ...cached on the original Devanagari lemma
    assert result.source == "pexels"


def test_zero_results_retries_once_with_context(monkeypatch: pytest.MonkeyPatch) -> None:
    queries: list[str] = []

    def _fake_try_pexels(query: str, *, client):
        queries.append(query)
        if len(queries) == 1:
            return map_images._PexelsResponse(photo_urls=[], rate_limit_remaining=100)
        return map_images._PexelsResponse(
            photo_urls=["https://images.pexels.com/skateboard-street.jpg"], rate_limit_remaining=99
        )

    monkeypatch.setattr(map_images, "_try_pexels", _fake_try_pexels)

    result = map_word_to_image("skateboard", context_lemma="street", cache=InMemoryImageCache())

    assert queries == ["skateboard", "skateboard street"]
    assert result.source == "pexels"


def test_exhausted_budget_skips_pexels_and_falls_back(
    monkeypatch: pytest.MonkeyPatch, placeholder_dir: Path
) -> None:
    monkeypatch.setattr(map_images, "_budget", map_images._PexelsBudget())
    map_images._budget.record(0)  # simulate an already-exhausted window

    def _unexpected_call(*args, **kwargs):
        raise AssertionError("Pexels should not be called when budget is exhausted")

    monkeypatch.setattr(map_images, "_try_pexels", _unexpected_call)

    result = map_word_to_image("anything", cache=InMemoryImageCache())

    assert result.source == "placeholder"
    assert Path(result.location).parent == placeholder_dir


def test_placeholder_fallback_is_not_cached(monkeypatch: pytest.MonkeyPatch, placeholder_dir: Path) -> None:
    cache = InMemoryImageCache()

    def _always_empty(query: str, *, client):
        return map_images._PexelsResponse(photo_urls=[], rate_limit_remaining=100)

    monkeypatch.setattr(map_images, "_try_pexels", _always_empty)

    result = map_word_to_image("zzznoresults", cache=cache)

    assert result.source == "placeholder"
    assert cache.get("zzznoresults") is None  # a transient miss shouldn't poison the persistent cache


@pytest.mark.network
def test_map_word_to_image_english_live() -> None:
    result = map_word_to_image("cat", cache=InMemoryImageCache())
    assert result.source == "pexels"
    assert result.location.startswith("https://")


@pytest.mark.network
def test_map_word_to_image_hindi_live() -> None:
    result = map_word_to_image("बिल्ली", cache=InMemoryImageCache())
    assert result.source == "pexels"
    assert result.location.startswith("https://")
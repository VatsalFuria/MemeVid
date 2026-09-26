"""Word -> image mapping stage (PROJECT_SPEC.md §5 Stage 4).

For each content word, this module:
  1. Checks a cross-job cache (lemma -> image URL) before hitting Pexels.
  2. Translates Devanagari lemmas to an English query first (Pexels' index
     is effectively English-only) -- this is where milestone 3d's
     `app.pipeline.translate_hi.translate_hindi_lemma` gets wired in.
  3. Caches on the *original* lemma, so repeated words skip re-translation
     too (PROJECT_SPEC.md §5 Stage 4).
  4. On a cache miss with zero Pexels results, retries once with a
     neighboring content word as extra context.
  5. Falls back to a small bundled set of neutral placeholder images on
     continued failure or exhausted Pexels budget, so one bad word never
     fails the whole render (PROJECT_SPEC.md §10).

The persistent, cross-job `ImageCache` table (PROJECT_SPEC.md §7) doesn't
exist yet -- that's milestone 5's SQLAlchemy + Alembic work. `ImageCacheStore`
below is the seam: `InMemoryImageCache` is today's process-local stand-in,
and milestone 5 drops in a DB-backed implementation of the same two
methods. Nothing here should need to change when that happens.

Placeholder fallback results are deliberately *not* written to the cache --
a placeholder chosen because Pexels' budget was exhausted this hour
shouldn't become a permanent answer for that word once the budget resets.
"""

from __future__ import annotations

import logging
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import httpx

from app.config import get_settings
from app.pipeline.translate_hi import translate_hindi_lemma

logger = logging.getLogger(__name__)

from app.pipeline.tokens import is_devanagari as _is_devanagari


class ImageCacheStore(Protocol):
    """Seam for PROJECT_SPEC.md §7's persistent, cross-job `ImageCache`
    table. `InMemoryImageCache` below satisfies this today; milestone 5
    swaps in a SQLAlchemy-backed implementation with the same two methods.
    """

    def get(self, lemma: str) -> str | None: ...

    def set(self, lemma: str, image_url: str) -> None: ...


class InMemoryImageCache:
    """Process-local stand-in for the real persistent cache (see module
    docstring). Fine for CLI runs and tests; not shared across worker
    processes or restarts -- that gap is exactly what milestone 5 closes.
    """

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def get(self, lemma: str) -> str | None:
        return self._store.get(lemma)

    def set(self, lemma: str, image_url: str) -> None:
        self._store[lemma] = image_url


_default_cache = InMemoryImageCache()


@dataclass(frozen=True)
class ImageResult:
    lemma: str
    query: str
    source: Literal["cache", "pexels", "placeholder"]
    location: str  # cache/pexels: image URL. placeholder: local filesystem path.


@dataclass(frozen=True)
class _PexelsResponse:
    photo_urls: list[str]
    rate_limit_remaining: int | None


class _PexelsBudget:
    """Tracks the last-seen `X-Ratelimit-Remaining` value so we can skip a
    doomed call once we already know the window is exhausted, rather than
    erroring out (PROJECT_SPEC.md §10). Process-local and best-effort --
    it only knows what the last response told it, same caveat as every
    other in-process cache in this codebase pre-milestone-5.
    """

    def __init__(self) -> None:
        self._remaining: int | None = None

    def has_budget(self) -> bool:
        settings = get_settings()
        return self._remaining is None or self._remaining > settings.pexels_min_rate_limit_remaining

    def record(self, remaining: int | None) -> None:
        if remaining is not None:
            self._remaining = remaining


_budget = _PexelsBudget()


def _search_pexels(query: str, *, client: httpx.Client) -> _PexelsResponse:
    settings = get_settings()
    response = client.get(
        settings.pexels_base_url,
        params={"query": query, "per_page": 5},
        # No "Bearer" prefix -- Pexels wants the raw key in this header.
        headers={"Authorization": settings.pexels_api_key},
        timeout=settings.pexels_request_timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    photo_urls = [photo["src"]["large"] for photo in payload.get("photos", [])]

    # Rate-limit headers are only present on 2xx responses (never on 429) --
    # if raise_for_status() above didn't already raise, they should be here,
    # but treat their absence as "unknown" rather than assuming exhausted.
    remaining_header = response.headers.get("X-Ratelimit-Remaining")
    remaining = int(remaining_header) if remaining_header is not None else None
    return _PexelsResponse(photo_urls=photo_urls, rate_limit_remaining=remaining)


def _try_pexels(query: str, *, client: httpx.Client) -> _PexelsResponse:
    try:
        result = _search_pexels(query, client=client)
    except httpx.HTTPError as exc:
        # A single flaky/failed Pexels call shouldn't fail the whole render
        # (PROJECT_SPEC.md §10) -- treat it as a miss and let the caller
        # fall through to retry-with-context or the placeholder.
        logger.warning("Pexels request failed for query %r: %s", query, exc)
        return _PexelsResponse(photo_urls=[], rate_limit_remaining=None)
    _budget.record(result.rate_limit_remaining)
    return result


def _placeholder_location(lemma: str) -> str:
    settings = get_settings()
    candidates = sorted(p for p in settings.placeholder_images_dir.glob("*") if p.is_file())
    if not candidates:
        raise RuntimeError(
            f"No placeholder images found in {settings.placeholder_images_dir} -- "
            "bundle at least one neutral fallback image (PROJECT_SPEC.md §5 Stage 4)."
        )
    # Deterministic per lemma so re-running on the same word within a job
    # doesn't flicker between different placeholders.
    index = zlib.crc32(lemma.encode("utf-8")) % len(candidates)
    return str(candidates[index])

def placeholder_location(seed: str) -> str:
    """Public entry point to Stage 4's placeholder-image selection, for
    other stages that need "some neutral image" without going through the
    full cache/Pexels/retry flow — e.g. render.py's leading-non-content-
    word and lead-in-gap edge cases (PROJECT_SPEC.md §5 Stage 5). Same
    deterministic-per-seed behavior as `_placeholder_location`'s other
    callers in this module.
    """
    return _placeholder_location(seed)


def _to_query(lemma: str) -> str:
    return translate_hindi_lemma(lemma) if _is_devanagari(lemma) else lemma


def map_word_to_image(
    lemma: str,
    *,
    context_lemma: str | None = None,
    cache: ImageCacheStore | None = None,
    client: httpx.Client | None = None,
) -> ImageResult:
    """Map a single content-word lemma to an image (PROJECT_SPEC.md §5 Stage 4).

    Args:
        lemma: a content word's lemma, in its original script (as produced
            by app/pipeline/tag_hi.py or the eventual English tagger).
            Cached and returned keyed on this exact, pre-translation value.
        context_lemma: an optional neighboring content word, used only if
            the first Pexels query comes back empty (PROJECT_SPEC.md §5
            Stage 4's "retry once with a neighboring content word").
        cache: defaults to a shared in-memory cache (see module docstring).
        client: defaults to a short-lived httpx.Client; pass your own to
            reuse a connection across many calls in one job.

    Returns:
        ImageResult noting where the image came from (cache/pexels/
        placeholder) so callers/tests can distinguish a real photo from a
        neutral fallback.
    """
    if not lemma or not lemma.strip():
        raise ValueError("Cannot map an empty lemma to an image.")

    cache = cache or _default_cache
    cached_url = cache.get(lemma)
    if cached_url is not None:
        return ImageResult(lemma=lemma, query=lemma, source="cache", location=cached_url)

    query = _to_query(lemma)
    if not query.strip():
        # An empty translation is a miss to fall through on, not an error
        # raised here (see translate_hi.py's own docstring on this point).
        return ImageResult(lemma=lemma, query=query, source="placeholder", location=_placeholder_location(lemma))

    owns_client = client is None
    client = client or httpx.Client()
    try:
        if _budget.has_budget():
            result = _try_pexels(query, client=client)
            if result.photo_urls:
                cache.set(lemma, result.photo_urls[0])
                return ImageResult(lemma=lemma, query=query, source="pexels", location=result.photo_urls[0])

            if context_lemma:
                retry_query = f"{query} {_to_query(context_lemma)}".strip()
                retry_result = _try_pexels(retry_query, client=client)
                if retry_result.photo_urls:
                    cache.set(lemma, retry_result.photo_urls[0])
                    return ImageResult(
                        lemma=lemma, query=retry_query, source="pexels", location=retry_result.photo_urls[0]
                    )
    finally:
        if owns_client:
            client.close()

    return ImageResult(lemma=lemma, query=query, source="placeholder", location=_placeholder_location(lemma))
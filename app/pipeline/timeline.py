"""Wiring Stage 2 (align) + Stage 3 (tag) + Stage 4 (map_images) into the
per-word timeline Stage 5 (render) expects (PROJECT_SPEC.md §5; see
app/pipeline/render.py's module docstring: "that zip is milestone 4's
wiring job, not this module's").

IMPORTANT — read before trusting this for Hindi: Stage 2's forced
alignment (app/pipeline/align.py) is called with `romanize=True`
regardless of language (its own docstring explains why: the MMS aligner
shares one romanized vocabulary across languages). It is NOT confirmed
whether ctc-forced-aligner's returned `WordTiming.text` reflects that
romanized form or the original script for Devanagari input -- so this
module never trusts that field for word identity or captions, only for
`.start`/`.end` timing. Word text (for matching against Stage 3's tokens,
and for what gets burned into captions) always comes from `script_text`
itself, split on whitespace the same way ctc-forced-aligner is assumed to
split it.

That assumption -- one `WordTiming` per whitespace-delimited word of
`script_text`, in order -- is verified against the checked-in English
fixture (tests/fixtures/sample_en + timings.json) but NOT yet verified
against a real Hindi run. If it doesn't hold, `build_render_words` raises
`TimelineMismatchError` loudly rather than silently mismapping words to
the wrong images/timings -- treat that error, if it ever fires, as a sign
this assumption needs revisiting for whatever input triggered it.
"""

from __future__ import annotations

import re

import httpx

from app.pipeline.align import WordTiming
from app.pipeline.map_images import ImageCacheStore, map_word_to_image
from app.pipeline.render import RenderWord
from app.pipeline.tokens import Token

_WORD_RE = re.compile(r"\S+")


class TimelineMismatchError(RuntimeError):
    """Raised when Stage 2's word count and Stage 3's tokenization of the
    same `script_text` can't be reconciled (PROJECT_SPEC.md §5: "surface a
    warning rather than silently producing garbage timing on mismatch" --
    applied here to the align/tag merge specifically).
    """


def _split_words(script_text: str) -> list[str]:
    """Whitespace-delimited words of `script_text`, in order -- see module
    docstring for why this, not Stage 2's own `WordTiming.text`, is the
    source of truth for word identity and captions.
    """
    return _WORD_RE.findall(script_text)


def _match_content_tokens(words: list[str], tokens: list[Token]) -> list[Token | None]:
    """For each whitespace word, walk `tokens` until their concatenated
    text reproduces that word exactly, and return the single *content*
    token inside that span (or `None`). Both `words` and `tokens` are
    non-overlapping partitions of the same `script_text`, in the same
    order, differing only in whether punctuation is split into its own
    token (e.g. "cat," vs. "cat" + ",") -- so this always terminates
    cleanly for well-formed input.
    """
    j = 0
    n = len(tokens)
    result: list[Token | None] = []

    for word in words:
        acc = ""
        content_token: Token | None = None
        while acc != word:
            if j >= n:
                raise TimelineMismatchError(
                    f"Ran out of tagged tokens while matching whitespace word {word!r}. "
                    "Stage 3's tokenization of script_text doesn't reconstruct it the "
                    "same way Stage 2's word count assumes -- see this module's docstring."
                )
            tok = tokens[j]
            acc += tok.text
            if tok.is_content and content_token is None:
                content_token = tok
            j += 1
            if len(acc) > len(word):
                raise TimelineMismatchError(
                    f"Tagged tokens overran whitespace word {word!r} (accumulated {acc!r}) -- "
                    "Stage 3's tokenization doesn't line up with script_text's whitespace split."
                )
        result.append(content_token)

    return result


def build_render_words(
    script_text: str,
    timings: list[WordTiming],
    tokens: list[Token],
    *,
    cache: ImageCacheStore | None = None,
    client: httpx.Client | None = None,
) -> list[RenderWord]:
    """Zip Stage 2's word timings with Stage 3's content-word flags and
    Stage 4's image lookups into the `RenderWord` list Stage 5 (render.py)
    consumes.

    Args:
        script_text: the exact text both `timings` and `tokens` were
            derived from -- the source of truth for word text (module
            docstring).
        timings: Stage 2's output, in start-time order, assumed to have
            exactly one entry per whitespace-delimited word of
            `script_text` (see module docstring).
        tokens: Stage 3's output for the same `script_text`, in reading
            order.
        cache: forwarded to `map_word_to_image` -- defaults to that
            function's own module-level in-memory cache.
        client: forwarded to `map_word_to_image`; reused across every
            content word in this job so they share one connection.

    Returns:
        One `RenderWord` per timing, with `text` taken from `script_text`
        (never from `WordTiming.text`) and `image_location` set for
        content words.
    """
    if not timings:
        raise ValueError("Cannot build a timeline from zero aligned words.")

    words = _split_words(script_text)
    if len(words) != len(timings):
        raise TimelineMismatchError(
            f"Aligned word count ({len(timings)}) doesn't match script_text's "
            f"whitespace-word count ({len(words)}) -- see this module's docstring "
            "on the assumption this violates."
        )

    content_tokens = _match_content_tokens(words, tokens)
    content_lemmas = [t.lemma if t is not None else None for t in content_tokens]

    owns_client = client is None
    client = client or httpx.Client()
    try:
        render_words: list[RenderWord] = []
        for index, (word, timing) in enumerate(zip(words, timings)):
            lemma = content_lemmas[index]
            if lemma is None:
                render_words.append(RenderWord(text=word, start=timing.start, end=timing.end))
                continue

            context_lemma = next((c for c in content_lemmas[index + 1 :] if c is not None), None)
            if context_lemma is None:
                context_lemma = next((c for c in reversed(content_lemmas[:index]) if c is not None), None)

            image = map_word_to_image(lemma, context_lemma=context_lemma, cache=cache, client=client)
            render_words.append(
                RenderWord(text=word, start=timing.start, end=timing.end, image_location=image.location)
            )
        return render_words
    finally:
        if owns_client:
            client.close()
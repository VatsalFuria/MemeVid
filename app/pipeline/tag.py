"""Unified tagging dispatch (PROJECT_SPEC.md §5 Stage 3).

Milestone-4 wiring: routes a job's script text to the matching backend by
its `language` (en/hi/mixed), so the rest of the shared pipeline
(app/pipeline/timeline.py) only ever calls one function regardless of
which job language it's handling.

  - en     -> app.pipeline.tag_en.tag_english (spaCy)
  - hi     -> app.pipeline.tag_hi.tag_hindi (Stanza)
  - mixed  -> best-effort script-based routing (see `_tag_mixed` below)

Known v1 limitation on `mixed`: each tagger does its own sentence-level
tokenization, so splitting the text into same-script runs first and
tagging each run in isolation loses whatever cross-run sentence context a
single end-to-end pass would have had. This is the "reasonable first-pass
heuristic" PROJECT_SPEC.md §5 Stage 3 calls for, not a claim it's
linguistically ideal -- and it's moot for milestone 4 specifically, since
app/flows.py's shared pipeline doesn't accept `mixed` yet either (Stage 2
alignment doesn't support it -- see that module's docstring).
"""

from __future__ import annotations

import re

from app.pipeline.tag_en import tag_english
from app.pipeline.tag_hi import tag_hindi
from app.pipeline.tokens import Token, is_devanagari

JobLanguage = str  # "en" | "hi" | "mixed" -- see app/core/asr.py's JobLanguage

# Splits on whitespace while keeping the whitespace itself as separate
# chunks, so runs can be rejoined with natural spacing before tagging.
_WORD_SPLIT_RE = re.compile(r"(\s+)")


def _script_of(word: str) -> str:
    """"hi" if `word` contains Devanagari, else "en" -- the same
    first-pass heuristic app/pipeline/map_images.py uses for routing
    lemmas to translation, reused here for routing whole words to a
    tagger (PROJECT_SPEC.md §5 Stage 3).
    """
    return "hi" if is_devanagari(word) else "en"


def _tag_mixed(text: str) -> list[Token]:
    pieces = _WORD_SPLIT_RE.split(text)  # alternating word / whitespace chunks
    words = [p for p in pieces if p and not p.isspace()]
    if not words:
        raise ValueError("Cannot tag empty text.")

    # Group consecutive same-script words into runs, then tag each run
    # with the matching backend and concatenate results in order.
    runs: list[tuple[str, list[str]]] = []
    for word in words:
        script = _script_of(word)
        if runs and runs[-1][0] == script:
            runs[-1][1].append(word)
        else:
            runs.append((script, [word]))

    tokens: list[Token] = []
    for script, run_words in runs:
        run_text = " ".join(run_words)
        tokens.extend(tag_english(run_text) if script == "en" else tag_hindi(run_text))
    return tokens


def tag_text(text: str, language: JobLanguage) -> list[Token]:
    """Tag `text` with whichever backend matches `language`
    (PROJECT_SPEC.md §5 Stage 3, §6). Single entry point the rest of the
    shared pipeline should call -- see module docstring.
    """
    if language == "en":
        return tag_english(text)
    if language == "hi":
        return tag_hindi(text)
    if language == "mixed":
        return _tag_mixed(text)
    raise ValueError(f"Unsupported tagging language: {language!r}")
"""Shared `Token` shape for all NLP tagging backends (PROJECT_SPEC.md §5
Stage 3).

Both the English tagger (spaCy, app/pipeline/tag_en.py) and the Hindi
tagger (Stanza, app/pipeline/tag_hi.py) normalize into this exact shape so
nothing downstream — word -> image mapping, the render-word wiring in
app/pipeline/timeline.py — needs to know or care which tagger produced a
given token. Previously duplicated per-module (tag_hi.py flagged this);
hoisted here now that app/pipeline/tag.py wires both backends behind one
dispatch (milestone 4).
"""

from __future__ import annotations

from dataclasses import dataclass

# Universal POS tags that count as a "content word" (PROJECT_SPEC.md §5
# Stage 3) -- consistent across both taggers since both emit UD-style
# tags, not treebank-specific ones.
CONTENT_POS = frozenset({"NOUN", "PROPN", "VERB", "ADJ", "ADV"})

_DEVANAGARI_RANGE = (0x0900, 0x097F)


def is_devanagari(text: str) -> bool:
    """First-pass script heuristic (PROJECT_SPEC.md §5 Stage 3): true if
    `text` contains any Devanagari codepoint. Shared by
    app/pipeline/map_images.py (routing a lemma to translation before a
    Pexels query) and app/pipeline/tag.py (routing a `mixed`-job word to
    the matching tagger) so the two never drift apart.
    """
    return any(_DEVANAGARI_RANGE[0] <= ord(ch) <= _DEVANAGARI_RANGE[1] for ch in text)


@dataclass(frozen=True)
class Token:
    text: str
    lemma: str
    pos: str
    is_content: bool
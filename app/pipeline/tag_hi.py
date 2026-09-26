"""Hindi NLP tagging stage (PROJECT_SPEC.md §5 Stage 3).

Standalone milestone-3c component: Hindi text in -> a list of unified
`Token`s out, in the same shape PROJECT_SPEC.md §5 mandates for *both*
language backends ("Nothing downstream should know or care which tagger
ran"): `text`, `lemma`, `pos`, `is_content`.

v1 uses Stanza's Hindi UD pipeline (PROJECT_SPEC.md §9) -- spaCy's Hindi
support is comparatively immature, whereas Stanza ships a model trained on
the Hindi Dependency Treebank (UD_Hindi-HDTB), the standard Hindi
Universal Dependencies corpus (verified current as of this writing -- §5
explicitly calls out re-checking model availability before assuming names
haven't moved).

Integration note: this module doesn't yet know about the English tagger
(spaCy, from milestone 2's shared pipeline) or the mixed-language,
script-based routing described in §5 Stage 3. Wire those together against
whichever `Token` shape milestone 2 already established -- if it differs
from the one below, change this file to match, not the other way around.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import stanza

from app.config import get_settings

# Universal POS tags that count as a "content word" (PROJECT_SPEC.md §5
# Stage 3) -- consistent across both taggers since both emit UD `upos`
# tags, not treebank-specific ones. Duplicated here and (presumably) in the
# English tagger for now; hoist to one shared constant once the two are
# wired together behind a single dispatch.
_CONTENT_POS = frozenset({"NOUN", "PROPN", "VERB", "ADJ", "ADV"})


@dataclass(frozen=True)
class Token:
    text: str
    lemma: str
    pos: str
    is_content: bool


@lru_cache(maxsize=1)
def _load_pipeline() -> stanza.Pipeline:
    """Load (and cache) the Hindi Stanza pipeline for this process.

    First use downloads the `hi` (HDTB) models to `~/stanza_resources` if
    they're not already there. Run `python -c "import stanza;
    stanza.download('hi')"` ahead of time (e.g. in the Docker build) to
    avoid eating that delay on the worker's first real job.

    `mwt` is included even though Hindi doesn't need multi-word-token
    expansion in practice -- it's a no-op for languages that don't need it,
    and `lemma` formally depends on it having run, matching the pattern in
    Stanza's own docs for every language, not just ones that need it.
    """
    settings = get_settings()
    return stanza.Pipeline(
        lang="hi",
        processors="tokenize,mwt,pos,lemma",
        use_gpu=settings.stanza_hi_use_gpu,
    )


def tag_hindi(text: str) -> list[Token]:
    """Tokenize + POS-tag + lemmatize Hindi `text` (PROJECT_SPEC.md §5 Stage 3).

    Args:
        text: known-good script text -- Stage 0 intake owns length/content
            validation; this function only refuses truly empty input.

    Returns:
        A flat list of `Token`s across all sentences in `text`, in reading
        order. Sentence boundaries aren't preserved in the return shape --
        nothing downstream (word -> image mapping, captioning) needs them,
        since Stage 2's forced alignment already gives each word its own
        timestamp independent of sentence structure.
    """
    if not text or not text.strip():
        raise ValueError("Cannot tag empty text.")

    nlp = _load_pipeline()
    doc = nlp(text)

    tokens: list[Token] = []
    for sentence in doc.sentences:
        for word in sentence.words:
            pos = word.upos or "X"  # UD's catch-all tag, just in case
            tokens.append(
                Token(
                    text=word.text,
                    lemma=word.lemma or word.text,
                    pos=pos,
                    is_content=pos in _CONTENT_POS,
                )
            )
    return tokens
"""English NLP tagging stage (PROJECT_SPEC.md §5 Stage 3).

Milestone-4 component completing the tagging half of the shared pipeline
that milestone 2 exercised manually: English text in -> a list of unified
`Token`s out, in the exact shape app/pipeline/tokens.py defines and
app/pipeline/tag_hi.py's Hindi backend also produces. Nothing downstream
should know or care which tagger ran (PROJECT_SPEC.md §5 Stage 3).

v1 uses spaCy's `en_core_web_sm` (PROJECT_SPEC.md §9) -- fast, mature
lemmatization and tagging out of the box. spaCy's `Token.pos_` is already
a Universal POS tag (the same tagset Stanza's Hindi pipeline emits via
`upos`), so no tag-translation table is needed here, unlike align.py's
ISO 639-3 mapping.

Setup note: `en_core_web_sm` is a separate download, not a pip extra --
run `python -m spacy download en_core_web_sm` once (e.g. in the Docker
build), same idea as tag_hi.py's `stanza.download('hi')` note.
"""

from __future__ import annotations

from functools import lru_cache

import spacy
from spacy.language import Language

from app.config import get_settings
from app.pipeline.tokens import CONTENT_POS, Token


@lru_cache(maxsize=1)
def _load_pipeline() -> Language:
    """Load (and cache) the English spaCy pipeline for this process --
    same reasoning as every other stage's model cache in this codebase.
    """
    settings = get_settings()
    return spacy.load(settings.spacy_model_id_en)


def tag_english(text: str) -> list[Token]:
    """Tokenize + POS-tag + lemmatize English `text` (PROJECT_SPEC.md §5
    Stage 3).

    Args:
        text: known-good script text -- Stage 0 intake owns length/content
            validation; this function only refuses truly empty input.

    Returns:
        A flat list of `Token`s in reading order, mirroring
        `tag_hi.tag_hindi`'s contract exactly.
    """
    if not text or not text.strip():
        raise ValueError("Cannot tag empty text.")

    nlp = _load_pipeline()
    doc = nlp(text)

    tokens: list[Token] = []
    for token in doc:
        if token.is_space:
            # spaCy emits whitespace as its own token in some edge cases
            # (e.g. double spaces); it carries no timing/content meaning
            # and Stage 2's alignment never produces a corresponding
            # entry for it, so drop it rather than confusing the
            # timeline merge in app/pipeline/timeline.py.
            continue
        pos = token.pos_ or "X"
        tokens.append(
            Token(
                text=token.text,
                lemma=token.lemma_ or token.text,
                pos=pos,
                is_content=pos in CONTENT_POS,
            )
        )
    return tokens
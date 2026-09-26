"""Hindi -> English lemma translation step (PROJECT_SPEC.md §5 Stage 4).

Standalone milestone-3d component. Pexels' index is effectively
English-only (§5), so a Devanagari lemma needs translating to an English
query before it can be used for image search. This module owns exactly
that translation step -- everything else in Stage 4 (checking/writing the
persistent cross-job ImageCache table, calling Pexels, retry-with-context
on a miss, falling back to a placeholder image) is later integration work
once the DB layer (milestone 5) exists.

v1 uses Helsinki-NLP's opus-mt-hi-en (MarianMT, Apache-2.0 -- no
non-commercial restriction, unlike the MMS-TTS checkpoints from Stage 1b).
It's trained on short/conversational text, a reasonable match for single
lemmas and short phrases rather than long-form prose.
"""

from __future__ import annotations

from functools import lru_cache

import torch
from transformers import MarianMTModel, MarianTokenizer

from app.config import get_settings


@lru_cache(maxsize=1)
def _load_model() -> tuple[MarianMTModel, MarianTokenizer]:
    """Load (and cache) the translation model for this process.

    Same reasoning as every other stage's model cache in this codebase:
    this is a real checkpoint, reloading it per lemma would be unusable.
    """
    settings = get_settings()
    tokenizer = MarianTokenizer.from_pretrained(settings.translation_model_id_hi_en)
    model = MarianMTModel.from_pretrained(settings.translation_model_id_hi_en)
    model.eval()
    return model, tokenizer


@lru_cache(maxsize=4096)
def translate_hindi_lemma(lemma: str) -> str:
    """Translate a single Hindi lemma (or short phrase) into an English
    image-search query (PROJECT_SPEC.md §5 Stage 4).

    This in-process `lru_cache` just avoids re-running the model on a
    repeated lemma within one worker process. It's deliberately separate
    from -- and sits in front of -- the persistent, cross-job `ImageCache`
    table (PROJECT_SPEC.md §7) that the full Stage 4 wiring will add later:
    that one survives process restarts and is shared across jobs, this one
    only saves a redundant model call within a single run.

    Args:
        lemma: a single Hindi (Devanagari) lemma, as produced by
            app/pipeline/tag_hi.py's `Token.lemma`. Not validated as
            Devanagari here -- callers route by script (PROJECT_SPEC.md §5
            Stage 3's mixed-language heuristic); this function just
            translates whatever text it's given.

    Returns:
        A short English string suitable as a Pexels search query. Not
        guaranteed non-empty for pathological input -- callers should
        treat an empty/whitespace result as a translation miss and fall
        through to Stage 4's retry/placeholder handling, not as an error
        raised here.
    """
    if not lemma or not lemma.strip():
        raise ValueError("Cannot translate an empty lemma.")

    model, tokenizer = _load_model()
    inputs = tokenizer(lemma, return_tensors="pt", padding=True)
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=16)

    return tokenizer.decode(generated[0], skip_special_tokens=True).strip()
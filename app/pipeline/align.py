"""Stage 2 — Forced alignment (PROJECT_SPEC.md §5).

Aligns a *known* script to its audio and returns word-level timestamps.
Uses MahmoudAshraf97/ctc-forced-aligner (MMS-based) — install via:

    pip install git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git

NOT `pip install ctc-forced-aligner`, which resolves to an unrelated
Deskpai package with a different API. See requirements.txt.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from ctc_forced_aligner import (
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    load_audio,
    postprocess_results,
    preprocess_text,
)

from app.config import get_settings

# ctc-forced-aligner expects ISO 639-3 codes; the rest of this app uses the
# 639-1 codes from PROJECT_SPEC.md §6 (en/hi/mixed). Keep the mapping local
# to this module — nothing downstream needs it.
_ISO_639_3 = {"en": "eng", "hi": "hin"}

# Loading the model re-downloads/re-initializes ~315M params (see the HF
# model card) — cache it per (device, dtype) so repeated calls in one
# process (or one pytest session) don't reload it every time.
_model_cache: dict[str, tuple] = {}


@dataclass(frozen=True)
class WordTiming:
    text: str
    start: float
    end: float
    score: float


def _load_model(device: str, dtype: torch.dtype):
    key = f"{device}:{dtype}"
    if key not in _model_cache:
        _model_cache[key] = load_alignment_model(device, dtype=dtype)
    return _model_cache[key]


def _to_word_timing(raw: dict) -> WordTiming:
    try:
        return WordTiming(
            text=raw["text"],
            start=float(raw["start"]),
            end=float(raw["end"]),
            score=float(raw["score"]),
        )
    except KeyError as exc:
        raise RuntimeError(
            "postprocess_results() returned an unexpected shape: "
            f"{raw!r}. This library's return format has changed across "
            "versions before — check github.com/MahmoudAshraf97/ctc-forced-aligner "
            "for the current schema and update _to_word_timing()."
        ) from exc


def align(script_text: str, audio_path: Path, language: str = "en") -> list[WordTiming]:
    """Force-align `script_text` to `audio_path`, returning word timings.

    `language` is the app's en/hi code. Mixed/code-switched text still goes
    through with romanize=True — the MMS aligner tokenizes on a shared
    romanized vocabulary regardless of script, so this same code path will
    cover `mixed` later without changes (PROJECT_SPEC.md §5 Stage 2).
    """
    if language not in _ISO_639_3:
        raise ValueError(f"Unsupported alignment language: {language!r}")

    settings = get_settings()
    device = settings.alignment_device
    dtype = torch.float16 if device == "cuda" else torch.float32
    model, tokenizer = _load_model(device, dtype)

    waveform = load_audio(str(audio_path), model.dtype, model.device)
    emissions, stride = generate_emissions(
        model, waveform, batch_size=settings.alignment_batch_size
    )

    tokens_starred, text_starred = preprocess_text(
        script_text, romanize=True, language=_ISO_639_3[language]
    )
    segments, scores, blank_token = get_alignments(emissions, tokens_starred, tokenizer)
    spans = get_spans(tokens_starred, segments, blank_token)
    raw_timings = postprocess_results(text_starred, spans, stride, scores)

    return [_to_word_timing(w) for w in raw_timings]
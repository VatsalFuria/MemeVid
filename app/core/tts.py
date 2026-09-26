"""Text-to-speech stage (PROJECT_SPEC.md §5 Stage 1b).

Standalone milestone-3a component: given a script and a language, produce a
WAV file. This becomes the "audio" half of the (script_text, audio_file)
pair that Stage 2 (forced alignment) consumes — identical in shape whether
it came from here or from a user's upload in Flow 1.

v1 uses Meta's MMS-TTS per-language checkpoints (PROJECT_SPEC.md §9). Model
checkpoint IDs are named config (app/config.py), never literals here, so a
benchmarked v2 upgrade (§13) is a config change, not a rewrite.

Known v1 limitation: MMS-TTS checkpoints are monolingual. Flow 2 requires an
explicit language/voice per PROJECT_SPEC.md §2, so "mixed" (Hinglish) input
is not a valid TTS target for v1 — narrate in the underlying `en` or `hi`
voice. Revisit if/when a Hinglish-native TTS checkpoint is evaluated (§13).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import scipy.io.wavfile
import torch
from transformers import AutoTokenizer, VitsModel

from app.config import get_settings

SupportedLanguage = Literal["en", "hi"]


class UnsupportedLanguageError(ValueError):
    """Raised for a language TTS doesn't (yet) support — see module docstring."""


@dataclass(frozen=True)
class TTSResult:
    audio_path: Path
    sample_rate: int
    duration_seconds: float


def _model_id(language: str) -> str:
    settings = get_settings()
    mapping = {
        "en": settings.tts_model_id_en,
        "hi": settings.tts_model_id_hi,
    }
    try:
        return mapping[language]
    except KeyError as exc:
        raise UnsupportedLanguageError(
            f"No MMS-TTS checkpoint configured for language={language!r}. "
            f"Supported: {sorted(mapping)}. 'mixed' is not a valid TTS target in v1 — "
            "pick the dominant language voice for narration (see module docstring)."
        ) from exc


@lru_cache(maxsize=None)
def _load(language: str) -> tuple[VitsModel, AutoTokenizer]:
    """Load (and cache) the VITS model + tokenizer for a language.

    Cached per-process, not per call — these are ~100MB+ checkpoints, and
    reloading on every synthesis call would make the worker unusably slow.
    """
    model_id = _model_id(language)
    model = VitsModel.from_pretrained(model_id)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer


def synthesize_speech(
    text: str,
    language: SupportedLanguage,
    output_path: str | Path,
    *,
    seed: int | None = 0,
) -> TTSResult:
    """Synthesize `text` in `language` to a WAV file at `output_path`.

    Args:
        text: script text, already known-good — Stage 0 intake owns the
            PROJECT_SPEC.md §10 word-cap/content validation; this function
            only refuses truly empty input.
        language: "en" or "hi". "mixed" raises UnsupportedLanguageError.
        output_path: where to write the WAV file. Parent dirs are created.
        seed: the model card notes MMS-TTS's duration predictor is
            stochastic (same text -> different timing each run). Fixing a
            seed makes output reproducible, which matters for a fixed-input
            smoke test. Pass None for natural run-to-run variation.

    Returns:
        TTSResult with the written path, sample rate, and clip duration.
    """
    if not text or not text.strip():
        raise ValueError("Cannot synthesize empty text.")

    model, tokenizer = _load(language)

    if seed is not None:
        torch.manual_seed(seed)

    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        waveform = model(**inputs).waveform

    audio = waveform[0].cpu().numpy()
    sample_rate = model.config.sampling_rate

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.wavfile.write(output_path, rate=sample_rate, data=audio)

    return TTSResult(
        audio_path=output_path,
        sample_rate=sample_rate,
        duration_seconds=len(audio) / sample_rate,
    )
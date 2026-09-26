"""Automatic speech recognition stage (PROJECT_SPEC.md §5 Stage 1a).

Standalone milestone-3b component: given uploaded audio, produce a draft
transcript. This is Flow 1's front door into the shared pipeline — it
produces the same `(script_text, audio_path)` pair that Stage 1b's TTS
(app/core/tts.py) produces for Flow 2. Nothing downstream of that pair
should know or care which front door produced it.

v1 uses faster-whisper — PROJECT_SPEC.md §9's "safe general default first"
pick over a Hinglish-tuned checkpoint, which is a benchmarked v2 upgrade
(§13).

Known v1 limitations (documented, not fixed here):
- Heavy background noise or music degrades transcription (§5 Stage 1a).
- Whisper has no native "mixed" (Hinglish) language code. For `mixed` jobs
  we transcribe without forcing a language and let Whisper's own detection
  do its best on the first window, rather than pretending vanilla Whisper
  is a real code-switch model.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from faster_whisper import WhisperModel

from app.config import get_settings

JobLanguage = Literal["en", "hi", "mixed"]

# Whisper's language codes are ISO 639-1 — same as this app's en/hi codes,
# so no translation table like align.py's ISO 639-3 one is needed. "mixed"
# has no Whisper code; None tells Whisper to auto-detect (see module docstring).
_WHISPER_LANGUAGE: dict[str, str | None] = {
    "en": "en",
    "hi": "hi",
    "mixed": None,
}


@dataclass(frozen=True)
class TranscriptSegment:
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class ASRResult:
    text: str
    segments: list[TranscriptSegment]
    detected_language: str
    detected_language_probability: float


@lru_cache(maxsize=None)
def _load_model() -> WhisperModel:
    """Load (and cache) the Whisper model for this process.

    Cached per-process, not per call — same reasoning as app/core/tts.py's
    per-language model cache: these are large checkpoints, reloading per
    transcription would make the worker unusably slow.
    """
    settings = get_settings()
    # float16 needs a GPU; int8 is the practical CPU default (faster, far
    # less memory, small accuracy cost) — mirrors the device-based dtype
    # branch in app/pipeline/align.py's _load_model().
    compute_type = "float16" if settings.asr_device == "cuda" else "int8"
    return WhisperModel(
        settings.asr_model_size,
        device=settings.asr_device,
        compute_type=compute_type,
    )


def transcribe_audio(audio_path: str | Path, language: JobLanguage) -> ASRResult:
    """Transcribe `audio_path` into a draft script (PROJECT_SPEC.md §5 Stage 1a).

    Args:
        audio_path: path to the uploaded narration audio. Duration-cap
            enforcement (PROJECT_SPEC.md §10) is Stage 0 intake's job, not
            this function's — it only requires the file to exist.
        language: the job's explicitly-selected language (en/hi/mixed —
            PROJECT_SPEC.md §6 has no auto-detect in v1). For en/hi this is
            passed to Whisper to lock decoding to that language, which is
            both faster and more accurate than free auto-detection. For
            `mixed` there's no such code, so Whisper auto-detects (module
            docstring).

    Returns:
        ASRResult with the full transcript, per-segment timings, and
        whatever language Whisper itself detected — surfaced so a caller
        can sanity-check it against the user's selection, though v1 doesn't
        enforce agreement between the two.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = _load_model()
    whisper_language = _WHISPER_LANGUAGE.get(language)

    segments_iter, info = model.transcribe(
        str(audio_path),
        language=whisper_language,
        beam_size=5,
        # Avoids Whisper's known repetition-loop failure mode on
        # silence/noise by not letting each segment's decoding lean on the
        # (possibly garbage) text of the one before it.
        condition_on_previous_text=False,
    )
    segments = [
        TranscriptSegment(text=s.text.strip(), start=s.start, end=s.end)
        for s in segments_iter
    ]

    full_text = " ".join(s.text for s in segments).strip()
    if not full_text:
        raise ValueError(
            f"Whisper produced no speech from {audio_path} — likely silence, "
            "heavy background noise, or an unsupported audio format (see "
            "PROJECT_SPEC.md §5 Stage 1a known limitations)."
        )

    return ASRResult(
        text=full_text,
        segments=segments,
        detected_language=info.language,
        detected_language_probability=info.language_probability,
    )
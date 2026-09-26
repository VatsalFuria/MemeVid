"""Flow orchestration (PROJECT_SPEC.md §2, §5, milestone 4).

Both flows converge on the shared pipeline (align -> tag -> map -> render)
the moment a `(script_text, audio_path)` pair exists (PROJECT_SPEC.md §2's
core architectural principle). This module is that convergence point:

  - `run_shared_pipeline` is the flow-agnostic align->tag->map->render
    pass -- the "zip" app/pipeline/render.py's module docstring calls
    "milestone 4's wiring job".
  - `run_flow1` is Flow 1's front door: ASR (app/core/asr.py) produces the
    script half of the pair from uploaded audio, using the *original
    uploaded audio* (PROJECT_SPEC.md §2).
  - `run_flow2` is Flow 2's front door: TTS (app/core/tts.py) produces the
    audio half of the pair from a given script, using the *generated*
    audio (PROJECT_SPEC.md §2).

`scripts/run_flow1.py` / `scripts/run_flow2.py` are thin CLI wrappers
around these two functions (milestone 4's "wired end-to-end via CLI" DoD)
-- the orchestration lives here so it's importable and testable without
argparse (see tests/test_flows.py).

Known v1 gap -- `language="mixed"` is not wired through yet: Stage 1b's
TTS has no mixed voice (app/core/tts.py's `UnsupportedLanguageError`) and
Stage 2's forced alignment doesn't accept "mixed" either, despite
align.py's own docstring anticipating it (its `_ISO_639_3` map only has
"en"/"hi"). Both flows below raise a clear error on "mixed" rather than
silently mishandling it -- closing this gap is follow-up work, not part
of milestone 4's "both languages" (en/hi) DoD.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal

import httpx

from app.core.asr import ASRResult, transcribe_audio
from app.core.tts import synthesize_speech
from app.pipeline.align import align
from app.pipeline.map_images import ImageCacheStore
from app.pipeline.render import RenderResult, render_video
from app.pipeline.tag import tag_text
from app.pipeline.timeline import build_render_words

FlowLanguage = Literal["en", "hi", "mixed"]

_ALIGN_SUPPORTED = {"en", "hi"}


def run_shared_pipeline(
    script_text: str,
    audio_path: str | Path,
    language: FlowLanguage,
    output_path: str | Path,
    *,
    cache: ImageCacheStore | None = None,
    client: httpx.Client | None = None,
) -> RenderResult:
    """The flow-agnostic align -> tag -> map -> render pass
    (PROJECT_SPEC.md §2, §5). Both flows call this once they each have a
    `(script_text, audio_path)` pair, however they produced it.
    """
    if language not in _ALIGN_SUPPORTED:
        raise NotImplementedError(
            f"Shared pipeline doesn't support language={language!r} yet -- "
            f"forced alignment (Stage 2) only covers {sorted(_ALIGN_SUPPORTED)} "
            "in v1 (see this module's docstring)."
        )

    audio_path = Path(audio_path)
    output_path = Path(output_path)

    timings = align(script_text, audio_path, language=language)
    tokens = tag_text(script_text, language)
    render_words = build_render_words(script_text, timings, tokens, cache=cache, client=client)
    return render_video(render_words, audio_path, output_path)


def run_flow1(
    audio_path: str | Path,
    language: FlowLanguage,
    output_path: str | Path,
    *,
    review_callback: Callable[[ASRResult], str] | None = None,
    cache: ImageCacheStore | None = None,
    client: httpx.Client | None = None,
) -> tuple[ASRResult, RenderResult]:
    """Flow 1 — audio-first (PROJECT_SPEC.md §2).

    Args:
        review_callback: the optional user-edit checkpoint
            (PROJECT_SPEC.md §5 Stage 1a / §8's
            `PATCH /jobs/{id}/transcript`) -- given the raw `ASRResult`,
            returns the script text to actually use. Defaults to the raw
            transcript unchanged. `scripts/run_flow1.py` is where a human
            actually gets to review/edit it.

    Returns:
        The raw `ASRResult` alongside the final `RenderResult`.
    """
    audio_path = Path(audio_path)
    asr_result = transcribe_audio(audio_path, language=language)
    script_text = review_callback(asr_result) if review_callback is not None else asr_result.text

    render_result = run_shared_pipeline(
        script_text, audio_path, language, output_path, cache=cache, client=client
    )
    return asr_result, render_result


def run_flow2(
    script_text: str,
    language: FlowLanguage,
    output_path: str | Path,
    *,
    generated_audio_path: str | Path | None = None,
    cache: ImageCacheStore | None = None,
    client: httpx.Client | None = None,
) -> RenderResult:
    """Flow 2 — text-first (PROJECT_SPEC.md §2).

    Args:
        language: TTS (app/core/tts.py) only supports "en"/"hi" in v1 --
            "mixed" raises `UnsupportedLanguageError` there.
        generated_audio_path: where to write Stage 1b's TTS output before
            Stage 2 consumes it. Defaults to a sibling of `output_path`.
    """
    output_path = Path(output_path)
    if generated_audio_path is None:
        generated_audio_path = output_path.with_suffix(".generated.wav")

    tts_result = synthesize_speech(script_text, language, generated_audio_path)
    return run_shared_pipeline(
        script_text, tts_result.audio_path, language, output_path, cache=cache, client=client
    )
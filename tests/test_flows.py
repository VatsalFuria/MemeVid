"""Milestone 4 (PROJECT_SPEC.md §12): both flows wired end-to-end via CLI,
one integration test per flow per language.

Real ASR/TTS/alignment/tagging/render all run for real — `slow` (ML
models) and `ffmpeg`-marked (shells out to a real ffmpeg binary). Word ->
image mapping is stubbed to a local fixture image so these tests don't
also need a live Pexels credential/budget (already covered by
tests/test_map_images.py) -- skip with `-m "not slow and not ffmpeg"`.
"""

from pathlib import Path

import pytest

from app.core.tts import synthesize_speech
from app.flows import run_flow1, run_flow2
from app.pipeline import timeline
from app.pipeline.map_images import ImageResult

pytestmark = [pytest.mark.slow, pytest.mark.ffmpeg]

FIXTURES = Path(__file__).parent / "fixtures"

_HINDI_SCRIPT = "यह एक छोटा परीक्षण वाक्य है।"  # "This is a short test sentence." (tests/test_tts.py)

# A hardcoded, minimal valid 1x1 PNG (same fixture as tests/test_render.py).
_TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02"
    b"\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x00\x03"
    b"\x00\x01\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
)


@pytest.fixture(autouse=True)
def _stub_image_mapping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub Stage 4 entirely -- these wiring tests don't need a live
    Pexels credential/budget. Every content word maps to one fixture image.
    """
    image_path = tmp_path / "stub.png"
    image_path.write_bytes(_TINY_PNG)

    def _fake_map_word_to_image(lemma, *, context_lemma=None, cache=None, client=None):
        return ImageResult(lemma=lemma, query=lemma, source="placeholder", location=str(image_path))

    monkeypatch.setattr(timeline, "map_word_to_image", _fake_map_word_to_image)


def test_flow2_english(tmp_path: Path) -> None:
    script_text = (FIXTURES / "sample_en" / "script.txt").read_text(encoding="utf-8").strip()
    output_path = tmp_path / "flow2_en.mp4"

    result = run_flow2(script_text, "en", output_path)

    assert result.output_path == output_path
    assert output_path.exists() and output_path.stat().st_size > 0


def test_flow2_hindi(tmp_path: Path) -> None:
    output_path = tmp_path / "flow2_hi.mp4"

    result = run_flow2(_HINDI_SCRIPT, "hi", output_path)

    assert output_path.exists() and output_path.stat().st_size > 0
    assert result.segment_count > 0


def test_flow1_english(tmp_path: Path) -> None:
    audio_path = FIXTURES / "sample_en" / "audio.wav"
    output_path = tmp_path / "flow1_en.mp4"

    asr_result, render_result = run_flow1(audio_path, "en", output_path)

    assert asr_result.text.strip() != ""
    assert output_path.exists() and output_path.stat().st_size > 0


def test_flow1_hindi(tmp_path: Path) -> None:
    """No pre-recorded Hindi fixture exists yet (same gap
    tests/test_asr.py's roundtrip test documents) -- manufacture one via
    TTS rather than requiring a checked-in binary asset.
    """
    audio_path = tmp_path / "hi_source.wav"
    synthesize_speech(_HINDI_SCRIPT, "hi", audio_path)
    output_path = tmp_path / "flow1_hi.mp4"

    asr_result, render_result = run_flow1(audio_path, "hi", output_path)

    assert asr_result.text.strip() != ""
    assert output_path.exists() and output_path.stat().st_size > 0
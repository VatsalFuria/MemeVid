from pathlib import Path

import pytest

from app.core.asr import transcribe_audio
from app.core.tts import synthesize_speech

pytestmark = pytest.mark.slow  # downloads + runs a real Whisper checkpoint

FIXTURES = Path(__file__).parent / "fixtures"


def test_transcribe_audio_english_fixture() -> None:
    """Uses the real recorded clip from milestone 2 (tests/fixtures/sample_en)."""
    audio_path = FIXTURES / "sample_en" / "audio.wav"

    result = transcribe_audio(audio_path, language="en")

    assert result.detected_language == "en"
    assert result.detected_language_probability > 0.5
    lowered = result.text.lower()
    for keyword in ("cat", "sunglasses", "skateboard"):
        assert keyword in lowered


def test_transcribe_audio_hindi_roundtrip(tmp_path: Path) -> None:
    """No pre-recorded Hindi fixture exists yet, so this test manufactures
    one via Stage 1b's TTS instead of requiring a checked-in binary asset.
    It's a roundtrip smoke test, not a WER benchmark — TTS -> ASR is lossy,
    so we only assert the pipeline runs and lands on the right language,
    not that it reproduces the input text exactly.
    """
    known_text = "यह एक छोटा परीक्षण वाक्य है।"
    audio_path = tmp_path / "hi.wav"
    synthesize_speech(known_text, "hi", audio_path)

    result = transcribe_audio(audio_path, language="hi")

    assert result.detected_language == "hi"
    assert result.text.strip() != ""


def test_transcribe_audio_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        transcribe_audio(tmp_path / "does-not-exist.wav", language="en")
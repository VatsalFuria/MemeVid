import wave
from pathlib import Path

import pytest

from app.core.tts import UnsupportedLanguageError, synthesize_speech

pytestmark = pytest.mark.slow  # downloads + runs a real MMS-TTS checkpoint


@pytest.mark.parametrize(
    "language,text",
    [
        ("en", "This is a short test sentence."),
        ("hi", "यह एक छोटा परीक्षण वाक्य है।"),
    ],
)
def test_synthesize_speech_produces_valid_wav(tmp_path: Path, language: str, text: str) -> None:
    output_path = tmp_path / f"{language}.wav"

    result = synthesize_speech(text, language, output_path)

    assert result.audio_path == output_path
    assert output_path.exists()
    assert result.sample_rate > 0
    assert result.duration_seconds > 0

    with wave.open(str(output_path), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getframerate() == result.sample_rate
        assert wav_file.getnframes() > 0


def test_synthesize_speech_rejects_empty_text(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        synthesize_speech("   ", "en", tmp_path / "out.wav")


def test_synthesize_speech_rejects_mixed_language(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedLanguageError):
        synthesize_speech("hello", "mixed", tmp_path / "out.wav")  # type: ignore[arg-type]
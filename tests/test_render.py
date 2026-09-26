"""Stage 5 — video assembly & render (PROJECT_SPEC.md §5 Stage 5).

Segment-grouping, concat-file, and SRT-generation logic are pure and
tested without ffmpeg or the network — fake image paths are enough. The
one `ffmpeg`-marked test actually shells out and needs a real ffmpeg
binary on PATH — skip it with `-m "not ffmpeg"`.
"""

from pathlib import Path

import pytest

from app.pipeline import render
from app.pipeline.render import RenderWord


def test_build_segments_rejects_empty_words() -> None:
    with pytest.raises(ValueError):
        render._build_segments([], {}, Path("/imgs/leading.jpg"))


def test_build_segments_rejects_out_of_order_words() -> None:
    words = [
        RenderWord(text="b", start=1.0, end=1.5, image_location="loc-b"),
        RenderWord(text="a", start=0.0, end=0.5, image_location="loc-a"),
    ]
    with pytest.raises(ValueError):
        render._build_segments(words, {"loc-a": Path("a.jpg"), "loc-b": Path("b.jpg")}, Path("/imgs/leading.jpg"))


def test_build_segments_carries_image_forward_across_non_content_words() -> None:
    words = [
        RenderWord(text="A", start=0.0, end=0.1),
        RenderWord(text="giant", start=0.1, end=0.4, image_location="loc-giant"),
        RenderWord(text="cat", start=0.4, end=0.7, image_location="loc-cat"),
        RenderWord(text="wearing", start=0.7, end=0.9),
        RenderWord(text="sunglasses", start=0.9, end=1.4, image_location="loc-sunglasses"),
    ]
    image_paths = {
        "loc-giant": Path("/imgs/giant.jpg"),
        "loc-cat": Path("/imgs/cat.jpg"),
        "loc-sunglasses": Path("/imgs/sunglasses.jpg"),
    }
    leading = Path("/imgs/leading-placeholder.jpg")

    segments = render._build_segments(words, image_paths, leading)

    assert [s.image_path for s in segments] == [
        leading,
        Path("/imgs/giant.jpg"),
        Path("/imgs/cat.jpg"),  # "wearing" (non-content) carries this forward
        Path("/imgs/sunglasses.jpg"),
    ]
    cat_segment = segments[2]
    assert [w.text for w in cat_segment.words] == ["cat", "wearing"]
    assert cat_segment.start == 0.4
    assert cat_segment.end == 0.9


def test_build_segments_inserts_leading_gap_when_audio_starts_before_first_word() -> None:
    words = [RenderWord(text="giant", start=0.14, end=0.44, image_location="loc-giant")]
    image_paths = {"loc-giant": Path("/imgs/giant.jpg")}
    leading = Path("/imgs/leading-placeholder.jpg")

    segments = render._build_segments(words, image_paths, leading)

    assert len(segments) == 2
    assert segments[0] == render.RenderSegment(image_path=leading, start=0.0, end=pytest.approx(0.14), words=())
    assert segments[1].image_path == Path("/imgs/giant.jpg")


def test_build_concat_content_repeats_last_file_without_duration() -> None:
    words = [
        RenderWord(text="a", start=0.0, end=0.5, image_location="loc-a"),
        RenderWord(text="b", start=0.5, end=1.2, image_location="loc-b"),
    ]
    image_paths = {"loc-a": Path("/imgs/a.jpg"), "loc-b": Path("/imgs/b.jpg")}
    segments = render._build_segments(words, image_paths, Path("/imgs/leading.jpg"))

    content = render._build_concat_content(segments)
    lines = content.strip().splitlines()

    assert lines[0] == "ffconcat version 1.0"
    assert lines[-1] == "file '/imgs/b.jpg'"
    assert "duration 0.500" in content
    assert "duration 0.700" in content


def test_build_concat_content_rejects_non_positive_duration() -> None:
    words = [RenderWord(text="x", start=1.0, end=1.0, image_location="loc")]
    image_paths = {"loc": Path("/imgs/x.jpg")}
    segments = render._build_segments(words, image_paths, Path("/imgs/leading.jpg"))
    with pytest.raises(ValueError):
        render._build_concat_content(segments)


def test_quote_concat_path_escapes_single_quotes() -> None:
    assert render._quote_concat_path(Path("/imgs/o'brien.jpg")) == "'/imgs/o'\\''brien.jpg'"


def test_build_srt_content_one_card_per_word() -> None:
    words = [
        RenderWord(text="giant", start=0.14, end=0.44),
        RenderWord(text="cat,", start=0.48, end=0.76),
    ]
    srt = render._build_srt_content(words)

    assert srt.strip().splitlines()[0] == "1"
    assert "00:00:00,140 --> 00:00:00,440" in srt
    assert "00:00:00,480 --> 00:00:00,760" in srt
    assert "giant" in srt and "cat," in srt


def test_materialize_images_downloads_urls_and_passes_through_local_paths(tmp_path: Path) -> None:
    local_dog = tmp_path / "local-dog.jpg"
    local_dog.write_bytes(b"fake")
    words = [
        RenderWord(text="cat", start=0.0, end=0.5, image_location="https://images.pexels.com/cat.jpg"),
        RenderWord(text="dog", start=0.5, end=1.0, image_location=str(local_dog)),
    ]

    class _FakeResponse:
        content = b"fake-downloaded-bytes"

        def raise_for_status(self) -> None:
            return None

    class _FakeClient:
        def get(self, url: str, timeout: float) -> "_FakeResponse":
            return _FakeResponse()

    resolved = render._materialize_images(
        words, work_dir=tmp_path, client=_FakeClient(), fallback_image_path=Path("/imgs/leading.jpg")
    )

    assert resolved["https://images.pexels.com/cat.jpg"].read_bytes() == b"fake-downloaded-bytes"
    assert resolved[str(local_dog)] == local_dog


def test_materialize_images_falls_back_on_download_failure(tmp_path: Path) -> None:
    import httpx

    words = [RenderWord(text="cat", start=0.0, end=0.5, image_location="https://images.pexels.com/dead.jpg")]

    class _FailingClient:
        def get(self, url: str, timeout: float):
            raise httpx.ConnectError("boom")

    fallback = Path("/imgs/leading.jpg")
    resolved = render._materialize_images(words, work_dir=tmp_path, client=_FailingClient(), fallback_image_path=fallback)

    assert resolved["https://images.pexels.com/dead.jpg"] == fallback


# A hardcoded, minimal valid 1x1 PNG — avoids depending on Pillow (not a
# project dependency) just to produce a throwaway fixture image.
_TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02"
    b"\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x00\x03"
    b"\x00\x01\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
)


@pytest.mark.ffmpeg
def test_render_video_produces_a_playable_mp4(tmp_path: Path) -> None:
    """Fixed-input smoke test (PROJECT_SPEC.md §12 milestone 2) — reuses
    the milestone-2 English sample's audio (tests/fixtures/sample_en) and
    its already-computed alignment timings (timings.json at repo root),
    with a hand-picked subset marked as content words pointing at one
    tiny local image. Needs ffmpeg but no network/Pexels credential.
    """
    image_path = tmp_path / "placeholder.png"
    image_path.write_bytes(_TINY_PNG)

    content_words = {"giant", "cat", "wearing", "sunglasses", "rides", "skateboard", "busy", "city", "street."}
    raw_timings = [
        {"text": "giant", "start": 0.14, "end": 0.44},
        {"text": "cat,", "start": 0.48, "end": 0.76},
        {"text": "A", "start": 1.18, "end": 1.2},
        {"text": "giant", "start": 1.28, "end": 1.6},
        {"text": "cat", "start": 1.64, "end": 1.92},
        {"text": "wearing", "start": 2.0, "end": 2.28},
        {"text": "sunglasses", "start": 2.34, "end": 2.96},
        {"text": "rides", "start": 3.06, "end": 3.28},
        {"text": "a", "start": 3.32, "end": 3.34},
        {"text": "skateboard", "start": 3.42, "end": 3.98},
        {"text": "down", "start": 4.12, "end": 4.34},
        {"text": "a", "start": 4.42, "end": 4.44},
        {"text": "busy", "start": 4.56, "end": 4.78},
        {"text": "city", "start": 4.86, "end": 5.04},
        {"text": "street.", "start": 5.1, "end": 5.74},
    ]
    words = [
        RenderWord(
            text=t["text"],
            start=t["start"],
            end=t["end"],
            image_location=str(image_path) if t["text"].rstrip(",") in content_words else None,
        )
        for t in raw_timings
    ]

    audio_path = Path(__file__).parent / "fixtures" / "sample_en" / "audio.wav"
    output_path = tmp_path / "out.mp4"

    result = render.render_video(words, audio_path, output_path)

    assert result.output_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert result.duration_seconds == pytest.approx(5.74)
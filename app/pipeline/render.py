"""Video assembly & render stage (PROJECT_SPEC.md §5 Stage 5).

Given the fully-materialized per-word timeline — Stage 2's (align.py)
word timings, zipped with Stage 3's (tag_hi.py / the English tagger)
is_content flags, zipped with Stage 4's (map_images.py) image locations
for content words — that zip is milestone 4's wiring job, not this
module's (PROJECT_SPEC.md §11: "build each new component ... as an
independently testable stage before wiring flows together"). This stage:

  1. Builds an ffmpeg concat-demuxer file sequencing one image per content
     word for exactly that word's [start, end) duration, carrying the
     most recent content word's image forward across any non-content
     words in between (PROJECT_SPEC.md §5 Stage 3: "everything else is
     captioned but doesn't trigger an image change").
  2. Builds an SRT subtitle file from the same per-word timestamps, one
     word per caption card.
  3. Runs a single ffmpeg call combining the concat-sequenced images, the
     burned-in subtitle track, and the audio track into the final MP4.

Devanagari requirement (PROJECT_SPEC.md §5, §10): the `subtitles` filter
renders through libass, which needs a font that actually covers
Devanagari glyphs (e.g. Noto Sans Devanagari) available to it via
`settings.subtitle_fonts_dir` — confirm it's present in the Docker image
before considering Hindi output "done"; otherwise captions render as
boxes.
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import httpx

from app.config import Settings, get_settings
from app.pipeline.map_images import placeholder_location

logger = logging.getLogger(__name__)

# Below this, a gap/segment isn't worth a distinct ffmpeg concat entry —
# guards against a near-zero duration rounding to "0.000" once formatted
# to 3 decimal places, which would otherwise look like a real error.
_MIN_SEGMENT_DURATION_SECONDS = 0.001


@dataclass(frozen=True)
class RenderWord:
    """One word, timed and (maybe) paired with an image.

    `image_location` is Stage 4's `ImageResult.location` for content words
    and `None` for non-content words — PROJECT_SPEC.md §5 Stage 3: a
    non-content word is captioned but "doesn't trigger an image change".
    """

    text: str
    start: float
    end: float
    image_location: str | None = None


@dataclass(frozen=True)
class RenderSegment:
    """One contiguous on-screen image window, spanning one or more
    `RenderWord`s (a content word plus any trailing non-content words that
    carried its image forward, or a synthetic lead-in/leading-placeholder
    window with no words of its own — see `_build_segments`).
    """

    image_path: Path
    start: float
    end: float
    words: tuple[RenderWord, ...]


@dataclass(frozen=True)
class RenderResult:
    output_path: Path
    duration_seconds: float  # approximate: the last word's end timestamp,
    # i.e. the video track's own length before ffmpeg's `-shortest`
    # reconciles it against the audio track's actual length.
    segment_count: int


def _build_segments(
    words: Sequence[RenderWord],
    image_paths: dict[str, Path],
    leading_image_path: Path,
) -> list[RenderSegment]:
    """Group `words` into image segments, carrying the most recent content
    word's image forward across intervening non-content words.

    Two edge cases both fall back to `leading_image_path`, a neutral
    placeholder, rather than failing:
      - a script that *opens* with one or more non-content words (e.g.
        "The cat..."), before any content word's image exists yet to
        carry forward;
      - a real gap between t=0 and the first word's timestamp (silence or
        lead-in before speech starts). The ffmpeg concat demuxer only
        knows relative *durations*, not absolute timestamps — without an
        explicit lead-in segment the image track would start ahead of the
        audio it's meant to match by however long that gap is.
    """
    if not words:
        raise ValueError("Cannot render zero words.")
    for prev, curr in zip(words, words[1:]):
        if curr.start < prev.start:
            raise ValueError(
                f"words must be in non-decreasing start-time order; "
                f"{curr.text!r} (start={curr.start}) comes after "
                f"{prev.text!r} (start={prev.start})."
            )

    segments: list[RenderSegment] = []
    current_path: Path | None = None
    current_words: list[RenderWord] = []

    def _flush() -> None:
        if not current_words:
            return
        segments.append(
            RenderSegment(
                image_path=current_path or leading_image_path,
                start=current_words[0].start,
                end=current_words[-1].end,
                words=tuple(current_words),
            )
        )

    for word in words:
        if word.image_location is not None:
            _flush()
            current_words = [word]
            current_path = image_paths[word.image_location]
        else:
            current_words.append(word)
    _flush()

    if segments and segments[0].start > _MIN_SEGMENT_DURATION_SECONDS:
        segments.insert(
            0,
            RenderSegment(image_path=leading_image_path, start=0.0, end=segments[0].start, words=()),
        )

    return segments


def _quote_concat_path(path: Path) -> str:
    """Quote a path for ffmpeg's concat-demuxer `file` directive. A
    literal single quote in the path is escaped as `'\\''` — close the
    quoted string, an escaped quote, reopen — the standard trick the
    concat demuxer's own docs recommend for this exact case.
    """
    escaped = str(path).replace("'", "'\\''")
    return f"'{escaped}'"


def _build_concat_content(segments: Sequence[RenderSegment]) -> str:
    if not segments:
        raise ValueError("Cannot build a concat file from zero segments.")

    lines = ["ffconcat version 1.0"]
    for segment in segments:
        duration = segment.end - segment.start
        if duration < _MIN_SEGMENT_DURATION_SECONDS:
            raise ValueError(
                f"Segment for image {segment.image_path} has a non-positive or "
                f"negligible duration ({duration:.4f}s, start={segment.start}, "
                f"end={segment.end}) — check upstream word timings."
            )
        lines.append(f"file {_quote_concat_path(segment.image_path)}")
        lines.append(f"duration {duration:.3f}")

    # ffmpeg's concat demuxer ignores the *last* duration directive unless
    # the final file is repeated once more without one — a well-documented
    # quirk of the format, not a bug here.
    lines.append(f"file {_quote_concat_path(segments[-1].image_path)}")
    return "\n".join(lines) + "\n"


def _format_srt_timestamp(seconds: float) -> str:
    seconds = max(seconds, 0.0)
    total_ms = round(seconds * 1000)
    hours, remainder_ms = divmod(total_ms, 3_600_000)
    minutes, remainder_ms = divmod(remainder_ms, 60_000)
    secs, ms = divmod(remainder_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _build_srt_content(words: Sequence[RenderWord]) -> str:
    """One caption card per word, using the exact same per-word
    timestamps Stage 2's forced alignment produced — captions and image
    changes are both driven off one shared timeline, never two
    independently-drifting ones.
    """
    if not words:
        raise ValueError("Cannot build subtitles from zero words.")

    lines: list[str] = []
    for index, word in enumerate(words, start=1):
        lines.append(str(index))
        lines.append(f"{_format_srt_timestamp(word.start)} --> {_format_srt_timestamp(word.end)}")
        lines.append(word.text)
        lines.append("")
    return "\n".join(lines)


def _is_url(location: str) -> bool:
    return location.startswith("http://") or location.startswith("https://")


def _download_image(url: str, *, work_dir: Path, client: httpx.Client) -> Path:
    settings = get_settings()
    response = client.get(url, timeout=settings.render_image_download_timeout_seconds)
    response.raise_for_status()
    suffix = Path(httpx.URL(url).path).suffix or ".jpg"
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    dest = work_dir / f"{digest}{suffix}"
    dest.write_bytes(response.content)
    return dest


def _materialize_images(
    words: Sequence[RenderWord],
    *,
    work_dir: Path,
    client: httpx.Client,
    fallback_image_path: Path,
) -> dict[str, Path]:
    """Resolve every distinct `image_location` in `words` to a local file
    ffmpeg can read. Pexels locations (Stage 4) are URLs and get
    downloaded once each into `work_dir`; placeholder locations (Stage 4)
    are already local filesystem paths and are used as-is.

    A download failure falls back to `fallback_image_path` rather than
    failing the whole render — PROJECT_SPEC.md §10's "one bad word never
    fails the whole render" applies just as much to a since-expired Pexels
    URL as to a Stage 4 zero-result miss.
    """
    resolved: dict[str, Path] = {}
    for word in words:
        location = word.image_location
        if location is None or location in resolved:
            continue
        if _is_url(location):
            try:
                resolved[location] = _download_image(location, work_dir=work_dir, client=client)
            except httpx.HTTPError as exc:
                logger.warning("Failed to download image %r for rendering: %s", location, exc)
                resolved[location] = fallback_image_path
        else:
            resolved[location] = Path(location)
    return resolved


def _escape_ffmpeg_filter_path(path: Path) -> str:
    """Escape a filesystem path for use inside an ffmpeg filtergraph
    argument (e.g. the `subtitles=...` filter). The filtergraph parser
    treats `:` as an option separator and `\\`/`'` as escape characters,
    so a path containing any of these (a Windows drive letter, or a POSIX
    path with an apostrophe) breaks the filter otherwise.
    """
    return str(path).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def _build_ffmpeg_command(
    *,
    concat_path: Path,
    audio_path: Path,
    srt_path: Path,
    output_path: Path,
    settings: Settings,
) -> list[str]:
    width, height = settings.render_width, settings.render_height
    subtitles_arg = (
        f"subtitles={_escape_ffmpeg_filter_path(srt_path)}"
        f":fontsdir={_escape_ffmpeg_filter_path(settings.subtitle_fonts_dir)}"
        f":force_style='FontName={settings.subtitle_font_name},"
        f"FontSize={settings.subtitle_font_size},Alignment=2,BorderStyle=1,Outline=1,Shadow=0'"
    )
    video_filter = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"setsar=1,{subtitles_arg}"
    )
    return [
        settings.ffmpeg_binary,
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_path),
        "-i", str(audio_path),
        "-vf", video_filter,
        "-c:v", settings.render_video_codec,
        "-crf", str(settings.render_crf),
        "-pix_fmt", "yuv420p",
        "-c:a", settings.render_audio_codec,
        "-shortest",
        str(output_path),
    ]


def render_video(
    words: Sequence[RenderWord],
    audio_path: str | Path,
    output_path: str | Path,
    *,
    work_dir: str | Path | None = None,
    client: httpx.Client | None = None,
) -> RenderResult:
    """Assemble the final MP4 (PROJECT_SPEC.md §5 Stage 5).

    Args:
        words: the full per-word timeline, in start-time order, each
            optionally carrying a Stage 4 image location (module
            docstring). Producing this from Stage 2/3/4's separate
            outputs is milestone 4's job, not this function's.
        audio_path: the shared (script, audio) pair's audio — the
            original upload for Flow 1, or Stage 1b's TTS output for
            Flow 2 (both already the same shape by the time this runs).
        output_path: where to write the final MP4. Parent dirs are
            created.
        work_dir: scratch directory for downloaded Pexels images and the
            generated concat/SRT files. Defaults to a temp directory
            cleaned up automatically afterward.
        client: reuse an httpx.Client if you have one; a short-lived one
            is created and closed otherwise.

    Returns:
        RenderResult with the output path, approximate duration, and how
        many distinct image segments were used.

    Raises:
        FileNotFoundError: `audio_path` doesn't exist.
        ValueError: `words` is empty, out of order, or produces a
            non-positive-duration segment.
        RuntimeError: ffmpeg exited non-zero; stderr is included so the
            failure is debuggable without re-running by hand.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    settings = get_settings()
    leading_image_path = Path(placeholder_location("__leading__"))

    owns_client = client is None
    client = client or httpx.Client(follow_redirects=True)
    owns_work_dir = work_dir is None
    work_dir_ctx = tempfile.TemporaryDirectory() if owns_work_dir else None
    resolved_work_dir = Path(work_dir_ctx.name) if work_dir_ctx else Path(work_dir)
    resolved_work_dir.mkdir(parents=True, exist_ok=True)

    try:
        image_paths = _materialize_images(
            words, work_dir=resolved_work_dir, client=client, fallback_image_path=leading_image_path
        )
        segments = _build_segments(words, image_paths, leading_image_path)

        concat_path = resolved_work_dir / "concat.ffconcat"
        concat_path.write_text(_build_concat_content(segments), encoding="utf-8")

        srt_path = resolved_work_dir / "captions.srt"
        srt_path.write_text(_build_srt_content(words), encoding="utf-8")

        command = _build_ffmpeg_command(
            concat_path=concat_path,
            audio_path=audio_path,
            srt_path=srt_path,
            output_path=output_path,
            settings=settings,
        )
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(
                f"ffmpeg exited with code {completed.returncode}.\n"
                f"Command: {' '.join(command)}\n"
                f"stderr (tail):\n{completed.stderr[-4000:]}"
            )
    finally:
        if owns_client:
            client.close()
        if work_dir_ctx:
            work_dir_ctx.cleanup()

    return RenderResult(
        output_path=output_path,
        duration_seconds=words[-1].end,
        segment_count=len(segments),
    )
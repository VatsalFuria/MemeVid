"""Manual runner for Stage 5 only (PROJECT_SPEC.md §12 milestone 2).

Takes a JSON file of already-timed, already-image-mapped words (the shape
milestone 4 will eventually produce by zipping Stage 2 + 3 + 4's outputs)
and renders the final MP4 — lets Stage 5 be exercised standalone, same as
scripts/run_alignment.py does for Stage 2.

Usage:
    python scripts/run_render.py \
        --words path/to/render_words.json \
        --audio tests/fixtures/sample_en/audio.wav \
        --out out/sample.mp4

Each entry in --words is:
    {"text": ..., "start": ..., "end": ..., "image_location": ... | null}
"""
import argparse
import json
from pathlib import Path

from app.pipeline.render import RenderWord, render_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 5 (render) on a known words+images timeline.")
    parser.add_argument("--words", type=Path, required=True, help="JSON list of timed, image-mapped words")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    raw_words = json.loads(args.words.read_text(encoding="utf-8"))
    words = [
        RenderWord(
            text=w["text"],
            start=float(w["start"]),
            end=float(w["end"]),
            image_location=w.get("image_location"),
        )
        for w in raw_words
    ]

    result = render_video(words, args.audio, args.out)
    print(
        f"Wrote ~{result.duration_seconds:.2f}s video "
        f"({result.segment_count} image segments) to {result.output_path}"
    )


if __name__ == "__main__":
    main()
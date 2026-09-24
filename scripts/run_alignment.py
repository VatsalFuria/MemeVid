"""Manual runner for Stage 2 only (PROJECT_SPEC.md §12 milestone 2).

Usage:
    python scripts/run_alignment.py \
        --audio tests/fixtures/sample_en/audio.wav \
        --script tests/fixtures/sample_en/script.txt \
        --language en
"""
import argparse
import json
from pathlib import Path

from app.pipeline.align import align


def main() -> None:
    parser = argparse.ArgumentParser(description="Run forced alignment on a known (script, audio) pair.")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True, help="Path to a .txt file with the exact spoken script")
    parser.add_argument("--language", default="en", choices=["en", "hi"])
    parser.add_argument("--out", type=Path, default=None, help="Optional path to write timings as JSON")
    args = parser.parse_args()

    script_text = args.script.read_text(encoding="utf-8").strip()
    timings = align(script_text, args.audio, language=args.language)

    for w in timings:
        print(f"{w.start:6.2f}s -> {w.end:6.2f}s  ({w.score:.2f})  {w.text}")

    if args.out:
        args.out.write_text(
            json.dumps([w.__dict__ for w in timings], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nWrote {len(timings)} word timings to {args.out}")


if __name__ == "__main__":
    main()
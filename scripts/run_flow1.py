"""Manual runner for Flow 1 — audio-first (PROJECT_SPEC.md §2, milestone 4).

Usage:
    python scripts/run_flow1.py \
        --audio path/to/narration.wav \
        --language en \
        --out out/flow1.mp4

By default, prints the ASR draft transcript and asks for confirmation
before alignment proceeds (PROJECT_SPEC.md §5 Stage 1a's optional
transcript-review checkpoint). Pass --auto-accept to skip the prompt and
proceed with the raw transcript as-is -- useful for scripted/CI runs.
"""
import argparse
from pathlib import Path

from app.core.asr import ASRResult
from app.flows import run_flow1


def _review_transcript(draft: str) -> str:
    print("\n--- Draft transcript (Stage 1a — ASR) ---")
    print(draft)
    print("---\n")
    choice = input("Proceed with this transcript? [Y]es / [e]dit / paste a replacement: ").strip()
    if choice.lower() in ("", "y", "yes"):
        return draft
    if choice.lower() in ("e", "edit"):
        print("Enter the corrected script, then an empty line to finish:")
        lines: list[str] = []
        while True:
            line = input()
            if not line:
                break
            lines.append(line)
        return "\n".join(lines).strip() or draft
    return choice


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Flow 1 (audio-first) end-to-end.")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--language", default="en", choices=["en", "hi", "mixed"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--auto-accept",
        action="store_true",
        help="Skip the transcript-review prompt and proceed with the raw ASR output.",
    )
    args = parser.parse_args()

    def _review(asr_result: ASRResult) -> str:
        if args.auto_accept:
            return asr_result.text
        return _review_transcript(asr_result.text)

    asr_result, render_result = run_flow1(args.audio, args.language, args.out, review_callback=_review)

    print(f"Detected language: {asr_result.detected_language} ({asr_result.detected_language_probability:.2f})")
    print(
        f"Wrote ~{render_result.duration_seconds:.2f}s video "
        f"({render_result.segment_count} image segments) to {render_result.output_path}"
    )


if __name__ == "__main__":
    main()
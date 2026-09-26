"""Manual runner for Flow 2 — text-first (PROJECT_SPEC.md §2, milestone 4).

Usage:
    python scripts/run_flow2.py \
        --script path/to/script.txt \
        --language en \
        --out out/flow2.mp4
"""
import argparse
from pathlib import Path

from app.flows import run_flow2


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Flow 2 (text-first) end-to-end.")
    parser.add_argument("--script", type=Path, required=True, help="Path to a .txt file with the script")
    parser.add_argument("--language", default="en", choices=["en", "hi"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    script_text = args.script.read_text(encoding="utf-8").strip()
    render_result = run_flow2(script_text, args.language, args.out)

    print(
        f"Wrote ~{render_result.duration_seconds:.2f}s video "
        f"({render_result.segment_count} image segments) to {render_result.output_path}"
    )


if __name__ == "__main__":
    main()
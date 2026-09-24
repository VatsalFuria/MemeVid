"""Shared core pipeline (PROJECT_SPEC.md §5): align -> tag -> map -> render.

Both flows (audio-first and text-first) converge on this package once a
(script_text, audio_path) pair exists. Nothing in here should know or care
which flow produced that pair.
"""
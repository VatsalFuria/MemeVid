from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # External services
    pexels_api_key: str

    # App metadata
    environment: str = "development"

    # v1 input caps (PROJECT_SPEC.md §10) — named constants, never literals downstream
    audio_max_duration_seconds: int = 180   # Flow 1 cap: 3 minutes
    script_max_words: int = 200             # Flow 2 cap: ~200 words

    # Stage 2 — forced alignment (PROJECT_SPEC.md §5, §9)
    alignment_device: str = "cpu"           # set ALIGNMENT_DEVICE=cuda in .env if you have a GPU
    alignment_batch_size: int = 16

    # Runtime data directory (gitignored — see .gitignore). For job artifacts only;
    # fixed test fixtures (e.g. the milestone-2 sample) live in tests/fixtures/ instead,
    # since those need to be committed for CI to see them.
    data_dir: Path = Path("data")

    # TTS (PROJECT_SPEC.md §5 Stage 1b, §9) — MMS-TTS per-language checkpoints.
    # Both are CC-BY-NC-4.0 (non-commercial) as of writing; §10 requires
    # re-confirming licensing before any "real service" use.
    tts_model_id_en: str = "facebook/mms-tts-eng"
    tts_model_id_hi: str = "facebook/mms-tts-hin"

        # Stage 1a — ASR (PROJECT_SPEC.md §5, §9). faster-whisper (CTranslate2
    # reimplementation of Whisper). "large-v3" is the safe default; "turbo"
    # (large-v3-turbo) is a comparably-accurate, much faster CPU-friendlier
    # swap — one config change, no code change, per PROJECT_SPEC.md §11.
    asr_model_size: str = "large-v3"
    asr_device: str = "cpu"          # set ASR_DEVICE=cuda in .env if you have a GPU

        # Stage 3 — Hindi NLP tagging (PROJECT_SPEC.md §5, §9). Stanza's Hindi
    # UD pipeline (HDTB treebank). False mirrors the CPU-safe default used
    # elsewhere (alignment_device, asr_device); flip via env once you've
    # confirmed a local GPU setup.
    stanza_hi_use_gpu: bool = False

    # Stage 3 — Hindi lemma -> English translation for Pexels image search
    # (PROJECT_SPEC.md §5, §9). Apache-2.0, unlike the MMS-TTS checkpoints —
    # no commercial-use caveat to flag here.
    translation_model_id_hi_en: str = "Helsinki-NLP/opus-mt-hi-en"

        # Stage 4 — word -> image mapping (PROJECT_SPEC.md §5 Stage 4, §9)
    pexels_base_url: str = "https://api.pexels.com/v1/search"
    pexels_request_timeout_seconds: float = 10.0
    # Stop calling Pexels once its own rate-limit header reports at or below
    # this many requests left this window; fall back to a placeholder
    # instead of risking a 429 mid-render (PROJECT_SPEC.md §10).
    pexels_min_rate_limit_remaining: int = 5
    # Bundled neutral fallback images (PROJECT_SPEC.md §5 Stage 4) -- ship
    # at least one real image file here; this repo doesn't check in binary
    # assets in this pass, so add them before deploying.
    placeholder_images_dir: Path = Path("app/assets/placeholders")

        # Stage 5 — video assembly & render (PROJECT_SPEC.md §5 Stage 5, §9).
    # Direct ffmpeg calls (concat demuxer + `subtitles` filter) — an
    # external system binary, not a pip dependency; must be on PATH (or
    # point ffmpeg_binary at an absolute path).
    ffmpeg_binary: str = "ffmpeg"
    render_width: int = 1280
    render_height: int = 720
    render_video_codec: str = "libx264"
    render_audio_codec: str = "aac"
    render_crf: int = 20                          # libx264 quality; lower = better/larger
    render_image_download_timeout_seconds: float = 15.0

    # Devanagari requirement (PROJECT_SPEC.md §5, §10): the `subtitles`
    # filter renders through libass, which needs a font actually covering
    # Devanagari glyphs available to it, or Hindi captions render as
    # boxes. Bundle the font file at this path into the Docker image —
    # like placeholder_images_dir above, no binary asset is checked in by
    # this pass.
    subtitle_fonts_dir: Path = Path("app/assets/fonts")
    subtitle_font_name: str = "Noto Sans Devanagari"
    subtitle_font_size: int = 28


@lru_cache
def get_settings() -> Settings:
    return Settings()
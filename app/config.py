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


@lru_cache
def get_settings() -> Settings:
    return Settings()
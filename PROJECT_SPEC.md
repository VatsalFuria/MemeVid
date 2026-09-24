# memevid — Technical Design & Build Guidelines (v2)

**Status:** authoritative spec as of 2026-09-25. Assumes no code exists yet. Read this in full before writing anything — it supersedes any earlier version or partial implementation you may find.

---

## 1. Objective

Build a web application, **memevid**, that turns narration into a word-synced, meme-style visual video. Two entry points are supported:

- give the system **audio**, get a script + video back
- give the system a **script**, get a voiceover + video back

In both cases, every content word (noun, verb, adjective, adverb, proper noun) in the resulting script is depicted on-screen by a relevant image at the exact moment it's spoken, captioned, cut together into a single MP4. **English and Hindi — including natural Hindi-English code-switching ("Hinglish") — are both first-class inputs**, not an English-only system with Hindi bolted on.

This is a portfolio project. Every architectural and library choice must be defensible on its own merits — able to survive "why did you choose X over Y" — and must leave real room to grow into a more complete, multi-user service later without requiring a rewrite to get there. Prototype fast, but don't cut corners that would make the eventual Postgres/multi-user/production version a rebuild instead of a graduation.

---

## 2. The two flows

**Flow 1 — Audio-first.** The user uploads a narration audio file (English, Hindi, or a natural code-switched mix) with no script. The system transcribes it (ASR) into a draft script, optionally lets the user review/edit that transcript, then proceeds through the same shared pipeline as Flow 2, using the **original uploaded audio**.

**Flow 2 — Text-first.** The user provides a script directly (English/Hindi/mixed) plus a target language/voice. The system synthesizes a voiceover (TTS) for that script, then proceeds through the same shared pipeline, using the **generated audio**.

**Core architectural principle:** both flows converge the moment a `(script_text, audio_file)` pair exists. They differ only in _how_ that pair is produced — transcribing existing audio vs. synthesizing new audio. Build the shared pipeline once; treat ASR and TTS purely as two different front doors into it.

```
Flow 1:  audio ──[ASR]──> draft script ──(optional user edit)──┐
                                                                  ├──> SHARED: align(script, audio)
Flow 2:  script ──[TTS]──> generated audio ────────────────────┘           │
                                                                             ▼
                                                       SHARED: NLP tag → image map → render → MP4
```

---

## 3. Scope

**In scope (v1):**

- Both flows, English + Hindi + Hinglish, one script + one audio per job
- User accounts (register/login), jobs owned by a user
- Word-level tokenization + POS tagging, language-aware
- Forced alignment of known script text to audio (word-level timestamps)
- Per-word image retrieval from a stock photo API, with a persistent cross-job cache
- Automated video rendering: sequenced images + burned-in captions (Latin _and_ Devanagari) + audio track
- Async job architecture: background worker, status polling, nothing blocks the request thread
- Postgres-backed persistence via SQLAlchemy + Alembic (SQLite acceptable for local dev only)
- A real, if minimal, frontend: both upload flows, job status, preview, download, login/register
- Dockerized; CI runs tests + lint on every push; a documented, working deploy path

**Out of scope for v1 (explicitly deferred, not forgotten — see §13):**

- Background-music/vocal separation and remixing
- Generative (diffusion) imagery — stock photos only for now
- LLM-driven scene composition / smarter query generation
- Stock video clips — photos only
- Languages beyond English/Hindi
- Social/OAuth login, password reset, billing, quotas beyond basic rate limiting
- A public gallery of past outputs

**Assumptions baked into this doc — flag any that are wrong before building against them:**

- v1 caps: audio ≤ 3 minutes, script ≤ ~200 words (tune later; keep as named constants, never literals)
- Language is **explicitly selected** by the user (`en` / `hi` / `mixed`) rather than auto-detected in v1
- Flow 1 always offers an optional transcript-review step before rendering proceeds
- "Multi-user" means real accounts + ownership + basic isolation — not a full roles/permissions system

---

## 4. System architecture

```
[Browser — React SPA]
   |  auth (login/register)
   |  POST flow1: audio file   |   POST flow2: script text + voice/lang
   v
[FastAPI]
   |  validate input, enforce caps, create Job row (status=queued), return job_id immediately
   v
[Background worker — pipeline runs async, never on the request thread]
   1. (Flow 1) ASR: audio -> draft script              -> Whisper family
      (Flow 1) optional user-edit checkpoint
      (Flow 2) TTS: script -> generated audio            -> MMS-TTS family
   2. Forced alignment: script <-> audio -> word timings -> ctc-forced-aligner (MMS)
   3. NLP: tokenize + POS tag, language-aware              -> spaCy (en) / Stanza (hi)
   4. Word -> image mapping                                 -> Pexels + cache + fallback
   5. Render: images + captions + audio -> MP4              -> ffmpeg
   6. Mark job complete, store output path
   v
[FastAPI: GET /jobs/{id}]  <---- polled by frontend every ~2s
   | {status, stage, progress%, video_url or error}
   v
[Browser: progress -> preview -> download]

[Postgres — users, jobs, image_cache: SQLAlchemy models, Alembic migrations]
```

---

## 5. Pipeline, stage by stage

**Stage 0 — Auth & job intake**
User registers/logs in (email + password, hashed with passlib/bcrypt, JWT issued on login). Every `/jobs` request requires a valid JWT; the created `Job` row is tied to `user_id`. Validate input per flow (§10 caps) before any processing starts — compute and API budget are the scarce resources, not code.

**Stage 1a — ASR (Flow 1 only)**
Feed the uploaded audio to Whisper (`faster-whisper`, `large-v3` or `turbo`) for a draft transcript. If the job's language is `hi` or `mixed`, prefer a Hinglish/code-switch-tuned Whisper checkpoint if you've benchmarked one in (v2 upgrade — §9); vanilla Whisper is the safe v1 default for both languages. Surface the draft transcript for an optional user edit; if skipped, proceed with the raw transcript. Known failure mode: heavy background noise or music degrades transcription — document as a v1 limitation, don't try to fix it in this stage.

**Stage 1b — TTS (Flow 2 only)**
Feed the script + selected language/voice to a TTS model (MMS-TTS per-language checkpoint as the v1 default) to produce a WAV file. That output becomes the "audio" input to Stage 2, exactly as if it had been uploaded.

**Stage 2 — Forced alignment**
Regardless of which flow produced it, align the now-known script to the audio using `ctc-forced-aligner` (Meta's MMS model family) to get `(word, start, end)` tuples. MMS's broad language coverage is exactly why this is the pick — English and Hindi run through the same code path, no per-language alignment logic needed. Validate that script length and audio duration are in a sane ratio before aligning; surface a warning rather than silently producing garbage timing on mismatch.

**Stage 3 — NLP: tokenize & tag (language-aware)**
Route by the job's language:

- English → spaCy (`en_core_web_sm`)
- Hindi → Stanza's Hindi pipeline — spaCy's Hindi support isn't as mature as its English pipeline; Stanza ships a trained Hindi UD model, the safer choice (verify current model availability when you build this; check before assuming names/versions are unchanged)
- Mixed/code-switched → tag each token by its own script (Devanagari vs. Latin is a reasonable first-pass heuristic) and route it to the matching tagger

Both backends must normalize into the same shape: `text`, `lemma`, `pos`, `is_content`. Nothing downstream should know or care which tagger ran. Content word = NOUN/PROPN/VERB/ADJ/ADV (universal POS tags, consistent across both taggers); everything else is captioned but doesn't trigger an image change — finding a meaningful photo for "the"/"का"/"में" is a losing battle not worth the engineering time.

**Stage 4 — Word → image mapping**
For each content word: check a persistent cross-job cache (lemma → image URL) before hitting Pexels. **Hindi-specific step:** Pexels' index is effectively English-only — a Devanagari lemma needs translation to English before querying (a lightweight Hindi→English step; even a bilingual dictionary lookup covers common words). Cache on the _original_ lemma so repeated words skip re-translation too. On a cache miss with zero Pexels results, retry once with a neighboring content word as context; on continued failure, fall back to a small bundled set of neutral placeholder images so one bad word never fails the whole render.

**Stage 5 — Video assembly & render**
An ffmpeg concat-demuxer file sequences images at exact per-word durations; an SRT/ASS subtitle track (from the same timestamp data) is burned in via ffmpeg's `subtitles` filter; one ffmpeg call combines images + captions + the audio track into the final MP4. **Devanagari requirement:** the `subtitles` filter renders through `libass`, which needs a font actually covering Devanagari glyphs (e.g., Noto Sans Devanagari) available to it — bundle this font into the Docker image or Hindi captions render as boxes.

**Stage 6 — Delivery & cleanup**
Serve the file for in-browser preview and download. Delete the source audio, any generated TTS audio, and the rendered video after a short retention window — this is voice data and personal script content tied to a real account now, not an anonymous one-off job; treat it accordingly.

---

## 6. Language handling — consolidated

| Concern          | v1 approach                                                                              |
| ---------------- | ---------------------------------------------------------------------------------------- |
| Language input   | explicit user selection: `en` / `hi` / `mixed` — no auto-detect in v1                    |
| ASR              | Whisper (large-v3/turbo) for both; Hinglish-tuned checkpoint as a benchmarked v2 upgrade |
| TTS              | MMS-TTS per-language checkpoints for both                                                |
| POS tagging      | spaCy for English, Stanza for Hindi, unified `Token` output shape                        |
| Forced alignment | ctc-forced-aligner (MMS) — same code path for both languages                             |
| Image search     | translate Hindi lemmas to English before querying Pexels; cache on original lemma        |
| Captions         | Devanagari-capable font (e.g. Noto Sans Devanagari) bundled in the Docker image          |

---

## 7. Data model

```
User {
  id: uuid
  email: str (unique)
  password_hash: str
  created_at: datetime
}

Job {
  id: uuid
  user_id: uuid (FK -> User.id)
  flow_type: audio_first | text_first
  language: en | hi | mixed
  status: queued | transcribing | synthesizing | aligning | tagging | mapping | rendering | done | failed
  progress_pct: int
  script_text: str | null            # null until ASR produces it (flow 1) or provided directly (flow 2)
  input_audio_path: str | null        # uploaded (flow 1) — null for flow 2
  generated_audio_path: str | null     # TTS output (flow 2) — null for flow 1
  output_video_path: str | null
  error_message: str | null
  created_at, updated_at: datetime
}

ImageCache {
  lemma: str (primary key)
  image_url: str
  fetched_at: datetime
}
```

---

## 8. API sketch

| Method | Path                    | Purpose                                                                                       |
| ------ | ----------------------- | --------------------------------------------------------------------------------------------- |
| POST   | `/auth/register`        | create account                                                                                |
| POST   | `/auth/login`           | issue JWT                                                                                     |
| POST   | `/jobs`                 | create a job — flow 1 (multipart audio + language) or flow 2 (JSON script + language + voice) |
| GET    | `/jobs/{id}`            | poll status/progress/result                                                                   |
| PATCH  | `/jobs/{id}/transcript` | (flow 1 only) submit an edited transcript before alignment proceeds                           |
| GET    | `/jobs/{id}/download`   | fetch the rendered MP4                                                                        |
| GET    | `/jobs`                 | list the current user's jobs                                                                  |
| GET    | `/health`               | liveness check                                                                                |

---

## 9. Technology choices

| Layer                          | Choice                                               | Alternative(s) considered                                                         | Why this one                                                                                                                                                                                                |
| ------------------------------ | ---------------------------------------------------- | --------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Backend                        | FastAPI                                              | Flask, Django                                                                     | Async-native; fits background jobs + polling without extra wiring                                                                                                                                           |
| Frontend                       | React SPA                                            | Plain JS, Vue                                                                     | Real component structure worth demonstrating; this is what a reviewer clicks first                                                                                                                          |
| Auth                           | Hand-rolled OAuth2PasswordBearer + JWT               | `fastapi-users`                                                                   | Better learning value at this stage; small surface area — graduate to a library once auth needs grow (password reset, social login)                                                                         |
| NLP (English)                  | spaCy `en_core_web_sm`                               | NLTK, Stanza                                                                      | Fast, mature lemmatization, good tagging out of the box                                                                                                                                                     |
| NLP (Hindi)                    | Stanza (Hindi UD pipeline)                           | spaCy Hindi                                                                       | spaCy's Hindi coverage is weaker; Stanza ships trained Hindi models                                                                                                                                         |
| ASR                            | Whisper (`faster-whisper`, large-v3/turbo)           | Hinglish-tuned Whisper checkpoint, NVIDIA Nemotron/Canary via NIM                 | Safe general default first; specialized code-switch model as a benchmarked upgrade; NIM is real infra/cost, evaluate but don't require it                                                                   |
| TTS                            | Meta MMS-TTS (per-language)                          | NVIDIA MagpieTTS, IndicF5-based Hinglish TTS, XTTS-v2 Hinglish fine-tune          | Same model family as the forced aligner (MMS) — one less ecosystem, consistent licensing baseline; upgrade candidates have stronger Hinglish naturalness but need a license check before "real service" use |
| Forced alignment               | `ctc-forced-aligner` (MMS)                           | torchaudio's own forced-align API (removed in torchaudio 2.9), WhisperX `align()` | Exact text is known — true forced alignment beats ASR-then-match; MMS covers both target languages through one code path                                                                                    |
| Image source                   | Pexels Photo Search API                              | Unsplash, Pixabay                                                                 | Free, no mandatory attribution, simplest license story                                                                                                                                                      |
| Video render                   | Direct ffmpeg (concat demuxer + `subtitles` filter)  | MoviePy                                                                           | Scales to hundreds of images without per-clip Python overhead; MoviePy is fine for prototyping only                                                                                                         |
| DB / ORM / migrations          | SQLAlchemy + Alembic, SQLite dev / Postgres prod     | raw `sqlite3`, Django ORM, peewee                                                 | Dialect-portable, versioned schema history from day one — the whole point given the Postgres target                                                                                                         |
| Job execution                  | In-process background task + status table            | Celery + Redis, RQ, arq                                                           | Right-sized for the current job volume; revisit if concurrent jobs become common                                                                                                                            |
| Source separation (v2 stretch) | Demucs                                               | Spleeter                                                                          | Standard, well-established, actively maintained                                                                                                                                                             |
| Containerization               | Docker + docker-compose (local Postgres)             | —                                                                                 | Needed anyway for ffmpeg + Devanagari font + ML model weights; also de-risks the SQLite→Postgres switch locally                                                                                             |
| CI                             | GitHub Actions (lint + test on push)                 | —                                                                                 | Standard, free for public/small private repos                                                                                                                                                               |
| Hosting                        | Render (Docker-based web service + managed Postgres) | Railway, Fly.io                                                                   | Best free/cheap-tier fit as of research time and its spin-down behavior matches the async job design; **re-verify current pricing/tier terms before finalizing — these change**                             |

---

## 10. Constraints

- **Input caps** (cost/predictability, not just UX): Flow 1 audio ≤ 3 minutes; Flow 2 script ≤ ~200 words. Keep as named constants in config, never hardcoded inline.
- **Pexels rate limits are real**: default key caps are low enough that a burst of demo traffic can hit them; the mapping stage must check remaining budget and fall back to a placeholder rather than error out. Rate-limit the `/jobs` endpoint itself (per-user, now that accounts exist) so one user can't burn the shared quota.
- **Privacy**: uploaded audio, generated TTS audio, and rendered video are personal data tied to a real account — delete on a short retention timer rather than indefinitely. Keep transcripts/scripts (plain text) only as long as genuinely useful for the account.
- **Licensing**: before treating any TTS/ASR model choice as part of a real, monetizable service, confirm its _current_ commercial-use terms — several strong Hinglish options have licensing caveats worth re-checking at that point, not assumed from this doc.
- **Hosting is cost-constrained**: assume ephemeral disk and idle spin-down; clean up uploaded/generated/rendered files aggressively. Verify the chosen host's current free/cheap-tier limits before finalizing the job/polling pattern — these shift often.
- **No blocking work on the request thread**: ASR, TTS, alignment, image mapping, and rendering all run in the background worker, never inline with an HTTP request.
- **Devanagari rendering**: the font requirement in Stage 5 is a hard requirement for Hindi captions to work at all, not a nice-to-have — confirm it in the Docker image before considering Hindi output "done."

---

## 11. Dos and Don'ts

**Do:**

- Build each new component (ASR, TTS, alignment, NLP-per-language, image mapping, render) as an independently testable stage before wiring flows together
- Keep the shared pipeline (align → tag → map → render) completely flow-agnostic
- Default to the safe, well-documented model choice first; treat specialized/regional models as a benchmarked, documented upgrade
- Cache word→image lookups persistently, cross-job, keyed on the original (pre-translation) lemma
- Keep all input caps, model names, and paths as named config, never literals scattered through the codebase
- Check every model's license before calling anything "production-ready" given the real-service ambition
- Surface pipeline stage and errors to the user rather than failing silently

**Don't:**

- Don't build source separation, generative imagery, or LLM scene composition before both core flows work end-to-end — explicitly good-to-have, not blocking (§13)
- Don't render, transcribe, or synthesize speech synchronously on the request thread
- Don't hand-roll a database abstraction beyond what SQLAlchemy already gives you
- Don't assume auto language detection is reliable enough to skip an explicit language input in v1
- Don't ship Hindi captions without first confirming the Devanagari font actually renders in the Docker environment
- Don't query Pexels with untranslated Devanagari text and assume the empty results are just "no good photo exists"
- Don't store uploaded audio, generated audio, or rendered video longer than needed for the user to retrieve it

---

## 12. Milestone roadmap (component-wise, tests required at each step)

| #   | Milestone                                          | Definition of done                                                                                                | Required tests                                               |
| --- | -------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| 1   | Repo scaffold                                      | project structure, `config.py`, `.env` handling, health-check endpoint                                            | —                                                            |
| 2   | Shared core pipeline, CLI only                     | given a known `(script, audio)` pair, align → tag → map → render produces a correct MP4, for one English sample   | manual run + one fixed-input smoke test                      |
| 3a  | TTS stage, standalone                              | script in → audio file out                                                                                        | unit test on a short fixed sentence, both `en` and `hi`      |
| 3b  | ASR stage, standalone                              | audio in → transcript out                                                                                         | unit test against a known short clip, both languages         |
| 3c  | Hindi NLP tagging, standalone                      | Hindi text in → correctly-shaped `Token` list out                                                                 | unit test comparing tags against a small hand-checked sample |
| 3d  | Hindi image mapping (translation step), standalone | Hindi lemma in → sensible English-query image out                                                                 | unit test with a handful of common Hindi nouns/verbs         |
| 4   | Both flows wired end-to-end via CLI                | `run_flow1.py` / `run_flow2.py` each produce a final video, both languages                                        | one integration test per flow per language (4 total)         |
| 5   | Multi-user data model                              | `users` + `jobs` + `image_cache` tables, Alembic migration applies cleanly on a fresh DB                          | migration up/down test                                       |
| 6   | FastAPI: auth + `/jobs` + worker + polling         | a real HTTP client can register, log in, submit a job (either flow), and poll it to completion                    | API test suite with a test client + temp DB                  |
| 7   | Frontend                                           | both upload flows, login/register, job status, preview, download all work manually                                | manual click-through per flow                                |
| 8   | CI                                                 | GitHub Actions runs the full test suite + lint on every push, green                                               | — (this milestone _is_ the test infrastructure)              |
| 9   | Docker + deploy                                    | `docker-compose up` runs the whole stack locally against Postgres; the same image deploys and works at a live URL | smoke test against the deployed URL                          |

---

## 13. Good-to-have / future roadmap (explicitly non-blocking)

- **LLM-driven scene composition** — group words into coherent visual "beats" and generate better, context-aware search queries, instead of one image per isolated word
- **Generative imagery** — swap stock-photo retrieval for a diffusion model once the retrieval version works, for a consistent visual style
- **Background-music/vocal separation (Demucs)** — isolate vocals for better ASR/alignment accuracy, and optionally remix the separated music stem back under the final video
- **Evaluation-driven model upgrades** — benchmark and document swapping in Hinglish-specialized ASR/TTS checkpoints against the v1 defaults
- **Service hardening** — per-user quotas, additional auth options (OAuth/social login), a public gallery, billing-readiness

None of the above should be started before every milestone in §12 is complete and tested.

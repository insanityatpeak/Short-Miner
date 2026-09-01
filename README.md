<div align="center">

# 🎬 Shorts Miner

**Paste a YouTube video URL → get 3 ready-to-upload Shorts** — automatically cut from the best moments, cropped to 9:16 with the speaker kept in frame, burned-in captions, hook-driven titles/descriptions/hashtags, and a data-backed best-posting-time suggestion.

![Python](https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![ffmpeg](https://img.shields.io/badge/video-ffmpeg-007808?logo=ffmpeg&logoColor=white)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)

</div>

<br>

<div align="center">

<a href="https://github.com/Priyanshu/Short-Miner/raw/main/assets/demo.mp4">
  <img src="assets/demo.gif" alt="Demo Video" />
</a>


<sub>End-to-end run on a real video — download, scoring, cropping, captions, and metadata generation, all live. If the player above doesn't load, open <a href="assets/demo.mp4">assets/demo.mp4</a> directly.</sub>

</div>

<br>

## Contents

- [Problem it solves](#problem-it-solves)
- [Tech stack](#tech-stack)
- [Setup](#setup)
- [How to run it](#how-to-run-it)
- [Architecture](#architecture)
- [Known limitations](#known-limitations)
- [License](#license)
- [Team](#team)

## Problem it solves

Turning a long-form video into Shorts normally means rewatching the whole thing to find good moments, manually cutting clips, writing titles/descriptions/hashtags for each one, and guessing when to post them — 1–3 hours of work per video. Shorts Miner collapses that into one automated pipeline you run from a single URL.

## Tech stack

| Layer | Tool |
|---|---|
| Language | Python 3.11+ |
| UI | Streamlit |
| Transcript | `youtube-transcript-api` (primary) → local `openai-whisper` fallback if a video has no captions |
| Video download / cut | `yt-dlp` + `ffmpeg-python` (wraps the system `ffmpeg` binary — not installed via pip) |
| Vertical crop | OpenCV Haar-cascade face detection for a stable, subject-centered 9:16 crop |
| Captions | Burned in via ffmpeg's `subtitles` filter, from a generated `.ass` file |
| LLM | Gemini (`gemini-3.5-flash-lite`) via `utils/llm.py` — see note below |
| YouTube analytics | `google-api-python-client` + `google-auth-oauthlib` (OAuth2, YouTube Data API v3), authenticated against the presenter's own channel |
| Tests | `pytest`, all network/LLM calls mocked |

> [!NOTE]
> **LLM provider:** `utils/llm.py` is a single choke point for all LLM calls, currently backed by Gemini — a temporary, free-tier substitute while Anthropic credits aren't available, not a permanent choice. Earlier free-tier candidates (`gemini-flash-latest`, `gemini-2.0-flash-lite-001`, `gemini-2.5-flash-lite`) hit daily quota exhaustion, zero free-tier entitlement, or new-user access restrictions respectively; `gemini-3.5-flash-lite` is confirmed working end-to-end (scorer, metadata, analytics all return 200s, no 429s). Swapping back to Claude (`claude-sonnet-4-6`) only requires replacing `call_llm()`'s body with an Anthropic Messages API call — no changes needed anywhere else in the pipeline; this swap is ready and documented in `utils/llm.py`'s own docstring, just gated on Anthropic credits.
>
> **opencv-python pin:** must stay below version 5 (`opencv-python<5` in `requirements.txt`) — the 5.x line dropped `cv2.CascadeClassifier` and ships no bundled Haar cascade files, which breaks face-centered cropping.

## Setup

```bash
git clone <this-repo>
cd shorts-miner
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env`:

| Variable | Required? | Notes |
|---|---|---|
| `GEMINI_API_KEY` | **Yes** | The pipeline's active LLM key. Get one free at [aistudio.google.com/apikey](https://aistudio.google.com/apikey). |
| `ANTHROPIC_API_KEY` | No | Present in `.env.example` for when the provider is swapped back to Claude (see note above). |
| `YOUTUBE_CLIENT_ID` / `YOUTUBE_CLIENT_SECRET` | No | Only needed for the best-posting-time analytics panel. Without them, the rest of the app still works and the panel shows "Analytics unavailable" instead of crashing. |
| `PROXY_URL` | No | Routes yt-dlp and caption fetching through an HTTP/SOCKS5 proxy (e.g. `http://user:pass@host:port`), for when YouTube blocks this host's IP outright rather than just one yt-dlp client. Not needed unless you're seeing persistent "Sign in to confirm you're not a bot" errors even after retries. **Strongly recommended for any Streamlit Community Cloud deployment** — see "Known limitations" below. |
| `WHISPER_MODEL_SIZE` | No | Overrides the local Whisper fallback's model size (default `tiny`). Set to `base` or larger for better accuracy on a host with more CPU than Streamlit Community Cloud's free tier. |

Make sure `ffmpeg` is installed and on your system `PATH` (`ffmpeg -version` should work in a terminal) — it's not installed via pip.

Then run:

```bash
streamlit run app.py
```

## How to run it

Paste a YouTube URL with captions into the input box and click **Run**. For a first test, a good sample video is:

```
https://www.youtube.com/watch?v=4TMPXK9tw5U
```

*(a TEDx talk with manually-authored captions, single on-camera speaker — exercises the full pipeline cleanly.)*

You'll see live status updates as each real pipeline stage completes (transcript loaded, moments identified, clips cut, metadata generated, posting time calculated), then three clip cards with playable video, title/description/hashtags, and the reason each moment was picked, followed by the best-posting-time panel and a "download all" zip button.

## Architecture

```
shorts-miner/
├── app.py                      # Streamlit UI — main entry point
├── pipeline/
│   ├── __init__.py
│   ├── transcript.py           # Pulls/parses video transcript (captions or Whisper)
│   ├── scorer.py                # LLM scores transcript segments, picks top N clips
│   ├── clipper.py               # Downloads video (yt-dlp), cuts clips (ffmpeg),
│   │                             # crops to 9:16 with face detection, burns in captions
│   ├── metadata.py              # LLM generates titles/descriptions/hashtags (batched)
│   └── analytics.py             # YouTube Data API — channel stats, best post time
├── utils/
│   ├── __init__.py
│   ├── config.py                # Loads env vars, API keys, constants
│   ├── llm.py                   # Single choke point for all LLM calls (Gemini/Claude)
│   └── youtube_auth.py          # OAuth2 flow for YouTube Data API
├── .streamlit/
│   └── config.toml              # Base Streamlit theme (light)
├── assets/                      # README demo video
├── output/                      # Generated clips, source cache, token cache (gitignored)
│   ├── source/                  # Cached downloaded source videos, keyed by video ID
│   ├── clips/                   # Final clip_1.mp4, clip_2.mp4, clip_3.mp4
│   └── .token.json              # Cached YouTube OAuth token
├── tests/
│   ├── test_transcript.py
│   ├── test_scorer.py
│   ├── test_clipper.py
│   └── test_metadata.py
├── .env.example
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

Each `pipeline/*.py` module is independently runnable from the command line, e.g.:

```bash
python -m pipeline.transcript <youtube_url>
python -m pipeline.scorer <youtube_url> [num_clips]
python -m pipeline.clipper <youtube_url> <start> <end> [output_name]
python -m pipeline.metadata <youtube_url> [num_clips]
python -m utils.youtube_auth
python -m pipeline.analytics
```

## Known limitations

- **YouTube analytics only works for a channel you own/can authorize** — the OAuth flow authenticates against the presenter's own Google account, so the best-posting-time panel is only meaningful when run by the channel owner. Without OAuth configured, it degrades gracefully to "Analytics unavailable" rather than breaking the rest of the app.
- **Whisper fallback is slower and CPU-bound** — used only when a video has no YouTube captions at all; expect it to noticeably extend the ~90s demo runtime target.
- **`yt-dlp` retries across a fallback list of player clients on a 403** (`utils/ytdlp_client.py`'s `CLIENT_FALLBACK_ORDER`, currently android → ios → tv → web_safari) — YouTube's default web client increasingly demands a PO token or bot-check sign-in and otherwise 403s, and which non-web client still serves formats without one shifts over time and by IP reputation (cloud/datacenter IPs, like Streamlit Community Cloud's, get blocked more aggressively than residential ones — so a video that downloads fine locally can still 403 from the deployed app if every fallback client is currently blocked for that IP range). This is an external, moving-target constraint on YouTube's side, not something fully fixable in this codebase; it also caps resolution at ~360p on clients affected by YouTube's SABR-only rollout, which withholds higher formats without a token.
- **The hosted demo needs `PROXY_URL` actually set, not just supported** — `youtube_transcript_api` (captions) is blocked outright on Streamlit Community Cloud's shared IP (confirmed via Cloud logs: every request gets YouTube's "requests from your IP" block, 100% of the time, not intermittently), which forces every run onto the slow Whisper fallback below. `PROXY_URL` support (`pipeline/transcript.py`, `utils/ytdlp_client.py`) fixes this, but it's a secret that has to be configured per-deployment in Streamlit Cloud's **Settings → Secrets** panel — it doesn't do anything just by existing in the code.
- **The Whisper fallback can be slow enough to look like the app is stuck** — with captions blocked (see above), essentially every run hits this path. On Streamlit Community Cloud's free tier (CPU-only, one shared vCPU, no GPU), a single transcription has been observed taking 8+ minutes, and concurrent visitors starve each other for that one vCPU rather than each just running proportionally slower (one session's audio download stalled 8 minutes waiting for CPU behind another session's transcription in the same log capture). `pipeline.transcript.get_transcript()` now takes an `on_progress` callback so the UI shows what phase it's in instead of a silent spinner, Whisper transcription is capped to the first `MAX_WHISPER_AUDIO_SECONDS` of audio, `WHISPER_MODEL_SIZE` defaults to `tiny` (not `base`) for speed, and concurrent transcriptions are serialized with a lock rather than run in parallel — but a proxy that makes captions succeed directly is still the highest-leverage fix for most videos.
- **Caption word timing is an estimate, not forced alignment** — burned-in captions split each existing caption segment's duration across its words proportionally by character count. This is fast and needs no extra model, but isn't frame-accurate word-level timing (that would require Whisper's `word_timestamps` or a forced-alignment tool, deliberately not used here for latency/dependency reasons).
- **Auto-generated (non-manual) captions can declare overlapping time windows** as a smoothing artifact of the auto-caption format — handled by clamping each segment's effective duration to the next segment's start before deriving word timing, but the underlying per-word timestamps are still an estimate.
- **The LLM provider is currently Gemini (`gemini-3.5-flash-lite`), not Claude** — see the tech-stack note above. Gemini's free tier is also strict per-model (some models return zero entitlement or 429 quota-exhausted depending on the account/project), so the specific model pinned here may need re-checking against [ai.dev/rate-limit](https://ai.dev/rate-limit) if it starts erroring.
- **`st.video()`'s native player chrome can't be fully restyled** to match the design system — only the container around it is styled.
- **Dark mode was attempted and reverted** — a toggle + derived dark token set were built, but didn't visually take effect for the user after a hard refresh and the cause wasn't isolated (no browser devtools access in that session to debug the live DOM/CSS). The revert has since been confirmed clean via `AppTest`: `app.py` loads with zero exceptions, `.streamlit/config.toml` carries only the light theme, and no dark/theme/toggle markup or widgets remain anywhere in the rendered app. Re-attempting dark mode would need visual debugging access to diagnose properly rather than another blind fix.

## License

[MIT](LICENSE)

## Team

- **Priyanshu R** — solo build

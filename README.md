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
| Transcript | `youtube-transcript-api` (primary) → Groq-hosted Whisper (if `GROQ_API_KEY` is set) → local `faster-whisper` fallback |
| Video download / cut | `yt-dlp` + `ffmpeg-python` (wraps the system `ffmpeg` binary — not installed via pip) |
| Vertical crop | OpenCV Haar-cascade face detection for a stable, subject-centered 9:16 crop |
| Captions | Burned in via ffmpeg's `subtitles` filter, from a generated `.ass` file |
| LLM | Gemini (`gemini-3.5-flash-lite`) → Groq (`openai/gpt-oss-120b`) → OpenRouter (free model), via `utils/llm.py` — see note below |
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
| `GROQ_API_KEY` | No (recommended) | Free at [console.groq.com/keys](https://console.groq.com/keys). Used for hosted Whisper transcription (seconds instead of minutes on a CPU host) and as the first LLM fallback when Gemini's quota runs out. |
| `OPENROUTER_API_KEY` | No | Free at [openrouter.ai/keys](https://openrouter.ai/keys). Last LLM fallback, using a `:free` model (override with `OPENROUTER_MODEL`). |
| `ANTHROPIC_API_KEY` | No | Present in `.env.example` for when the provider is swapped back to Claude (see note above). |
| `YOUTUBE_CLIENT_ID` / `YOUTUBE_CLIENT_SECRET` | No | Only needed for the best-posting-time analytics panel. Without them, the rest of the app still works and the panel shows "Analytics unavailable" instead of crashing. |
| `PROXY_URL` | No | Routes yt-dlp and caption fetching through an HTTP/SOCKS5 proxy (e.g. `http://user:pass@host:port`), for when YouTube blocks this host's IP outright rather than just one yt-dlp client. Not needed unless you're seeing persistent "Sign in to confirm you're not a bot" errors even after retries. Would sidestep the captions block described in "Known limitations" below, but requires a paid proxy service — deliberately left unset on the hosted demo to keep it free to run. |
| `WHISPER_MODEL_SIZE` | No | Overrides the local faster-whisper fallback's model size (default `tiny`). Set to `base` or larger for better accuracy on a host with more CPU than Streamlit Community Cloud's free tier. |

Make sure `ffmpeg` is installed and on your system `PATH` (`ffmpeg -version` should work in a terminal) — it's not installed via pip.

Then run:

```bash
streamlit run app.py
```

### Share it publicly from your own machine (Cloudflare Tunnel, free)

YouTube blocks cloud/datacenter IPs (see "Known limitations"), but not home connections. Running the app on your own machine and exposing it with a free Cloudflare quick tunnel gives you a public `https://….trycloudflare.com` link, and YouTube sees your residential IP. No Cloudflare account is needed.

```bash
winget install --id Cloudflare.cloudflared   # macOS: brew install cloudflared
streamlit run app.py                          # terminal 1
cloudflared tunnel --url http://localhost:8501   # terminal 2: prints the public URL
```

The link only works while both commands are running and your machine is awake, and it changes on each restart. For a fixed URL, create a named tunnel on a domain you own in Cloudflare (`cloudflared tunnel login`, then `cloudflared tunnel create shorts-miner`).

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
- **`yt-dlp` retries across a fallback list of player clients on a 403** (`utils/ytdlp_client.py`'s `CLIENT_FALLBACK_ORDER`, currently android → ios → visionos → tv → web_safari) — YouTube's default web client increasingly demands a PO token or bot-check sign-in and otherwise 403s, and which non-web client still serves formats without one shifts over time and by IP reputation (cloud/datacenter IPs, like Streamlit Community Cloud's, get blocked more aggressively than residential ones — so a video that downloads fine locally can still 403 from the deployed app if every fallback client is currently blocked for that IP range). This is an external, moving-target constraint on YouTube's side, not something fully fixable in this codebase; it also caps resolution at ~360p on clients affected by YouTube's SABR-only rollout, which withholds higher formats without a token.
- **`youtube_transcript_api` (captions) is blocked outright on Streamlit Community Cloud's shared IP** (confirmed via Cloud logs: every request gets YouTube's "requests from your IP" block, 100% of the time, not intermittently), which forces every run onto the Whisper fallback below. The free fix is to host from a residential connection via Cloudflare Tunnel (see "Setup"); `PROXY_URL` also works but needs a proxy service.
- **The local Whisper fallback can take several minutes to run, and can queue behind other visitors**. Setting `GROQ_API_KEY` avoids this: Groq transcribes in seconds, and local `faster-whisper` is only used if Groq is unset or fails (e.g. its free daily audio quota is used up). On Streamlit Community Cloud's free tier (CPU-only, one shared vCPU, no GPU), local transcription of a full video has been observed taking 10+ minutes with `openai-whisper`; `faster-whisper` with int8 is roughly 3-4x faster. `pipeline.transcript.get_transcript()` takes an `on_progress` callback so the UI shows what phase it's in instead of a silent spinner, `WHISPER_MODEL_SIZE` defaults to `tiny` (not `base`) for speed, and only one Whisper transcription runs at a time per app instance (a lock, not a duration cap — the full audio is always transcribed) — a visitor whose video queues behind another's sees an explicit "please wait" message instead of an unexplained stall.
- **Caption word timing is only exact for Whisper transcripts** — when the transcript comes from Groq or local Whisper, burned-in captions use Whisper's real per-word timestamps (display text and punctuation still come from the segment text; split tokens like "93" + "%" are merged back). YouTube captions have no word timings, so for those each segment's duration is split across its words proportionally by character count, which is close but not frame-accurate. Any segment whose word timings can't be matched to its text falls back to the same estimate.
- **Auto-generated (non-manual) captions can declare overlapping time windows** as a smoothing artifact of the auto-caption format — handled by clamping each segment's effective duration to the next segment's start before deriving word timing, but the underlying per-word timestamps are still an estimate.
- **The LLM provider is currently Gemini (`gemini-3.5-flash-lite`), not Claude** — see the tech-stack note above. Gemini's free tier is also strict per-model (some models return zero entitlement or 429 quota-exhausted depending on the account/project), so the specific model pinned here may need re-checking against [ai.dev/rate-limit](https://ai.dev/rate-limit) if it starts erroring. With `GROQ_API_KEY` or `OPENROUTER_API_KEY` set, any Gemini failure falls through to those providers instead of failing the run.
- **`st.video()`'s native player chrome can't be fully restyled** to match the design system — only the container around it is styled.
- **Dark mode was attempted and reverted** — a toggle + derived dark token set were built, but didn't visually take effect for the user after a hard refresh and the cause wasn't isolated (no browser devtools access in that session to debug the live DOM/CSS). The revert has since been confirmed clean via `AppTest`: `app.py` loads with zero exceptions, `.streamlit/config.toml` carries only the light theme, and no dark/theme/toggle markup or widgets remain anywhere in the rendered app. Re-attempting dark mode would need visual debugging access to diagnose properly rather than another blind fix.

## License

[MIT](LICENSE)

## Team

- **Priyanshu R** — solo build

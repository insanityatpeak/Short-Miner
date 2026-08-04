# 🎬 Shorts Miner

Paste a YouTube video URL and get 3 ready-to-upload Shorts back — automatically cut from the best moments, cropped to 9:16 with the speaker kept in frame, burned-in captions, hook-driven titles/descriptions/hashtags, and a data-backed best-posting-time suggestion pulled from your own channel's history.

## Problem it solves

Turning a long-form video into Shorts normally means rewatching the whole thing to find good moments, manually cutting clips, writing titles/descriptions/hashtags for each one, and guessing when to post them — 1–3 hours of work per video. Shorts Miner collapses that into one automated pipeline you run from a single URL.

## Tech stack

- **Language:** Python 3.11+
- **UI:** Streamlit
- **Transcript:** `youtube-transcript-api` (primary), local `openai-whisper` fallback if a video has no captions
- **Video download/cut:** `yt-dlp` + `ffmpeg-python` (wraps the system `ffmpeg` binary — not installed via pip)
- **Vertical crop + captions:** OpenCV Haar-cascade face detection for subject-centered cropping; burned-in captions via ffmpeg's `subtitles` filter with a generated `.ass` file (word timing is a proportional character-count estimate against existing caption segments — no Whisper/forced alignment involved)
- **LLM:** `utils/llm.py` is a single choke point for all LLM calls. **Currently backed by Gemini** (`google-genai`, free tier) as a documented temporary override — swap `call_llm()`'s body back to an Anthropic Messages API call using `claude-sonnet-4-6` to match the original spec; no changes needed anywhere else in the pipeline.
- **YouTube analytics:** `google-api-python-client` + `google-auth-oauthlib` (OAuth2, YouTube Data API v3), authenticated against the presenter's own channel
- **Design system:** `DESIGN.md` (a Vercel-inspired token set) drives `app.py`'s CSS — see that file for the full color/type/spacing/component spec
- **Tests:** `pytest`, all network/LLM calls mocked

Note: `opencv-python` must stay pinned below version 5 (`opencv-python<5` in requirements.txt) — the 5.x line dropped `cv2.CascadeClassifier` and ships no bundled Haar cascade files, which breaks face-centered cropping.

## Setup

```bash
git clone <this-repo>
cd shorts-miner
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env`:
- `GEMINI_API_KEY` — required, the pipeline's current active LLM key (get one free at https://aistudio.google.com/apikey). `ANTHROPIC_API_KEY` is also present in `.env.example` for when the provider is swapped back to Claude per the note above.
- `YOUTUBE_CLIENT_ID` / `YOUTUBE_CLIENT_SECRET` — optional. Only needed for the best-posting-time analytics panel; the rest of the app works without them and the panel just shows "Analytics unavailable" instead of crashing.

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

(a TEDx talk with manually-authored captions, single on-camera speaker — exercises the full pipeline cleanly).

You'll see live status updates as each real pipeline stage completes (transcript loaded, moments identified, clips cut, metadata generated, posting time calculated), then three clip cards with playable video, title/description/hashtags, and the reason each moment was picked, followed by the best-posting-time panel and a "download all" zip button.

## Architecture

```
shorts-miner/
├── app.py                      # Streamlit UI — main entry point
├── DESIGN.md                   # Design token set driving app.py's CSS
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
│   └── config.toml              # Base Streamlit theme (light, matches DESIGN.md)
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
├── requirements.txt
├── README.md
└── CLAUDE.md                    # Build spec this project was implemented against
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
- **Caption word timing is an estimate, not forced alignment** — burned-in captions split each existing caption segment's duration across its words proportionally by character count. This is fast and needs no extra model, but isn't frame-accurate word-level timing (that would require Whisper's `word_timestamps` or a forced-alignment tool, deliberately not used here for latency/dependency reasons).
- **Auto-generated (non-manual) captions can declare overlapping time windows** as a smoothing artifact of the auto-caption format — handled by clamping each segment's effective duration to the next segment's start before deriving word timing, but the underlying per-word timestamps are still an estimate.
- **The LLM provider is currently Gemini, not Claude** — a documented temporary swap in `utils/llm.py`; the original spec called for Claude (`claude-sonnet-4-6`). Swapping back only requires changing `call_llm()`'s implementation.
- **`st.video()`'s native player chrome can't be fully restyled** to match the design system — only the container around it is styled.
- **Dark mode was attempted and reverted** — a toggle + derived dark token set were built and functionally verified, but didn't visually take effect for the user after a hard refresh and the cause wasn't isolated (no browser devtools access in that session to debug the live DOM/CSS). The codebase is back to the confirmed-working light-only version; re-attempting dark mode would need visual debugging access to diagnose properly rather than another blind fix.

## Team

_Placeholder — add names and who built what before submitting._

- **Name** — role / what they built
- **Name** — role / what they built

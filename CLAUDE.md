# CLAUDE.md — Shorts Miner

This file is the build spec for Claude Code. Read this fully before writing any code. Build the project end-to-end, in the order laid out below, verifying each stage works before moving to the next.

---

## 1. Project Overview

**Name:** Shorts Miner

**One-liner:** Paste a YouTube video URL → get 3 ready-to-upload Shorts, automatically cut, titled, captioned, and paired with a data-backed best-posting-time suggestion.

**Context:** This is a hackathon submission for the "YouTube Automation Hackathon" (community-run, unaffiliated with YouTube/Google). Judging criteria are functionality, creativity, technical execution, and real-world usefulness. Bonus points go to tools that run live and produce real output (not mockups, not pre-recorded demos). Build accordingly: every pipeline stage must produce a visible, real artifact when run live.

**Target demo runtime:** under 90 seconds end-to-end for a ~15–20 minute source video.

---

## 2. Problem Statement

Creators who want to repurpose long-form videos into Shorts currently have to:
1. Rewatch the full video to find good moments
2. Manually cut clips in an editor
3. Write titles, descriptions, and hashtags for each clip
4. Guess when to post them

This takes 1–3 hours per video. Shorts Miner collapses this into one automated pipeline.

---

## 3. Architecture

```
shorts-miner/
├── app.py                  # Streamlit UI — main entry point
├── pipeline/
│   ├── __init__.py
│   ├── transcript.py       # Pulls/parses video transcript
│   ├── scorer.py           # LLM scores transcript segments, picks top 3
│   ├── clipper.py          # Downloads video (yt-dlp), cuts clips (ffmpeg)
│   ├── metadata.py         # LLM generates titles/descriptions/hashtags
│   └── analytics.py        # YouTube Data API — channel stats, best post time
├── utils/
│   ├── __init__.py
│   ├── config.py           # Loads env vars, API keys, constants
│   └── youtube_auth.py     # OAuth2 flow for YouTube Data API
├── output/                 # Generated clips + metadata.json land here (gitignored)
├── tests/
│   ├── test_transcript.py
│   ├── test_scorer.py
│   ├── test_clipper.py
│   └── test_metadata.py
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
└── CLAUDE.md                # this file
```

Each `pipeline/*.py` module should be independently runnable and testable from the command line (e.g. `python -m pipeline.transcript <url>`) BEFORE being wired into the Streamlit app. Build and verify bottom-up: transcript → scorer → clipper → metadata → analytics → app.py glues it together.

---

## 4. Tech Stack

- **Language:** Python 3.11+
- **UI:** Streamlit
- **Transcript:** `youtube-transcript-api` (primary), `openai-whisper` local fallback if no captions exist
- **Video download:** `yt-dlp`
- **Video cutting:** `ffmpeg-python` (wraps system `ffmpeg` — assume ffmpeg binary is installed on the host; do not attempt to install it via pip)
- **LLM:** Anthropic Claude API (`anthropic` Python SDK), model `claude-sonnet-4-6`
- **YouTube analytics:** `google-api-python-client` + `google-auth-oauthlib` (OAuth2, YouTube Data API v3)
- **Env management:** `python-dotenv`
- **HTTP:** `httpx` (only if needed beyond the above SDKs)

Full `requirements.txt`:
```
streamlit
youtube-transcript-api
yt-dlp
ffmpeg-python
openai-whisper
anthropic
google-api-python-client
google-auth-oauthlib
opencv-python
python-dotenv
httpx
```

---

## 5. Environment & Secrets

Create `.env.example` (committed) and `.env` (gitignored, user fills in locally):

```
ANTHROPIC_API_KEY=
YOUTUBE_CLIENT_ID=
YOUTUBE_CLIENT_SECRET=
```

`utils/config.py` loads these via `python-dotenv` and raises a clear error at startup if `ANTHROPIC_API_KEY` is missing (this key is required for the core pipeline to run at all). YouTube OAuth credentials are only required for the analytics module — the rest of the app should degrade gracefully (skip the analytics panel with a clear message) if they're absent, so the core demo never breaks due to auth issues.

Never hardcode API keys anywhere. Never print API keys to logs or the Streamlit UI.

---

## 6. Module Specs

### 6.1 `pipeline/transcript.py`

**Function:** `get_transcript(youtube_url: str) -> list[dict]`

- Extract the video ID from the URL (handle `youtube.com/watch?v=`, `youtu.be/`, and `youtube.com/shorts/` formats).
- Try `youtube_transcript_api` first. Return a list of segments: `{"text": str, "start": float, "duration": float}`.
- If no transcript is available (captions disabled), fall back to:
  - Download audio only via `yt-dlp` (`-x --audio-format mp3`)
  - Run local `openai-whisper` (`base` model is fine for demo speed) to generate a transcript with timestamps
- Log which method was used (captions vs. Whisper) so the UI can display it.
- Raise a clear, catchable exception if both methods fail (e.g. private/unavailable video).

### 6.2 `pipeline/scorer.py`

**Function:** `score_segments(transcript: list[dict], num_clips: int = 3) -> list[dict]`

- Group raw transcript lines into candidate windows of 30–60 seconds (merge adjacent segments by timestamp).
- Send the full transcript (with timestamps) to Claude in a single call. Prompt Claude to:
  - Identify the `num_clips` best segments for standalone YouTube Shorts, scoring each on: hook strength (does it open mid-action or with a strong statement?), self-containedness (makes sense without prior context), emotional/energy peak, and quotability.
  - Return strict JSON: a list of objects with `start_time`, `end_time`, `score` (0–100), and `reason` (one sentence).
- Use Claude's JSON mode / explicit instruction ("respond with ONLY valid JSON, no markdown fences, no preamble") and parse defensively — strip code fences if present before `json.loads`.
- Sort by score descending, return top `num_clips`, each clip 30–60 seconds long. Enforce non-overlapping time ranges.

### 6.3 `pipeline/clipper.py`

**Function:** `download_video(youtube_url: str, output_path: str) -> str`
**Function:** `cut_clip(source_path: str, start: float, end: float, output_path: str) -> str`
**Function:** `reformat_vertical(clip_path: str, output_path: str) -> str`

- `download_video`: use `yt-dlp` to download the source video once. Prefer 1080p; if the video doesn't offer a 1080p format (checked via yt-dlp's format list), fall back to the best available resolution below 1080p. Cache it in `output/source/` keyed by video ID so re-running the same demo video doesn't re-download.
- `cut_clip`: use `ffmpeg-python` to cut `[start, end]` from the source into a standalone `.mp4`. Use stream copy (`-c copy`) where possible for speed; fall back to re-encode only if stream copy produces a broken clip (test this — keyframe alignment can cause copy-cut clips to start black or drift from the requested boundaries entirely if the cut point isn't near a keyframe; if so, re-encode with `libx264` at fast preset).
- `reformat_vertical`: crop a cut 16:9 clip down to a 9:16 vertical frame for Shorts, keeping the speaking subject in frame instead of blindly center-cropping:
  - **Sample:** grab frames from the clip every 1–2 seconds (via OpenCV `VideoCapture`) and run a face detector — Haar cascade (`cv2.data.haarcascades` + `haarcascade_frontalface_default.xml`, bundled with `opencv-python`) or a DNN face detector — on each sampled frame. Record the horizontal center (x-coordinate) of the detected face, if any, per sample.
  - **Compute crop center:** from the set of per-sample face x-centers, take the median (not a rolling/per-frame average — the crop must not jitter frame-to-frame) to get one stable horizontal center for the whole clip. Convert that into a fixed x-offset for a 9:16 crop window (`width = source_height * 9/16`, `height = source_height`), clamped so the window stays fully inside the source frame.
  - **Fallback:** if faces are detected in fewer than ~30% of sampled frames (speaker turned away, a slide/graphic on screen, detector failure), skip subject-centering for that clip and fall back to a plain center-crop. Log clearly that this clip fell back, so the UI can surface it honestly rather than silently mis-cropping.
  - **Apply:** cut with ffmpeg's `crop=w:h:x:y` filter using the single computed x-offset for the clip's entire duration, then re-encode with `libx264` (crop always requires re-encoding — no stream-copy path here). No per-frame dynamic recropping or virtual camera panning — explicitly out of scope for the hackathon timeline.
- Output clips to `output/clips/clip_1.mp4`, `clip_2.mp4`, `clip_3.mp4`. `reformat_vertical` overwrites the cut clip in place with the vertical version — there is no separate `_vertical.mp4` file; the UI only ever plays the final 9:16 clip.
- Return the file paths so the UI can render them directly.

### 6.4 `pipeline/metadata.py`

**Function:** `generate_metadata(clip_text: str, clip_reason: str) -> dict`

- One Claude call per clip (or batch all 3 into a single call with structured JSON output — prefer batching to save latency and cost).
- For each clip, generate:
  - `title`: hook-driven, under 60 characters
  - `description`: 1–2 sentences
  - `hashtags`: 3–5 relevant tags
- Return strict JSON, same defensive parsing as `scorer.py`.

### 6.5 `pipeline/analytics.py`

**Function:** `get_best_posting_time(channel_id: str = None) -> dict`

- Authenticate via OAuth2 (`utils/youtube_auth.py`) against the authenticated user's own channel — this only works for a channel the demo presenter owns/can authorize, which is expected and fine for a hackathon demo.
- Pull the channel's recent video list and basic stats (views, likes, published time) via YouTube Data API v3.
- Compute a simple heuristic: bucket published times by hour-of-day and day-of-week, weight by view count, return the top 1–2 time windows as the "best posting time" recommendation.
- If OAuth isn't configured or the call fails, return `None` and let the UI display "Analytics unavailable — connect a YouTube account to enable this" rather than crashing.

### 6.6 `utils/youtube_auth.py`

- Standard `google-auth-oauthlib` installed-app flow.
- Store the resulting token in `output/.token.json` (gitignored) so re-auth isn't required on every run during the hackathon.
- Provide a `get_authenticated_service()` function that returns a ready-to-use `googleapiclient.discovery` client, or raises a clear exception if not authenticated.

---

## 7. `app.py` — Streamlit UI Spec

Build this last, after every pipeline module works standalone from the CLI.

**Layout:**
1. Title: "🎬 Shorts Miner"
2. Text input for a YouTube URL + "Run" button
3. As the pipeline runs, show live status updates using `st.status()` or sequential `st.success()` calls:
   - "✅ Transcript loaded (N segments, via [captions/Whisper])"
   - "✅ Top 3 moments identified"
   - "⏳ Cutting clips..." → "✅ 3 clips ready"
   - "✅ Titles & metadata generated"
   - "✅ Best posting time calculated" (or the graceful fallback message)
4. Three-column layout, one per clip:
   - `st.video()` playing the actual cut clip
   - Generated title, description, hashtags displayed below
   - The one-sentence "reason" this segment was chosen
5. Bottom panel: best posting time recommendation with the underlying data shown (e.g. a small bar chart of views by hour using `st.bar_chart`)
6. A "Download all" option that zips the `output/clips/` folder for the user

**Important for the live-demo bonus:** every status line above must reflect real work completing, not a fake progress bar. Wire `st.status()` directly to the return values of each pipeline function — don't simulate delays.

---

## 8. Error Handling Requirements

- Every pipeline function should raise specific, catchable exceptions (not bare `Exception`) with clear messages — the UI will catch these and show a friendly `st.error()` instead of a stack trace.
- Handle common live-demo failure modes explicitly:
  - Video has no transcript and Whisper fails → clear error, suggest a different URL
  - Video is age-restricted / private / unavailable → clear error
  - `ANTHROPIC_API_KEY` missing → fail fast at app startup with a clear message
  - YouTube OAuth not configured → analytics panel degrades gracefully, rest of app still works

---

## 9. Testing

Write basic tests in `tests/` using `pytest`:
- `test_transcript.py`: transcript ID extraction from various URL formats (pure function, no network needed)
- `test_scorer.py`: JSON parsing/defensive-parsing logic works given a mocked Claude response (mock the API call)
- `test_clipper.py`: timestamp math for clip boundaries (non-overlap enforcement)
- `test_metadata.py`: JSON parsing logic given a mocked Claude response

Keep network/API-dependent code covered by mocks, not live calls, so tests run without secrets.

---

## 10. README.md Requirements

Generate a README that includes:
1. What the tool does (2–3 sentences)
2. Problem it solves
3. Tech stack used
4. Setup instructions: clone, `pip install -r requirements.txt`, copy `.env.example` to `.env` and fill in keys, ensure `ffmpeg` is installed on the system, run `streamlit run app.py`
5. How to run it (with a sample YouTube URL that has captions, for first-time testers)
6. Architecture diagram (ASCII, matching section 3 above)
7. Known limitations (e.g. YouTube analytics requires the presenter's own channel/OAuth; Whisper fallback is slower)
8. Team info section (placeholder for names + who built what)

---

## 11. Build Order (follow exactly)

1. Scaffold the directory structure and `requirements.txt`, `.env.example`, `.gitignore`
2. Build and CLI-test `pipeline/transcript.py` against a real public video URL
3. Build and CLI-test `pipeline/scorer.py` using the transcript output from step 2
4. Build and CLI-test `pipeline/clipper.py`, producing real playable `.mp4` files
5. Build and CLI-test `pipeline/metadata.py`
6. Build `utils/youtube_auth.py` and `pipeline/analytics.py`, test with a real (or gracefully-failing) OAuth flow
7. Wire everything into `app.py`
8. Write tests
9. Write the README
10. Do a full live dry run end-to-end and time it — optimize the slowest step if total runtime exceeds ~90 seconds (likely candidates: video download resolution, Whisper fallback, or splitting the metadata call into multiple round trips instead of one batched call)

Do not skip ahead to `app.py` before each pipeline module has been verified working standalone — debugging the full app before the pieces work individually wastes hackathon time.

---

## 12. Explicit Non-Goals (do not build these)

- No database
- No user accounts / multi-tenant support
- No Docker setup
- No cloud deployment / hosting config
- No auto-upload to YouTube (out of scope, adds real risk, not needed for judging)
- No support for platforms other than YouTube

Keep the build tight and focused on a reliable, fast, live-demoable pipeline.

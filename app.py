"""Shorts Miner — Streamlit UI.

Glues together transcript -> scorer -> clipper -> metadata -> analytics.
Run: streamlit run app.py
"""
import html
import io
import os
import uuid
import zipfile
from contextlib import contextmanager

import streamlit as st

from pipeline.analytics import get_best_posting_time
from pipeline.clipper import ClipperError, VideoDownloadError, cut_and_reformat, download_video
from pipeline.metadata import (
    ClaudeMetadataResponseError,
    MetadataError,
    clip_transcript_text,
    generate_metadata_batch,
)
from pipeline.scorer import (
    ClaudeResponseError,
    InsufficientSegmentsError,
    ScorerError,
    score_segments,
)
from pipeline.transcript import (
    InvalidYouTubeURLError,
    TranscriptError,
    TranscriptUnavailableError,
    VideoUnavailableError,
    get_last_transcript_method,
    get_transcript,
)
from utils.config import CLIPS_DIR, ConfigError
from utils.llm import LLMQuotaExceededError, require_llm_key

NUM_CLIPS = 3
ACCENT_BLUE = "#0070f3"

st.set_page_config(page_title="Shorts Miner", page_icon="🎬", layout="wide")

# --- Design tokens -----------------------------------------------------------
# Colors, typography, spacing, and radii below are a small Vercel-inspired
# token set. Streamlit's own widgets are restyled via data-testid selectors
# (best-effort — Streamlit doesn't expose a public API for this, so testids
# can shift between versions); custom elements (hero, cards, chips, callouts)
# are plain HTML injected via st.markdown so they can match the component
# specs exactly. st.container(key=...) is used to visually group native
# widgets (st.video, st.bar_chart) inside a styled card, since Streamlit
# elements can't otherwise be nested inside hand-written HTML.
STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400&display=swap');

:root {
  --color-primary: #171717;
  --color-on-primary: #ffffff;
  --color-ink: #171717;
  --color-body: #4d4d4d;
  --color-mute: #888888;
  --color-hairline: #ebebeb;
  --color-canvas: #ffffff;
  --color-canvas-soft: #fafafa;
  --color-canvas-soft-2: #f5f5f5;
  --color-link: #0070f3;
  --color-link-deep: #0761d1;
  --color-link-bg-soft: #d3e5ff;
  --color-warning: #f5a623;
  --color-warning-deep: #ab570a;
  --color-warning-soft: #ffefcf;
  --gradient-develop-start: #007cf0;
  --gradient-develop-end: #00dfd8;

  --radius-sm: 6px;
  --radius-md: 8px;
  --radius-lg: 12px;
  --radius-pill: 100px;
  --radius-full: 9999px;

  --space-xs: 8px;
  --space-sm: 12px;
  --space-md: 16px;
  --space-lg: 24px;
  --space-xl: 32px;
  --space-2xl: 40px;
  --space-3xl: 48px;
  --space-4xl: 64px;
}

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
  font-family: 'Inter', system-ui, -apple-system, sans-serif;
  color: var(--color-ink);
  -webkit-font-smoothing: antialiased;
  text-rendering: optimizeLegibility;
}

[data-testid="stAppViewContainer"], [data-testid="stMain"] {
  background: var(--color-canvas-soft);
}

/* Custom scrollbar (webkit) */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
  background: var(--color-hairline);
  border-radius: var(--radius-full);
  border: 2px solid var(--color-canvas-soft);
}
::-webkit-scrollbar-thumb:hover { background: var(--color-mute); }

@keyframes sm-fade-up {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}

.block-container {
  padding-top: var(--space-2xl);
  padding-bottom: var(--space-4xl);
  max-width: 1200px;
}

/* Hero */
.sm-hero {
  margin-bottom: var(--space-4xl);
  animation: sm-fade-up 0.5s ease-out both;
}
[data-testid="stMarkdownContainer"] .sm-hero-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 12px;
  letter-spacing: 0.02em;
  text-transform: uppercase;
  color: var(--color-body);
  background: var(--color-canvas-soft-2);
  border: 1px solid var(--color-hairline);
  border-radius: var(--radius-full);
  padding: 4px var(--space-sm);
  margin-bottom: var(--space-md);
}
.sm-hero-badge::before {
  content: "";
  display: inline-block;
  width: 6px;
  height: 6px;
  border-radius: var(--radius-full);
  background: linear-gradient(135deg, var(--gradient-develop-start), var(--gradient-develop-end));
}
[data-testid="stMarkdownContainer"] .sm-hero-title {
  font-family: 'Inter', system-ui, -apple-system, sans-serif;
  font-size: clamp(32px, 5vw, 48px);
  font-weight: 600;
  line-height: 1.05;
  letter-spacing: -2.4px;
  color: var(--color-ink);
  margin: 0 0 var(--space-sm) 0;
}
[data-testid="stMarkdownContainer"] .sm-hero-lead {
  font-family: 'Inter', system-ui, -apple-system, sans-serif;
  font-size: 18px;
  font-weight: 400;
  line-height: 28px;
  color: var(--color-body);
  max-width: 640px;
  margin: 0 0 var(--space-lg) 0;
}
.sm-hero-rule {
  width: 64px;
  height: 3px;
  border-radius: var(--radius-full);
  background: linear-gradient(90deg, var(--gradient-develop-start), var(--gradient-develop-end));
  margin: 0;
}

/* Section rhythm */
[data-testid="stMarkdownContainer"] .sm-section-title {
  position: relative;
  font-family: 'Inter', system-ui, -apple-system, sans-serif;
  font-size: 24px;
  font-weight: 600;
  line-height: 32px;
  letter-spacing: -0.96px;
  color: var(--color-ink);
  margin: 0 0 var(--space-lg) 0;
  padding-left: var(--space-md);
}
.sm-section-title::before {
  content: "";
  position: absolute;
  left: 0;
  top: 2px;
  bottom: 2px;
  width: 3px;
  border-radius: var(--radius-full);
  background: linear-gradient(180deg, var(--gradient-develop-start), var(--gradient-develop-end));
}
.sm-section {
  margin-top: var(--space-4xl);
  padding-top: var(--space-3xl);
  border-top: 1px solid var(--color-hairline);
  animation: sm-fade-up 0.5s ease-out both;
}

/* Clip cards */
div[class*="st-key-clip-card-"] {
  background: var(--color-canvas);
  border-radius: var(--radius-md);
  padding: var(--space-lg);
  box-shadow: 0px 1px 1px #00000005, 0px 2px 2px #0000000a;
  border: 1px solid var(--color-hairline);
  height: 100%;
  transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}
div[class*="st-key-clip-card-"]:hover {
  transform: translateY(-3px);
  box-shadow: 0px 4px 8px #00000008, 0px 12px 24px -8px #00000014;
  border-color: #dcdcdc;
}
div[class*="st-key-clip-card-"] [data-testid="stVideo"] video {
  border-radius: var(--radius-sm);
  display: block;
}
[data-testid="stMarkdownContainer"] .sm-eyebrow {
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 12px;
  line-height: 16px;
  color: var(--color-mute);
  text-transform: uppercase;
  letter-spacing: 0.02em;
  margin: var(--space-md) 0 var(--space-xs) 0;
}
[data-testid="stMarkdownContainer"] .sm-card-title {
  font-family: 'Inter', system-ui, -apple-system, sans-serif;
  font-size: 20px;
  font-weight: 600;
  line-height: 28px;
  letter-spacing: -0.6px;
  color: var(--color-ink);
  margin: 0 0 var(--space-xs) 0;
}
[data-testid="stMarkdownContainer"] .sm-card-desc {
  font-family: 'Inter', system-ui, -apple-system, sans-serif;
  font-size: 16px;
  font-weight: 400;
  line-height: 24px;
  color: var(--color-body);
  margin: 0 0 var(--space-sm) 0;
}
.sm-chip-row { margin-bottom: var(--space-sm); line-height: 2.2; }
[data-testid="stMarkdownContainer"] .sm-chip {
  display: inline-block;
  font-size: 12px;
  line-height: 16px;
  color: var(--color-link-deep);
  background: var(--color-link-bg-soft);
  border-radius: var(--radius-full);
  padding: 4px var(--space-xs);
  margin: 0 var(--space-xs) var(--space-xs) 0;
  transition: background 0.15s ease, transform 0.15s ease;
}
.sm-chip:hover {
  background: #c3daff;
  transform: translateY(-1px);
}
[data-testid="stMarkdownContainer"] .sm-card-reason {
  font-family: 'Inter', system-ui, -apple-system, sans-serif;
  font-size: 14px;
  line-height: 20px;
  color: var(--color-mute);
  border-top: 1px solid var(--color-hairline);
  padding-top: var(--space-sm);
  margin-top: var(--space-sm);
}

/* Posting-time panel */
div[class*="st-key-posting-panel"] {
  background: var(--color-canvas);
  border-radius: var(--radius-lg);
  padding: var(--space-xl);
  box-shadow: 0px 2px 2px #0000000a, 0px 8px 16px -4px #0000000a;
  border: 1px solid var(--color-hairline);
}
div[class*="st-key-callout-0"], div[class*="st-key-callout-1"] {
  border-radius: var(--radius-md);
  padding: var(--space-md) var(--space-lg);
  height: 100%;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}
div[class*="st-key-callout-0"]:hover, div[class*="st-key-callout-1"]:hover {
  transform: translateY(-2px);
  box-shadow: 0px 6px 16px -6px #00000022;
}
div[class*="st-key-callout-0"] { background: var(--color-link-bg-soft); }
div[class*="st-key-callout-1"] { background: var(--color-warning-soft); }
[data-testid="stMarkdownContainer"] .sm-callout-label {
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 12px;
  color: var(--color-body);
  text-transform: uppercase;
  margin: 0 0 4px 0;
}
[data-testid="stMarkdownContainer"] .sm-callout-time {
  font-family: 'Inter', system-ui, -apple-system, sans-serif;
  font-size: 24px;
  font-weight: 600;
  line-height: 32px;
  letter-spacing: -0.96px;
  margin: 0 0 4px 0;
}
div[class*="st-key-callout-0"] [data-testid="stMarkdownContainer"] .sm-callout-time { color: var(--color-link-deep); }
div[class*="st-key-callout-1"] [data-testid="stMarkdownContainer"] .sm-callout-time { color: var(--color-warning-deep); }
[data-testid="stMarkdownContainer"] .sm-callout-meta {
  font-family: 'Inter', system-ui, -apple-system, sans-serif;
  font-size: 14px;
  line-height: 20px;
  color: var(--color-body);
  margin: 0;
}

/* Native widget restyling (best-effort across Streamlit versions) */
[data-testid="stTextInput"] input {
  border-radius: var(--radius-sm);
  border: 1px solid var(--color-hairline);
  font-size: 14px;
  transition: border-color 0.15s ease, box-shadow 0.15s ease;
}
[data-testid="stTextInput"] input:hover {
  border-color: #d4d4d4;
}
[data-testid="stTextInput"] input:focus {
  border-color: var(--color-link) !important;
  box-shadow: 0 0 0 3px var(--color-link-bg-soft) !important;
}
button[kind="primary"], [data-testid="stBaseButton-primary"] {
  background: var(--color-primary) !important;
  border-radius: var(--radius-pill) !important;
  border: none !important;
  font-weight: 500 !important;
  transition: transform 0.15s ease, opacity 0.15s ease, box-shadow 0.15s ease !important;
}
button[kind="primary"]:hover:not(:disabled), [data-testid="stBaseButton-primary"]:hover:not(:disabled) {
  opacity: 0.88;
  transform: translateY(-1px);
  box-shadow: 0px 6px 16px -6px #00000040;
}
button[kind="primary"]:active:not(:disabled), [data-testid="stBaseButton-primary"]:active:not(:disabled) {
  transform: translateY(0);
}
button[kind="primary"]:disabled, [data-testid="stBaseButton-primary"]:disabled {
  background: var(--color-canvas-soft-2) !important;
  color: var(--color-mute) !important;
  border: 1px solid var(--color-hairline) !important;
}
button[kind="secondary"], [data-testid="stBaseButton-secondary"] {
  border-radius: var(--radius-pill) !important;
  border: 1px solid var(--color-hairline) !important;
  font-weight: 500 !important;
  transition: border-color 0.15s ease, background 0.15s ease, transform 0.15s ease !important;
}
button[kind="secondary"]:hover:not(:disabled), [data-testid="stBaseButton-secondary"]:hover:not(:disabled) {
  border-color: var(--color-ink) !important;
  background: var(--color-canvas-soft-2) !important;
  transform: translateY(-1px);
}
[data-testid="stStatusWidget"], [data-testid="stExpander"] {
  border-radius: var(--radius-md) !important;
  border: 1px solid var(--color-hairline) !important;
}
[data-testid="stAlert"] {
  border-radius: var(--radius-md) !important;
  border: 1px solid var(--color-hairline) !important;
  box-shadow: 0px 1px 1px #00000005, 0px 2px 2px #0000000a;
}
[data-testid="stVideo"] video, [data-testid="stBaseButton-primary"], [data-testid="stBaseButton-secondary"], [data-testid="stTextInput"] input {
  outline-offset: 2px;
}
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

try:
    require_llm_key()
except ConfigError as e:
    st.error(f"⚠️ {e}")
    st.stop()

st.markdown(
    """
    <div class="sm-hero">
      <span class="sm-hero-badge">Automated Shorts Pipeline</span>
      <p class="sm-hero-title">🎬 Shorts Miner</p>
      <p class="sm-hero-lead">Paste a YouTube video URL → get 3 ready-to-upload Shorts,
      automatically cut, titled, captioned, and paired with a data-backed
      best-posting-time suggestion.</p>
      <div class="sm-hero-rule"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

url = st.text_input("YouTube video URL", placeholder="https://www.youtube.com/watch?v=...")
run_clicked = st.button("Run", type="primary", disabled=not url)


@contextmanager
def sm_section(title: str | None = None):
    """Wrap a page section in the `.sm-section` div, with an optional
    `.sm-section-title` heading. Keeps the open/close markup pair together
    at one call site instead of three copies that can drift apart."""
    st.markdown('<div class="sm-section">', unsafe_allow_html=True)
    if title:
        st.markdown(f'<p class="sm-section-title">{title}</p>', unsafe_allow_html=True)
    yield
    st.markdown("</div>", unsafe_allow_html=True)


def _friendly_reason(exc: Exception) -> str:
    """Plain-language likely cause for a pipeline failure, shown under the
    error so it reads as "here's what's going on" rather than "the app is
    broken" — most failures here are external (YouTube, the AI provider, or
    the specific video), not bugs in Shorts Miner."""
    msg = str(exc)
    msg_lower = msg.lower()

    if "sign in to confirm" in msg_lower or "not a bot" in msg_lower:
        return (
            "YouTube's bot-check is challenging this server's shared IP address "
            "(common on cloud hosting like this demo) — it isn't specific to this "
            "video or a bug in Shorts Miner. It sometimes clears up on retry, but "
            "may keep happening for videos without captions (which need this "
            "download step) until the host's IP falls out of YouTube's suspicion "
            "window."
        )
    if "403" in msg or "forbidden" in msg_lower:
        return (
            "YouTube is temporarily blocking video downloads from this server's IP "
            "address — common on shared/cloud hosting, and not a bug in Shorts "
            "Miner. It usually clears up on its own; try again in a bit, or with "
            "a different video."
        )
    if isinstance(exc, VideoUnavailableError):
        return (
            "The video itself is private, deleted, age-restricted, or blocked in "
            "this region — nothing Shorts Miner can work around."
        )
    if isinstance(exc, InvalidYouTubeURLError):
        return (
            "That doesn't look like a URL Shorts Miner recognizes — check it's a "
            "youtube.com/watch, youtu.be, or /shorts/ link."
        )
    if isinstance(exc, TranscriptUnavailableError):
        return (
            "This video has no YouTube captions, and the local speech-to-text "
            "fallback couldn't transcribe it either (e.g. no spoken dialogue, or "
            "unsupported audio) — some videos just can't be processed this way."
        )
    if isinstance(exc, InsufficientSegmentsError):
        return "This video (or its usable transcript) is too short to pull out separate highlight clips."
    if isinstance(exc, VideoDownloadError):
        return (
            "The source video couldn't be downloaded from YouTube right now — "
            "usually a transient block or hiccup on YouTube's side, not a bug "
            "here. Try again in a bit."
        )
    if isinstance(exc, (ClaudeResponseError, ClaudeMetadataResponseError)):
        return (
            "The AI model returned a response Shorts Miner couldn't parse — an "
            "occasional hiccup with the model, not a bug. Try again."
        )
    return (
        "This is most likely a temporary issue with YouTube, the AI model "
        "provider, or this specific video — not a bug in Shorts Miner. Try "
        "again, or with a different video."
    )


def _show_pipeline_error(headline: str, exc: Exception) -> None:
    """Show a pipeline failure as a plain-language headline first, with the raw
    exception tucked into a collapsed expander — not the reverse. Most of these
    failures are external (YouTube, the AI provider) and the raw yt-dlp/API text
    (cookie-export links, stack-shaped messages) reads as "the app is broken"
    when shown as the primary line."""
    st.error(f"{headline} {_friendly_reason(exc)}")
    with st.expander("Technical details"):
        st.code(str(exc))


def _build_clips_zip(clip_paths: list[str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in clip_paths:
            zf.write(path, arcname=os.path.basename(path))
    return buf.getvalue()


if run_clicked:
    try:
        with st.status("Running Shorts Miner...", expanded=True) as status:
            transcript = get_transcript(url, on_progress=status.write)
            method = get_last_transcript_method() or "unknown"
            status.write(f"✅ Transcript loaded ({len(transcript)} segments, via {method})")

            scored = score_segments(transcript, num_clips=NUM_CLIPS)
            status.write(f"✅ Top {len(scored)} moments identified")

            status.write("⏳ Cutting clips...")
            source_path = download_video(url)
            if "session_id" not in st.session_state:
                st.session_state["session_id"] = uuid.uuid4().hex
            session_clips_dir = os.path.join(CLIPS_DIR, st.session_state["session_id"])
            clip_paths = []
            for i, clip in enumerate(scored, start=1):
                out_path = os.path.join(session_clips_dir, f"clip_{i}.mp4")
                cut_and_reformat(
                    source_path, clip["start_time"], clip["end_time"], out_path,
                    transcript=transcript,
                )
                clip_paths.append(out_path)
            status.write(f"✅ {len(clip_paths)} clips ready")

            meta_clips = [
                {
                    "text": clip_transcript_text(transcript, c["start_time"], c["end_time"]),
                    "reason": c["reason"],
                }
                for c in scored
            ]
            metadata = generate_metadata_batch(meta_clips)
            status.write("✅ Titles & metadata generated")

            posting_time = get_best_posting_time()
            if posting_time:
                status.write("✅ Best posting time calculated")
            else:
                status.write("ℹ️ Analytics unavailable — connect a YouTube account to enable this")

            status.update(label="✅ Done", state="complete")

        st.session_state["results"] = {
            "scored": scored,
            "clip_paths": clip_paths,
            "metadata": metadata,
            "posting_time": posting_time,
        }
    except LLMQuotaExceededError as e:
        st.warning(f"🕒 {e}")
    except TranscriptError as e:
        _show_pipeline_error("Couldn't get a transcript for this video.", e)
    except ScorerError as e:
        _show_pipeline_error("Couldn't identify moments in this video.", e)
    except ClipperError as e:
        _show_pipeline_error("Couldn't cut clips from this video.", e)
    except MetadataError as e:
        _show_pipeline_error("Couldn't generate titles/descriptions for these clips.", e)

if "results" in st.session_state:
    results = st.session_state["results"]

    with sm_section("Your Shorts"):
        cols = st.columns(len(results["clip_paths"]) or 1)
        for i, col in enumerate(cols):
            if i >= len(results["clip_paths"]):
                continue
            with col:
                with st.container(key=f"clip-card-{i}"):
                    st.video(results["clip_paths"][i])
                    meta = results["metadata"][i]
                    title = html.escape(meta["title"])
                    description = html.escape(meta["description"])
                    reason = html.escape(results["scored"][i]["reason"])
                    st.markdown(f'<p class="sm-eyebrow">Clip {i + 1:02d}</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="sm-card-title">{title}</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="sm-card-desc">{description}</p>', unsafe_allow_html=True)
                    chips = "".join(f'<span class="sm-chip">{html.escape(h)}</span>' for h in meta["hashtags"])
                    st.markdown(f'<div class="sm-chip-row">{chips}</div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<p class="sm-card-reason">Why this clip: {reason}</p>',
                        unsafe_allow_html=True,
                    )

    with sm_section("📈 Best posting time"):
        posting_time = results["posting_time"]
        if posting_time is None:
            st.info("Analytics unavailable — connect a YouTube account to enable this.")
        else:
            with st.container(key="posting-panel"):
                windows = posting_time["top_windows"]
                callout_cols = st.columns(len(windows) or 1)
                for i, col in enumerate(callout_cols):
                    w = windows[i]
                    with col:
                        with st.container(key=f"callout-{i}"):
                            rank_label = "Best window" if i == 0 else "Runner-up"
                            st.markdown(f'<p class="sm-callout-label">{rank_label}</p>', unsafe_allow_html=True)
                            st.markdown(f'<p class="sm-callout-time">{w["label"]}</p>', unsafe_allow_html=True)
                            st.markdown(
                                f'<p class="sm-callout-meta">{w["weighted_views"]} views across '
                                f'{w["video_count"]} video(s)</p>',
                                unsafe_allow_html=True,
                            )
                st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)
                st.bar_chart(posting_time["views_by_hour"], color=ACCENT_BLUE)

    with sm_section():
        st.download_button(
            "⬇️ Download all clips (.zip)",
            data=_build_clips_zip(results["clip_paths"]),
            file_name="shorts_miner_clips.zip",
            mime="application/zip",
        )

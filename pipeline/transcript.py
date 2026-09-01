"""Pulls/parses video transcripts: YouTube captions first, local Whisper as a fallback.

Runnable standalone: python -m pipeline.transcript <youtube_url>
"""
import logging
import os
import re
import sys
import threading
from typing import Callable
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    CouldNotRetrieveTranscript,
    TranscriptsDisabled,
    VideoUnavailable,
)
from youtube_transcript_api.proxies import GenericProxyConfig

from utils.config import PROXY_URL, WHISPER_MODEL_SIZE

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

VIDEO_ID_RE = re.compile(r"^[0-9A-Za-z_-]{11}$")
PREFERRED_LANGUAGES = ("en", "en-US", "en-GB", "en-orig")

# Cap on how much audio Whisper actually transcribes. Uncapped, a full
# long-form video has no realistic time ceiling on a CPU-only host (measured
# 8+ minutes for a single short video under light concurrent load on
# Streamlit Community Cloud's free tier) — enough runway for the scorer's
# candidate windows either way, since it only needs a handful of usable
# 30-60s moments, not the whole transcript.
MAX_WHISPER_AUDIO_SECONDS = 600

# Only one Whisper transcription runs at a time per process. Concurrent
# visitors hitting the fallback simultaneously were observed starving each
# other for the host's single shared vCPU (one session's yt-dlp download
# stalled 8 minutes waiting for CPU behind another session's transcription)
# rather than each just running proportionally slower — serializing turns
# that into a predictable queue instead.
_whisper_semaphore = threading.Semaphore(1)

ProgressCallback = Callable[[str], None]


def _noop_progress(_msg: str) -> None:
    pass


# Set by get_transcript() on its most recent call; lets callers (e.g. the
# Streamlit UI) report which method produced the transcript without changing
# get_transcript's return type, which existing callers already depend on.
_last_method: str | None = None


def get_last_transcript_method() -> str | None:
    """Return 'captions' or 'whisper' for the most recent get_transcript() call,
    or None if get_transcript hasn't been called yet this process."""
    return _last_method


class TranscriptError(Exception):
    """Base exception for transcript retrieval failures."""


class InvalidYouTubeURLError(TranscriptError):
    """Raised when a video ID can't be extracted from the given URL."""


class VideoUnavailableError(TranscriptError):
    """Raised when the video is private, age-restricted, region-locked, or otherwise unavailable."""


class TranscriptUnavailableError(TranscriptError):
    """Raised when neither YouTube captions nor the Whisper fallback could produce a transcript."""


def extract_video_id(youtube_url: str) -> str:
    """Extract an 11-character YouTube video ID from common URL formats.

    Handles youtube.com/watch?v=, youtu.be/, youtube.com/shorts/, youtube.com/embed/,
    and youtube.com/v/ (with or without www./m. prefixes and extra query params).
    """
    parsed = urlparse(youtube_url.strip())
    host = (parsed.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    elif host.startswith("m."):
        host = host[2:]

    candidate = ""
    if host == "youtu.be":
        candidate = parsed.path.lstrip("/").split("/")[0]
    elif host in ("youtube.com", "youtube-nocookie.com"):
        if parsed.path == "/watch":
            candidate = parse_qs(parsed.query).get("v", [""])[0]
        elif parsed.path.startswith("/shorts/"):
            candidate = parsed.path[len("/shorts/"):].split("/")[0]
        elif parsed.path.startswith("/embed/"):
            candidate = parsed.path[len("/embed/"):].split("/")[0]
        elif parsed.path.startswith("/v/"):
            candidate = parsed.path[len("/v/"):].split("/")[0]

    if not candidate and VIDEO_ID_RE.match(youtube_url.strip()):
        # Allow passing a bare video ID directly.
        candidate = youtube_url.strip()

    if not candidate or not VIDEO_ID_RE.match(candidate):
        raise InvalidYouTubeURLError(f"Could not extract a video ID from URL: {youtube_url!r}")

    return candidate


def _fetch_captions(video_id: str) -> list[dict]:
    """Fetch captions via youtube_transcript_api, preferring English but falling back to
    whatever transcript is available (manual or auto-generated).

    Routed through PROXY_URL (utils.config) when set, same as the yt-dlp paths —
    useful when this host's IP itself is blocked, not just one yt-dlp client.
    """
    proxy_config = GenericProxyConfig(http_url=PROXY_URL, https_url=PROXY_URL) if PROXY_URL else None
    api = YouTubeTranscriptApi(proxy_config=proxy_config)
    transcript_list = api.list(video_id)

    try:
        transcript = transcript_list.find_transcript(PREFERRED_LANGUAGES)
    except Exception:
        transcript = next(iter(transcript_list), None)
        if transcript is None:
            raise TranscriptsDisabled(video_id)

    fetched = transcript.fetch()
    logger.info(
        "Fetched captions via youtube_transcript_api (language=%s, generated=%s)",
        fetched.language_code,
        fetched.is_generated,
    )
    return [
        {"text": snip["text"], "start": snip["start"], "duration": snip["duration"]}
        for snip in fetched.to_raw_data()
    ]


def _trim_audio(audio_path: str, max_seconds: int) -> str:
    """Truncate audio_path to at most max_seconds in place, via a fast stream-copy
    (no re-encode). Returns audio_path unchanged if trimming fails for any reason —
    transcribing the untrimmed file is preferable to hard-failing the fallback."""
    import ffmpeg

    trimmed_path = f"{audio_path}.trimmed.mp3"
    try:
        ffmpeg.input(audio_path, t=max_seconds).output(
            trimmed_path, acodec="copy"
        ).run(overwrite_output=True, quiet=True)
    except ffmpeg.Error:
        return audio_path
    if not os.path.exists(trimmed_path):
        return audio_path
    return trimmed_path


def _fetch_via_whisper(
    youtube_url: str, video_id: str, on_progress: ProgressCallback = _noop_progress
) -> list[dict]:
    """Fallback: download audio with yt-dlp, transcribe locally with Whisper."""
    import tempfile

    try:
        import whisper
    except ImportError as exc:
        raise TranscriptUnavailableError(
            "No captions available and openai-whisper is not installed for the fallback."
        ) from exc

    import yt_dlp

    from utils.ytdlp_client import extract_with_client_fallback

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path_template = os.path.join(tmpdir, f"{video_id}.%(ext)s")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": audio_path_template,
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
            }],
            "quiet": True,
            "no_warnings": True,
        }
        on_progress("⏳ No captions found — downloading audio for local transcription...")
        try:
            extract_with_client_fallback(ydl_opts, youtube_url, download=True)
        except yt_dlp.utils.DownloadError as exc:
            # Not VideoUnavailableError: the video itself may be perfectly fine —
            # this is a download-side failure (bot-check, IP block, etc.), same
            # category as captions already having failed above.
            raise TranscriptUnavailableError(
                f"Could not download audio for video {video_id}: {exc}"
            ) from exc

        audio_path = os.path.join(tmpdir, f"{video_id}.mp3")
        if not os.path.exists(audio_path):
            raise TranscriptUnavailableError(
                f"Audio download for video {video_id} did not produce an mp3 file."
            )
        audio_path = _trim_audio(audio_path, MAX_WHISPER_AUDIO_SECONDS)

        if not _whisper_semaphore.acquire(blocking=False):
            on_progress(
                "⏳ Another transcription is already running on this server — "
                "waiting for it to finish before starting..."
            )
            _whisper_semaphore.acquire()
        try:
            on_progress(f"⏳ Transcribing audio locally with Whisper ({WHISPER_MODEL_SIZE} model)...")
            logger.info("Transcribing audio locally with Whisper (%s model)...", WHISPER_MODEL_SIZE)
            model = whisper.load_model(WHISPER_MODEL_SIZE)
            result = model.transcribe(audio_path)
        finally:
            _whisper_semaphore.release()

        segments = [
            {
                "text": seg["text"].strip(),
                "start": float(seg["start"]),
                "duration": float(seg["end"]) - float(seg["start"]),
            }
            for seg in result["segments"]
        ]
        logger.info("Transcribed via Whisper (%d segments)", len(segments))
        return segments


def get_transcript(
    youtube_url: str, on_progress: ProgressCallback = _noop_progress
) -> list[dict]:
    """Get a transcript for a YouTube video as a list of {text, start, duration} dicts.

    Tries YouTube captions first; falls back to a local Whisper transcription of the
    downloaded audio if no captions are available. Raises TranscriptError subclasses
    on failure so callers (e.g. the Streamlit UI) can show a friendly message.

    on_progress, if given, is called with short human-readable strings at each phase
    transition (fetching captions, falling back to Whisper, downloading audio,
    transcribing) — the Whisper fallback in particular can take several minutes, and
    without visible progress a caller has no way to distinguish "still working" from
    "stuck".
    """
    global _last_method
    video_id = extract_video_id(youtube_url)

    on_progress("⏳ Fetching transcript...")
    try:
        segments = _fetch_captions(video_id)
        _last_method = "captions"
        return segments
    except VideoUnavailable as exc:
        raise VideoUnavailableError(
            f"Video {video_id} is unavailable (private, deleted, or region-locked): {exc}"
        ) from exc
    except (TranscriptsDisabled, CouldNotRetrieveTranscript) as exc:
        logger.info("No captions available for %s (%s); falling back to Whisper.", video_id, exc)
    except Exception as exc:
        logger.info("Caption fetch failed for %s (%s); falling back to Whisper.", video_id, exc)

    try:
        segments = _fetch_via_whisper(youtube_url, video_id, on_progress=on_progress)
        _last_method = "whisper"
        return segments
    except TranscriptError:
        raise
    except Exception as exc:
        raise TranscriptUnavailableError(
            f"Both YouTube captions and Whisper fallback failed for video {video_id}: {exc}"
        ) from exc


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m pipeline.transcript <youtube_url>")
        sys.exit(1)

    url = sys.argv[1]
    try:
        segments = get_transcript(url)
    except TranscriptError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    total_duration = segments[-1]["start"] + segments[-1]["duration"] if segments else 0
    print(f"\nGot {len(segments)} segments, ~{total_duration:.0f}s of video.\n")
    for seg in segments[:10]:
        print(f"[{seg['start']:7.2f}s +{seg['duration']:.2f}s] {seg['text']}")
    if len(segments) > 10:
        print(f"... ({len(segments) - 10} more segments)")

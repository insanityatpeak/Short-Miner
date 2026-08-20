"""Pulls/parses video transcripts: YouTube captions first, local Whisper as a fallback.

Runnable standalone: python -m pipeline.transcript <youtube_url>
"""
import logging
import os
import re
import sys
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    CouldNotRetrieveTranscript,
    TranscriptsDisabled,
    VideoUnavailable,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

VIDEO_ID_RE = re.compile(r"^[0-9A-Za-z_-]{11}$")
PREFERRED_LANGUAGES = ("en", "en-US", "en-GB", "en-orig")
WHISPER_MODEL_SIZE = "base"

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
    whatever transcript is available (manual or auto-generated)."""
    api = YouTubeTranscriptApi()
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


def _fetch_via_whisper(youtube_url: str, video_id: str) -> list[dict]:
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
        try:
            extract_with_client_fallback(ydl_opts, youtube_url, download=True)
        except yt_dlp.utils.DownloadError as exc:
            raise VideoUnavailableError(
                f"Could not download audio for video {video_id}: {exc}"
            ) from exc

        audio_path = os.path.join(tmpdir, f"{video_id}.mp3")
        if not os.path.exists(audio_path):
            raise TranscriptUnavailableError(
                f"Audio download for video {video_id} did not produce an mp3 file."
            )

        logger.info("Transcribing audio locally with Whisper (%s model)...", WHISPER_MODEL_SIZE)
        model = whisper.load_model(WHISPER_MODEL_SIZE)
        result = model.transcribe(audio_path)

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


def get_transcript(youtube_url: str) -> list[dict]:
    """Get a transcript for a YouTube video as a list of {text, start, duration} dicts.

    Tries YouTube captions first; falls back to a local Whisper transcription of the
    downloaded audio if no captions are available. Raises TranscriptError subclasses
    on failure so callers (e.g. the Streamlit UI) can show a friendly message.
    """
    global _last_method
    video_id = extract_video_id(youtube_url)

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
        segments = _fetch_via_whisper(youtube_url, video_id)
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

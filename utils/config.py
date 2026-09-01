"""Loads env vars, API keys, and shared constants for Shorts Miner."""
import os

from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
YOUTUBE_CLIENT_ID = os.getenv("YOUTUBE_CLIENT_ID")
YOUTUBE_CLIENT_SECRET = os.getenv("YOUTUBE_CLIENT_SECRET")

# Optional: route yt-dlp and youtube_transcript_api through an HTTP/SOCKS5
# proxy (e.g. http://user:pass@host:port) to work around YouTube blocking
# this host's IP outright — see utils.ytdlp_client and pipeline.transcript.
# Unset by default; the pipeline works without it as long as the host IP
# isn't currently in YouTube's bad graces.
PROXY_URL = os.getenv("PROXY_URL")

# Whisper model size for the local speech-to-text fallback (pipeline.transcript).
# "tiny" by default: on Streamlit Community Cloud's CPU-only, single-shared-vCPU
# free tier, "base" measured 8+ minutes to transcribe a single short video under
# light concurrent load — "tiny" trades some accuracy for finishing in a
# realistic time. Override via secret/env if a host with more CPU is used.
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "tiny")

CLAUDE_MODEL = "claude-sonnet-4-6"
GEMINI_MODEL = "gemini-3.5-flash-lite"

OUTPUT_DIR = "output"
SOURCE_DIR = os.path.join(OUTPUT_DIR, "source")
CLIPS_DIR = os.path.join(OUTPUT_DIR, "clips")

# YouTube's default (web) extraction client increasingly demands a PO token
# or bot-check sign-in and 403s/blocks yt-dlp without one. Which non-web
# client still serves formats without one shifts over time and by IP
# reputation (cloud/datacenter IPs like Streamlit Community Cloud's get
# blocked more aggressively than residential ones); every yt-dlp call in the
# pipeline goes through utils.ytdlp_client.extract_with_client_fallback,
# which retries across CLIENT_FALLBACK_ORDER on a 403 instead of pinning one
# client.

YOUTUBE_ANALYTICS_CONFIGURED = bool(YOUTUBE_CLIENT_ID and YOUTUBE_CLIENT_SECRET)


class ConfigError(Exception):
    """Raised when required configuration is missing."""


def require_anthropic_key() -> str:
    if not ANTHROPIC_API_KEY:
        raise ConfigError(
            "ANTHROPIC_API_KEY is missing. Add it to your .env file — "
            "this key is required for the core pipeline to run."
        )
    return ANTHROPIC_API_KEY


def require_gemini_key() -> str:
    if not GEMINI_API_KEY:
        raise ConfigError(
            "GEMINI_API_KEY is missing. Add it to your .env file — this key is required "
            "while the LLM provider is set to Gemini (see utils/llm.py). Get a free key "
            "at https://aistudio.google.com/apikey."
        )
    return GEMINI_API_KEY

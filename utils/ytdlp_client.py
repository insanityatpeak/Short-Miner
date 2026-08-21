"""Shared yt-dlp download-with-client-fallback helper.

YouTube periodically blocks one player client's requests — with a plain 403,
its "Sign in to confirm you're not a bot" challenge, a broken player response
("The page needs to be reloaded"), or a formats list that's missing anything
usable ("Requested format is not available") — while others still work, and
which client is currently blocked shifts without notice (see utils.config's
note on the android client for the last-known state). Both video and audio
download paths in this codebase go through extract_with_client_fallback()
instead of hard-failing on the first client that gets blocked.
"""
import logging

import yt_dlp

logger = logging.getLogger(__name__)

# Ordered by observed reliability without a PO token. Kept short: each retry
# re-does a full extract_info/download, so this trades a bit of latency on a
# bad day for actually working instead of hard-failing on one blocked client.
CLIENT_FALLBACK_ORDER = ["android", "ios", "tv", "web_safari"]

# Substrings (lowercased) that mark a DownloadError as "this client is
# blocked, try the next one" rather than a real failure (bad URL, private
# video, etc.) that no client swap will fix.
_BLOCKED_CLIENT_MARKERS = (
    "403",
    "forbidden",
    "sign in to confirm",
    "not a bot",
    "the page needs to be reloaded",
    "requested format is not available",
)


def _is_blocked_client_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(marker in msg for marker in _BLOCKED_CLIENT_MARKERS)


def extract_with_client_fallback(ydl_opts: dict, youtube_url: str, download: bool = True) -> dict:
    """Run yt_dlp.YoutubeDL.extract_info, retrying across CLIENT_FALLBACK_ORDER
    when a client is blocked (403, or YouTube's bot-check challenge).

    ydl_opts is used as given except "extractor_args" is set per attempt to
    pin a single client. Other errors (bad URL, private video, etc.) are not
    retried — they raise immediately since a different client won't fix
    them. Raises the last DownloadError if every client is blocked.
    """
    last_exc: yt_dlp.utils.DownloadError | None = None
    for i, client in enumerate(CLIENT_FALLBACK_ORDER):
        opts = {**ydl_opts, "extractor_args": {"youtube": {"player_client": [client]}}}
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                return ydl.extract_info(youtube_url, download=download)
        except yt_dlp.utils.DownloadError as exc:
            if not _is_blocked_client_error(exc):
                raise
            last_exc = exc
            remaining = CLIENT_FALLBACK_ORDER[i + 1:]
            if remaining:
                logger.warning(
                    "yt-dlp client %r blocked; retrying with %r.",
                    client, remaining[0],
                )
    raise last_exc

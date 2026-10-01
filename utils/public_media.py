"""Free media-acquisition fallback via public Piped / Invidious instances.

On cloud hosts YouTube blocks yt-dlp, but a public Piped/Invidious instance
resolves the stream URLs from its own IP and hands back googlevideo.com links
that this host can usually fetch directly. This module finds such links and
downloads them; callers (pipeline.clipper, pipeline.transcript) only use it
after yt-dlp has failed.

Instances are discovered dynamically (Invidious's instance API, Piped's public
instance list) with a small built-in seed list as a last resort, and probed in
parallel with short timeouts: most public instances are dead, rate-limited or
behind a bot-check at any given moment, and some Piped instances only return
Odysee streams (rejected here). No key, cookie or proxy is used.
"""
import logging
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

INVIDIOUS_LIST_URL = "https://api.invidious.io/instances.json?sort_by=type,users"
PIPED_LIST_URL = (
    "https://raw.githubusercontent.com/TeamPiped/documentation/main/content/docs/public-instances/index.md"
)
SEED_INVIDIOUS = ["https://invidious.f5.si", "https://inv.nadeko.net", "https://invidious.nerdvpn.de"]
SEED_PIPED = ["https://api.piped.private.coffee", "https://pipedapi.ducks.party", "https://pipedapi.kavin.rocks"]

MAX_INSTANCES = 12  # per kind
PROBE_TIMEOUT = 10  # seconds to resolve stream metadata from one instance
CHUNK_BYTES = 5_000_000  # googlevideo throttles/rejects large unchunked reads
CHUNK_TIMEOUT = 60


class PublicMediaError(Exception):
    """Raised when no public instance could provide the requested media."""


@dataclass
class Streams:
    source: str  # instance base URL, for logging
    video_url: str | None
    audio_url: str


def _discover(kind: str) -> list[str]:
    try:
        if kind == "invidious":
            data = httpx.get(INVIDIOUS_LIST_URL, timeout=10).json()
            found = [
                info["uri"] for _name, info in data
                if info.get("type") == "https" and info.get("api")
            ]
            seed = SEED_INVIDIOUS
        else:
            text = httpx.get(PIPED_LIST_URL, timeout=10).text
            found = re.findall(r"\|\s*(https://[^\s|]+)\s*\|", text)
            seed = SEED_PIPED
    except Exception as exc:
        logger.info("Could not fetch %s instance list (%s); using seed list.", kind, exc)
        found, seed = [], (SEED_INVIDIOUS if kind == "invidious" else SEED_PIPED)
    return list(dict.fromkeys(found + seed))[:MAX_INSTANCES]


def _height(size: str | None) -> int:
    try:
        return int(size.split("x")[1])
    except (AttributeError, IndexError, ValueError):
        return 0


def _resolve_invidious(base: str, video_id: str, max_height: int, want_video: bool) -> Streams:
    resp = httpx.get(f"{base}/api/v1/videos/{video_id}", timeout=PROBE_TIMEOUT)
    resp.raise_for_status()
    formats = resp.json()["adaptiveFormats"]
    audio = [f for f in formats if f["type"].startswith("audio/mp4")]
    if not audio:
        raise PublicMediaError("no audio stream")
    video_url = None
    if want_video:
        videos = [
            f for f in formats
            if f["type"].startswith("video/mp4") and "avc1" in f["type"]
            and 0 < _height(f.get("size")) <= max_height
        ]
        if not videos:
            raise PublicMediaError("no usable video stream")
        video_url = max(videos, key=lambda f: _height(f["size"]))["url"]
    return Streams(base, video_url, audio[0]["url"])


def _resolve_piped(base: str, video_id: str, max_height: int, want_video: bool) -> Streams:
    resp = httpx.get(f"{base}/streams/{video_id}", timeout=PROBE_TIMEOUT)
    resp.raise_for_status()
    body = resp.json()

    def usable(s):  # some instances only return Odysee/LBRY mirrors, which reject us
        return "googlevideo" in s.get("url", "") or "/videoplayback" in s.get("url", "")

    audio = [s for s in body.get("audioStreams", []) if s.get("mimeType", "").startswith("audio/mp4") and usable(s)]
    if not audio:
        raise PublicMediaError("no usable audio stream")
    video_url = None
    if want_video:
        videos = [
            s for s in body.get("videoStreams", [])
            if s.get("videoOnly") and s.get("mimeType", "").startswith("video/mp4")
            and "avc1" in s.get("codec", "") and 0 < (s.get("height") or 0) <= max_height and usable(s)
        ]
        if not videos:
            raise PublicMediaError("no usable video stream")
        video_url = max(videos, key=lambda s: s["height"])["url"]
    return Streams(base, video_url, audio[0]["url"])


def _download(url: str, dest: str) -> None:
    """Ranged chunked download; raises on any non-2xx (e.g. 429 rate limit)."""
    total = None
    pos = 0
    with open(dest, "wb") as f:
        while total is None or pos < total:
            resp = httpx.get(
                url, timeout=CHUNK_TIMEOUT, follow_redirects=True,
                headers={"Range": f"bytes={pos}-{pos + CHUNK_BYTES - 1}"},
            )
            resp.raise_for_status()
            if total is None:
                m = re.search(r"/(\d+)$", resp.headers.get("content-range", ""))
                total = int(m.group(1)) if m else len(resp.content)
            f.write(resp.content)
            pos += len(resp.content)
            if not resp.content:
                raise PublicMediaError("empty chunk")


def download_from_public_frontends(
    video_id: str, dest_path: str, max_height: int = 1080, audio_only: bool = False
) -> str:
    """Download a video (merged mp4, or audio-only .m4a if audio_only) to
    dest_path via the first public instance whose streams actually download.

    Raises PublicMediaError with a per-instance summary if every instance fails.
    """
    candidates = [("invidious", b) for b in _discover("invidious")] + [("piped", b) for b in _discover("piped")]
    resolvers = {"invidious": _resolve_invidious, "piped": _resolve_piped}
    failures: list[str] = []

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(resolvers[kind], base, video_id, max_height, not audio_only): f"{kind}:{base}"
            for kind, base in candidates
        }
        for fut in as_completed(futures):
            label = futures[fut]
            try:
                streams = fut.result()
            except Exception as exc:
                failures.append(f"{label}: {type(exc).__name__} {exc}"[:160])
                continue
            try:
                result = _fetch_streams(streams, dest_path, audio_only)
            except Exception as exc:
                failures.append(f"{label}: download failed ({type(exc).__name__} {exc})"[:160])
                continue
            for pending in futures:
                pending.cancel()
            logger.info("Fetched media for %s via %s", video_id, label)
            return result

    raise PublicMediaError(
        f"No public Piped/Invidious instance could provide video {video_id} "
        f"({len(failures)} tried): " + "; ".join(failures[:5])
    )


def _fetch_streams(streams: Streams, dest_path: str, audio_only: bool) -> str:
    # "_dl_" prefix keeps temp files from matching pipeline.clipper._find_cached
    # (which treats any "<video_id>.*" file as a finished download).
    folder, name = os.path.split(dest_path)
    base = os.path.join(folder, "_dl_" + os.path.splitext(name)[0])
    audio_tmp = f"{base}.audio.m4a"
    try:
        if audio_only:
            _download(streams.audio_url, dest_path)
            return dest_path
        video_tmp = f"{base}.video.mp4"
        try:
            _download(streams.video_url, video_tmp)
            _download(streams.audio_url, audio_tmp)
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-i", video_tmp, "-i", audio_tmp, "-c", "copy", dest_path],
                check=True,
            )
        finally:
            if os.path.exists(video_tmp):
                os.remove(video_tmp)
        return dest_path
    except Exception:
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise
    finally:
        if os.path.exists(audio_tmp) and not audio_only:
            os.remove(audio_tmp)

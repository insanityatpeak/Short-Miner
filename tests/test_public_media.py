"""Tests for the public Piped/Invidious media fallback. No network: HTTP and ffmpeg are mocked."""
import pytest

from utils import public_media as pm

INVIDIOUS_BODY = {"adaptiveFormats": [
    {"type": 'video/mp4; codecs="avc1.640028"', "size": "1920x1080", "url": "https://gv/v1080"},
    {"type": 'video/mp4; codecs="avc1.4d401f"', "size": "1280x720", "url": "https://gv/v720"},
    {"type": 'video/webm; codecs="vp9"', "size": "3840x2160", "url": "https://gv/webm"},
    {"type": 'audio/mp4; codecs="mp4a.40.2"', "url": "https://gv/audio"},
]}


class _Resp:
    def __init__(self, json_body=None, status=200):
        self._json, self.status_code = json_body, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._json


def test_invidious_picks_best_avc_mp4_within_height_cap(monkeypatch):
    monkeypatch.setattr(pm.httpx, "get", lambda *a, **kw: _Resp(INVIDIOUS_BODY))
    assert pm._resolve_invidious("https://i", "id", 1080, True).video_url == "https://gv/v1080"
    assert pm._resolve_invidious("https://i", "id", 720, True).video_url == "https://gv/v720"
    assert pm._resolve_invidious("https://i", "id", 720, False).video_url is None


def test_piped_rejects_odysee_only_streams(monkeypatch):
    body = {"videoStreams": [{"url": "https://player.odycdn.com/x", "mimeType": "video/mp4",
                              "codec": "avc1", "height": 720, "videoOnly": False}],
            "audioStreams": []}
    monkeypatch.setattr(pm.httpx, "get", lambda *a, **kw: _Resp(body))
    with pytest.raises(pm.PublicMediaError):
        pm._resolve_piped("https://p", "id", 1080, True)


def test_piped_accepts_googlevideo_streams(monkeypatch):
    body = {"videoStreams": [{"url": "https://x.googlevideo.com/videoplayback?a", "mimeType": "video/mp4",
                              "codec": "avc1.64", "height": 1080, "videoOnly": True}],
            "audioStreams": [{"url": "https://x.googlevideo.com/videoplayback?b", "mimeType": "audio/mp4"}]}
    monkeypatch.setattr(pm.httpx, "get", lambda *a, **kw: _Resp(body))
    s = pm._resolve_piped("https://p", "id", 1080, True)
    assert s.video_url.endswith("?a") and s.audio_url.endswith("?b")


def test_failover_skips_dead_and_rate_limited_instances(monkeypatch, tmp_path):
    monkeypatch.setattr(pm, "_discover", lambda kind: ["https://dead", "https://limited", "https://good"]
                        if kind == "invidious" else [])

    def resolve(base, video_id, max_height, want_video):
        if base == "https://dead":
            raise TimeoutError("timeout")
        return pm.Streams(base, "https://gv/v", "https://gv/a")

    def fetch(streams, dest, audio_only):
        if streams.source == "https://limited":
            raise RuntimeError("HTTP 429")
        return dest

    monkeypatch.setitem(pm.__dict__, "_fetch_streams", fetch)
    monkeypatch.setattr(pm, "_resolve_invidious", resolve)

    dest = str(tmp_path / "id.mp4")
    assert pm.download_from_public_frontends("id", dest) == dest


def test_all_instances_failing_raises_with_summary(monkeypatch, tmp_path):
    monkeypatch.setattr(pm, "_discover", lambda kind: ["https://a", "https://b"])

    def boom(*a, **kw):
        raise ConnectionError("down")

    monkeypatch.setattr(pm, "_resolve_invidious", boom)
    monkeypatch.setattr(pm, "_resolve_piped", boom)

    with pytest.raises(pm.PublicMediaError, match="4 tried"):
        pm.download_from_public_frontends("id", str(tmp_path / "id.mp4"))


def test_download_video_falls_back_to_public_instances(monkeypatch, tmp_path):
    import yt_dlp

    from pipeline import clipper

    def blocked(*a, **kw):
        raise yt_dlp.utils.DownloadError("Sign in to confirm you're not a bot")

    called = {}

    def fake_public(video_id, dest, max_height=1080, audio_only=False):
        called["dest"] = dest
        open(dest, "wb").close()
        return dest

    monkeypatch.setattr(clipper, "extract_with_client_fallback", blocked)
    monkeypatch.setattr(clipper, "download_from_public_frontends", fake_public)

    path = clipper.download_video("https://youtu.be/dQw4w9WgXcQ", output_path=str(tmp_path))

    assert path == called["dest"] == str(tmp_path / "dQw4w9WgXcQ.mp4")

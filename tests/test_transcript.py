"""Tests for video-ID extraction from YouTube URL formats. Pure function, no network."""
import pytest

from pipeline.transcript import InvalidYouTubeURLError, extract_video_id

VIDEO_ID = "dQw4w9WgXcQ"


@pytest.mark.parametrize("url", [
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://youtube.com/watch?v=dQw4w9WgXcQ",
    "http://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PLxyz&index=3",
    "https://m.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://youtu.be/dQw4w9WgXcQ",
    "https://youtu.be/dQw4w9WgXcQ?t=42",
    "https://www.youtube.com/shorts/dQw4w9WgXcQ",
    "https://www.youtube.com/embed/dQw4w9WgXcQ",
    "https://www.youtube.com/v/dQw4w9WgXcQ",
    "  https://www.youtube.com/watch?v=dQw4w9WgXcQ  ",
    "dQw4w9WgXcQ",
])
def test_extract_video_id_valid_formats(url):
    assert extract_video_id(url) == VIDEO_ID


@pytest.mark.parametrize("url", [
    "https://www.youtube.com/watch?v=short",
    "https://example.com/watch?v=dQw4w9WgXcQ",
    "not a url at all",
    "",
    "https://www.youtube.com/watch",
])
def test_extract_video_id_invalid_formats_raise(url):
    with pytest.raises(InvalidYouTubeURLError):
        extract_video_id(url)


def test_get_transcript_uses_local_media_instead_of_downloading(monkeypatch):
    from youtube_transcript_api._errors import TranscriptsDisabled

    from pipeline import transcript as t

    def no_captions(video_id):
        raise TranscriptsDisabled(video_id)

    def fail_download(*args, **kwargs):
        raise AssertionError("should not download audio when a local file is given")

    transcribed = []
    def no_public(video_id):
        raise RuntimeError("public service down")

    monkeypatch.setattr(t, "_fetch_captions", no_captions)
    monkeypatch.setattr(t, "_fetch_captions_public", no_public)
    monkeypatch.setattr(t, "_fetch_via_whisper", fail_download)
    monkeypatch.setattr(
        t, "_transcribe_file",
        lambda path, on_progress=None: transcribed.append(path) or [{"text": "hi", "start": 0.0, "duration": 1.0}],
    )

    segments = t.get_transcript("https://youtu.be/dQw4w9WgXcQ", local_media_path="upload.mp4")

    assert transcribed == ["upload.mp4"]
    assert segments == [{"text": "hi", "start": 0.0, "duration": 1.0}]
    assert t.get_last_transcript_method() == "whisper"


PUBLIC_MARKDOWN = """# Transcript: Some Title

Source video: https://www.youtube.com/watch?v=dQw4w9WgXcQ
Language: en · Duration: 1:05:30 · Words: 481

[0:00] First paragraph.

[0:19] Second paragraph, with: a colon.

[1:02:03] Hour-mark paragraph.
"""


def test_parse_public_transcript():
    from pipeline import transcript as t

    assert t._parse_public_transcript(PUBLIC_MARKDOWN) == [
        {"text": "First paragraph.", "start": 0.0, "duration": 19.0},
        {"text": "Second paragraph, with: a colon.", "start": 19.0, "duration": 3704.0},
        {"text": "Hour-mark paragraph.", "start": 3723.0, "duration": 207.0},
    ]


def test_parse_public_transcript_without_timestamps_is_empty():
    from pipeline import transcript as t

    assert t._parse_public_transcript("# Transcript\n\nno timestamps here") == []


def test_parse_public_transcript_last_duration_fallback():
    from pipeline import transcript as t

    assert t._parse_public_transcript("[0:10] only line") == [
        {"text": "only line", "start": 10.0, "duration": 5.0}
    ]


def test_fetch_captions_public_uses_mocked_http(monkeypatch):
    from unittest.mock import MagicMock

    from pipeline import transcript as t

    resp = MagicMock(text=PUBLIC_MARKDOWN)
    monkeypatch.setattr("httpx.get", lambda *a, **kw: resp)

    assert len(t._fetch_captions_public("dQw4w9WgXcQ")) == 3


def test_get_transcript_falls_back_direct_then_public_then_whisper(monkeypatch):
    from pipeline import transcript as t

    calls = []

    def direct_blocked(video_id):
        calls.append("direct")
        raise RuntimeError("blocked IP")

    def public_ok(video_id):
        calls.append("public")
        return [{"text": "hi", "start": 0.0, "duration": 1.0}]

    def whisper_unexpected(*a, **kw):
        raise AssertionError("whisper should not run when the public service works")

    monkeypatch.setattr(t, "_fetch_captions", direct_blocked)
    monkeypatch.setattr(t, "_fetch_captions_public", public_ok)
    monkeypatch.setattr(t, "_fetch_via_whisper", whisper_unexpected)

    segments = t.get_transcript("https://youtu.be/dQw4w9WgXcQ")

    assert calls == ["direct", "public"]
    assert segments[0]["text"] == "hi"
    assert "youtube-transcript.ai" in t.get_last_transcript_method()


def test_get_transcript_direct_success_skips_public(monkeypatch):
    from pipeline import transcript as t

    monkeypatch.setattr(t, "_fetch_captions", lambda v: [{"text": "x", "start": 0.0, "duration": 1.0}])
    monkeypatch.setattr(t, "_fetch_captions_public", lambda v: (_ for _ in ()).throw(AssertionError("no")))

    t.get_transcript("dQw4w9WgXcQ")

    assert t.get_last_transcript_method() == "captions"


def test_groq_transcription_offsets_each_chunk(monkeypatch):
    from unittest.mock import MagicMock

    from pipeline import transcript as t

    monkeypatch.setattr(t, "GROQ_API_KEY", "g")
    monkeypatch.setattr(t, "GROQ_CHUNK_SECONDS", 100)
    monkeypatch.setattr(t, "_media_duration", lambda path: 150.0)
    monkeypatch.setattr(t.subprocess, "run", lambda cmd, **kw: open(cmd[-1], "wb").close())
    resp = MagicMock()
    resp.json.return_value = {
        "segments": [{"text": " hello ", "start": 1.0, "end": 3.0}],
        "words": [{"word": "hello", "start": 1.5, "end": 2.5}],
    }
    monkeypatch.setattr("httpx.post", lambda *a, **kw: resp)

    segments = t._transcribe_file("video.mp4")

    assert segments == [
        {"text": "hello", "start": 1.0, "duration": 2.0,
         "words": [{"text": "hello", "start": 1.5, "end": 2.5}]},
        {"text": "hello", "start": 101.0, "duration": 2.0,
         "words": [{"text": "hello", "start": 101.5, "end": 102.5}]},
    ]


def test_groq_failure_falls_back_to_local_whisper(monkeypatch):
    from pipeline import transcript as t

    def boom(path):
        raise RuntimeError("429")

    monkeypatch.setattr(t, "GROQ_API_KEY", "g")
    monkeypatch.setattr(t, "_transcribe_with_groq", boom)
    monkeypatch.setattr(t, "_transcribe_locally", lambda path, on_progress=None: ["local"])

    assert t._transcribe_file("video.mp4") == ["local"]

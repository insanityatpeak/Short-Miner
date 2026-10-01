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
    monkeypatch.setattr(t, "_fetch_captions", no_captions)
    monkeypatch.setattr(t, "_fetch_via_whisper", fail_download)
    monkeypatch.setattr(
        t, "_transcribe_file",
        lambda path, on_progress=None: transcribed.append(path) or [{"text": "hi", "start": 0.0, "duration": 1.0}],
    )

    segments = t.get_transcript("https://youtu.be/dQw4w9WgXcQ", local_media_path="upload.mp4")

    assert transcribed == ["upload.mp4"]
    assert segments == [{"text": "hi", "start": 0.0, "duration": 1.0}]
    assert t.get_last_transcript_method() == "whisper"

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

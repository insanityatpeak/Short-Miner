"""Tests for clipper.py's pure timestamp/boundary math: clip-range validation,
caption-chunk windowing and non-overlap, and unbreakable-unit chunking. No
ffmpeg/yt-dlp/network calls are exercised here."""
import pytest

from pipeline.clipper import (
    ClipCutError,
    _build_unbreakable_units,
    _clamp_segment_durations,
    _format_ass_timestamp,
    _round_even,
    build_caption_chunks,
    cut_clip,
)


def _transcript_segment(text, start, duration):
    return {"text": text, "start": start, "duration": duration}


def test_cut_clip_rejects_end_before_start(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"not a real video, but the range check happens first")
    with pytest.raises(ClipCutError):
        cut_clip(str(source), start=30.0, end=10.0, output_path=str(tmp_path / "out.mp4"))


def test_cut_clip_rejects_zero_length_range(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"x")
    with pytest.raises(ClipCutError):
        cut_clip(str(source), start=10.0, end=10.0, output_path=str(tmp_path / "out.mp4"))


def test_cut_clip_rejects_missing_source(tmp_path):
    with pytest.raises(ClipCutError):
        cut_clip(
            str(tmp_path / "does_not_exist.mp4"), start=0.0, end=10.0,
            output_path=str(tmp_path / "out.mp4"),
        )


@pytest.mark.parametrize("value", [0.4, 1.5, 2.5, 3.49, 100.0, 0.0])
def test_round_even_always_returns_even(value):
    assert _round_even(value) % 2 == 0


@pytest.mark.parametrize("seconds,expected", [
    (0, "0:00:00.00"),
    (65.5, "0:01:05.50"),
    (3661.239, "1:01:01.24"),
])
def test_format_ass_timestamp(seconds, expected):
    assert _format_ass_timestamp(seconds) == expected


def test_build_caption_chunks_chunks_are_ordered_and_non_overlapping():
    transcript = [
        _transcript_segment(
            "But today the average eighteen year old in the United States is on pace to spend",
            171.07, 5.9,
        ),
        _transcript_segment("ninety three percent of their remaining free time looking at a screen", 179.58, 5.1),
    ]
    chunks = build_caption_chunks(transcript, clip_start=171.1, clip_end=225.4)

    assert len(chunks) > 1
    for i in range(len(chunks) - 1):
        assert chunks[i]["end"] <= chunks[i + 1]["start"] + 1e-9
    for c in chunks:
        assert 0.0 <= c["start"] < c["end"] <= (225.4 - 171.1) + 1e-6


def test_build_caption_chunks_preserves_document_order_across_overlapping_segments():
    # Regression test: real YouTube caption segments sometimes overlap in time
    # (one segment's window hasn't ended before the next begins). Word order
    # in the rendered caption must follow transcript array order — true
    # speaking order — not a re-sort by each word's proportionally-estimated
    # start time, which can interleave words out of order across the overlap.
    transcript = [
        _transcript_segment("But before Computer", 0.0, 9.0),
        _transcript_segment("modern science", 3.0, 4.0),  # overlaps the segment above
    ]
    chunks = build_caption_chunks(transcript, clip_start=0.0, clip_end=20.0)

    all_words = [w for c in chunks for w in c["words"]]
    stripped = [w.strip(".,!?").lower() for w in all_words]
    assert stripped == ["but", "before", "computer", "modern", "science"]


def test_clamp_segment_durations_prevents_bleed_into_next_segment():
    # Regression test: real auto-generated captions declare overlapping windows
    # (a "rolling" smoothing artifact) even though text never repeats between
    # segments. Segment i's effective end must never exceed segment i+1's start.
    segments = [
        _transcript_segment("understood as a mathematical machine,", 20.80, 4.36),  # ends 25.16
        _transcript_segment("laying the groundwork for both modern", 23.40, 3.60),   # overlaps prev
        _transcript_segment("computer science and artificial", 25.16, 4.68),         # overlaps prev
        _transcript_segment("intelligence.", 27.00, 2.84),                            # overlaps prev
    ]
    clamped = _clamp_segment_durations(segments)
    for i in range(len(clamped) - 1):
        this_end = clamped[i]["start"] + clamped[i]["duration"]
        next_start = clamped[i + 1]["start"]
        assert this_end <= next_start + 1e-9


def test_build_caption_chunks_no_overlap_with_real_style_overlapping_segments():
    # Same overlapping-window shape as real auto-captions, reproducing the
    # user-reported "But before Computer... modern science... Intelligence"
    # scramble and its accompanying overlapping-timestamp bug.
    transcript = [
        _transcript_segment("understood as a mathematical machine,", 20.80, 4.36),
        _transcript_segment("laying the groundwork for both modern", 23.40, 3.60),
        _transcript_segment("computer science and artificial", 25.16, 4.68),
        _transcript_segment("intelligence.", 27.00, 2.84),
        _transcript_segment("But before we get into the math, you", 30.48, 4.52),
    ]
    chunks = build_caption_chunks(transcript, clip_start=20.80, clip_end=35.00)

    for i in range(len(chunks) - 1):
        assert chunks[i]["end"] <= chunks[i + 1]["start"] + 1e-9

    all_words = [w.strip(".,!?").lower() for c in chunks for w in c["words"]]
    computer_idx = all_words.index("computer")
    modern_idx = all_words.index("modern")
    intelligence_idx = all_words.index("intelligence")
    before_idx = all_words.index("before")
    assert modern_idx < computer_idx < intelligence_idx < before_idx


def test_build_caption_chunks_excludes_segments_outside_clip_window():
    transcript = [
        _transcript_segment("inside the clip window", 10.0, 3.0),
        _transcript_segment("way outside the clip window", 500.0, 3.0),
    ]
    chunks = build_caption_chunks(transcript, clip_start=0.0, clip_end=20.0)

    all_words = " ".join(c["text"] for c in chunks)
    assert "outside" not in all_words or "way" not in all_words
    assert "inside" in all_words.lower() or "window" in all_words.lower()


def test_build_caption_chunks_empty_window_returns_empty_list():
    transcript = [_transcript_segment("only word", 500.0, 3.0)]
    chunks = build_caption_chunks(transcript, clip_start=0.0, clip_end=20.0)
    assert chunks == []


def test_unbreakable_units_keeps_proper_noun_together():
    # Regression test: chunk boundaries must never split "United States".
    words = [
        {"text": "the", "start": 0.0, "end": 0.3},
        {"text": "United", "start": 0.3, "end": 0.7},
        {"text": "States", "start": 0.7, "end": 1.1},
        {"text": "is", "start": 1.1, "end": 1.3},
    ]
    units = _build_unbreakable_units(words)
    unit_texts = [[w["text"] for w in u] for u in units]
    assert ["United", "States"] in unit_texts


def test_unbreakable_units_merges_number_with_next_word():
    words = [
        {"text": "the", "start": 0.0, "end": 0.3},
        {"text": "18", "start": 0.3, "end": 0.6},
        {"text": "year", "start": 0.6, "end": 0.9},
        {"text": "old", "start": 0.9, "end": 1.1},
    ]
    units = _build_unbreakable_units(words)
    unit_texts = [[w["text"] for w in u] for u in units]
    assert ["18", "year"] in unit_texts


def test_unbreakable_units_does_not_merge_sentence_initial_article():
    # "The United States" should merge as "United States" only, not "The United".
    words = [
        {"text": "The", "start": 0.0, "end": 0.2},
        {"text": "United", "start": 0.2, "end": 0.5},
        {"text": "States", "start": 0.5, "end": 0.9},
    ]
    units = _build_unbreakable_units(words)
    unit_texts = [[w["text"] for w in u] for u in units]
    assert ["The"] in unit_texts
    assert ["United", "States"] in unit_texts

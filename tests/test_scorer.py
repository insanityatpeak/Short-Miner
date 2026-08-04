"""Tests for scorer.py's defensive JSON parsing and clip-selection logic.
The Claude/LLM call is always mocked — no live API calls."""
import json

import pytest

import pipeline.scorer as scorer
from pipeline.scorer import (
    ClaudeResponseError,
    InsufficientSegmentsError,
    ScorerError,
    score_segments,
)


def _transcript(num_lines=40, line_duration=5.0):
    return [
        {"text": f"This is line number {i} of the transcript.", "start": i * line_duration, "duration": line_duration}
        for i in range(num_lines)
    ]


def test_parse_json_response_strips_code_fences():
    raw = '```json\n[{"start_time": 0, "end_time": 30, "score": 90, "reason": "hook"}]\n```'
    data = scorer._parse_json_response(raw)
    assert data == [{"start_time": 0, "end_time": 30, "score": 90, "reason": "hook"}]


def test_parse_json_response_plain_json_no_fences():
    raw = '[{"start_time": 0, "end_time": 30, "score": 90, "reason": "hook"}]'
    data = scorer._parse_json_response(raw)
    assert len(data) == 1


def test_parse_json_response_invalid_json_raises():
    with pytest.raises(ClaudeResponseError):
        scorer._parse_json_response("this is not json")


def test_parse_json_response_non_list_raises():
    with pytest.raises(ClaudeResponseError):
        scorer._parse_json_response('{"start_time": 0}')


def test_clamp_duration_shrinks_overlong_clip():
    clip = {"start_time": 10.0, "end_time": 200.0, "score": 80, "reason": "x"}
    clamped = scorer._clamp_duration(clip, transcript_end=300.0)
    assert clamped["end_time"] - clamped["start_time"] == scorer.MAX_CLIP_DURATION


def test_clamp_duration_grows_undershort_clip():
    clip = {"start_time": 10.0, "end_time": 15.0, "score": 80, "reason": "x"}
    clamped = scorer._clamp_duration(clip, transcript_end=300.0)
    assert clamped["end_time"] - clamped["start_time"] == scorer.MIN_CLIP_DURATION


def test_enforce_non_overlap_picks_highest_scoring_non_overlapping():
    clips = [
        {"start_time": 0, "end_time": 40, "score": 90, "reason": "a"},
        {"start_time": 20, "end_time": 60, "score": 95, "reason": "b (overlaps a)"},
        {"start_time": 100, "end_time": 140, "score": 70, "reason": "c"},
    ]
    selected = scorer._enforce_non_overlap(clips, num_clips=3)
    starts_ends = [(c["start_time"], c["end_time"]) for c in selected]
    # b (score 95) wins over a (score 90) since they overlap; c never overlaps either.
    assert {"b (overlaps a)", "c"} == {c["reason"] for c in selected}
    for i in range(len(starts_ends)):
        for j in range(i + 1, len(starts_ends)):
            s1, e1 = starts_ends[i]
            s2, e2 = starts_ends[j]
            assert not (s1 < e2 and e1 > s2), "selected clips must not overlap"


def test_score_segments_empty_transcript_raises():
    with pytest.raises(ScorerError):
        score_segments([], num_clips=3)


def test_score_segments_too_few_windows_raises(monkeypatch):
    monkeypatch.setattr(scorer, "call_llm", lambda prompt: "[]")
    with pytest.raises(InsufficientSegmentsError):
        score_segments(_transcript(num_lines=2, line_duration=2.0), num_clips=3)


def test_score_segments_happy_path_mocked(monkeypatch):
    fake_response = json.dumps([
        {"start_time": 0.0, "end_time": 40.0, "score": 95, "reason": "strong hook"},
        {"start_time": 60.0, "end_time": 100.0, "score": 85, "reason": "quotable"},
        {"start_time": 120.0, "end_time": 155.0, "score": 75, "reason": "emotional peak"},
    ])
    monkeypatch.setattr(scorer, "call_llm", lambda prompt: fake_response)

    clips = score_segments(_transcript(), num_clips=3)

    assert len(clips) == 3
    assert [c["score"] for c in clips] == sorted((c["score"] for c in clips), reverse=True)
    for c in clips:
        assert scorer.MIN_CLIP_DURATION <= (c["end_time"] - c["start_time"]) <= scorer.MAX_CLIP_DURATION


def test_score_segments_malformed_items_are_skipped(monkeypatch):
    fake_response = json.dumps([
        {"start_time": 0.0, "end_time": 40.0, "score": 95, "reason": "ok"},
        {"start_time": "not-a-number", "end_time": 40.0, "score": 50, "reason": "bad"},
        {"end_time": 40.0, "score": 50, "reason": "missing start_time"},
    ])
    monkeypatch.setattr(scorer, "call_llm", lambda prompt: fake_response)

    clips = score_segments(_transcript(), num_clips=3)
    assert len(clips) == 1
    assert clips[0]["reason"] == "ok"

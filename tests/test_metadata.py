"""Tests for metadata.py's defensive JSON parsing and validation logic.
The Claude/LLM call is always mocked — no live API calls."""
import json

import pytest

import pipeline.metadata as metadata
from pipeline.metadata import (
    ClaudeMetadataResponseError,
    MetadataError,
    generate_metadata,
    generate_metadata_batch,
)


def test_parse_json_response_strips_code_fences():
    raw = '```json\n[{"title": "Hook", "description": "d.", "hashtags": ["a", "b"]}]\n```'
    data = metadata._parse_json_response(raw)
    assert data == [{"title": "Hook", "description": "d.", "hashtags": ["a", "b"]}]


def test_parse_json_response_invalid_json_raises():
    with pytest.raises(ClaudeMetadataResponseError):
        metadata._parse_json_response("not json")


def test_parse_json_response_non_list_raises():
    with pytest.raises(ClaudeMetadataResponseError):
        metadata._parse_json_response('{"title": "x"}')


def test_validate_metadata_item_adds_missing_hash_prefix():
    item = {"title": "Hook title", "description": "desc.", "hashtags": ["tag1", "#tag2"]}
    result = metadata._validate_metadata_item(item)
    assert result["hashtags"] == ["#tag1", "#tag2"]


def test_validate_metadata_item_empty_title_raises():
    with pytest.raises(ClaudeMetadataResponseError):
        metadata._validate_metadata_item({"title": "", "description": "d", "hashtags": ["a"]})


def test_validate_metadata_item_missing_key_raises():
    with pytest.raises(ClaudeMetadataResponseError):
        metadata._validate_metadata_item({"title": "t", "hashtags": ["a"]})


def test_validate_metadata_item_no_usable_hashtags_raises():
    with pytest.raises(ClaudeMetadataResponseError):
        metadata._validate_metadata_item({"title": "t", "description": "d", "hashtags": ["#", ""]})


def test_generate_metadata_batch_empty_list_raises():
    with pytest.raises(MetadataError):
        generate_metadata_batch([])


def test_generate_metadata_batch_happy_path_mocked(monkeypatch):
    fake_response = json.dumps([
        {"title": "You'll Spend 93% on Screens", "description": "A sobering stat.", "hashtags": ["screentime", "mindfulness"]},
        {"title": "TikTok Costs $1,200/Month", "description": "The real cost of scrolling.", "hashtags": ["#tiktok", "#money"]},
    ])
    monkeypatch.setattr(metadata, "call_llm", lambda prompt: fake_response)

    clips = [
        {"text": "transcript one", "reason": "hook"},
        {"text": "transcript two", "reason": "quotable"},
    ]
    results = generate_metadata_batch(clips)

    assert len(results) == 2
    assert results[0]["title"] == "You'll Spend 93% on Screens"
    assert results[0]["hashtags"] == ["#screentime", "#mindfulness"]
    assert results[1]["hashtags"] == ["#tiktok", "#money"]


def test_generate_metadata_batch_fewer_items_than_clips_raises(monkeypatch):
    fake_response = json.dumps([
        {"title": "Only one", "description": "d.", "hashtags": ["a"]},
    ])
    monkeypatch.setattr(metadata, "call_llm", lambda prompt: fake_response)

    clips = [{"text": "one", "reason": "r1"}, {"text": "two", "reason": "r2"}]
    with pytest.raises(ClaudeMetadataResponseError):
        generate_metadata_batch(clips)


def test_generate_metadata_single_clip_wrapper_mocked(monkeypatch):
    fake_response = json.dumps([
        {"title": "Single Clip Title", "description": "desc.", "hashtags": ["a", "b", "c"]},
    ])
    monkeypatch.setattr(metadata, "call_llm", lambda prompt: fake_response)

    result = generate_metadata("some transcript text", "why this clip was chosen")
    assert result["title"] == "Single Clip Title"
    assert len(result["hashtags"]) == 3

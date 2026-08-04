"""Scores transcript segments with Claude and picks the best standalone-Short candidates.

Runnable standalone: python -m pipeline.scorer <youtube_url> [num_clips]
"""
import json
import logging
import sys

from utils.llm import LLMError, call_llm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

MIN_CLIP_DURATION = 30.0
MAX_CLIP_DURATION = 60.0


class ScorerError(Exception):
    """Base exception for segment scoring failures."""


class InsufficientSegmentsError(ScorerError):
    """Raised when the transcript is too short to produce the requested number of clips."""


class ClaudeResponseError(ScorerError):
    """Raised when Claude's response can't be parsed into usable clip candidates."""


def _merge_into_windows(
    transcript: list[dict],
    min_duration: float = MIN_CLIP_DURATION,
    max_duration: float = MAX_CLIP_DURATION,
) -> list[dict]:
    """Merge adjacent transcript lines into candidate windows of roughly min-max seconds."""
    windows = []
    current = None

    for seg in transcript:
        start = seg["start"]
        end = seg["start"] + seg["duration"]
        text = seg["text"].replace("\n", " ")

        if current is None:
            current = {"start": start, "end": end, "texts": [text]}
            continue

        would_span = end - current["start"]
        already_long_enough = (current["end"] - current["start"]) >= min_duration
        if would_span > max_duration and already_long_enough:
            windows.append(current)
            current = {"start": start, "end": end, "texts": [text]}
        else:
            current["end"] = end
            current["texts"].append(text)

    if current is not None:
        windows.append(current)

    return [
        {"start": w["start"], "end": w["end"], "text": " ".join(w["texts"])}
        for w in windows
    ]


def _build_prompt(windows: list[dict], num_clips: int) -> str:
    numbered = "\n\n".join(
        f"[Window {i}] {w['start']:.1f}s - {w['end']:.1f}s:\n{w['text']}"
        for i, w in enumerate(windows)
    )
    return f"""You are selecting the best moments from a long-form YouTube video transcript to turn into standalone YouTube Shorts.

Below is the transcript, split into candidate time windows with start/end timestamps in seconds:

{numbered}

Identify the {num_clips} best candidate windows for standalone YouTube Shorts. Score each 0-100 based on:
- Hook strength: does it open mid-action or with a strong statement?
- Self-containedness: does it make sense without prior context?
- Emotional/energy peak: is this a high-energy or emotionally resonant moment?
- Quotability: does it contain a memorable, shareable line?

You may use a candidate window's timestamps as-is, or tighten the start/end within its boundaries to sharpen the hook. Each final clip must be between {MIN_CLIP_DURATION:.0f} and {MAX_CLIP_DURATION:.0f} seconds long, and no two clips may overlap in time.

Respond with ONLY valid JSON, no markdown code fences, no preamble, no explanation outside the JSON. The JSON must be a list of exactly {num_clips} objects, each with these keys:
- "start_time": number (seconds)
- "end_time": number (seconds)
- "score": number (0-100)
- "reason": string (one sentence explaining why this moment works as a Short)
"""


def _call_llm(prompt: str) -> str:
    try:
        return call_llm(prompt)
    except LLMError as exc:
        raise ScorerError(str(exc)) from exc


def _parse_json_response(raw: str) -> list:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ClaudeResponseError(
            f"Claude did not return valid JSON: {exc}\nRaw response: {raw[:500]}"
        ) from exc

    if not isinstance(data, list):
        raise ClaudeResponseError(
            f"Expected a JSON list of clip candidates from Claude, got {type(data).__name__}"
        )
    return data


def _clamp_duration(clip: dict, transcript_end: float) -> dict:
    start, end = clip["start_time"], clip["end_time"]
    duration = end - start

    if duration > MAX_CLIP_DURATION:
        end = start + MAX_CLIP_DURATION
    elif duration < MIN_CLIP_DURATION:
        end = min(start + MIN_CLIP_DURATION, transcript_end)
        if end - start < MIN_CLIP_DURATION:
            start = max(0.0, end - MIN_CLIP_DURATION)

    clip["start_time"], clip["end_time"] = start, end
    return clip


def _enforce_non_overlap(clips: list[dict], num_clips: int) -> list[dict]:
    """Greedily pick the highest-scoring clips that don't overlap in time."""
    clips_by_score = sorted(clips, key=lambda c: c["score"], reverse=True)
    selected = []
    for clip in clips_by_score:
        overlaps = any(
            clip["start_time"] < s["end_time"] and clip["end_time"] > s["start_time"]
            for s in selected
        )
        if overlaps:
            continue
        selected.append(clip)
        if len(selected) == num_clips:
            break
    return selected


def score_segments(transcript: list[dict], num_clips: int = 3) -> list[dict]:
    """Ask Claude to pick the num_clips best standalone-Short moments from a transcript.

    Returns a list of dicts sorted by score descending, each with start_time, end_time,
    score (0-100), and reason. Raises ScorerError subclasses on failure.
    """
    if not transcript:
        raise ScorerError("Cannot score an empty transcript.")

    windows = _merge_into_windows(transcript)
    if len(windows) < num_clips:
        raise InsufficientSegmentsError(
            f"Transcript only yields {len(windows)} candidate window(s) of "
            f"{MIN_CLIP_DURATION:.0f}-{MAX_CLIP_DURATION:.0f}s; need at least {num_clips}."
        )

    transcript_end = transcript[-1]["start"] + transcript[-1]["duration"]

    prompt = _build_prompt(windows, num_clips)
    raw = _call_llm(prompt)
    data = _parse_json_response(raw)

    clips = []
    for item in data:
        try:
            clip = {
                "start_time": float(item["start_time"]),
                "end_time": float(item["end_time"]),
                "score": float(item["score"]),
                "reason": str(item["reason"]).strip(),
            }
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Skipping malformed clip candidate from Claude: %s (%s)", item, exc)
            continue
        if clip["end_time"] <= clip["start_time"]:
            continue
        clips.append(_clamp_duration(clip, transcript_end))

    if not clips:
        raise ClaudeResponseError("Claude returned no usable clip candidates.")

    selected = _enforce_non_overlap(clips, num_clips)
    if not selected:
        raise ScorerError("No non-overlapping clip candidates could be selected.")
    if len(selected) < num_clips:
        logger.warning(
            "Only found %d non-overlapping clip(s) (requested %d).", len(selected), num_clips
        )

    selected.sort(key=lambda c: c["score"], reverse=True)
    return selected


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("Usage: python -m pipeline.scorer <youtube_url> [num_clips]")
        sys.exit(1)

    from pipeline.transcript import TranscriptError, get_transcript

    url = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) == 3 else 3

    try:
        transcript = get_transcript(url)
        clips = score_segments(transcript, num_clips=n)
    except (TranscriptError, ScorerError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"\nTop {len(clips)} clip(s):\n")
    for i, clip in enumerate(clips, 1):
        duration = clip["end_time"] - clip["start_time"]
        print(
            f"#{i}  score={clip['score']:.0f}  "
            f"[{clip['start_time']:.1f}s - {clip['end_time']:.1f}s]  ({duration:.1f}s)"
        )
        print(f"    reason: {clip['reason']}\n")

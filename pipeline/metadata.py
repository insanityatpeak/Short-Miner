"""Generates hook-driven titles, descriptions, and hashtags for cut clips.

Runnable standalone: python -m pipeline.metadata <youtube_url> [num_clips]
"""
import logging
import sys

from utils.llm import LLMError, call_llm, parse_json_list

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

TITLE_MAX_CHARS = 60
MIN_HASHTAGS = 3
MAX_HASHTAGS = 5


class MetadataError(Exception):
    """Base exception for metadata generation failures."""


class ClaudeMetadataResponseError(MetadataError):
    """Raised when Claude's response can't be parsed into usable metadata."""


def _build_prompt(clips: list[dict]) -> str:
    numbered = "\n\n".join(
        f"[Clip {i}]\nTranscript: {c['text']}\nWhy this clip was chosen: {c['reason']}"
        for i, c in enumerate(clips)
    )
    return f"""You are writing YouTube Shorts metadata for {len(clips)} short clips cut from a longer video.

{numbered}

For each clip, write:
- "title": a hook-driven title under {TITLE_MAX_CHARS} characters that would make someone stop scrolling
- "description": 1-2 sentences describing the clip
- "hashtags": {MIN_HASHTAGS}-{MAX_HASHTAGS} relevant hashtags (each starting with #, no spaces)

Respond with ONLY valid JSON, no markdown code fences, no preamble, no explanation outside the JSON. The JSON must be a list of exactly {len(clips)} objects, in the same order as the clips above, each with keys "title", "description", "hashtags" (a list of strings).
"""


def _call_llm(prompt: str) -> str:
    try:
        return call_llm(prompt)
    except LLMError as exc:
        raise MetadataError(str(exc)) from exc


def _parse_json_response(raw: str) -> list:
    return parse_json_list(raw, ClaudeMetadataResponseError)


def _validate_metadata_item(item: dict) -> dict:
    try:
        title = str(item["title"]).strip()
        description = str(item["description"]).strip()
        hashtags = [str(h).strip() for h in item["hashtags"]]
    except (KeyError, TypeError) as exc:
        raise ClaudeMetadataResponseError(
            f"Malformed metadata item from Claude: {item} ({exc})"
        ) from exc

    if not title or not description:
        raise ClaudeMetadataResponseError(f"Empty title/description in metadata item: {item}")

    hashtags = [h if h.startswith("#") else f"#{h}" for h in hashtags if h.strip("#")]
    if not hashtags:
        raise ClaudeMetadataResponseError(f"No usable hashtags in metadata item: {item}")

    return {"title": title, "description": description, "hashtags": hashtags}


def generate_metadata_batch(clips: list[dict]) -> list[dict]:
    """Generate title/description/hashtags for multiple clips in a single Claude call.

    clips: list of {"text": str, "reason": str} — transcript text and the reason
    the clip was chosen (e.g. from scorer.py's output). Batched into one LLM
    call to save latency/cost vs. one call per clip. Returns a list of
    {"title", "description", "hashtags"} dicts in the same order. Raises
    MetadataError subclasses on failure.
    """
    if not clips:
        raise MetadataError("Cannot generate metadata for an empty clip list.")

    prompt = _build_prompt(clips)
    raw = _call_llm(prompt)
    data = _parse_json_response(raw)

    if len(data) < len(clips):
        raise ClaudeMetadataResponseError(
            f"Claude returned {len(data)} metadata item(s) for {len(clips)} clips."
        )

    results = [_validate_metadata_item(item) for item in data[:len(clips)]]
    return results


def generate_metadata(clip_text: str, clip_reason: str) -> dict:
    """Generate title/description/hashtags for a single clip.

    Thin wrapper around generate_metadata_batch. Prefer calling the batch
    function directly when generating metadata for multiple clips — one
    Claude call instead of several.
    """
    return generate_metadata_batch([{"text": clip_text, "reason": clip_reason}])[0]


def clip_transcript_text(transcript: list[dict], start: float, end: float) -> str:
    return " ".join(
        seg["text"].replace("\n", " ")
        for seg in transcript
        if seg["start"] < end and seg["start"] + seg["duration"] > start
    )


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("Usage: python -m pipeline.metadata <youtube_url> [num_clips]")
        sys.exit(1)

    from pipeline.scorer import ScorerError, score_segments
    from pipeline.transcript import TranscriptError, get_transcript

    url = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) == 3 else 3

    try:
        transcript = get_transcript(url)
        scored = score_segments(transcript, num_clips=n)
        clips = [
            {"text": clip_transcript_text(transcript, c["start_time"], c["end_time"]), "reason": c["reason"]}
            for c in scored
        ]
        results = generate_metadata_batch(clips)
    except (TranscriptError, ScorerError, MetadataError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    for i, (scored_clip, meta) in enumerate(zip(scored, results), 1):
        print(f"\n#{i}  [{scored_clip['start_time']:.1f}s - {scored_clip['end_time']:.1f}s]  score={scored_clip['score']:.0f}")
        print(f"  Title: {meta['title']}")
        print(f"  Description: {meta['description']}")
        print(f"  Hashtags: {' '.join(meta['hashtags'])}")

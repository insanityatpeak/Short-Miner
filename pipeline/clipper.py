"""Downloads the source video (yt-dlp), cuts timestamped clips (ffmpeg), and
reformats them to vertical 9:16 for Shorts.

Runnable standalone: python -m pipeline.clipper <youtube_url> <start> <end> [output_name]
"""
import logging
import os
import re
import statistics
import sys
import tempfile

import ffmpeg

from pipeline.transcript import extract_video_id
from utils.config import CLIPS_DIR, SOURCE_DIR, YTDLP_EXTRACTOR_ARGS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

PREFERRED_HEIGHT = 1080
BLACK_FRAME_MEAN_THRESHOLD = 10.0
MAX_START_DRIFT_SECONDS = 0.75
FACE_SAMPLE_INTERVAL_SECONDS = 1.5
MIN_FACE_DETECTION_RATIO = 0.30
VERTICAL_ASPECT_W = 9
VERTICAL_ASPECT_H = 16

CAPTION_MIN_WORDS_PER_CHUNK = 3
CAPTION_MAX_WORDS_PER_CHUNK = 5
CAPTION_FONT_SIZE_RATIO = 0.050
CAPTION_OUTLINE_RATIO = 0.08
CAPTION_MARGIN_V_RATIO = 0.14
CAPTION_MARGIN_LR_RATIO = 0.06
CAPTION_MAX_HIGHLIGHTS = 2
CAPTION_HIGHLIGHT_COLOR = "&H00FFFF&"  # ASS &HBBGGRR& — bright yellow
CAPTION_DEFAULT_COLOR = "&HFFFFFF&"  # white

_NUMERIC_RE = re.compile(r"^\$?[\d][\d,]*(\.\d+)?%?$")

# Words that can legitimately be capitalized just because they start a
# sentence — excluded from starting a "proper noun run" merge so "The United
# States" merges as "United States" only, not "The United".
_SENTENCE_STARTER_STOPWORDS = {
    "the", "a", "an", "this", "that", "these", "those", "it", "i", "but", "so",
    "and", "or", "if", "when", "then", "there",
}

_SUPERLATIVE_WORDS = {
    "most", "least", "best", "worst", "first", "last", "only", "never", "always",
    "biggest", "smallest", "greatest", "fastest", "slowest", "hardest", "easiest",
    "hottest", "coldest", "strongest", "weakest", "richest", "poorest", "oldest",
    "youngest", "highest", "lowest", "longest", "shortest", "incredible", "insane",
    "massive", "huge", "amazing", "shocking", "unbelievable", "crazy", "wild",
}

_CAPTION_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "but",
    "is", "are", "was", "were", "be", "been", "this", "that", "these", "those",
    "it", "its", "he", "she", "they", "we", "you", "i", "my", "your", "his",
    "her", "their", "our", "as", "by", "with", "from", "not", "no", "so", "if",
    "than", "then", "there", "here", "just", "get", "got", "let", "do", "did",
    "does", "have", "has", "had", "will", "would", "could", "should", "can",
}


class ClipperError(Exception):
    """Base exception for download/cut/reformat failures."""


class VideoDownloadError(ClipperError):
    """Raised when yt-dlp fails to download the source video."""


class ClipCutError(ClipperError):
    """Raised when ffmpeg fails to cut a clip from the source video."""


class VerticalReformatError(ClipperError):
    """Raised when ffmpeg fails to reformat a clip to vertical 9:16."""


class CaptionGenerationError(ClipperError):
    """Raised when caption timing/subtitle-file generation fails."""


def _find_cached(output_path: str, video_id: str) -> str | None:
    if not os.path.isdir(output_path):
        return None
    for name in os.listdir(output_path):
        if name.startswith(f"{video_id}."):
            return os.path.join(output_path, name)
    return None


def download_video(youtube_url: str, output_path: str = SOURCE_DIR) -> str:
    """Download a YouTube video via yt-dlp into output_path.

    Prefers 1080p; if the video's format list doesn't offer 1080p, falls back
    to the best resolution available below it. Cached by video ID: re-running
    with the same video skips re-downloading. Returns the local path to the
    downloaded (or cached) video file.
    """
    import yt_dlp

    video_id = extract_video_id(youtube_url)
    os.makedirs(output_path, exist_ok=True)

    cached = _find_cached(output_path, video_id)
    if cached:
        logger.info("Using cached source video: %s", cached)
        return cached

    outtmpl = os.path.join(output_path, f"{video_id}.%(ext)s")
    ydl_opts = {
        "format": (
            f"bestvideo[height<={PREFERRED_HEIGHT}]+bestaudio"
            f"/best[height<={PREFERRED_HEIGHT}]"
        ),
        "merge_output_format": "mp4",
        "outtmpl": outtmpl,
        "quiet": True,
        "no_warnings": True,
        # See utils.config.YTDLP_EXTRACTOR_ARGS. Caps resolution at ~360p
        # (YouTube's SABR-only rollout withholds higher formats from the
        # android client too without a token) — an external constraint,
        # not something fixable on our end.
        "extractor_args": YTDLP_EXTRACTOR_ARGS,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
    except yt_dlp.utils.DownloadError as exc:
        raise VideoDownloadError(f"Could not download video {video_id}: {exc}") from exc

    downloaded = _find_cached(output_path, video_id)
    if not downloaded:
        raise VideoDownloadError(
            f"yt-dlp reported success but no output file was found for video {video_id}."
        )

    available_heights = sorted({
        f.get("height") for f in (info.get("formats") or [])
        if f.get("height") and f.get("vcodec") not in (None, "none")
    })
    requested = info.get("requested_formats") or []
    achieved_height = (
        max((f.get("height") or 0) for f in requested) if requested else info.get("height")
    )
    logger.info(
        "Downloaded source video: %s (available heights: %s; selected %sp against a "
        "%dp preference)",
        downloaded, available_heights, achieved_height, PREFERRED_HEIGHT,
    )
    return downloaded


def _starts_black(path: str, threshold: float = BLACK_FRAME_MEAN_THRESHOLD) -> bool:
    """Check whether a clip's first frame is (near-)black — a sign of a bad copy-cut."""
    import cv2

    cap = cv2.VideoCapture(path)
    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            return True
        return float(frame.mean()) < threshold
    finally:
        cap.release()


def _drifted_from_requested_duration(
    path: str, expected_duration: float, max_drift: float = MAX_START_DRIFT_SECONDS
) -> bool:
    """Check whether a copy-cut clip's actual duration overshoots what was requested.

    Input-seeked stream copy snaps the start backward to the nearest preceding
    keyframe, silently prepending unwanted lead-in footage — the clip plays fine
    but no longer opens on the intended hook line. A longer-than-requested
    duration is the signal that this happened.
    """
    try:
        probe = ffmpeg.probe(path)
        actual_duration = float(probe["format"]["duration"])
    except (ffmpeg.Error, KeyError, ValueError):
        return True
    return (actual_duration - expected_duration) > max_drift


def _run_ffmpeg_cut(source_path: str, start: float, end: float, output_path: str, copy: bool) -> None:
    duration = end - start
    stream = ffmpeg.input(source_path, ss=start, t=duration)
    kwargs = {"avoid_negative_ts": "make_zero"}
    if copy:
        kwargs["c"] = "copy"
    else:
        kwargs.update({"vcodec": "libx264", "preset": "fast", "acodec": "aac"})
    node = ffmpeg.output(stream, output_path, **kwargs)
    try:
        ffmpeg.run(node, overwrite_output=True, quiet=True)
    except ffmpeg.Error as exc:
        stderr = exc.stderr.decode(errors="replace") if exc.stderr else str(exc)
        raise ClipCutError(f"ffmpeg failed cutting {output_path}: {stderr[-500:]}") from exc


def cut_clip(source_path: str, start: float, end: float, output_path: str) -> str:
    """Cut [start, end] (seconds) from source_path into a standalone mp4 at output_path.

    Tries a fast stream-copy cut first. If the result is missing, starts on a
    black frame, or drifted earlier than requested (keyframe misalignment),
    re-encodes with libx264 instead so the clip is frame-accurate.
    """
    if end <= start:
        raise ClipCutError(f"Invalid clip range: start={start} end={end}")
    if not os.path.exists(source_path):
        raise ClipCutError(f"Source video not found: {source_path}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    _run_ffmpeg_cut(source_path, start, end, output_path, copy=True)

    needs_reencode = (
        not os.path.exists(output_path)
        or _starts_black(output_path)
        or _drifted_from_requested_duration(output_path, end - start)
    )
    if needs_reencode:
        logger.info(
            "Stream-copy cut for %s was black, drifted, or failed; re-encoding.", output_path
        )
        _run_ffmpeg_cut(source_path, start, end, output_path, copy=False)

    if not os.path.exists(output_path):
        raise ClipCutError(f"ffmpeg did not produce an output file: {output_path}")

    logger.info("Cut clip: %s [%.1fs - %.1fs] -> %s", source_path, start, end, output_path)
    return output_path


def _round_even(x: float) -> int:
    n = int(round(x))
    return n if n % 2 == 0 else n - 1


def _split_segment_into_words(segment: dict) -> list[dict]:
    """Split one transcript segment's [start, start+duration] across its words.

    Proportional-by-character-count split, not real speech alignment: no
    Whisper/forced-alignment involved. Returns word dicts with absolute
    (source-video-timeline) start/end in seconds.
    """
    words = segment["text"].split()
    total_chars = sum(len(w) for w in words)
    if not words or total_chars == 0:
        return []

    cursor = segment["start"]
    out = []
    for word in words:
        word_duration = segment["duration"] * (len(word) / total_chars)
        out.append({"text": word, "start": cursor, "end": cursor + word_duration})
        cursor += word_duration
    return out


def _strip_punct(text: str) -> str:
    return text.strip(".,!?;:\"'()[]{}“”‘’")


def _looks_numeric(text: str) -> bool:
    return bool(_NUMERIC_RE.match(_strip_punct(text)))


def _looks_capitalized(text: str) -> bool:
    core = _strip_punct(text)
    return bool(core) and core[0].isalpha() and core[0].isupper()


def _is_superlative(text: str) -> bool:
    return _strip_punct(text).lower() in _SUPERLATIVE_WORDS


def _build_unbreakable_units(words: list[dict]) -> list[list[dict]]:
    """Merge words into indivisible units so a chunk boundary never splits a
    proper-noun run (consecutive Title-Case words, e.g. "United States") or a
    number from the word immediately after it (e.g. "18 year")."""
    n = len(words)
    units: list[list[dict]] = []
    i = 0
    while i < n:
        text_i = words[i]["text"]

        if (
            _looks_capitalized(text_i)
            and _strip_punct(text_i).lower() not in _SENTENCE_STARTER_STOPWORDS
        ):
            j = i + 1
            while j < n and _looks_capitalized(words[j]["text"]):
                j += 1
            if j - i >= 2:
                units.append(words[i:j])
                i = j
                continue

        if _looks_numeric(text_i) and i + 1 < n:
            units.append(words[i:i + 2])
            i += 2
            continue

        units.append([words[i]])
        i += 1
    return units


def _group_units_into_chunks(
    units: list[list[dict]],
    min_words: int = CAPTION_MIN_WORDS_PER_CHUNK,
    max_words: int = CAPTION_MAX_WORDS_PER_CHUNK,
) -> list[list[dict]]:
    """Greedily pack units into ~min_words-max_words chunks, never splitting a
    unit (an oversized unbreakable unit becomes its own chunk regardless)."""
    chunks: list[list[dict]] = []
    current: list[dict] = []
    current_count = 0
    for unit in units:
        if current and current_count >= min_words and current_count + len(unit) > max_words:
            chunks.append(current)
            current = list(unit)
            current_count = len(unit)
        else:
            current.extend(unit)
            current_count += len(unit)
    if current:
        if chunks and current_count < min_words:
            chunks[-1].extend(current)
        else:
            chunks.append(current)
    return chunks


def _select_highlight_words(words: list[str], max_highlights: int = CAPTION_MAX_HIGHLIGHTS) -> set[int]:
    """Pick up to max_highlights word indices to emphasize: numbers first, then
    superlatives, then the longest remaining content word(s) as a main-noun proxy."""
    selected: list[int] = []

    for i, w in enumerate(words):
        if _looks_numeric(w):
            selected.append(i)
    for i, w in enumerate(words):
        if len(selected) >= max_highlights:
            break
        if i not in selected and _is_superlative(w):
            selected.append(i)
    if len(selected) < max_highlights:
        candidates = [
            i for i, w in enumerate(words)
            if i not in selected and _strip_punct(w) and _strip_punct(w).lower() not in _CAPTION_STOPWORDS
        ]
        candidates.sort(key=lambda i: -len(_strip_punct(words[i])))
        selected.extend(candidates[:max_highlights - len(selected)])

    return set(selected[:max_highlights])


def _clamp_segment_durations(segments: list[dict]) -> list[dict]:
    """Clamp each segment's effective end time so it never runs past the next
    segment's start.

    Auto-generated YouTube captions routinely declare overlapping windows —
    segment i's duration bleeds several seconds into segment i+1 — as a
    smoothing artifact of the auto-caption format, even though the two
    segments' text never actually repeats. Left alone, that overlap flows
    through to derived word/chunk timestamps and produces two captions
    on screen at once. Segment *start* times are trustworthy (they're in
    true document order); duration is only an upper bound.
    """
    clamped = []
    for i, seg in enumerate(segments):
        end = seg["start"] + seg["duration"]
        if i + 1 < len(segments):
            end = min(end, segments[i + 1]["start"])
        duration = max(end - seg["start"], 0.05)
        clamped.append({**seg, "duration": duration})
    return clamped


def build_caption_chunks(
    transcript: list[dict], clip_start: float, clip_end: float
) -> list[dict]:
    """Build phrase-level burned-caption chunks for a clip from existing transcript segments.

    Uses only the segment-level {text, start, duration} data transcript.py already
    produces (YouTube captions) — no Whisper or forced alignment. Segment durations
    are first clamped so overlapping auto-caption windows can't bleed into the next
    segment (see _clamp_segment_durations). Per-word timing within each segment is
    then a proportional-by-character-count split; words are merged into unbreakable
    units (proper nouns, number+word) before being packed into 3-5 word phrase
    chunks, so entities and numbers never split across a chunk boundary. Each
    chunk's first word is capitalized; existing punctuation from the source
    transcript is never stripped or fabricated. Returns chunks with start/end
    relative to the clip (0 = clip_start), a plain "text" for logging, "words"
    (display strings) and "highlight" (indices into "words" to render in the
    accent color) for the renderer.
    """
    relevant_segments = [
        seg for seg in transcript
        if not (seg["start"] + seg["duration"] <= clip_start or seg["start"] >= clip_end)
    ]
    relevant_segments = _clamp_segment_durations(relevant_segments)

    words = []
    for seg in relevant_segments:
        words.extend(_split_segment_into_words(seg))

    words = [w for w in words if w["end"] > clip_start and w["start"] < clip_end]
    # Deliberately not re-sorted by start time: transcript segments already arrive
    # in true reading/speaking order from the source API, and per-word start times
    # here are only a proportional *estimate* within each segment — sorting by
    # them can reorder words whenever segments overlap even slightly (real caption
    # data does this), scrambling display text while leaving it looking plausible.

    chunks = []
    for group in _group_units_into_chunks(_build_unbreakable_units(words)):
        start = max(0.0, group[0]["start"] - clip_start)
        end = min(clip_end - clip_start, group[-1]["end"] - clip_start)
        if end <= start:
            continue

        display_words = [w["text"] for w in group]
        display_words[0] = display_words[0][:1].upper() + display_words[0][1:]
        highlight = _select_highlight_words(display_words)

        chunks.append({
            "text": " ".join(display_words),
            "words": display_words,
            "highlight": highlight,
            "start": start,
            "end": end,
        })
    return chunks


def _sample_face_x_centers(
    path: str,
    interval: float = FACE_SAMPLE_INTERVAL_SECONDS,
    start_offset: float = 0.0,
    duration: float | None = None,
) -> tuple[list[float], int]:
    """Sample frames every `interval` seconds and return detected face x-centers.

    Samples the window [start_offset, start_offset + duration) of `path`
    (defaults to the whole file, i.e. start_offset=0 / duration=None, for
    callers that already have an isolated clip file). Returns
    (x_centers, total_samples) so callers can compute a detection ratio. Only
    the largest detected face per sampled frame is kept (most likely the
    on-camera speaker rather than a person in the background).

    Frames are grabbed via ffmpeg's own seek + single-frame extraction rather
    than OpenCV's VideoCapture.set(CAP_PROP_POS_MSEC, ...): measured ~7x
    faster when seeking to a late offset in a large source file (38 samples
    at ~587s into a ~106MB 1080p source: 53.2s via OpenCV vs 7.3s via ffmpeg).
    OpenCV's ffmpeg backend doesn't reliably keyframe-seek here and can
    effectively re-decode from an earlier point on every .set() call.
    """
    import cv2

    if duration is None:
        try:
            probe = ffmpeg.probe(path)
            file_duration = float(probe["format"]["duration"])
        except (ffmpeg.Error, KeyError, ValueError):
            return [], 0
        duration = file_duration - start_offset
    duration = max(0.0, duration)

    sample_times = []
    t = 0.0
    while t < duration:
        sample_times.append(t)
        t += interval
    if not sample_times:
        return [], 0

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    x_centers = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, t in enumerate(sample_times):
            frame_path = os.path.join(tmpdir, f"sample_{i}.jpg")
            try:
                ffmpeg.input(path, ss=start_offset + t).output(
                    frame_path, vframes=1
                ).run(overwrite_output=True, quiet=True)
            except ffmpeg.Error:
                continue
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
            if len(faces) == 0:
                continue
            fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            x_centers.append(fx + fw / 2)

    return x_centers, len(sample_times)


def _format_ass_timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    cs = round(seconds * 100)
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _ass_escape_text(text: str) -> str:
    return text.replace("{", "(").replace("}", ")").replace("\n", " ")


def _render_ass_dialogue_text(chunk: dict) -> str:
    """Build a chunk's Dialogue Text field, wrapping highlighted words in inline
    \\c color-override tags (bright yellow) so the rest stays the style's white."""
    words = chunk.get("words") or chunk["text"].split()
    highlight = chunk.get("highlight") or set()
    parts = []
    for i, word in enumerate(words):
        escaped = _ass_escape_text(word)
        if i in highlight:
            parts.append(
                f"{{\\c{CAPTION_HIGHLIGHT_COLOR}}}{escaped}{{\\c{CAPTION_DEFAULT_COLOR}}}"
            )
        else:
            parts.append(escaped)
    return " ".join(parts)


def _write_ass_file(chunks: list[dict], width: int, height: int, ass_path: str) -> None:
    """Write a styled .ass subtitle file: bold white text, black outline, positioned
    in the lower-middle third so it clears the face (which sits in the upper portion
    of the face-centered vertical crop)."""
    font_size = max(24, round(height * CAPTION_FONT_SIZE_RATIO))
    outline = max(2, round(font_size * CAPTION_OUTLINE_RATIO))
    margin_v = round(height * CAPTION_MARGIN_V_RATIO)
    margin_lr = round(width * CAPTION_MARGIN_LR_RATIO)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
ScaledBorderAndShadow: yes
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,Arial,{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,{outline},1,2,{margin_lr},{margin_lr},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header]
    for chunk in chunks:
        start_ts = _format_ass_timestamp(chunk["start"])
        end_ts = _format_ass_timestamp(chunk["end"])
        text = _render_ass_dialogue_text(chunk)
        lines.append(f"Dialogue: 0,{start_ts},{end_ts},Caption,,0,0,0,,{text}\n")

    with open(ass_path, "w", encoding="utf-8") as f:
        f.write("".join(lines))


def _escape_path_for_subtitles_filter(path: str) -> str:
    """Build a path for the ffmpeg subtitles filter's filename= value.

    Prefers a path relative to the current working directory: on Windows, an
    absolute path's drive-letter colon collides with ffmpeg's filtergraph
    option-separator syntax, and no combination of backslash-escaping the
    colon actually parses (verified directly against ffmpeg — not just an
    escaping mistake). A cwd-relative path has no drive letter, so the
    problem doesn't arise. Falls back to an escaped absolute path only if the
    file lives on a different drive than the cwd (relpath can't express that).
    """
    abs_path = os.path.abspath(path)
    try:
        return os.path.relpath(abs_path, os.getcwd()).replace("\\", "/")
    except ValueError:
        # Different drive than cwd — relpath can't express it; best effort.
        return abs_path.replace("\\", "/").replace(":", "\\:")


def _probe_video_dimensions(path: str) -> tuple[int, int]:
    try:
        probe = ffmpeg.probe(path)
        video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")
        return int(video_info["width"]), int(video_info["height"])
    except (ffmpeg.Error, KeyError, StopIteration, ValueError) as exc:
        raise VerticalReformatError(f"Could not probe video dimensions for {path}: {exc}") from exc


def _compute_vertical_crop(
    sample_path: str,
    source_width: int,
    source_height: int,
    start_offset: float = 0.0,
    duration: float | None = None,
    log_context: str = "",
) -> tuple[int, int, int, int]:
    """Compute the (crop_width, crop_height, crop_x, crop_y) window for a 9:16
    vertical crop of a source_width x source_height frame.

    Samples face positions every ~1.5s (optionally restricted to
    [start_offset, start_offset + duration) within sample_path) and uses the
    median horizontal center as a single fixed crop offset — no per-frame
    recropping, so the crop can't jitter. Falls back to a plain center-crop
    if a face isn't reliably detected (below MIN_FACE_DETECTION_RATIO of
    sampled frames). log_context is prefixed onto the log line so callers can
    identify which clip/source window it refers to.
    """
    crop_width = _round_even(source_height * VERTICAL_ASPECT_W / VERTICAL_ASPECT_H)
    if crop_width > source_width:
        # Source is already narrower than 9:16 needs — crop height instead, keep full width.
        crop_width = source_width
        crop_height = _round_even(source_width * VERTICAL_ASPECT_H / VERTICAL_ASPECT_W)
    else:
        crop_height = source_height

    x_centers, total_samples = _sample_face_x_centers(
        sample_path, start_offset=start_offset, duration=duration
    )
    detection_ratio = (len(x_centers) / total_samples) if total_samples else 0.0

    if detection_ratio >= MIN_FACE_DETECTION_RATIO:
        crop_x = int(round(statistics.median(x_centers) - crop_width / 2))
        crop_x = max(0, min(crop_x, source_width - crop_width))
        logger.info(
            "%sface detected in %d/%d sampled frames (%.0f%%); centering crop at x=%d",
            log_context, len(x_centers), total_samples, detection_ratio * 100, crop_x,
        )
    else:
        crop_x = max(0, (source_width - crop_width) // 2)
        logger.info(
            "%sface detected in only %d/%d sampled frames (%.0f%%, below %.0f%% threshold); "
            "falling back to center-crop.",
            log_context, len(x_centers), total_samples, detection_ratio * 100,
            MIN_FACE_DETECTION_RATIO * 100,
        )
    crop_y = max(0, (source_height - crop_height) // 2)
    return crop_width, crop_height, crop_x, crop_y


def _maybe_write_captions_file(
    captions: list[dict] | None, crop_width: int, crop_height: int, output_path: str
) -> str | None:
    if not captions:
        return None
    ass_path = f"{output_path}.tmp_captions.ass"
    try:
        _write_ass_file(captions, crop_width, crop_height, ass_path)
    except OSError as exc:
        raise CaptionGenerationError(f"Could not write subtitle file {ass_path}: {exc}") from exc
    return ass_path


def _encode_crop(
    in_stream,
    crop_width: int,
    crop_height: int,
    crop_x: int,
    crop_y: int,
    ass_path: str | None,
    output_path: str,
    error_context: str,
    extra_output_kwargs: dict | None = None,
) -> None:
    """Run the crop (+ optional burned-in subtitles) ffmpeg filtergraph and
    encode it to output_path, cleaning up the subtitle file afterward."""
    video = ffmpeg.filter(in_stream.video, "crop", crop_width, crop_height, crop_x, crop_y)
    if ass_path:
        video = ffmpeg.filter(
            video, "subtitles", filename=_escape_path_for_subtitles_filter(ass_path)
        )
    kwargs = {"vcodec": "libx264", "preset": "fast", "acodec": "aac"}
    kwargs.update(extra_output_kwargs or {})
    node = ffmpeg.output(video, in_stream.audio, output_path, **kwargs)
    try:
        ffmpeg.run(node, overwrite_output=True, quiet=True)
    except ffmpeg.Error as exc:
        stderr = exc.stderr.decode(errors="replace") if exc.stderr else str(exc)
        raise VerticalReformatError(
            f"ffmpeg failed {error_context}: {stderr[-500:]}"
        ) from exc
    finally:
        if ass_path and os.path.exists(ass_path):
            os.remove(ass_path)


def reformat_vertical(clip_path: str, output_path: str, captions: list[dict] | None = None) -> str:
    """Crop a 16:9 clip to a 9:16 vertical frame, keeping the speaking subject in view,
    and optionally burn in phrase-level captions in the same encode pass.

    Samples face positions every ~1.5s across the clip and uses the median
    horizontal center as a single fixed crop offset for the clip's full
    duration (no per-frame recropping, so the crop can't jitter). Falls back
    to a plain center-crop if a face isn't reliably detected (below
    MIN_FACE_DETECTION_RATIO of sampled frames).

    If `captions` is given (list of {text, start, end} relative to the clip,
    e.g. from build_caption_chunks), an .ass subtitle file is generated and
    chained into this same ffmpeg pass via the subtitles filter — no second
    encode.
    """
    if not os.path.exists(clip_path):
        raise VerticalReformatError(f"Clip not found: {clip_path}")
    if os.path.abspath(clip_path) == os.path.abspath(output_path):
        raise VerticalReformatError(
            "reformat_vertical requires clip_path and output_path to differ "
            "(ffmpeg cannot read and overwrite the same file in one pass)."
        )

    source_width, source_height = _probe_video_dimensions(clip_path)
    crop_width, crop_height, crop_x, crop_y = _compute_vertical_crop(
        clip_path, source_width, source_height, log_context=f"{clip_path}: "
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    ass_path = _maybe_write_captions_file(captions, crop_width, crop_height, output_path)

    in_stream = ffmpeg.input(clip_path)
    _encode_crop(
        in_stream, crop_width, crop_height, crop_x, crop_y, ass_path, output_path,
        error_context=f"reformatting {clip_path}",
    )

    if not os.path.exists(output_path):
        raise VerticalReformatError(f"ffmpeg did not produce an output file: {output_path}")

    logger.info(
        "Reformatted to vertical: %s -> %s (%dx%d, %d caption chunks)",
        clip_path, output_path, crop_width, crop_height, len(captions) if captions else 0,
    )
    return output_path


def cut_and_reformat(
    source_path: str,
    start: float,
    end: float,
    output_path: str,
    transcript: list[dict] | None = None,
) -> str:
    """Cut [start, end] from source_path directly into a vertically-cropped,
    optionally-captioned clip at output_path — trim + crop + burned-in
    captions in a single ffmpeg encode pass.

    Deliberately does NOT route through cut_clip() + reformat_vertical()
    sequentially: reformat_vertical's crop filter always forces a re-encode
    (crop can't stream-copy), so cutting first and reformatting second means
    encoding the same footage twice for no benefit — measured to roughly
    double per-clip time in the full pipeline. Re-encoding directly with
    input-seeked -ss is frame-accurate by default (ffmpeg's "accurate seek"
    for transcodes, unlike stream copy), so the black-frame/drift checks
    cut_clip needs for its stream-copy path don't apply here. Face detection
    samples directly from source_path within [start, end] instead of
    requiring an intermediate cut file first.
    """
    if end <= start:
        raise ClipCutError(f"Invalid clip range: start={start} end={end}")
    if not os.path.exists(source_path):
        raise ClipCutError(f"Source video not found: {source_path}")

    source_width, source_height = _probe_video_dimensions(source_path)
    duration = end - start
    crop_width, crop_height, crop_x, crop_y = _compute_vertical_crop(
        source_path, source_width, source_height,
        start_offset=start, duration=duration,
        log_context=f"{source_path} [{start:.1f}s-{end:.1f}s]: ",
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    captions = build_caption_chunks(transcript, start, end) if transcript else None
    ass_path = _maybe_write_captions_file(captions, crop_width, crop_height, output_path)

    in_stream = ffmpeg.input(source_path, ss=start, t=duration)
    _encode_crop(
        in_stream, crop_width, crop_height, crop_x, crop_y, ass_path, output_path,
        error_context=f"cutting/reformatting {source_path}",
        extra_output_kwargs={"avoid_negative_ts": "make_zero"},
    )

    if not os.path.exists(output_path):
        raise VerticalReformatError(f"ffmpeg did not produce an output file: {output_path}")

    logger.info(
        "Cut+reformatted: %s [%.1fs-%.1fs] -> %s (%dx%d, %d caption chunks)",
        source_path, start, end, output_path, crop_width, crop_height,
        len(captions) if captions else 0,
    )
    return output_path


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5):
        print("Usage: python -m pipeline.clipper <youtube_url> <start> <end> [output_name]")
        sys.exit(1)

    url, start_s, end_s = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    name = sys.argv[4] if len(sys.argv) == 5 else "clip_1.mp4"

    try:
        source = download_video(url)
        clip_path = cut_and_reformat(source, start_s, end_s, os.path.join(CLIPS_DIR, name))
    except ClipperError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"\nClip ready: {clip_path}")

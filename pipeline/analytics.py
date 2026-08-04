"""Computes a best-posting-time recommendation from the authenticated user's
own YouTube channel via the YouTube Data API v3.

Runnable standalone: python -m pipeline.analytics
"""
import calendar
import datetime
import logging
import sys
from collections import defaultdict

from utils.config import YOUTUBE_ANALYTICS_CONFIGURED
from utils.youtube_auth import YouTubeAuthError, get_authenticated_service

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

MAX_VIDEOS = 50
TOP_WINDOWS = 2


def _fetch_recent_videos(service, channel_id: str | None) -> list[dict]:
    """Return up to MAX_VIDEOS recent videos with publishedAt/viewCount/likeCount."""
    if channel_id:
        channels_resp = service.channels().list(part="contentDetails", id=channel_id).execute()
    else:
        channels_resp = service.channels().list(part="contentDetails", mine=True).execute()

    items = channels_resp.get("items", [])
    if not items:
        return []

    uploads_playlist_id = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]

    playlist_resp = service.playlistItems().list(
        part="contentDetails",
        playlistId=uploads_playlist_id,
        maxResults=MAX_VIDEOS,
    ).execute()
    video_ids = [item["contentDetails"]["videoId"] for item in playlist_resp.get("items", [])]
    if not video_ids:
        return []

    videos_resp = service.videos().list(part="snippet,statistics", id=",".join(video_ids)).execute()

    videos = []
    for item in videos_resp.get("items", []):
        published_at = item.get("snippet", {}).get("publishedAt")
        stats = item.get("statistics", {})
        if not published_at:
            continue
        videos.append({
            "published_at": published_at,
            "views": int(stats.get("viewCount", 0)),
            "likes": int(stats.get("likeCount", 0)),
        })
    return videos


def _bucket_by_day_hour(videos: list[dict]) -> dict[tuple[int, int], dict]:
    """Bucket videos by (day_of_week, hour_of_day) in UTC, weighted by view count."""
    buckets: dict[tuple[int, int], dict] = defaultdict(lambda: {"views": 0, "count": 0})
    for v in videos:
        dt = datetime.datetime.fromisoformat(v["published_at"].replace("Z", "+00:00"))
        key = (dt.weekday(), dt.hour)  # Monday=0
        buckets[key]["views"] += v["views"]
        buckets[key]["count"] += 1
    return buckets


def _format_hour(hour: int) -> str:
    period = "AM" if hour < 12 else "PM"
    display = hour % 12 or 12
    return f"{display} {period}"


def get_best_posting_time(channel_id: str | None = None) -> dict | None:
    """Recommend the best posting time window(s) for the authenticated channel.

    Buckets the channel's recent uploads (up to MAX_VIDEOS) by day-of-week and
    hour-of-day (UTC), weighted by view count, and returns the top 1-2
    windows. Returns None (never raises) if OAuth isn't configured or the API
    call fails, so the UI can show "Analytics unavailable" instead of crashing.
    """
    if not YOUTUBE_ANALYTICS_CONFIGURED:
        logger.info("YouTube OAuth not configured; analytics unavailable.")
        return None

    try:
        service = get_authenticated_service()
        videos = _fetch_recent_videos(service, channel_id)
    except YouTubeAuthError as exc:
        logger.warning("YouTube analytics unavailable (auth): %s", exc)
        return None
    except Exception as exc:
        logger.warning("YouTube analytics unavailable (API call failed): %s", exc)
        return None

    if not videos:
        logger.info("No videos found for this channel; analytics unavailable.")
        return None

    buckets = _bucket_by_day_hour(videos)
    ranked = sorted(buckets.items(), key=lambda kv: kv[1]["views"], reverse=True)

    top_windows = []
    for (day, hour), data in ranked[:TOP_WINDOWS]:
        top_windows.append({
            "day_of_week": calendar.day_name[day],
            "hour": hour,
            "label": (
                f"{calendar.day_name[day]} {_format_hour(hour)}"
                f"-{_format_hour((hour + 1) % 24)} UTC"
            ),
            "weighted_views": data["views"],
            "video_count": data["count"],
        })

    views_by_hour = defaultdict(int)
    for (_, hour), data in buckets.items():
        views_by_hour[hour] += data["views"]

    return {
        "top_windows": top_windows,
        "views_by_hour": {h: views_by_hour.get(h, 0) for h in range(24)},
        "video_count": len(videos),
    }


if __name__ == "__main__":
    result = get_best_posting_time()
    if result is None:
        print("Analytics unavailable — connect a YouTube account to enable this.")
        sys.exit(0)

    print(f"\nAnalyzed {result['video_count']} recent video(s).\n")
    print("Top posting time window(s):")
    for w in result["top_windows"]:
        print(f"  {w['label']}  ({w['weighted_views']} views across {w['video_count']} video(s))")

    print("\nViews by hour (UTC):")
    for hour, views in result["views_by_hour"].items():
        if views:
            print(f"  {_format_hour(hour):>7}: {views}")

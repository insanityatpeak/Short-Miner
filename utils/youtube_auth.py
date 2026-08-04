"""OAuth2 authentication for the YouTube Data API v3 (used by pipeline/analytics.py).

Standard google-auth-oauthlib installed-app flow, authenticated against the
presenter's own YouTube channel. Caches the resulting token in
output/.token.json so re-auth isn't required on every run during the demo.

Runnable standalone: python -m utils.youtube_auth
"""
import logging
import os
import sys

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from utils.config import OUTPUT_DIR, YOUTUBE_CLIENT_ID, YOUTUBE_CLIENT_SECRET

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

SCOPES = ["https://www.googleapis.com/auth/youtube.readonly"]
TOKEN_PATH = os.path.join(OUTPUT_DIR, ".token.json")


class YouTubeAuthError(Exception):
    """Raised when YouTube OAuth isn't configured or authentication fails."""


def _client_config() -> dict:
    if not YOUTUBE_CLIENT_ID or not YOUTUBE_CLIENT_SECRET:
        raise YouTubeAuthError(
            "YOUTUBE_CLIENT_ID / YOUTUBE_CLIENT_SECRET are missing. Add them to your "
            ".env file to enable YouTube analytics."
        )
    return {
        "installed": {
            "client_id": YOUTUBE_CLIENT_ID,
            "client_secret": YOUTUBE_CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost"],
        }
    }


def _load_saved_credentials() -> Credentials | None:
    if not os.path.exists(TOKEN_PATH):
        return None
    try:
        return Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    except (ValueError, OSError) as exc:
        logger.warning("Could not load saved YouTube token (%s); re-authenticating.", exc)
        return None


def _save_credentials(creds: Credentials) -> None:
    os.makedirs(os.path.dirname(TOKEN_PATH) or ".", exist_ok=True)
    with open(TOKEN_PATH, "w", encoding="utf-8") as f:
        f.write(creds.to_json())


def _authenticate() -> Credentials:
    creds = _load_saved_credentials()

    if creds and creds.valid:
        logger.info("Using cached YouTube OAuth token: %s", TOKEN_PATH)
        return creds

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            _save_credentials(creds)
            logger.info("Refreshed cached YouTube OAuth token.")
            return creds
        except Exception as exc:
            logger.warning("Token refresh failed (%s); running a fresh consent flow.", exc)

    logger.info("No valid cached token; opening browser for YouTube OAuth consent...")
    flow = InstalledAppFlow.from_client_config(_client_config(), SCOPES)
    creds = flow.run_local_server(port=0)
    _save_credentials(creds)
    logger.info("YouTube OAuth complete; token cached at %s", TOKEN_PATH)
    return creds


def get_authenticated_service():
    """Return a ready-to-use YouTube Data API v3 client (googleapiclient.discovery),
    running the OAuth consent flow (opens a browser) if no valid cached token
    exists. Raises YouTubeAuthError if OAuth isn't configured or auth fails."""
    creds = _authenticate()
    try:
        return build("youtube", "v3", credentials=creds)
    except Exception as exc:
        raise YouTubeAuthError(f"Could not build YouTube API client: {exc}") from exc


if __name__ == "__main__":
    try:
        service = get_authenticated_service()
        response = service.channels().list(part="snippet,statistics", mine=True).execute()
    except YouTubeAuthError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    items = response.get("items", [])
    if not items:
        print("\nAuthenticated, but no channel found for this account.")
    else:
        channel = items[0]
        stats = channel.get("statistics", {})
        print(f"\nAuthenticated as: {channel['snippet']['title']}")
        print(f"Subscribers: {stats.get('subscriberCount', 'hidden')}")
        print(f"Total views: {stats.get('viewCount', 'n/a')}")
        print(f"Video count: {stats.get('videoCount', 'n/a')}")

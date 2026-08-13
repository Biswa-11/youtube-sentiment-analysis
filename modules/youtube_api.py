from __future__ import annotations
from datetime import datetime
from typing import Any
from urllib.parse import parse_qs, urlparse
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from config import MAX_COMMENTS, YOUTUBE_API_KEY


class YouTubeAPI:

    def __init__(self) -> None:
        self.client = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        self._video_cache: dict[str, dict[str, Any]] = {}

    @staticmethod
    def extract_video_id(video_url: str) -> str:
        parsed_url = urlparse(video_url.strip())
        host = parsed_url.netloc.lower()

        if "youtu.be" in host:
            return parsed_url.path.lstrip("/").split("/")[0]

        if "youtube.com" in host:
            if parsed_url.path == "/watch":
                return parse_qs(parsed_url.query).get("v", [""])[0]
            if parsed_url.path.startswith("/embed/"):
                return parsed_url.path.split("/embed/")[1].split("/")[0]
            if parsed_url.path.startswith("/shorts/"):
                return parsed_url.path.split("/shorts/")[1].split("/")[0]

        return ""

    @staticmethod
    def _format_count(value: Any) -> str:
        try:
            return f"{int(value):,}"
        except (TypeError, ValueError):
            return "0"

    @staticmethod
    def _human_date(date_text: str) -> str:
        try:
            dt = datetime.fromisoformat(date_text.replace("Z", "+00:00"))
            return dt.strftime("%B %d, %Y")
        except ValueError:
            return date_text

    def get_video_details(self, video_url: str) -> dict[str, Any]:
        video_id = self.extract_video_id(video_url)
        if not video_id:
            raise ValueError("INVALID_URL")
        if video_id in self._video_cache:
            return self._video_cache[video_id]

        try:
            response = (
                self.client.videos()
                .list(
                    part="snippet,statistics",
                    id=video_id,
                    fields=(
                        "items(snippet(title,channelTitle,publishedAt,description,thumbnails),"
                        "statistics(viewCount,likeCount,commentCount))"
                    ),
                )
                .execute()
            )
        except HttpError as exc:
            self._raise_api_error(exc)

        items = response.get("items", [])
        if not items:
            raise ValueError("VIDEO_NOT_FOUND")

        item = items[0]
        snippet = item.get("snippet", {})
        stats = item.get("statistics", {})

        thumbnails = snippet.get("thumbnails", {})
        thumb_url = (
            thumbnails.get("high", {}).get("url")
            or thumbnails.get("medium", {}).get("url")
            or thumbnails.get("default", {}).get("url", "")
        )

        details = {
            "video_id": video_id,
            "title": snippet.get("title", "Untitled Video"),
            "channel": snippet.get("channelTitle", "Unknown Channel"),
            "views": self._format_count(stats.get("viewCount", 0)),
            "likes": self._format_count(stats.get("likeCount", 0)),
            "total_comments": int(stats.get("commentCount", 0)),
            "published_at": self._human_date(snippet.get("publishedAt", "")),
            "thumbnail_url": thumb_url,
            "description": snippet.get("description", "")[:200],
        }
        self._video_cache[video_id] = details
        return details

    def get_all_comments(
        self, video_id: str, max_results: int = MAX_COMMENTS
    ) -> list[dict[str, Any]]:
        comments: list[dict[str, Any]] = []
        page_token: str | None = None

        try:
            while len(comments) < max_results:
                request = self.client.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=100,
                    pageToken=page_token,
                    textFormat="plainText",
                    order="time",
                    fields=(
                        "nextPageToken,items(snippet(totalReplyCount,topLevelComment(snippet("
                        "textDisplay,authorDisplayName,likeCount,publishedAt))))"
                    ),
                )
                response = request.execute()

                for item in response.get("items", []):
                    top_level = item.get("snippet", {}).get("topLevelComment", {}).get(
                        "snippet", {}
                    )
                    published_raw = top_level.get("publishedAt", "")
                    try:
                        published_at = datetime.fromisoformat(
                            published_raw.replace("Z", "+00:00")
                        )
                    except ValueError:
                        published_at = datetime.utcnow()

                    if len(comments) >= max_results:
                        break

                    comments.append(
                        {
                            "text": top_level.get("textDisplay", "").strip(),
                            "author": top_level.get("authorDisplayName", "Unknown"),
                            "likes": int(top_level.get("likeCount", 0)),
                            "published_at": published_at,
                            "reply_count": int(item.get("snippet", {}).get("totalReplyCount", 0)),
                        }
                    )

                page_token = response.get("nextPageToken")
                if not page_token:
                    break
        except HttpError as exc:
            reason = str(exc).lower()
            if "commentsdisabled" in reason:
                raise ValueError("COMMENTS_DISABLED") from exc
            self._raise_api_error(exc)
        except TimeoutError as exc:
            raise ValueError("NETWORK_TIMEOUT") from exc

        comments.sort(key=lambda x: x["published_at"])
        return comments

    @staticmethod
    def _raise_api_error(exc: HttpError) -> None:
        reason = str(exc).lower()
        if "quota" in reason or "daily limit" in reason:
            raise ValueError("QUOTA_EXCEEDED") from exc
        if "notfound" in reason or "video not found" in reason or "404" in reason:
            raise ValueError("VIDEO_NOT_FOUND") from exc
        raise ValueError("API_ERROR") from exc
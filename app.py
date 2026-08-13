from __future__ import annotations
from typing import Any
from flask import Flask, flash, jsonify, redirect, render_template, request, url_for
from flask_caching import Cache

import config
from modules.analysis_service import AnalysisService
from modules.sentiment_analyzer import SentimentAnalyzer
from modules.youtube_api import YouTubeAPI

app = Flask(__name__)
app.config["SECRET_KEY"] = config.SECRET_KEY
app.config["CACHE_TYPE"] = "SimpleCache"
app.config["CACHE_DEFAULT_TIMEOUT"] = config.CACHE_TIMEOUT
cache = Cache(app)

youtube_api = YouTubeAPI()
sentiment_analyzer = SentimentAnalyzer()
analysis_service = AnalysisService(youtube_api, sentiment_analyzer, cache)


def _map_error_message(error_code: str) -> str:
    mapping = {
        "INVALID_URL": "Please enter a valid YouTube video URL",
        "VIDEO_NOT_FOUND": "This video could not be found. It may be private or deleted.",
        "COMMENTS_DISABLED": "Comments are disabled for this video.",
        "QUOTA_EXCEEDED": "Daily API limit reached. Please try again tomorrow.",
        "NETWORK_TIMEOUT": "Connection timed out. Please check your internet and try again.",
        "ZERO_COMMENTS": "This video has no comments to analyze.",
    }
    return mapping.get(error_code, "Something went wrong while processing the video.")


def _serialize_comment(comment: dict[str, Any]) -> dict[str, Any]:
    serialized = comment.copy()
    published = serialized.get("published_at")
    if published is not None:
        serialized["published_at"] = published.isoformat()
    return serialized


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze() -> Any:
    video_url = request.form.get("video_url", "").strip()
    try:
        data = analysis_service.run(video_url)
        return render_template("results.html", **data)
    except ValueError as exc:
        flash(_map_error_message(str(exc)), "error")
        return redirect(url_for("index"))
    except Exception:
        flash("Something unexpected happened. Please try again.", "error")
        return redirect(url_for("index"))


@app.route("/api/analyze", methods=["POST"])
def api_analyze() -> Any:
    payload = request.get_json(silent=True) or {}
    video_url = payload.get("url", "")
    try:
        data = analysis_service.run(video_url)
        serializable_comments = [_serialize_comment(c) for c in data["analyzed_comments"]]
        serializable_positive = [_serialize_comment(c) for c in data["top_positive"]]
        serializable_negative = [_serialize_comment(c) for c in data["top_negative"]]
        response = {
            **data,
            "analyzed_comments": serializable_comments,
            "top_positive": serializable_positive,
            "top_negative": serializable_negative,
        }
        return jsonify(response), 200
    except ValueError as exc:
        return jsonify({"error": _map_error_message(str(exc))}), 400
    except Exception:
        return jsonify({"error": "Something unexpected happened. Please try again."}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
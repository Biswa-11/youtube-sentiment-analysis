from __future__ import annotations
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import config
from modules.local_translator import SUPPORTED, batch_translate
from modules.preprocessing import clean_comment_text
from modules.sentiment_analyzer import SentimentAnalyzer
from modules.spam_detector import filter_spam
from modules.visualizer import sentiment_pie_chart, sentiment_timeline_chart
from modules.wordcloud_gen import generate_wordcloud
from modules.youtube_api import YouTubeAPI

try:
    from langdetect import LangDetectException, detect
except Exception:  
    detect = None
    LangDetectException = Exception


class AnalysisService:

    def __init__(self, youtube_api: YouTubeAPI, sentiment_analyzer: SentimentAnalyzer, cache: Any) -> None:
        self.youtube_api = youtube_api
        self.sentiment_analyzer = sentiment_analyzer
        self.cache = cache

    @staticmethod
    def _detect_language(text: str) -> str:
        if not text or detect is None:
            return "en"
        try:
            return detect(text)
        except LangDetectException:
            return "en"

    def _translate_comments(self, comments: list[dict[str, Any]]) -> int:
        translation_failed_count = 0
        if not comments:
            return translation_failed_count

        texts = [comment.get("clean_text", "") for comment in comments]
        with ThreadPoolExecutor(max_workers=8) as executor:
            languages = list(executor.map(self._detect_language, texts))

        lang_groups: dict[str, list[int]] = defaultdict(list)
        for i, lang in enumerate(languages):
            comments[i]["language"] = lang
            if lang == "en" or lang not in SUPPORTED:
                comments[i]["translated_text"] = comments[i].get("clean_text", "")
            else:
                lang_groups[lang].append(i)

        for lang, indices in lang_groups.items():
            try:
                translated = batch_translate([comments[i]["clean_text"] for i in indices], lang)
                for i, translated_text in zip(indices, translated):
                    comments[i]["translated_text"] = translated_text
            except Exception:
                for i in indices:
                    comments[i]["translated_text"] = comments[i].get("clean_text", "")
                    translation_failed_count += 1

        return translation_failed_count

    @staticmethod
    def _prepare_comments(raw_comments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prepared: list[dict[str, Any]] = []
        for comment in raw_comments:
            cleaned = clean_comment_text(comment.get("text", ""))
            if not cleaned:
                continue
            updated = dict(comment)
            updated["clean_text"] = cleaned
            prepared.append(updated)
        return prepared

    @staticmethod
    def _is_valid_youtube_url(video_url: str) -> bool:
        lowered = (video_url or "").lower()
        return "youtube.com" in lowered or "youtu.be" in lowered

    def run(self, video_url: str) -> dict[str, Any]:
        if not self._is_valid_youtube_url(video_url):
            raise ValueError("INVALID_URL")

        video_info = self.youtube_api.get_video_details(video_url)
        video_id = video_info["video_id"]

        cached = self.cache.get(video_id)
        if cached:
            return cached

        raw_comments = self.youtube_api.get_all_comments(video_id, max_results=config.MAX_COMMENTS)
        if not raw_comments:
            raise ValueError("ZERO_COMMENTS")

        non_spam_comments, spam_count = filter_spam(raw_comments)
        prepared_comments = self._prepare_comments(non_spam_comments)
        if not prepared_comments:
            raise ValueError("ZERO_COMMENTS")

        translation_failed_count = self._translate_comments(prepared_comments)
        analyzed_comments, summary = self.sentiment_analyzer.analyze_all(prepared_comments)

        timeline_data = self.sentiment_analyzer.get_sentiment_over_time(analyzed_comments)
        top_positive, top_negative = self.sentiment_analyzer.get_top_comments(analyzed_comments)

        pie_chart_b64 = sentiment_pie_chart(
            summary["positive_count"],
            summary["negative_count"],
            summary["neutral_count"],
        )
        timeline_chart_b64 = sentiment_timeline_chart(timeline_data)
        wordcloud_path = generate_wordcloud(analyzed_comments)

        result = {
            "video_info": video_info,
            "summary": summary,
            "analyzed_comments": analyzed_comments,
            "timeline_data": timeline_data,
            "top_positive": top_positive,
            "top_negative": top_negative,
            "pie_chart_b64": pie_chart_b64,
            "timeline_chart_b64": timeline_chart_b64,
            "wordcloud_path": wordcloud_path,
            "spam_count": spam_count,
            "translation_failed_count": translation_failed_count,
            "positive": summary["positive_count"],
            "negative": summary["negative_count"],
            "neutral": summary["neutral_count"],
            "positive_percent": summary["positive_percent"],
            "negative_percent": summary["negative_percent"],
            "neutral_percent": summary["neutral_percent"],
            "prediction": summary["prediction"],
        }

        self.cache.set(video_id, result, timeout=config.CACHE_TIMEOUT)
        return result

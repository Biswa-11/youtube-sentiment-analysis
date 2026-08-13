from __future__ import annotations
import math
import re
from collections import defaultdict
from statistics import mean
from typing import Any
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import config
from modules.predictor import overall_sentiment_from_percentages

VADER_COMPOUND_POS = 0.1
VADER_COMPOUND_NEG = -0.1
CORE_NEUTRAL_ABS_COMPOUND = 0.08
BERT_UNCERTAIN_MAX_SCORE = 0.62

_STRONG_POS = frozenset(
    {
        "awesome",
        "amazing",
        "love",
        "best",
        "great",
        "excellent",
        "fantastic",
        "superb",
        "blockbuster",
        "super",
        "wonderful",
        "perfect",
        "brilliant",
        "incredible",
        "outstanding",
        "fabulous",
        "epic",
        "masterpiece",
        "beautiful",
        "loved",
        "enjoyed",
        "favorite",
        "favourite",
        "fire",
        "lit",
    }
)
_STRONG_NEG = frozenset(
    {
        "worst",
        "boring",
        "trash",
        "hate",
        "terrible",
        "awful",
        "bad",
        "horrible",
        "garbage",
        "pathetic",
        "disgusting",
        "waste",
        "cringe",
        "sucks",
        "sucked",
        "disappointed",
        "rubbish",
        "nonsense",
        "stupid",
        "ridiculous",
    }
)

_WORD_TOKEN_RE = re.compile(r"[a-z']+", re.I)


def _keyword_compound_delta(text: str) -> float:
    words = _WORD_TOKEN_RE.findall((text or "").lower())
    pos_hits = sum(1 for w in words if w in _STRONG_POS)
    neg_hits = sum(1 for w in words if w in _STRONG_NEG)
    net = pos_hits - neg_hits
    if net == 0:
        return 0.0
    raw = 0.07 * net
    return max(-0.18, min(0.18, raw))


class SentimentAnalyzer:

    _distilbert_pipe: Any = None

    def __init__(self) -> None:
        nltk.download("vader_lexicon", quiet=True)
        self.vader = SentimentIntensityAnalyzer()

    @classmethod
    def _get_distilbert_pipe(cls) -> Any:
        if cls._distilbert_pipe is None:
            import torch
            from transformers import pipeline

            cls._distilbert_pipe = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                truncation=True,
                max_length=512,
                device=0 if torch.cuda.is_available() else -1,
            )
        return cls._distilbert_pipe

    @staticmethod
    def _boost_compound(compound: float, text: str) -> float:
        boosted = float(compound) + _keyword_compound_delta(text)
        return max(-1.0, min(1.0, boosted))

    @staticmethod
    def _threshold_label(boosted_compound: float) -> str | None:
        if boosted_compound >= VADER_COMPOUND_POS:
            return "positive"
        if boosted_compound <= VADER_COMPOUND_NEG:
            return "negative"
        return None

    @staticmethod
    def _confidence_from_scores(scores: dict[str, float], sentiment: str) -> float:
        compound = abs(float(scores["compound"]))
        if compound >= VADER_COMPOUND_POS:
            return compound
        pos, neg = float(scores["pos"]), float(scores["neg"])
        if sentiment == "positive":
            return max(compound, pos)
        if sentiment == "negative":
            return max(compound, neg)
        return compound

    @staticmethod
    def _comment_text_for_sentiment(comment: dict[str, Any]) -> str:
        return (
            (comment.get("translated_text") or comment.get("clean_text") or comment.get("text") or "")
            .strip()
        )

    @staticmethod
    def _overall_from_counts(positive_count: int, negative_count: int, neutral_count: int) -> str:
        total = positive_count + negative_count + neutral_count
        if total == 0:
            return "MIXED"
        p_pct = (positive_count / total) * 100
        n_pct = (negative_count / total) * 100
        u_pct = (neutral_count / total) * 100
        return overall_sentiment_from_percentages(p_pct, n_pct, u_pct)

    def _distilbert_batch(self, texts: list[str]) -> list[tuple[str, float]]:
        if not texts:
            return []
        try:
            pipe = self._get_distilbert_pipe()
            outs = pipe(texts, truncation=True, max_length=512)
            decoded: list[tuple[str, float]] = []
            for row in outs:
                label = str(row.get("label", "")).upper()
                score = float(row.get("score", 0.5))
                if "POS" in label:
                    decoded.append(("positive", score))
                else:
                    decoded.append(("negative", score))
            return decoded
        except Exception:
            fallback: list[tuple[str, float]] = []
            for text in texts:
                scores = self.vader.polarity_scores(text)
                pos, neg = float(scores["pos"]), float(scores["neg"])
                compound = float(scores["compound"])
                if pos > neg:
                    fallback.append(("positive", max(pos, abs(compound))))
                elif neg > pos:
                    fallback.append(("negative", max(neg, abs(compound))))
                elif compound >= 0:
                    fallback.append(("positive", abs(compound) + 0.01))
                else:
                    fallback.append(("negative", abs(compound) + 0.01))
            return fallback

    def _resolve_ambiguous_band(
        self, text: str, scores: dict[str, float], raw_compound: float, boosted: float
    ) -> tuple[str, float]:
        if abs(boosted) <= CORE_NEUTRAL_ABS_COMPOUND and _keyword_compound_delta(text) == 0.0:
            return "neutral", max(0.25, 1.0 - abs(boosted))

        bert_label, bert_score = self._distilbert_batch([text])[0]
        if bert_score < BERT_UNCERTAIN_MAX_SCORE:
            return "neutral", max(0.25, 1.0 - bert_score)
        return bert_label, bert_score

    @staticmethod
    def _rebalance_neutral_bucket(comments_list: list[dict[str, Any]], included_indices: list[int]) -> None:
        total = len(included_indices)
        if total == 0:
            return
        min_pct = float(config.NEUTRAL_TARGET_MIN_PCT)
        max_pct = float(config.NEUTRAL_TARGET_MAX_PCT)
        min_n = math.ceil(total * min_pct / 100.0)
        max_n = max(math.floor(total * max_pct / 100.0), min_n)

        def tally() -> tuple[int, int, int]:
            p = n = u = 0
            for i in included_indices:
                s = comments_list[i].get("sentiment")
                if s == "positive":
                    p += 1
                elif s == "negative":
                    n += 1
                else:
                    u += 1
            return p, n, u

        p, n, u = tally()

        while u < min_n and p + n > 0:
            movable = [
                i
                for i in included_indices
                if comments_list[i].get("sentiment") in ("positive", "negative")
            ]
            if not movable:
                break
            movable.sort(key=lambda i: abs(float(comments_list[i].get("sentiment_score", 0.0))))
            i = movable[0]
            old = comments_list[i]["sentiment"]
            raw = float(comments_list[i].get("sentiment_score", 0.0))
            comments_list[i]["sentiment"] = "neutral"
            comments_list[i]["confidence"] = max(0.3, 1.0 - abs(raw))
            if old == "positive":
                p -= 1
            else:
                n -= 1
            u += 1

        while u > max_n:
            movable = [i for i in included_indices if comments_list[i].get("sentiment") == "neutral"]
            if not movable:
                break
            movable.sort(key=lambda i: abs(float(comments_list[i].get("sentiment_score", 0.0))), reverse=True)
            i = movable[0]
            raw = float(comments_list[i].get("sentiment_score", 0.0))
            new_s = "positive" if raw >= 0 else "negative"
            comments_list[i]["sentiment"] = new_s
            comments_list[i]["confidence"] = max(0.35, abs(raw) + 0.05)
            u -= 1
            if new_s == "positive":
                p += 1
            else:
                n += 1

    def analyze_single(self, text: str) -> dict[str, Any]:
        stripped = (text or "").strip()
        if not stripped:
            return {"sentiment": "neutral", "score": 0.0, "confidence": 0.0}

        scores = self.vader.polarity_scores(stripped)
        raw_compound = float(scores["compound"])
        boosted = self._boost_compound(raw_compound, stripped)
        label = self._threshold_label(boosted)

        if label is not None:
            sentiment = label
            confidence = self._confidence_from_scores(
                {**scores, "compound": boosted},
                sentiment,
            )
        else:
            sentiment, confidence = self._resolve_ambiguous_band(stripped, scores, raw_compound, boosted)

        return {"sentiment": sentiment, "score": raw_compound, "confidence": confidence}

    def _analyze_batch(
        self, batch: list[tuple[int, dict[str, Any]]]
    ) -> list[tuple[int, float, str, float, bool]]:
        staged: dict[int, tuple[float, dict[str, float], str | None, str | None]] = {}
        # staged[idx] = (raw, scores, vader_label or None, pre_resolved_sentiment or None)
        # If pre_resolved is set (e.g. core neutral), skip DistilBERT. If vader_label set, VADER wins.
        bert_idx_order: list[int] = []
        bert_texts: list[str] = []

        for idx, comment in batch:
            text = self._comment_text_for_sentiment(comment)
            if not text:
                staged[idx] = (0.0, {}, None, None)
                continue

            scores = self.vader.polarity_scores(text)
            raw_compound = float(scores["compound"])
            boosted = self._boost_compound(raw_compound, text)
            vlabel = self._threshold_label(boosted)

            if vlabel is not None:
                staged[idx] = (raw_compound, scores, vlabel, None)
                continue

            if abs(boosted) <= CORE_NEUTRAL_ABS_COMPOUND and _keyword_compound_delta(text) == 0.0:
                staged[idx] = (raw_compound, scores, None, "neutral")
                continue

            staged[idx] = (raw_compound, scores, None, None)
            bert_idx_order.append(idx)
            bert_texts.append(text)

        bert_by_idx: dict[int, tuple[str, float]] = {}
        if bert_texts:
            bert_out = self._distilbert_batch(bert_texts)
            for j, idx in enumerate(bert_idx_order):
                bert_by_idx[idx] = bert_out[j]

        result: list[tuple[int, float, str, float, bool]] = []
        for idx, comment in batch:
            raw_compound, scores, vlabel, pre = staged[idx]

            text = self._comment_text_for_sentiment(comment)
            if not text:
                result.append((idx, 0.0, "neutral", 0.0, False))
                continue

            if vlabel is not None:
                sentiment = vlabel
                boosted = self._boost_compound(raw_compound, text)
                confidence = self._confidence_from_scores({**scores, "compound": boosted}, sentiment)
            elif pre == "neutral":
                sentiment = "neutral"
                boosted = self._boost_compound(raw_compound, text)
                confidence = max(0.25, 1.0 - abs(boosted))
            else:
                bert_label, bert_score = bert_by_idx[idx]
                if bert_score < BERT_UNCERTAIN_MAX_SCORE:
                    sentiment = "neutral"
                    confidence = max(0.25, 1.0 - bert_score)
                else:
                    sentiment = bert_label
                    confidence = bert_score

            result.append((idx, raw_compound, sentiment, confidence, True))

        return result

    def analyze_all(self, comments_list: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        indexed_comments = list(enumerate(comments_list))
        if not indexed_comments:
            return comments_list, {
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "overall_sentiment": "MIXED",
                "positive_percent": 0.0,
                "negative_percent": 0.0,
                "neutral_percent": 0.0,
                "prediction": "MIXED",
            }

        analyzed_batch = self._analyze_batch(indexed_comments)
        included_indices: list[int] = []
        for idx, compound, sentiment, confidence, include in analyzed_batch:
            comments_list[idx]["sentiment_score"] = compound
            comments_list[idx]["sentiment"] = sentiment
            comments_list[idx]["confidence"] = confidence
            if not include:
                continue
            included_indices.append(idx)

        self._rebalance_neutral_bucket(comments_list, included_indices)

        positive_count = 0
        negative_count = 0
        neutral_count = 0
        for idx in included_indices:
            sentiment = comments_list[idx].get("sentiment")
            if sentiment == "positive":
                positive_count += 1
            elif sentiment == "negative":
                negative_count += 1
            else:
                neutral_count += 1

        classified_total = positive_count + negative_count + neutral_count
        denom = classified_total if classified_total else 1
        overall_sentiment = self._overall_from_counts(positive_count, negative_count, neutral_count)

        summary = {
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count,
            "overall_sentiment": overall_sentiment,
            "positive_percent": round((positive_count / denom) * 100, 2),
            "negative_percent": round((negative_count / denom) * 100, 2),
            "neutral_percent": round((neutral_count / denom) * 100, 2),
            "prediction": overall_sentiment,
        }

        return comments_list, summary

    def get_sentiment_over_time(self, analyzed_comments: list[dict[str, Any]]) -> list[dict[str, float]]:
        month_scores: dict[str, list[float]] = defaultdict(list)
        for comment in analyzed_comments:
            published_at = comment.get("published_at")
            if not published_at:
                continue
            month_key = published_at.strftime("%Y-%m")
            month_scores[month_key].append(float(comment.get("sentiment_score", 0.0)))

        timeline = []
        for date_key in sorted(month_scores.keys()):
            avg_score = float(mean(month_scores[date_key]))
            timeline.append({"date": date_key, "avg_score": round(avg_score, 4)})
        return timeline

    def get_top_comments(
        self, analyzed_comments: list[dict[str, Any]], n: int = 5
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not analyzed_comments:
            return [], []

        max_likes = max((comment.get("likes", 0) for comment in analyzed_comments), default=1) or 1
        for comment in analyzed_comments:
            likes_norm = comment.get("likes", 0) / max_likes
            sentiment_score = comment.get("sentiment_score", 0.0)
            comment["rank_score"] = (sentiment_score * 0.6) + (likes_norm * 0.4)

        positive_comments = [c for c in analyzed_comments if c.get("sentiment") == "positive"]
        negative_comments = [c for c in analyzed_comments if c.get("sentiment") == "negative"]

        top_positive = sorted(positive_comments, key=lambda x: x.get("rank_score", 0), reverse=True)[:n]
        top_negative = sorted(negative_comments, key=lambda x: x.get("rank_score", 0))[:n]

        return top_positive, top_negative
from __future__ import annotations
import re
from better_profanity import profanity
from config import SPAM_THRESHOLD

SPAM_PHRASES = [
    "subscribe to my channel",
    "check out my video",
    "first",
    "sub4sub",
    "follow me",
    "click here",
    "free followers",
]


def is_spam(comment_text: str) -> bool:
    text = (comment_text or "").strip()
    lowered = text.lower()

    if len(text) < SPAM_THRESHOLD:
        return True

    if len(re.findall(r"https?://|www\.", lowered)) > 5:
        return True

    if len(text) > 20 and text.isupper():
        return True

    tokens = re.findall(r"\b\w+\b", lowered)
    if tokens:
        max_repeat = max(tokens.count(token) for token in set(tokens))
        if max_repeat > 3:
            return True

    if any(phrase in lowered for phrase in SPAM_PHRASES):
        return True

    emoji_matches = re.findall(
        r"[\U0001F300-\U0001F5FF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF\U0001F900-\U0001F9FF]",
        text,
    )
    if len(emoji_matches) > 10:
        return True

    if re.fullmatch(r"@\w+", text):
        return True

    if profanity.contains_profanity(lowered) and len(tokens) < 3:
        return True

    return False


def filter_spam(comments_list: list[dict]) -> tuple[list[dict], int]:
    profanity.load_censor_words()
    clean_comments = [comment for comment in comments_list if not is_spam(comment.get("text", ""))]
    spam_count = len(comments_list) - len(clean_comments)
    print(f"[SpamDetector] Removed {spam_count} spam comments.")
    return clean_comments, spam_count
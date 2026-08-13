from __future__ import annotations
import re

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MULTI_SPACE_RE = re.compile(r"\s+")


def clean_comment_text(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""

    cleaned = URL_RE.sub(" ", cleaned)
    cleaned = MULTI_SPACE_RE.sub(" ", cleaned).strip()
    return cleaned
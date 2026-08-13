from __future__ import annotations
from functools import lru_cache
from googletrans import Translator
from langdetect import LangDetectException, detect

_translator = Translator()

@lru_cache(maxsize=2048)
def detect_and_translate(text: str) -> dict[str, str]:
    cleaned = (text or "").strip()
    if not cleaned:
        return {"original": "", "translated": "", "language": "unknown"}

    try:
        language = detect(cleaned)
    except LangDetectException:
        language = "unknown"

    translated = cleaned
    if language not in {"en", "unknown"}:
        try:
            translated = _translator.translate(cleaned, dest="en").text
        except Exception:
            translated = cleaned

    return {"original": cleaned, "translated": translated, "language": language}

from __future__ import annotations
import torch
from transformers import MarianMTModel, MarianTokenizer
from langdetect import LangDetectException, detect

SUPPORTED = {
    "hi": "Helsinki-NLP/opus-mt-hi-en",  # Hindi
    "bn": "Helsinki-NLP/opus-mt-bn-en",  # Bengali
    "ta": "Helsinki-NLP/opus-mt-ta-en",  # Tamil
    "te": "Helsinki-NLP/opus-mt-mul-en",  # Telugu
    "mr": "Helsinki-NLP/opus-mt-mul-en",  # Marathi
    "ur": "Helsinki-NLP/opus-mt-ur-en",  # Urdu
    "or": "Helsinki-NLP/opus-mt-or-en",  # Odia / Oriya
}

_model_cache = {}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _get_model(lang_code: str):
    model_name = SUPPORTED.get(lang_code)
    if not model_name:
        raise ValueError(f"Unsupported language for local translation: {lang_code}")
    if model_name not in _model_cache:
        print(f"[Translator] Loading model for {lang_code}...")
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name).to(DEVICE)
        _model_cache[model_name] = (tokenizer, model)
    return _model_cache[model_name]


def translate_to_english(text: str, lang: str) -> str:
    try:
        tokenizer, model = _get_model(lang)
        tokens = tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(DEVICE)
        translated = model.generate(**tokens)
        return tokenizer.decode(translated[0], skip_special_tokens=True)
    except Exception:
        return text


def batch_translate(comments: list[str], lang: str) -> list[str]:
    if lang not in SUPPORTED:
        return comments
    try:
        tokenizer, model = _get_model(lang)
        tokens = tokenizer(
            comments,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(DEVICE)
        translated = model.generate(**tokens)
        return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    except Exception:
        return comments
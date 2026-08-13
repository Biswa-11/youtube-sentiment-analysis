from __future__ import annotations
import re
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud


def generate_wordcloud(
    comments_list: list[dict], output_path: str = "static/images/wordcloud.png"
) -> str | None:
    try:
        nltk.download("stopwords", quiet=True)
        texts = " ".join(comment.get("translated_text", comment.get("text", "")) for comment in comments_list)
        if not texts.strip():
            return None

        texts = re.sub(r"https?://\S+|www\.\S+", " ", texts)
        texts = re.sub(r"@\w+|#\w+", " ", texts)
        texts = re.sub(r"[^\w\s]", " ", texts)
        texts = re.sub(r"\s+", " ", texts).strip()

        stop_words = set(stopwords.words("english"))
        stop_words.update({"movie", "film", "video", "watch"})

        cloud = WordCloud(
            background_color="black",
            max_words=150,
            colormap="Set2",
            width=800,
            height=400,
            min_font_size=10,
            stopwords=stop_words,
        ).generate(texts)

        cloud.to_file(output_path)
        return output_path
    except Exception:
        return None
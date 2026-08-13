from __future__ import annotations
import base64
from io import BytesIO
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _fig_to_base64() -> str:
    buffer = BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight", transparent=True)
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    buffer.close()
    plt.close()
    return encoded


def sentiment_pie_chart(positive: int, negative: int, neutral: int) -> str:
    labels = ["Positive", "Negative", "Neutral"]
    values = [positive, negative, neutral]
    colors = ["#4CAF50", "#F44336", "#FFC107"]
    explode = [0.06, 0.04, 0.04]

    if sum(values) == 0:
        values = [1, 1, 1]

    plt.figure(figsize=(7, 6), facecolor="white")
    plt.pie(
        values,
        labels=labels,
        colors=colors,
        explode=explode,
        autopct="%1.1f%%",
        startangle=140,
        shadow=True,
        textprops={"color": "#111111", "fontsize": 14},
    )
    plt.axis("equal")
    plt.title("Sentiment Distribution", fontsize=24, fontweight="bold")
    plt.legend(labels, loc="upper right", fontsize=12, framealpha=0.8)
    return _fig_to_base64()


def sentiment_timeline_chart(timeline_data: list[dict]) -> str:
    dates = [item.get("date", "") for item in timeline_data]
    scores = [float(item.get("avg_score", 0.0)) for item in timeline_data]

    if not dates:
        dates = ["N/A"]
        scores = [0.0]

    plt.figure(figsize=(10, 4.5), facecolor="#0d1117")
    axis = plt.gca()
    axis.set_facecolor("#0d1117")
    plt.plot(dates, scores, color="#58a6ff", marker="o", linewidth=2)
    plt.axhline(0, color="#8b949e", linestyle="--", linewidth=1)
    plt.ylim(-1, 1)
    plt.xticks(rotation=45, ha="right", color="white")
    plt.yticks(color="white")
    plt.grid(alpha=0.2, color="white")
    plt.title("Sentiment Over Time", color="white")
    plt.xlabel("Date", color="white")
    plt.ylabel("Average Sentiment Score", color="white")
    return _fig_to_base64()
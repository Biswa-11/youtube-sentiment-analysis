from __future__ import annotations
from typing import Any

TOP_TWO_GAP_PCT = 5.0


def overall_sentiment_from_percentages(
    positive_percent: float,
    negative_percent: float,
    neutral_percent: float,
) -> str:
    p_pct = float(positive_percent)
    n_pct = float(negative_percent)
    u_pct = float(neutral_percent)

    if u_pct > 50.0:
        return "NEUTRAL"

    sorted_pcts = sorted((p_pct, n_pct, u_pct), reverse=True)
    if sorted_pcts[0] - sorted_pcts[1] < TOP_TWO_GAP_PCT:
        return "NEUTRAL"

    if p_pct > n_pct and p_pct > u_pct:
        return "POSITIVE"
    if n_pct > p_pct and n_pct > u_pct:
        return "NEGATIVE"
    return "NEUTRAL"


def predict_box_office(features: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "not_implemented",
        "message": "Box office prediction is a placeholder in this version.",
        "input_features": features,
    }
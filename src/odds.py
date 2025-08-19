from __future__ import annotations

from typing import Tuple


def prob_to_american_odds(probability: float) -> int:
    p = max(1e-6, min(1 - 1e-6, float(probability)))
    if p >= 0.5:
        return int(round(-100 * p / (1 - p)))
    return int(round(100 * (1 - p) / p))


def american_odds_to_prob(odds: int) -> float:
    if odds < 0:
        return (-odds) / ((-odds) + 100)
    return 100 / (odds + 100)


def format_moneyline(odds: int) -> str:
    return f"{odds:+d}" 
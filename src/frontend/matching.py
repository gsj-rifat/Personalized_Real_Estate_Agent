from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ListingMatch:
    neighborhood: str
    location: str
    bedrooms: int
    bathrooms: float
    house_size_sqft: int
    price_k_usd: float
    score: float


def _normalize_tokens(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {w for w in words if len(w) > 2}


def _score_row(preferences_text: str, row_text: str) -> float:
    pref_tokens = _normalize_tokens(preferences_text)
    row_tokens = _normalize_tokens(row_text)
    if not pref_tokens:
        return 0.0
    overlap = len(pref_tokens.intersection(row_tokens))
    return overlap / len(pref_tokens)


def top_listing_matches(csv_path: str, preferences_text: str, limit: int = 3) -> list[ListingMatch]:
    path = Path(csv_path)
    if not path.exists():
        return []

    df = pd.read_csv(path)
    columns = [
        "Neighborhood",
        "Location",
        "Bedrooms",
        "Bathrooms",
        "House Size (sqft)",
        "Price (k$)",
    ]
    for col in columns:
        if col not in df.columns:
            return []

    scored: list[ListingMatch] = []
    for _, row in df.iterrows():
        row_text = " ".join(str(row[col]) for col in columns)
        score = _score_row(preferences_text, row_text)
        scored.append(
            ListingMatch(
                neighborhood=str(row["Neighborhood"]),
                location=str(row["Location"]),
                bedrooms=int(row["Bedrooms"]),
                bathrooms=float(row["Bathrooms"]),
                house_size_sqft=int(row["House Size (sqft)"]),
                price_k_usd=float(row["Price (k$)"]),
                score=score,
            )
        )

    return sorted(scored, key=lambda x: x.score, reverse=True)[:limit]


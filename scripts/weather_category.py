"""Map `Sighter Observed Weather Data` free text to a single 4-category column.

Output values: {sunny, cloudy, rainy, unknown}

Priority order resolves multi-descriptor cells:
  rainy > cloudy > sunny > unknown

Examples:
  '70º F, sunny'             -> sunny
  '65º F, cloudy, drizzle'   -> rainy   (rain > cloud)
  '55º F, partly cloudy'     -> cloudy
  '60º F, breezy'            -> unknown (wind-only is not in the 4-category set)
  '70º F'                    -> unknown

Run as a script for a sanity-check breakdown on data/hectare.csv:
    python scripts/weather_category.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


# Priority-ordered rules: first match wins.
RULES: list[tuple[str, list[str]]] = [
    ("rainy",  [r"\brain", r"\bdrizzl", r"\bsprinkl", r"\bmist", r"\bfog", r"\bshower"]),
    ("cloudy", [r"\bcloud\w*", r"\bovercast\b"]),
    ("sunny",  [r"\bsunny\b", r"\bsun\b", r"\bclear\b", r"\bblue sk"]),
]

# Common typos in the raw data, normalised before pattern matching.
TYPO_FIXES: dict[str, str] = {
    r"\bcoudy\b": "cloudy",
    r"\bsuny\b": "sunny",
}


_COMPILED = [(label, [re.compile(p, re.IGNORECASE) for p in pats]) for label, pats in RULES]
_TYPO = [(re.compile(k, re.IGNORECASE), v) for k, v in TYPO_FIXES.items()]


def _normalise(text: str) -> str:
    for pat, repl in _TYPO:
        text = pat.sub(repl, text)
    return text


def _label(text: object) -> str:
    if not isinstance(text, str):
        return "unknown"
    text = _normalise(text)
    for label, patterns in _COMPILED:
        for p in patterns:
            if p.search(text):
                return label
    return "unknown"


def weather_category(series: pd.Series) -> pd.Series:
    """Return a Series of {sunny, cloudy, rainy, unknown} aligned to ``series``."""
    return series.apply(_label).rename("weather")


if __name__ == "__main__":
    path = Path(__file__).resolve().parents[1] / "data" / "hectare.csv"
    raw = pd.read_csv(path)["Sighter Observed Weather Data"]
    out = weather_category(raw)

    n = len(out)
    print(f"weather  ({n} rows)")
    for k, v in out.value_counts(dropna=False).items():
        print(f"  {k:8s} {v:5d}  ({v / n:.1%})")

    unk_non_null = (out == "unknown") & raw.notna()
    print(f"\n{int(unk_non_null.sum())} non-null rows still 'unknown' (sample):")
    for v in raw[unk_non_null]:
        print(f"  - {v!r}")

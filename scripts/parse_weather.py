"""Parse `Sighter Observed Weather Data` free text into a small set of features.

Strategy: two-layer mapping.
  Layer 1 (fine):   raw tokens / regex patterns extracted from the text.
  Layer 2 (broad):  fine tokens are folded into a small number of orthogonal
                    categorical axes. The fitted pipeline one-hot-encodes
                    these later, so we want few categories per axis.

Three axes plus one boolean — total ~10 OHE columns instead of 20 booleans:
  - sky_cover : clear / partly / cloudy / overcast / unknown
  - precip    : none / mist / drizzle / rain
  - wind      : calm / breezy / windy / unknown
  - is_humid  : 0/1 (orthogonal to temperature_f, so kept separate)

Per-axis priority rules: the first pattern that matches in priority order
wins, so e.g. 'partly cloudy' is assigned to `cloudy` (heavier-cover label
listed first). Adjust the priority order in the *_RULES lists below if you
prefer 'partly' to win over 'cloudy'.

Run as a script for a sanity-check breakdown on data/hectare.csv:
    python scripts/parse_weather.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Layer 1 -> Layer 2 rules. Each rule is (broad_label, [regex patterns]).
# Priority: earlier rules win over later ones.
# Patterns use word boundaries to avoid matching inside other words.
# ---------------------------------------------------------------------------

SKY_COVER_RULES: list[tuple[str, list[str]]] = [
    ("overcast", [r"\bovercast\b"]),
    ("cloudy",   [r"\bcloudy\b", r"\bclouds?\b"]),
    ("partly",   [r"\bpartly\b", r"\bmostly\b", r"\bsome sun\b"]),
    ("clear",    [r"\bsunny\b", r"\bsun\b", r"\bclear\b", r"\bblue sk"]),
]

PRECIP_RULES: list[tuple[str, list[str]]] = [
    # rain trumps drizzle trumps mist (heaviest first).
    ("rain",    [r"\brain", r"\bshower", r"\bpour"]),          # rain/rainy/raining
    ("drizzle", [r"\bdrizzl", r"\bsprinkl"]),                  # drizzle/drizzling/sprinkling
    ("mist",    [r"\bmist", r"\bfog", r"\bdamp\b", r"\bmoist\b", r"\bwet\b"]),
]

WIND_RULES: list[tuple[str, list[str]]] = [
    ("windy",  [r"\bwindy\b", r"\bwind\b", r"\bgust"]),
    ("breezy", [r"\bbreez"]),                                  # breeze/breezy
    ("calm",   [r"\bcalm\b", r"\bstill\b", r"\bno wind\b"]),
]

HUMID_PATTERNS = [r"\bhumid", r"\bmuggy\b", r"\bsticky\b"]

# Compact single-axis rules. Priority order resolves multi-descriptor cells:
#   "sunny, breezy"          -> sunny  (wind dropped)
#   "cloudy, light drizzle"  -> rainy  (cloud dropped — rain is more behaviourally relevant)
#   "70º F"                  -> unknown
WEATHER_COMPACT_RULES: list[tuple[str, list[str]]] = [
    ("rainy",  [r"\brain", r"\bdrizzl", r"\bsprinkl", r"\bmist", r"\bfog", r"\bshower"]),
    ("cloudy", [r"\bcloudy\b", r"\bclouds?\b", r"\bovercast\b", r"\bpartly\b", r"\bmostly\b"]),
    ("sunny",  [r"\bsunny\b", r"\bsun\b", r"\bclear\b", r"\bblue sk"]),
    ("windy",  [r"\bwindy\b", r"\bwind\b", r"\bbreez", r"\bgust"]),
]

# Common typos found in the raw data, applied before pattern matching.
TYPO_FIXES = {
    r"\bcoudy\b": "cloudy",
    r"\bsuny\b": "sunny",
}


# ---------------------------------------------------------------------------
# Compile once.
# ---------------------------------------------------------------------------

def _compile(rules: list[tuple[str, list[str]]]):
    return [(label, [re.compile(p, re.IGNORECASE) for p in pats]) for label, pats in rules]


_SKY = _compile(SKY_COVER_RULES)
_PRECIP = _compile(PRECIP_RULES)
_WIND = _compile(WIND_RULES)
_COMPACT = _compile(WEATHER_COMPACT_RULES)
_HUMID = [re.compile(p, re.IGNORECASE) for p in HUMID_PATTERNS]
_TYPO = [(re.compile(k, re.IGNORECASE), v) for k, v in TYPO_FIXES.items()]


def _normalise(text: str) -> str:
    for pat, repl in _TYPO:
        text = pat.sub(repl, text)
    return text


def _first_match(text: str, compiled_rules, default: str) -> str:
    if not isinstance(text, str):
        return default
    text = _normalise(text)
    for label, pats in compiled_rules:
        for p in pats:
            if p.search(text):
                return label
    return default


def _any_match(text: str, compiled_pats) -> int:
    if not isinstance(text, str):
        return 0
    return int(any(p.search(text) for p in compiled_pats))


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------

def parse_weather(
    series: pd.Series,
    *,
    include_wind: bool = False,
    infer_sky_from_precip: bool = False,
) -> pd.DataFrame:
    """Map a free-text weather Series to engineered columns.

    NaNs map to: sky_cover='unknown', precip='none', wind='unknown', is_humid=0.

    Parameters
    ----------
    include_wind : default False. Wind is mentioned in only ~20% of cells, so
        the resulting column is ~80% 'unknown' and unlikely to carry signal at
        n≈643 train rows. Enable only if you specifically want it.
    infer_sky_from_precip : default False. If True, rows where sky_cover would
        be 'unknown' but precip != 'none' are reassigned to 'overcast' (rain /
        drizzle / fog implies non-clear sky). Reduces sky_cover unknowns at
        the cost of inducing correlation between sky_cover and precip.
    """
    out = pd.DataFrame(index=series.index)
    out["sky_cover"] = series.apply(lambda x: _first_match(x, _SKY, "unknown"))
    out["precip"] = series.apply(lambda x: _first_match(x, _PRECIP, "none"))
    out["is_humid"] = series.apply(lambda x: _any_match(x, _HUMID))
    if include_wind:
        out["wind"] = series.apply(lambda x: _first_match(x, _WIND, "unknown"))
    if infer_sky_from_precip:
        mask = (out["sky_cover"] == "unknown") & (out["precip"] != "none")
        out.loc[mask, "sky_cover"] = "overcast"
    return out


def parse_weather_compact(
    series: pd.Series,
    *,
    drop_windy: bool = False,
) -> pd.DataFrame:
    """Map weather text to a single 5-category column plus is_humid.

    Returns columns:
      - weather  : {sunny, cloudy, rainy, windy, unknown}  (4 if drop_windy)
      - is_humid : 0/1

    Priority order (rainy > cloudy > sunny > windy) means cells with multiple
    descriptors get the higher-priority label only. See WEATHER_COMPACT_RULES.

    Parameters
    ----------
    drop_windy : if True, the `windy` rules are skipped so wind-only cells
        fall through to 'unknown'. Use when the 'windy' bucket is too small
        to be useful (~2% on hectare.csv).
    """
    rules = [r for r in _COMPACT if not (drop_windy and r[0] == "windy")]

    def _label(t):
        if not isinstance(t, str):
            return "unknown"
        t = _normalise(t)
        for lab, pats in rules:
            for p in pats:
                if p.search(t):
                    return lab
        return "unknown"

    out = pd.DataFrame(index=series.index)
    out["weather"] = series.apply(_label)
    out["is_humid"] = series.apply(lambda x: _any_match(x, _HUMID))
    return out


# ---------------------------------------------------------------------------
# CLI sanity check.
# ---------------------------------------------------------------------------

def _summarise(df: pd.DataFrame) -> None:
    n = len(df)
    for col in [c for c in ("sky_cover", "precip", "wind") if c in df.columns]:
        counts = df[col].value_counts(dropna=False)
        print(f"\n{col}  ({n} rows)")
        for k, v in counts.items():
            print(f"  {k:10s} {v:5d}  ({v / n:.1%})")
    h = int(df["is_humid"].sum())
    print(f"\nis_humid     {h:5d}  ({h / n:.1%})")


def _flag_unknowns(df: pd.DataFrame, raw: pd.Series) -> None:
    """Show raw text for rows that ended up 'unknown' on sky_cover, so you can
    decide whether to extend the rules."""
    mask = (df["sky_cover"] == "unknown") & raw.notna()
    if mask.any():
        print(f"\n{int(mask.sum())} non-null rows with sky_cover='unknown':")
        for v in raw[mask].head(15):
            print(f"  - {v!r}")


if __name__ == "__main__":
    path = Path(__file__).resolve().parents[1] / "data" / "hectare.csv"
    raw = pd.read_csv(path)["Sighter Observed Weather Data"]

    print("=== Default (no wind, no precip→sky inference) ===")
    parsed = parse_weather(raw)
    _summarise(parsed)
    _flag_unknowns(parsed, raw)

    print("\n=== With infer_sky_from_precip=True ===")
    parsed2 = parse_weather(raw, infer_sky_from_precip=True)
    _summarise(parsed2)

    print("\n=== Compact (single 5-cat column) ===")
    compact = parse_weather_compact(raw)
    print(compact["weather"].value_counts(dropna=False))
    print(f"is_humid: {int(compact['is_humid'].sum())} / {len(compact)}")

    print("\n=== Compact with drop_windy=True (4-cat) ===")
    compact4 = parse_weather_compact(raw, drop_windy=True)
    print(compact4["weather"].value_counts(dropna=False))
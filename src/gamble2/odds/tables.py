"""Load and query precomputed preflop equity tables."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from pathlib import Path

from gamble2.odds.enumerate import Equity
from gamble2.vision.cards import Card, canonical_preflop_key

# Preferred locations: package data, then repo data/
_PACKAGE_DATA = "gamble2.data"
_FALLBACK = Path(__file__).resolve().parents[3] / "data" / "preflop_vs_random.json"


@lru_cache(maxsize=1)
def _load_vs_random() -> dict[str, dict[str, float]]:
    # Try package resource first
    try:
        root = resources.files(_PACKAGE_DATA)
        text = (root / "preflop_vs_random.json").read_text(encoding="utf-8")
        return json.loads(text)
    except (FileNotFoundError, ModuleNotFoundError, TypeError, OSError):
        pass

    if _FALLBACK.exists():
        return json.loads(_FALLBACK.read_text(encoding="utf-8"))

    raise FileNotFoundError(
        "Missing preflop table. Run: python -m gamble2.odds.build_tables"
    )


@lru_cache(maxsize=1)
def _load_matchups() -> dict[str, dict[str, dict[str, float]]] | None:
    try:
        root = resources.files(_PACKAGE_DATA)
        path = root / "preflop_matchups.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ModuleNotFoundError, TypeError, OSError):
        pass

    matchup_path = _FALLBACK.parent / "preflop_matchups.json"
    if matchup_path.exists():
        return json.loads(matchup_path.read_text(encoding="utf-8"))
    return None


def preflop_vs_random(hero: list[Card]) -> Equity:
    key = canonical_preflop_key(hero)
    row = _load_vs_random()[key]
    return Equity(row["win"], row["tie"], row["lose"], int(row.get("samples", 0)))


def preflop_matchup(hero: list[Card], dealer: list[Card]) -> Equity | None:
    tables = _load_matchups()
    if tables is None:
        return None
    hk = canonical_preflop_key(hero)
    dk = canonical_preflop_key(dealer)
    row = tables.get(hk, {}).get(dk)
    if row is None:
        return None
    return Equity(row["win"], row["tie"], row["lose"], int(row.get("samples", 0)))

"""Card model and notation helpers (e.g. Ah, Td, 2c)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

RANKS = "23456789TJQKA"
SUITS = "cdhs"  # clubs, diamonds, hearts, spades
SUIT_SYMBOLS = {"c": "♣", "d": "♦", "h": "♥", "s": "♠"}
RANK_NAMES = {
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
    "6": "6",
    "7": "7",
    "8": "8",
    "9": "9",
    "T": "T",
    "J": "J",
    "Q": "Q",
    "K": "K",
    "A": "A",
}


@dataclass(frozen=True, order=True)
class Card:
    rank: str
    suit: str

    def __post_init__(self) -> None:
        rank = self.rank.upper()
        suit = self.suit.lower()
        if rank not in RANKS:
            raise ValueError(f"Invalid rank: {self.rank}")
        if suit not in SUITS:
            raise ValueError(f"Invalid suit: {self.suit}")
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "suit", suit)

    @classmethod
    def parse(cls, text: str) -> Card:
        text = text.strip()
        if len(text) != 2:
            raise ValueError(f"Expected 2-char card like Ah, got {text!r}")
        return cls(text[0], text[1])

    @property
    def code(self) -> str:
        return f"{self.rank}{self.suit}"

    def __str__(self) -> str:
        return self.code

    def pretty(self) -> str:
        return f"{self.rank}{SUIT_SYMBOLS[self.suit]}"


def parse_cards(text: str | Iterable[str]) -> list[Card]:
    if isinstance(text, str):
        parts = text.replace(",", " ").split()
    else:
        parts = list(text)
    return [Card.parse(p) for p in parts]


def full_deck() -> list[Card]:
    return [Card(r, s) for r in RANKS for s in SUITS]


def canonical_preflop_key(cards: list[Card]) -> str:
    """Map two hole cards to a 169-key like AA, AKs, AKo."""
    if len(cards) != 2:
        raise ValueError("Need exactly two hole cards")
    a, b = cards
    r1, r2 = a.rank, b.rank
    i1, i2 = RANKS.index(r1), RANKS.index(r2)
    if i1 < i2:
        r1, r2 = r2, r1
        a, b = b, a
    if r1 == r2:
        return f"{r1}{r2}"
    suited = a.suit == b.suit
    return f"{r1}{r2}{'s' if suited else 'o'}"


def all_preflop_keys() -> list[str]:
    keys: list[str] = []
    for i, r1 in enumerate(RANKS):
        for j, r2 in enumerate(RANKS):
            if i == j:
                keys.append(f"{r1}{r2}")
            elif i > j:
                keys.append(f"{r1}{r2}s")
                keys.append(f"{r1}{r2}o")
    return keys

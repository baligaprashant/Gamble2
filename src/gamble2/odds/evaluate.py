"""Hand strength evaluation via treys."""

from __future__ import annotations

from treys import Card as TreysCard
from treys import Evaluator

from gamble2.vision.cards import Card

_EVALUATOR = Evaluator()


def to_treys(card: Card) -> int:
    return TreysCard.new(card.code)


def to_treys_list(cards: list[Card]) -> list[int]:
    return [to_treys(c) for c in cards]


def evaluate_seven(hole: list[Card], board: list[Card]) -> int:
    """Lower score is stronger (treys convention)."""
    if len(hole) != 2:
        raise ValueError("hole must have 2 cards")
    if len(board) != 5:
        raise ValueError("board must have 5 cards")
    return _EVALUATOR.evaluate(to_treys_list(board), to_treys_list(hole))

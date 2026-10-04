"""Exact heads-up equity enumeration for flop / turn / river."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from gamble2.odds.evaluate import evaluate_seven
from gamble2.vision.cards import Card, full_deck


@dataclass(frozen=True)
class Equity:
    win: float
    tie: float
    lose: float
    samples: int

    @property
    def equity(self) -> float:
        """Win probability counting ties as half."""
        return self.win + 0.5 * self.tie

    def as_dict(self) -> dict[str, float | int]:
        return {
            "win": round(self.win * 100, 2),
            "tie": round(self.tie * 100, 2),
            "lose": round(self.lose * 100, 2),
            "equity": round(self.equity * 100, 2),
            "samples": self.samples,
        }


def _remaining(dead: set[Card]) -> list[Card]:
    return [c for c in full_deck() if c not in dead]


def _normalize(wins: int, ties: int, losses: int) -> Equity:
    total = wins + ties + losses
    if total == 0:
        return Equity(0.0, 0.0, 0.0, 0)
    return Equity(wins / total, ties / total, losses / total, total)


def equity_vs_dealer(
    hero: list[Card],
    board: list[Card],
    dealer: list[Card] | None = None,
) -> Equity:
    """Exact equity vs dealer for postflop (or river).

    Preflop should use the lookup table; calling this with an empty board
    is supported but slow (full enumeration of boards).
    """
    if len(hero) != 2:
        raise ValueError("hero needs 2 hole cards")
    if len(board) not in (0, 3, 4, 5):
        raise ValueError("board must be 0, 3, 4, or 5 cards")
    if dealer is not None and len(dealer) != 2:
        raise ValueError("dealer needs 2 hole cards when provided")

    dead = set(hero) | set(board)
    if dealer:
        dead |= set(dealer)
        return _equity_known_dealer(hero, board, dealer, dead)
    return _equity_unknown_dealer(hero, board, dead)


def _equity_known_dealer(
    hero: list[Card],
    board: list[Card],
    dealer: list[Card],
    dead: set[Card],
) -> Equity:
    remaining = _remaining(dead)
    need = 5 - len(board)
    wins = ties = losses = 0

    if need == 0:
        h = evaluate_seven(hero, board)
        d = evaluate_seven(dealer, board)
        if h < d:
            wins = 1
        elif h > d:
            losses = 1
        else:
            ties = 1
        return _normalize(wins, ties, losses)

    for fill in itertools.combinations(remaining, need):
        full_board = board + list(fill)
        h = evaluate_seven(hero, full_board)
        d = evaluate_seven(dealer, full_board)
        if h < d:
            wins += 1
        elif h > d:
            losses += 1
        else:
            ties += 1
    return _normalize(wins, ties, losses)


def _equity_unknown_dealer(
    hero: list[Card],
    board: list[Card],
    dead: set[Card],
) -> Equity:
    remaining = _remaining(dead)
    need = 5 - len(board)
    wins = ties = losses = 0

    for dealer_hole in itertools.combinations(remaining, 2):
        dealer = list(dealer_hole)
        after_dealer = [c for c in remaining if c not in dealer]
        if need == 0:
            h = evaluate_seven(hero, board)
            d = evaluate_seven(dealer, board)
            if h < d:
                wins += 1
            elif h > d:
                losses += 1
            else:
                ties += 1
            continue
        for fill in itertools.combinations(after_dealer, need):
            full_board = board + list(fill)
            h = evaluate_seven(hero, full_board)
            d = evaluate_seven(dealer, full_board)
            if h < d:
                wins += 1
            elif h > d:
                losses += 1
            else:
                ties += 1
    return _normalize(wins, ties, losses)

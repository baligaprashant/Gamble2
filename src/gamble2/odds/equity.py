"""Equity engine: preflop lookup + exact postflop enumeration."""

from __future__ import annotations

import random

from gamble2.odds.enumerate import Equity, equity_vs_dealer
from gamble2.odds.evaluate import evaluate_seven
from gamble2.odds.tables import preflop_matchup, preflop_vs_random
from gamble2.vision.cards import Card, full_deck

# Full flop×unknown-dealer enumeration is ~1M boards; sample for latency.
FLOP_UNKNOWN_SAMPLES = 40_000


def compute_equity(
    hero: list[Card],
    board: list[Card] | None = None,
    dealer: list[Card] | None = None,
    *,
    flop_samples: int = FLOP_UNKNOWN_SAMPLES,
) -> Equity:
    """Heads-up equity vs dealer.

    - Preflop + unknown dealer → precomputed 169-hand table (instant)
    - Preflop + known dealer → 169×169 table if present
    - Turn / river (and known dealer) → exact enumeration
    - Flop + unknown dealer → high-sample Monte Carlo (near-instant)
    """
    board = board or []
    if len(hero) != 2:
        raise ValueError("hero needs exactly 2 cards")
    if len(board) not in (0, 3, 4, 5):
        raise ValueError("board must have 0, 3, 4, or 5 cards")
    if dealer is not None and len(dealer) != 2:
        raise ValueError("dealer needs exactly 2 cards when provided")

    dead = set(hero) | set(board)
    if dealer:
        dead |= set(dealer)
    if len(dead) != len(hero) + len(board) + (2 if dealer else 0):
        raise ValueError("duplicate cards in hero/board/dealer")

    if len(board) == 0:
        if dealer is None:
            return preflop_vs_random(hero)
        matchup = preflop_matchup(hero, dealer)
        if matchup is not None:
            return matchup
        return equity_vs_dealer(hero, board, dealer)

    if len(board) == 3 and dealer is None:
        return _flop_unknown_mc(hero, board, flop_samples)

    return equity_vs_dealer(hero, board, dealer)


def _flop_unknown_mc(hero: list[Card], board: list[Card], samples: int) -> Equity:
    dead = set(hero) | set(board)
    remaining = [c for c in full_deck() if c not in dead]
    rng = random.Random(hash(tuple(sorted(c.code for c in hero + board))) & 0xFFFFFFFF)
    wins = ties = losses = 0
    for _ in range(samples):
        pick = rng.sample(remaining, 4)  # 2 dealer + turn + river
        dealer = pick[:2]
        full_board = board + pick[2:]
        h = evaluate_seven(hero, full_board)
        d = evaluate_seven(dealer, full_board)
        if h < d:
            wins += 1
        elif h > d:
            losses += 1
        else:
            ties += 1
    total = wins + ties + losses
    return Equity(wins / total, ties / total, losses / total, total)

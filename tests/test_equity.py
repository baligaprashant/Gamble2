import pytest

from gamble2.odds.enumerate import equity_vs_dealer
from gamble2.odds.equity import compute_equity
from gamble2.vision.cards import parse_cards


def test_river_known_dealer_aa_vs_kk():
    hero = parse_cards("Ah Ad")
    board = parse_cards("2c 7d 9h 3s 5c")
    dealer = parse_cards("Kh Kd")
    eq = equity_vs_dealer(hero, board, dealer)
    assert eq.win == 1.0
    assert eq.lose == 0.0
    assert eq.samples == 1


def test_river_tie():
    hero = parse_cards("Ah Kd")
    dealer = parse_cards("As Kc")
    board = parse_cards("2c 7d 9h 3s 5c")  # both high-card AK, Ace kicker same board
    eq = equity_vs_dealer(hero, board, dealer)
    # Both play A-K-9-7-5 → tie
    assert eq.tie == 1.0


def test_turn_exact_runs():
    hero = parse_cards("Ah Kh")
    board = parse_cards("Qh Jh 2c 7d")
    dealer = parse_cards("9c 9d")
    eq = equity_vs_dealer(hero, board, dealer)
    assert eq.samples == 44  # 52 - 2 - 4 - 2 = 44
    assert 0.0 <= eq.equity <= 1.0


def test_compute_equity_flop_unknown():
    hero = parse_cards("Ah Kh")
    board = parse_cards("As Td 2c")
    eq = compute_equity(hero, board, flop_samples=2_000)
    assert eq.samples == 2_000
    assert eq.win + eq.tie + eq.lose == pytest.approx(1.0)


def test_duplicate_cards_rejected():
    with pytest.raises(ValueError):
        compute_equity(parse_cards("Ah Kh"), parse_cards("Ah Td 2c"))

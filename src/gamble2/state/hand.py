"""Mutable hand state: hero, board, optional known dealer cards."""

from __future__ import annotations

from dataclasses import dataclass, field

from gamble2.odds.equity import compute_equity
from gamble2.odds.enumerate import Equity
from gamble2.vision.cards import Card


@dataclass
class HandState:
    hero: list[Card] = field(default_factory=list)
    board: list[Card] = field(default_factory=list)
    dealer: list[Card] = field(default_factory=list)

    def clear(self) -> None:
        self.hero.clear()
        self.board.clear()
        self.dealer.clear()

    def set_hero(self, cards: list[Card]) -> None:
        if len(cards) != 2:
            raise ValueError("hero needs 2 cards")
        self.hero = list(cards)

    def set_board(self, cards: list[Card]) -> None:
        if len(cards) not in (0, 3, 4, 5):
            raise ValueError("board must be 0, 3, 4, or 5 cards")
        self.board = list(cards)

    def set_dealer(self, cards: list[Card]) -> None:
        if cards and len(cards) != 2:
            raise ValueError("dealer needs 0 or 2 cards")
        self.dealer = list(cards)

    @property
    def ready(self) -> bool:
        return len(self.hero) == 2

    def street_name(self) -> str:
        n = len(self.board)
        return {0: "preflop", 3: "flop", 4: "turn", 5: "river"}.get(n, "invalid")

    def equity(self) -> Equity | None:
        if not self.ready:
            return None
        return compute_equity(
            self.hero,
            self.board,
            self.dealer if self.dealer else None,
        )

    def summary(self) -> str:
        hero = " ".join(c.code for c in self.hero) or "--"
        board = " ".join(c.code for c in self.board) or "--"
        dealer = " ".join(c.code for c in self.dealer) or "random"
        return f"Hero[{hero}] Board[{board}] Dealer[{dealer}] ({self.street_name()})"

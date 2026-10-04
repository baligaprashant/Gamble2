"""Work out which detected cards are the hole cards and which are the board
from where they sit on screen.

TV poker graphics (and most poker tables) show the two hole cards as one short
row and the community cards as a separate row of three to five.
"""

from __future__ import annotations

from collections import deque

from gamble2.vision.base import DetectedCard
from gamble2.vision.cards import Card


def _rows(dets: list[DetectedCard]) -> list[list[DetectedCard]]:
    if not dets:
        return []
    heights = sorted(d.bbox[3] for d in dets)
    tol = 0.6 * heights[len(heights) // 2]
    rows: list[list[DetectedCard]] = []
    for d in sorted(dets, key=lambda d: d.bbox[1] + d.bbox[3] / 2):
        yc = d.bbox[1] + d.bbox[3] / 2
        for row in rows:
            ry = sum(r.bbox[1] + r.bbox[3] / 2 for r in row) / len(row)
            if abs(yc - ry) <= tol:
                row.append(d)
                break
        else:
            rows.append([d])
    for row in rows:
        row.sort(key=lambda d: d.bbox[0])
    return rows


def split_rows(dets: list[DetectedCard]) -> tuple[list[DetectedCard] | None, list[DetectedCard] | None]:
    """Return (hole_row, board_row); either may be None when no such row is visible."""
    rows = _rows(dets)
    pairs = [r for r in rows if len(r) == 2]
    boards = [r for r in rows if 3 <= len(r) <= 5]
    hole = pairs[0] if len(pairs) == 1 else None
    board = max(boards, key=len) if boards else None
    return hole, board


class StableCards:
    """Cards seen on at least ``need`` of the last ``window`` frames."""

    def __init__(self, need: int = 4, window: int = 14) -> None:
        self.need = need
        self._frames: deque[list[tuple[Card, float]]] = deque(maxlen=window)

    def observe(self, items: list[tuple[Card, float]]) -> list[Card]:
        """``items`` are (card, x position) pairs for this frame; returns the stable
        cards ordered left to right."""
        self._frames.append(list(items))
        votes: dict[str, int] = {}
        last: dict[str, tuple[Card, float, int]] = {}
        for i, frame in enumerate(self._frames):
            for card, x in frame:
                votes[card.code] = votes.get(card.code, 0) + 1
                last[card.code] = (card, x, i)
        keep = [last[c] for c, n in votes.items() if n >= self.need]
        keep.sort(key=lambda t: -t[2])  # most recently seen first
        keep = keep[:5]
        keep.sort(key=lambda t: t[1])
        return [c for c, _x, _i in keep]

    def clear(self) -> None:
        self._frames.clear()

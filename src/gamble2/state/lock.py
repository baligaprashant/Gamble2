"""Turn noisy per-frame detections into a stable pair of hole cards.

Per-frame recognition is never perfect (motion blur, a finger over a corner,
one card momentarily lost), so instead of demanding the same pair N frames in
a row we *vote*: every frame adds its detected cards to a sliding window and
the two cards seen most often win.
"""

from __future__ import annotations

from collections import Counter, deque

from gamble2.vision.cards import Card


class StableHero:
    def __init__(self, need_frames: int = 4, window: int = 14) -> None:
        self.need_frames = need_frames
        self.window = window
        self._frames: deque[list[Card]] = deque(maxlen=window)
        self.locked: list[Card] | None = None

    def observe(self, cards: list[Card] | None) -> list[Card] | None:
        """Feed one frame's detections (may be empty). Returns the locked pair."""
        uniq: dict[str, Card] = {c.code: c for c in (cards or [])}
        self._frames.append(list(uniq.values()))

        votes: Counter[str] = Counter()
        by_code: dict[str, Card] = {}
        for frame in self._frames:
            for c in frame:
                votes[c.code] += 1
                by_code[c.code] = c

        ranked = [code for code, n in votes.most_common() if n >= self.need_frames]
        if len(ranked) >= 2:
            pair = [by_code[ranked[0]], by_code[ranked[1]]]
            if self.locked is None or {c.code for c in pair} != {c.code for c in self.locked}:
                # only switch away from an existing lock when the newcomers
                # clearly out-vote the cards we were holding
                if self.locked is None or self._beats_lock(votes, pair):
                    self.locked = pair
        return self.locked

    def _beats_lock(self, votes: Counter[str], pair: list[Card]) -> bool:
        assert self.locked is not None
        old = min(votes.get(c.code, 0) for c in self.locked)
        new = min(votes[c.code] for c in pair)
        return new > old

    def progress(self) -> int:
        """How many frames the best pair has been seen (for UI feedback)."""
        votes: Counter[str] = Counter()
        for frame in self._frames:
            for c in frame:
                votes[c.code] += 1
        top = votes.most_common(2)
        return min(n for _, n in top) if len(top) == 2 else (top[0][1] if top else 0)

    def clear(self) -> None:
        self._frames.clear()
        self.locked = None

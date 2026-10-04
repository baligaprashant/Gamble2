"""Card detector: find card-shaped regions, read each one's index corner."""

from __future__ import annotations

import cv2
import numpy as np

from gamble2.vision.base import CardDetector, CardRole, DetectedCard
from gamble2.vision.cards import Card
from gamble2.vision.glyph_bank import GlyphBank, default_bank, save_user_glyph
from gamble2.vision.index_finder import find_index_reads, index_pairs
from gamble2.vision.locate import Candidate, iou


class TemplateDetector(CardDetector):
    def __init__(self, bank: GlyphBank | None = None, match_threshold: float = 0.70) -> None:
        self.bank = bank or default_bank()
        self.match_threshold = match_threshold
        #: filled on every detect() call - used by the debug view
        self.last_candidates: list[Candidate] = []
        #: every reading this frame, even below threshold: (bbox, card, confidence)
        self.last_reads: list[tuple[tuple[int, int, int, int], Card, float]] = []

    @property
    def ready(self) -> bool:
        return self.bank.count() > 0

    def reload_bank(self) -> None:
        from gamble2.vision.glyph_bank import load_bank

        self.bank = load_bank()

    def detect(self, frame: np.ndarray) -> list[DetectedCard]:
        """Read every index corner in view (works with fanned / overlapped cards)."""
        reads = find_index_reads(frame, self.bank, 0.0)
        self.last_reads = [(r.bbox, r.card, r.confidence) for r in reads if r.confidence >= 0.45]
        self.last_candidates = []
        hits = [
            DetectedCard(card=r.card, confidence=r.confidence, bbox=r.bbox, role=CardRole.HOLE)
            for r in reads
            if r.confidence >= self.match_threshold
        ]
        return _suppress(hits)

    def teach(self, frame: np.ndarray, card: Card) -> bool:
        """Learn the rank and suit glyphs of ``card`` from the card visible in ``frame``."""
        pair = self.best_pair_for(frame, card)
        if pair is None:
            return False
        g = pair.glyphs
        for kind, label, img in (("rank", card.rank, g.rank), ("suit", card.suit, g.suit)):
            save_user_glyph(kind, label, img)
            for _ in range(USER_WEIGHT):
                self.bank.add(kind, label, img)
        return True

    def best_pair_for(self, frame: np.ndarray, card: Card, min_score: float = 0.25, require_top: bool = False):
        """The index corner in ``frame`` that most plausibly is ``card`` (right
        colour, best match to what we know about its rank and suit)."""
        best = None
        for pair in index_pairs(frame):
            if pair.glyphs.red != (card.suit in "hd"):
                continue
            rs = self.bank.match("rank", pair.glyphs.rank, card.rank)[1]
            ss = self.bank.match("suit", pair.glyphs.suit, card.suit)[1]
            score = min(rs, ss)
            if require_top:
                # only learn from corners the reader already agrees are this card -
                # never from junk that merely resembles it
                from gamble2.vision.classify import match_glyphs

                hit = match_glyphs(pair.glyphs, self.bank)
                if hit is None or hit[0].code != card.code or hit[1] < 0.75 or pair.bbox[3] < 28:
                    continue
            # prefer larger (closer, sharper) corners when scores are close
            score += 0.0005 * pair.bbox[3]
            if best is None or score > best[0]:
                best = (score, pair)
        if best is None or best[0] < min_score:
            return None
        return best[1]


USER_WEIGHT = 2


def _suppress(cards: list[DetectedCard]) -> list[DetectedCard]:
    """Best-confidence first; drop overlapping boxes and repeated identities."""
    kept: list[DetectedCard] = []
    for c in sorted(cards, key=lambda d: d.confidence, reverse=True):
        if any(iou(c.bbox, k.bbox) > 0.25 for k in kept):
            continue
        if any(c.card.code == k.card.code for k in kept):
            continue
        kept.append(c)
    return kept


def annotate_candidates(frame: np.ndarray, cands: list[Candidate]) -> None:
    for c in cands:
        cv2.polylines(frame, [c.quad.astype(np.int32)], True, (0, 180, 255), 1, cv2.LINE_AA)

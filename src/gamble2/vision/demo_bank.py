"""Glyph bank for the synthetic ``--demo`` cards (drawn with OpenCV's built-in font).

The demo cards look nothing like a real deck, so they get their own bank, built
on the fly by reading our own rendered cards with the same index finder.
"""

from __future__ import annotations

from functools import lru_cache

from gamble2.vision.cards import RANKS, SUITS
from gamble2.vision.glyph_bank import GlyphBank


@lru_cache(maxsize=1)
def demo_bank() -> GlyphBank:
    from gamble2.capture.demo import DemoCapture
    from gamble2.vision.index_finder import index_pairs

    bank = GlyphBank()
    for r in RANKS:
        for s in SUITS:
            code = r + s
            other = "2c" if code != "2c" else "3d"
            for p in index_pairs(DemoCapture(f"{code} {other}").read()):
                x, y, _w, h = p.bbox
                if 190 <= x <= 215 and 200 <= y <= 212 and 40 <= h <= 60 and p.upright and p.glyphs.red == (s in "hd"):
                    bank.add("rank", r, p.glyphs.rank)
                    bank.add("suit", s, p.glyphs.suit)
                    break
    return bank

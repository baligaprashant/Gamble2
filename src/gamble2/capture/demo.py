"""Laptop webcam capture via OpenCV."""

from __future__ import annotations

import cv2
import numpy as np

from gamble2.vision.cards import parse_cards
from gamble2.vision.render import place_cards


class DemoCapture:
    """Synthetic table: two hole cards on a dark felt (no camera)."""

    def __init__(self, cards: str = "Ah Kh") -> None:
        self.hero = parse_cards(cards)
        self._frame = self._build()

    @property
    def name(self) -> str:
        return "demo"

    def read(self) -> np.ndarray | None:
        return self._frame.copy()

    def release(self) -> None:
        return None

    def _build(self) -> np.ndarray:
        frame = np.full((720, 980, 3), (28, 72, 36), dtype=np.uint8)
        cv2.putText(
            frame,
            "Demo  ·  your 2 hole cards",
            (70, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
        place_cards(frame, self.hero, origin=(180, 180), card_size=(220, 308), gap=48)
        return frame

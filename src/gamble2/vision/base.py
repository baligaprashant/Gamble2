"""Card detector interface (OpenCV now, ML later)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np

from gamble2.vision.cards import Card


class CardRole(str, Enum):
    HOLE = "hole"
    BOARD = "board"
    DEALER = "dealer"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class DetectedCard:
    card: Card
    confidence: float
    bbox: tuple[int, int, int, int]  # x, y, w, h
    role: CardRole = CardRole.UNKNOWN


class CardDetector(ABC):
    """Pluggable detector: template matching today, ML tomorrow."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[DetectedCard]:
        """Return detected cards in the frame."""

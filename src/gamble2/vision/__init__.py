"""Vision package."""

from gamble2.vision.base import CardDetector, CardRole, DetectedCard
from gamble2.vision.cards import Card, parse_cards
from gamble2.vision.ml_stub import MLDetector
from gamble2.vision.template import TemplateDetector

__all__ = [
    "Card",
    "CardDetector",
    "CardRole",
    "DetectedCard",
    "MLDetector",
    "TemplateDetector",
    "parse_cards",
]

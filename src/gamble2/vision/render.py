"""Simple poker-card faces for the HUD and vision tests."""

from __future__ import annotations

import cv2
import numpy as np

from gamble2.vision.cards import Card
from gamble2.vision.glyphs import draw_rank, draw_suit, suit_bgr


def render_card(card: Card, width: int = 200, height: int = 280) -> np.ndarray:
    img = np.full((height, width, 3), 248, dtype=np.uint8)
    radius = max(8, width // 14)
    _rounded_fill(img, radius, (248, 248, 248))
    cv2.rectangle(img, (2, 2), (width - 3, height - 3), (40, 40, 40), 2)
    color = suit_bgr(card.suit)
    scale = (0.85 if card.rank == "T" else 0.9) * width / 200
    draw_rank(img, int(width * 0.07), int(height * 0.15), card.rank, color, scale, max(2, width // 90))
    pip = max(6, width // 15)
    draw_suit(
        img,
        int(width * 0.14) if card.rank != "T" else int(width * 0.19),
        int(height * 0.205),
        pip,
        card.suit,
        color,
    )
    draw_suit(
        img,
        width // 2,
        height // 2 + height // 18,
        max(18, width // 5),
        card.suit,
        color,
    )
    return img


def place_cards(
    canvas: np.ndarray,
    cards: list[Card],
    origin: tuple[int, int] = (80, 120),
    card_size: tuple[int, int] = (200, 280),
    gap: int = 36,
) -> np.ndarray:
    x, y = origin
    cw, ch = card_size
    for card in cards:
        face = render_card(card, cw, ch)
        canvas[y : y + ch, x : x + cw] = face
        x += cw + gap
    return canvas


def _rounded_fill(img: np.ndarray, radius: int, color: tuple[int, int, int]) -> None:
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (radius, 0), (w - radius, h), 255, -1)
    cv2.rectangle(mask, (0, radius), (w, h - radius), 255, -1)
    cv2.circle(mask, (radius, radius), radius, 255, -1)
    cv2.circle(mask, (w - radius - 1, radius), radius, 255, -1)
    cv2.circle(mask, (radius, h - radius - 1), radius, 255, -1)
    cv2.circle(mask, (w - radius - 1, h - radius - 1), radius, 255, -1)
    img[mask == 0] = (0, 0, 0)
    img[mask == 255] = color

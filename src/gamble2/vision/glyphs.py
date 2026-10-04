"""Rank letters and suit pips drawn with OpenCV (used for cards + matching)."""

from __future__ import annotations

import cv2
import numpy as np

from gamble2.vision.cards import RANKS, SUITS

_RED_BGR = (45, 45, 210)
_BLACK_BGR = (28, 28, 28)


def suit_bgr(suit: str) -> tuple[int, int, int]:
    return _RED_BGR if suit in "hd" else _BLACK_BGR


def rank_label(rank: str) -> str:
    return "10" if rank == "T" else rank


def draw_suit(
    img: np.ndarray,
    cx: int,
    cy: int,
    size: int,
    suit: str,
    color: tuple[int, int, int] | int,
) -> None:
    s = max(4, size)
    if suit == "d":
        pts = np.array(
            [[cx, cy - s], [cx + int(s * 0.72), cy], [cx, cy + s], [cx - int(s * 0.72), cy]],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(img, pts, color, lineType=cv2.LINE_AA)
    elif suit == "h":
        r = max(3, int(s * 0.42))
        cv2.circle(img, (cx - r, cy - int(s * 0.15)), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (cx + r, cy - int(s * 0.15)), r, color, -1, cv2.LINE_AA)
        pts = np.array(
            [
                [cx - int(s * 0.85), cy - int(s * 0.05)],
                [cx + int(s * 0.85), cy - int(s * 0.05)],
                [cx, cy + s],
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(img, pts, color, lineType=cv2.LINE_AA)
    elif suit == "s":
        r = max(3, int(s * 0.38))
        cv2.circle(img, (cx - r, cy + int(s * 0.08)), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (cx + r, cy + int(s * 0.08)), r, color, -1, cv2.LINE_AA)
        pts = np.array(
            [
                [cx, cy - s],
                [cx + int(s * 0.85), cy + int(s * 0.15)],
                [cx - int(s * 0.85), cy + int(s * 0.15)],
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(img, pts, color, lineType=cv2.LINE_AA)
        cv2.rectangle(
            img,
            (cx - max(2, s // 8), cy + int(s * 0.1)),
            (cx + max(2, s // 8), cy + s),
            color,
            -1,
        )
    else:  # clubs
        r = max(3, int(s * 0.36))
        cv2.circle(img, (cx, cy - r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (cx - r, cy + int(r * 0.35)), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (cx + r, cy + int(r * 0.35)), r, color, -1, cv2.LINE_AA)
        cv2.rectangle(
            img,
            (cx - max(2, s // 8), cy + int(s * 0.05)),
            (cx + max(2, s // 8), cy + s),
            color,
            -1,
        )


def draw_rank(
    img: np.ndarray,
    x: int,
    y: int,
    rank: str,
    color: tuple[int, int, int] | int,
    scale: float,
    thickness: int,
) -> None:
    cv2.putText(
        img,
        rank_label(rank),
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def rank_templates() -> dict[str, np.ndarray]:
    """White-background, black-ink glyphs for template matching."""
    out: dict[str, np.ndarray] = {}
    for rank in RANKS:
        text = rank_label(rank)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.15 if text == "10" else 1.7
        thickness = 3
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        pad = 8
        img = np.full((th + pad * 2, tw + pad * 2), 255, dtype=np.uint8)
        cv2.putText(
            img,
            text,
            (pad, pad + th),
            font,
            scale,
            0,
            thickness,
            cv2.LINE_AA,
        )
        out[rank] = img
    return out


def suit_templates() -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for suit in SUITS:
        size = 36
        img = np.full((size * 2 + 8, size * 2 + 8), 255, dtype=np.uint8)
        draw_suit(img, img.shape[1] // 2, img.shape[0] // 2, size, suit, 0)
        out[suit] = img
    return out

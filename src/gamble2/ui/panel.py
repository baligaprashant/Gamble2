"""Right-hand odds panel for the live overlay."""

from __future__ import annotations

import cv2
import numpy as np

from gamble2.odds.enumerate import Equity
from gamble2.state.hand import HandState
from gamble2.vision.render import render_card


PANEL_W = 420


def draw_odds_panel(
    height: int,
    state: HandState,
    equity: Equity | None,
    status: str,
    seeing: int,
    manual_hint: str | None = None,
) -> np.ndarray:
    h = max(height, 560)
    panel = np.full((h, PANEL_W, 3), (32, 28, 26), dtype=np.uint8)
    cv2.line(panel, (0, 0), (0, h), (70, 70, 70), 2)

    y = 42
    _text(panel, "YOUR HAND", 28, y, 0.7, (200, 200, 200), 1)
    y = 58
    _text(panel, "Heads-up vs dealer", 28, y + 28, 0.55, (150, 150, 150), 1)

    y = 110
    if len(state.hero) == 2:
        left = render_card(state.hero[0], 150, 210)
        right = render_card(state.hero[1], 150, 210)
        panel[y : y + 210, 40 : 190] = left
        panel[y : y + 210, 220 : 370] = right
    else:
        _card_placeholder(panel, 40, y, "Card 1")
        _card_placeholder(panel, 220, y, "Card 2")

    y = 350
    _text(panel, "WIN PROBABILITY", 28, y, 0.55, (160, 160, 160), 1)

    if equity is not None:
        d = equity.as_dict()
        _text(
            panel,
            f"{d['equity']:.1f}%",
            28,
            y + 70,
            2.1,
            (50, 220, 255),
            4,
        )
        y = 460
        _bar(panel, 28, y, PANEL_W - 56, "Win", float(d["win"]), (70, 190, 80))
        _bar(panel, 28, y + 38, PANEL_W - 56, "Tie", float(d["tie"]), (50, 180, 220))
        _bar(panel, 28, y + 76, PANEL_W - 56, "Lose", float(d["lose"]), (70, 70, 210))
        who = (
            "vs dealer " + " ".join(c.code for c in state.dealer)
            if state.dealer
            else "vs random dealer hand"
        )
        board = " ".join(c.code for c in state.board)
        _text(
            panel,
            f"{who}  ·  {state.street_name()}" + (f"  [{board}]" if board else ""),
            28,
            h - 78,
            0.45,
            (140, 140, 140),
            1,
        )
    else:
        _text(panel, "—", 28, y + 70, 2.1, (90, 90, 90), 3)
        _text(
            panel,
            "Hold 2 hole cards to the camera",
            28,
            y + 120,
            0.55,
            (180, 180, 180),
            1,
        )

    seeing_color = (80, 200, 80) if seeing >= 2 else (120, 120, 120)
    _text(panel, f"Camera seeing {seeing} card(s)", 28, h - 48, 0.5, seeing_color, 1)
    hint = manual_hint or status
    _text(panel, hint[:48], 28, h - 22, 0.42, (170, 170, 170), 1)
    return panel


def compose_view(frame: np.ndarray, panel: np.ndarray) -> np.ndarray:
    fh, fw = frame.shape[:2]
    ph, _pw = panel.shape[:2]
    scale = ph / max(fh, 1)
    resized = cv2.resize(frame, (max(1, int(fw * scale)), ph))
    return np.hstack([resized, panel])


def _text(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    cv2.putText(
        img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA
    )


def _card_placeholder(panel: np.ndarray, x: int, y: int, label: str) -> None:
    cv2.rectangle(panel, (x, y), (x + 150, y + 210), (70, 70, 70), 2)
    _text(panel, label, x + 28, y + 110, 0.55, (110, 110, 110), 1)


def _bar(
    img: np.ndarray,
    x: int,
    y: int,
    width: int,
    label: str,
    pct: float,
    color: tuple[int, int, int],
) -> None:
    _text(img, f"{label}  {pct:.1f}%", x, y - 4, 0.42, (190, 190, 190), 1)
    cv2.rectangle(img, (x, y), (x + width, y + 10), (50, 50, 50), -1)
    fill = int(width * max(0.0, min(pct, 100.0)) / 100.0)
    if fill:
        cv2.rectangle(img, (x, y), (x + fill, y + 10), color, -1)

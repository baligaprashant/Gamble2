"""Live OpenCV overlay: camera on the left, win probability on the right."""

from __future__ import annotations

import time
from typing import Callable

import cv2
import numpy as np

from gamble2.odds.enumerate import Equity
from gamble2.state.hand import HandState
from gamble2.state.lock import StableHero
from gamble2.ui.panel import compose_view, draw_odds_panel
from gamble2.vision.base import DetectedCard
from gamble2.vision.cards import Card, parse_cards


MIN_CONFIDENCE = 0.75
_RANK_WORDS = {"A": "Ace", "K": "King", "Q": "Queen", "J": "Jack", "T": "10", "9": "9", "8": "8",
               "7": "7", "6": "6", "5": "5", "4": "4", "3": "3", "2": "2"}
_SUIT_WORDS = {"c": "clubs", "d": "diamonds", "h": "hearts", "s": "spades"}


class OverlayApp:
    def __init__(
        self,
        read_frame: Callable[[], np.ndarray | None],
        detect: Callable[[np.ndarray], list[DetectedCard]],
        source_name: str = "live",
        window: str = "Gamble2",
        lock_frames: int = 4,
        detector: object | None = None,
        window_pos: tuple[int, int] | None = None,
        debug: bool = False,
    ) -> None:
        self.window_pos = window_pos
        self.detector = detector  # optional: enables the debug view / teach mode
        self.read_frame = read_frame
        self.detect = detect
        self.source_name = source_name
        self.window = window
        self.state = HandState()
        self._hero_lock = StableHero(need_frames=lock_frames)
        self._last_equity: Equity | None = None
        self._last_key: tuple | None = None
        self._status = "Hold your 2 hole cards to the camera"
        self._manual_buf = ""
        self._manual_mode: str | None = None
        self._prefer_manual = False
        self._last_seen = 0
        self._debug = debug
        self._calib: list[Card] | None = None
        self._calib_i = 0
        self._calib_hits = 0
        self._calib_best = None
        self._last_frame: np.ndarray | None = None
        self._empty_frames = 0

    def run(self) -> None:
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        if self.window_pos is not None:
            cv2.moveWindow(self.window, *self.window_pos)
        while True:
            frame = self.read_frame()
            self._last_frame = None if frame is None else frame.copy()
            display = self.tick(frame)
            cv2.imshow(self.window, display)
            cv2.resizeWindow(self.window, display.shape[1], display.shape[0])

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            self._handle_key(key)

        cv2.destroyWindow(self.window)

    def tick(self, frame: np.ndarray | None) -> np.ndarray:
        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                frame,
                "No camera frame",
                (40, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )
            detections: list[DetectedCard] = []
        elif self._calib is not None:
            detections = []
            self._calibrate_step(frame)
        else:
            detections = self.detect(frame)
            if self._debug and self.detector is not None:
                from gamble2.vision.template import annotate_candidates

                annotate_candidates(frame, getattr(self.detector, "last_candidates", []))
                k = _scale(frame)
                reads = getattr(self.detector, "last_reads", [])
                for (x, y, w, h), card, conf in reads:
                    cv2.rectangle(frame, (x - 3, y - 3), (x + w + 3, y + h + 3), (0, 140, 255), max(1, int(k)))
                    cv2.putText(frame, f"{card.code} {conf:.2f}", (x + w + 6, y + int(18 * k)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6 * k, (0, 140, 255), max(1, int(2 * k)), cv2.LINE_AA)
                self._draw_ribbon(frame, reads, detections)
            self._draw_detections(frame, detections)
            self._maybe_apply_detections(detections)

        self._last_seen = len(detections)
        self._refresh_equity()
        self._draw_camera_hint(frame)
        hint = (
            f"Manual[{self._manual_mode}]: {self._manual_buf}_"
            if self._manual_mode
            else None
        )
        panel = draw_odds_panel(
            frame.shape[0],
            self.state,
            self._last_equity,
            self._status,
            seeing=self._last_seen,
            manual_hint=hint,
        )
        view = compose_view(frame, panel)
        vh = view.shape[0]
        if vh > _MAX_VIEW_H:  # e.g. a Retina screen grab: keep the window on the display
            f = _MAX_VIEW_H / float(vh)
            view = cv2.resize(view, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        return view

    def _state_key(self) -> tuple:
        return (
            tuple(c.code for c in self.state.hero),
            tuple(c.code for c in self.state.board),
            tuple(c.code for c in self.state.dealer),
        )

    def _refresh_equity(self) -> None:
        if not self.state.ready:
            self._last_equity = None
            return
        key = self._state_key()
        if key == self._last_key and self._last_equity is not None:
            return
        t0 = time.perf_counter()
        try:
            self._last_equity = self.state.equity()
        except FileNotFoundError as exc:
            self._status = str(exc)
            self._last_equity = None
            return
        ms = (time.perf_counter() - t0) * 1000
        self._last_key = key
        if self._last_equity:
            d = self._last_equity.as_dict()
            self._status = (
                f"{self.state.summary()}  {d['equity']:.1f}% eq  ({ms:.0f}ms)"
            )

    def _draw_detections(self, frame: np.ndarray, dets: list[DetectedCard]) -> None:
        for d in dets:
            x, y, w, h = d.bbox
            k = _scale(frame)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 80), max(2, int(2 * k)))
            cv2.putText(
                frame,
                f"{d.card.code} {d.confidence:.2f}",
                (x, max(int(20 * k), y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6 * k,
                (0, 200, 80),
                max(2, int(2 * k)),
                cv2.LINE_AA,
            )

    def _draw_ribbon(self, frame: np.ndarray, reads, dets: list[DetectedCard]) -> None:
        """Top strip that says what the reader is seeing right now."""
        h, w = frame.shape[:2]
        k = _scale(frame)
        top = sorted(reads, key=lambda r: -r[2])[:4]
        if top:
            guesses = "  ".join(f"{c.code} {conf:.2f}" for _b, c, conf in top)
            line = f"corners {len(reads)} | {guesses}"
        else:
            line = "no card corners found - enlarge the video / pause on a sharp frame"
        bar_h = int(34 * k)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, f"{w}x{h}  {line}", (int(10 * k), int(23 * k)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55 * k, (80, 220, 255), max(1, int(k)), cv2.LINE_AA)

    def _draw_camera_hint(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        k = _scale(frame)
        bar = int(42 * k)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - bar), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.putText(
            frame,
            "K calibrate  H hole  B board  D dealer  V debug  S save  C clear  Q quit",
            (int(16 * k), h - int(14 * k)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55 * k,
            (230, 230, 230),
            max(1, int(k)),
            cv2.LINE_AA,
        )

    def _maybe_apply_detections(self, dets: list[DetectedCard]) -> None:
        if self._prefer_manual or self._manual_mode:
            return
        cards = [d.card for d in dets if d.confidence >= MIN_CONFIDENCE]
        locked = self._hero_lock.observe(cards)
        if locked:
            if {c.code for c in locked} != {c.code for c in self.state.hero}:
                try:
                    self.state.set_hero(locked)
                    self._last_key = None
                except ValueError:
                    pass
        elif not self.state.hero:
            seen = self._hero_lock.progress()
            self._status = (
                f"Reading cards... ({len(cards)} in view)"
                if cards
                else "Hold your 2 hole cards up to the camera (index corners visible)"
            )
            _ = seen

    def _handle_key(self, key: int) -> None:
        if self._manual_mode:
            if key in (8, 127):
                self._manual_buf = self._manual_buf[:-1]
            elif key in (13, 10):
                self._commit_manual()
            elif key == 27:
                self._manual_mode = None
                self._manual_buf = ""
            elif 32 <= key < 127:
                self._manual_buf += chr(key)
            return

        if self._calib is not None:
            if key == ord(" "):
                self._calib_next("skipped")
            elif key == 27:
                self._calib_stop("Calibration stopped")
            return

        if key == ord("k"):
            self._calib_start()
        elif key == ord("h"):
            self._manual_mode = "hole"
            self._manual_buf = ""
            self._status = "Type hole cards (e.g. Ah Kh) then Enter"
        elif key == ord("b"):
            self._manual_mode = "board"
            self._manual_buf = ""
            self._status = "Enter board (e.g. As Td 2c) then Enter"
        elif key == ord("d"):
            self._manual_mode = "dealer"
            self._manual_buf = ""
            self._status = "Enter dealer cards or blank for random"
        elif key == ord("v"):
            self._debug = not self._debug
            self._status = f"Debug view {'on' if self._debug else 'off'}"
        elif key == ord("s"):
            self._status = self._save_frame()
        elif key == ord("t"):
            self._manual_mode = "teach"
            self._manual_buf = ""
            self._status = "Teach: show ONE card, type its code (e.g. Kd), Enter"
        elif key == ord("c"):
            self.state.clear()
            self._hero_lock.clear()
            self._last_equity = None
            self._last_key = None
            self._prefer_manual = False
            self._status = "Cleared hand — show 2 cards to the camera"

    def _commit_manual(self) -> None:
        mode = self._manual_mode
        raw = self._manual_buf.strip()
        self._manual_mode = None
        self._manual_buf = ""
        try:
            if mode == "hole":
                self.state.set_hero(parse_cards(raw))
                self._prefer_manual = True
                self._hero_lock.clear()
            elif mode == "board":
                self.state.set_board(parse_cards(raw) if raw else [])
            elif mode == "dealer":
                self.state.set_dealer(parse_cards(raw) if raw else [])
            elif mode == "teach":
                self._teach(parse_cards(raw))
                return
            self._status = f"Updated {mode}: {self.state.summary()}"
            self._last_key = None
        except ValueError as exc:
            self._status = f"Invalid input: {exc}"


    # ---- calibration wizard ------------------------------------------
    def _calib_start(self) -> None:
        if self.detector is None:
            self._status = "Calibration needs the live detector"
            return
        order = "AKQJT98765432"
        suits = "chds"
        self._calib = [Card(r, suits[i % 4]) for i, r in enumerate(order)]
        self._calib_i = 0
        self._calib_hits = 0
        self._calib_best = None
        self._status = "Calibration started"

    def _calib_stop(self, msg: str) -> None:
        self._calib = None
        self._status = msg
        self._last_key = None

    def _calib_next(self, why: str) -> None:
        assert self._calib is not None
        self._calib_i += 1
        self._calib_hits = 0
        self._calib_best = None
        if self._calib_i >= len(self._calib):
            self._calib_stop("Calibration done - your deck's glyphs are saved")
        else:
            self._status = f"{why}: next card"

    def _calibrate_step(self, frame: np.ndarray) -> None:
        if self._calib is None or self.detector is None:
            return
        target = self._calib[self._calib_i]
        pair = self.detector.best_pair_for(frame, target, min_score=0.55, require_top=True)  # type: ignore[attr-defined]
        if pair is not None:
            x, y, w, h = pair.bbox
            cv2.rectangle(frame, (x - 4, y - 4), (x + w + 4, y + h + 4), (0, 200, 80), 2)
            self._calib_hits += 1
            self._calib_best = pair
            if self._calib_hits >= 8:
                from gamble2.vision.glyph_bank import save_user_glyph

                g = pair.glyphs
                for kind, label, img in (("rank", target.rank, g.rank), ("suit", target.suit, g.suit)):
                    save_user_glyph(kind, label, img)
                    for _ in range(2):
                        self.detector.bank.add(kind, label, img)  # type: ignore[attr-defined]
                self._calib_next(f"Saved {target.code}")
                return  # the wizard may have just finished (self._calib is None)
        else:
            self._calib_hits = max(0, self._calib_hits - 1)
        msg = (
            f"CALIBRATE {self._calib_i + 1}/{len(self._calib)}: show ONLY the "
            f"{_RANK_WORDS[target.rank]} of {_SUIT_WORDS[target.suit]}  "
            f"(hold still, corner visible)  SPACE=skip ESC=stop"
        )
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 44), (30, 30, 30), -1)
        cv2.putText(frame, msg, (14, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (80, 220, 255), 2, cv2.LINE_AA)
        bar = int(frame.shape[1] * min(self._calib_hits, 8) / 8)
        cv2.rectangle(frame, (0, 44), (bar, 50), (0, 200, 80), -1)

    def _save_frame(self) -> str:
        """Save the raw camera frame (for debugging recognition) next to the project."""
        from pathlib import Path

        if self._last_frame is None:
            return "No frame to save"
        folder = Path(__file__).resolve().parents[3] / "captures"
        folder.mkdir(exist_ok=True)
        path = folder / f"frame_{int(time.time())}.png"
        cv2.imwrite(str(path), self._last_frame)
        return f"Saved {path.name} to captures/"

    def _teach(self, cards: list[Card]) -> None:
        """Save the rank/suit glyphs of the card in view as templates for ``cards[0]``."""
        if len(cards) != 1:
            raise ValueError("teach needs exactly one card, e.g. Kd")
        if self.detector is None or self._last_frame is None:
            self._status = "Teach mode needs the live detector"
            return
        saved = self.detector.teach(self._last_frame, cards[0])  # type: ignore[attr-defined]
        self._status = (
            f"Taught {cards[0].code} from the card in view"
            if saved
            else "No card found - hold it flat, index corner visible, then try again"
        )


_MAX_VIEW_H = 820


def _scale(frame: np.ndarray) -> float:
    """Drawing scale so labels stay readable on big (e.g. Retina screen) frames."""
    return max(1.0, frame.shape[0] / 540.0)


def _best_hole_pair(dets: list[DetectedCard]) -> list[DetectedCard] | None:
    if len(dets) < 2:
        return None
    ranked = sorted(dets, key=lambda d: d.confidence, reverse=True)
    picked: list[DetectedCard] = []
    seen: set[str] = set()
    for d in ranked:
        if d.card.code in seen:
            continue
        picked.append(d)
        seen.add(d.card.code)
        if len(picked) == 2:
            break
    if len(picked) < 2:
        return None
    if min(d.confidence for d in picked) < 0.52:
        return None
    return sorted(picked, key=lambda d: d.bbox[0])

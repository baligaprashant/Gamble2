"""Screen / window region capture via mss."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import mss
except ImportError as exc:  # pragma: no cover
    raise ImportError("mss is required for screen capture") from exc


@dataclass
class Region:
    left: int
    top: int
    width: int
    height: int

    @classmethod
    def parse(cls, text: str) -> Region:
        parts = [int(p.strip()) for p in text.split(",")]
        if len(parts) != 4:
            raise ValueError("Region must be left,top,width,height")
        return cls(*parts)

    def as_mss(self) -> dict[str, int]:
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }


class ScreenCapture:
    def __init__(self, region: Region | None = None, monitor: int = 1) -> None:
        self._sct = mss.mss()
        self._region = region
        self._monitor = monitor

    @property
    def region(self) -> Region | None:
        return self._region

    @property
    def name(self) -> str:
        if self._region:
            r = self._region
            return f"screen:{r.left},{r.top},{r.width},{r.height}"
        return f"screen:monitor{self._monitor}"

    def read(self) -> np.ndarray | None:
        if self._region:
            shot = self._sct.grab(self._region.as_mss())
        else:
            shot = self._sct.grab(self._sct.monitors[self._monitor])
        # mss returns BGRA
        frame = np.array(shot)
        return frame[:, :, :3].copy()  # BGR

    def release(self) -> None:
        self._sct.close()


def region_from_pixels(box: tuple[int, int, int, int], monitor: dict, shot_width: int) -> Region:
    """Convert a rectangle drawn on a screenshot (physical pixels) into a capture
    region in screen coordinates.  Retina screens grab 2x the pixels of their
    coordinate space, so divide by that scale."""
    x, y, w, h = box
    scale = shot_width / float(monitor["width"])
    return Region(
        left=int(monitor["left"] + x / scale),
        top=int(monitor["top"] + y / scale),
        width=max(1, int(w / scale)),
        height=max(1, int(h / scale)),
    )


def list_displays() -> list[dict]:
    """The real displays (mss numbers them from 1)."""
    with mss.mss() as sct:
        mons = list(sct.monitors[1:]) or [sct.monitors[0]]
    return mons


def select_region(monitor: int = 1) -> Region | None:
    """Show a still screenshot of a display; the user clicks and drags a box
    around the video, then presses ENTER.  With several displays, TAB switches
    between them.  ENTER without a box takes the whole display; ESC cancels
    (returns None)."""
    import cv2

    with mss.mss() as sct:
        mons = list(sct.monitors[1:]) or [sct.monitors[0]]
        shots = [np.array(sct.grab(m))[:, :, :3].copy() for m in mons]
    idx = min(max(monitor - 1, 0), len(mons) - 1)
    title = "Gamble2 - choose capture area"
    state = {"start": None, "end": None, "drag": False}

    def fit_for(i: int) -> float:
        h, w = shots[i].shape[:2]
        return min(1.0, 1400.0 / w, 800.0 / h)

    def view_for(i: int):
        f = fit_for(i)
        shot = shots[i]
        return cv2.resize(shot, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else shot.copy()

    views = [view_for(i) for i in range(len(mons))]

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state.update(start=(x, y), end=(x, y), drag=True)
        elif event == cv2.EVENT_MOUSEMOVE and state["drag"]:
            state["end"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and state["drag"]:
            state.update(end=(x, y), drag=False)

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, on_mouse)
    result = None
    while True:
        canvas = views[idx].copy()
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 40), (30, 30, 30), -1)
        which = f"DISPLAY {idx + 1} of {len(mons)}  (TAB = other display)   " if len(mons) > 1 else ""
        cv2.putText(canvas, which + "CLICK AND DRAG a box around the video, then ENTER  (ENTER alone = whole display, ESC = cancel)",
                    (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 220, 255), 1, cv2.LINE_AA)
        if state["start"] and state["end"]:
            cv2.rectangle(canvas, state["start"], state["end"], (0, 220, 0), 2)
        cv2.imshow(title, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            break
        if key == 9 and len(mons) > 1:  # TAB
            idx = (idx + 1) % len(mons)
            state.update(start=None, end=None, drag=False)
            continue
        if key in (13, 10):
            mon, w = mons[idx], shots[idx].shape[1]
            if state["start"] and state["end"]:
                x0, y0 = state["start"]
                x1, y1 = state["end"]
                x, y = min(x0, x1), min(y0, y1)
                bw, bh = abs(x1 - x0), abs(y1 - y0)
                if bw >= 20 and bh >= 20:
                    f = fit_for(idx)
                    box = (int(x / f), int(y / f), int(bw / f), int(bh / f))
                    result = region_from_pixels(box, mon, w)
                    break
                # a stray click, not a box: ignore it and let them try again
                state.update(start=None, end=None)
                continue
            result = Region(mon["left"], mon["top"], mon["width"], mon["height"])
            break
    cv2.destroyWindow(title)
    cv2.waitKey(1)
    return result


def display_containing(region: Region, mons: list[dict]) -> dict:
    """The display a region sits on (by its top-left corner)."""
    for m in mons:
        if m["left"] <= region.left < m["left"] + m["width"] and m["top"] <= region.top < m["top"] + m["height"]:
            return m
    return mons[0]


def suggest_window_pos(region: Region, screen_width: int, origin_left: int = 0) -> tuple[int, int]:
    """Where to put our own window so it does not sit on top of the captured
    area: on whichever side of it has more room, on the same display."""
    left_room = region.left - origin_left
    right_room = origin_left + screen_width - (region.left + region.width)
    if right_room >= left_room:
        return (min(region.left + region.width + 10, origin_left + max(0, screen_width - 200)), 40)
    return (origin_left, 40)


def screen_capture_allowed(request: bool = False) -> bool | None:
    """macOS only: has this app been given Screen Recording permission?

    Without it macOS silently hands back a picture of the bare desktop wallpaper
    with every window (including the video) missing.  Returns None when the
    answer cannot be determined (not macOS / old macOS)."""
    import sys

    if sys.platform != "darwin":
        return None
    try:
        import ctypes

        cg = ctypes.CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
        cg.CGPreflightScreenCaptureAccess.restype = ctypes.c_bool
        ok = bool(cg.CGPreflightScreenCaptureAccess())
        if not ok and request:
            cg.CGRequestScreenCaptureAccess.restype = ctypes.c_bool
            cg.CGRequestScreenCaptureAccess()
        return ok
    except (OSError, AttributeError):
        return None


PERMISSION_HELP = (
    "macOS is not letting this app record the screen, so it only sees the desktop wallpaper "
    "(no windows, no video).\n"
    "Fix:\n"
    "  1. System Settings > Privacy & Security > Screen & System Audio Recording\n"
    "  2. Turn ON the app you run this from (Terminal, iTerm, Cursor or PyCharm)\n"
    "  3. Quit that app completely (Cmd+Q) and reopen it, then run the command again."
)

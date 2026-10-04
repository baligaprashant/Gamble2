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


def select_region(monitor: int = 1) -> Region | None:
    """Show a screenshot of ``monitor`` and let the user drag a box around the
    video.  Returns None if they cancel."""
    import cv2

    with mss.mss() as sct:
        mon = sct.monitors[monitor]
        shot = np.array(sct.grab(mon))[:, :, :3].copy()
    h, w = shot.shape[:2]
    fit = min(1.0, 1400.0 / w, 800.0 / h)
    view = cv2.resize(shot, None, fx=fit, fy=fit, interpolation=cv2.INTER_AREA) if fit < 1 else shot
    title = "Drag a box around the video, then press ENTER (C to cancel)"
    x, y, bw, bh = cv2.selectROI(title, view, showCrosshair=False, fromCenter=False)
    cv2.destroyWindow(title)
    if bw < 20 or bh < 20:
        return None
    box = (int(x / fit), int(y / fit), int(bw / fit), int(bh / fit))
    return region_from_pixels(box, mon, w)

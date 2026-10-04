"""Laptop webcam capture via OpenCV."""

from __future__ import annotations

import cv2
import numpy as np


class CameraCapture:
    def __init__(self, device: int = 0, width: int = 1280, height: int = 720) -> None:
        self._device = device
        import sys

        # AVFoundation is the reliable backend on macOS; default elsewhere.
        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(device, backend)
        if not self._cap.isOpened():
            self._cap.release()
            self._cap = cv2.VideoCapture(device)
        if width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera device {device}")
        for _ in range(5):  # let auto-exposure settle; first frames are often black
            self._cap.read()

    @property
    def name(self) -> str:
        return f"camera:{self._device}"

    def read(self) -> np.ndarray | None:
        ok, frame = self._cap.read()
        return frame if ok else None

    def release(self) -> None:
        self._cap.release()

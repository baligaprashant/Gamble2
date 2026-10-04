"""Frame sources: camera and screen capture share this protocol."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class FrameSource(Protocol):
    def read(self) -> np.ndarray | None:
        """Return a BGR frame, or None if unavailable."""

    def release(self) -> None:
        """Free device / resources."""

    @property
    def name(self) -> str:
        ...

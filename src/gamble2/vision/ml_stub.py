"""ML detector stub — same CardDetector interface for a future YOLO/CNN."""

from __future__ import annotations

import numpy as np

from gamble2.vision.base import CardDetector, DetectedCard


class MLDetector(CardDetector):
    """Placeholder for a trained model.

    Swap this in once you have weights; keep TemplateDetector as fallback.
    """

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path
        self._model = None
        if model_path:
            raise NotImplementedError(
                "MLDetector is a stub. Load your model in detect() when ready."
            )

    def detect(self, frame: np.ndarray) -> list[DetectedCard]:
        if self._model is None:
            return []
        raise NotImplementedError("Wire your inference pipeline here.")

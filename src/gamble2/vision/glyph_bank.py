"""Reference glyphs (rank letters / suit pips) used for matching.

A bank holds several example images per label.  A built-in bank ships with the
package (``data/glyph_bank.npz``); glyphs taught from the user's own deck live in
``assets/templates/user`` and are merged in at load time.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

USER_DIR = Path(__file__).resolve().parents[3] / "assets" / "templates" / "user_v2"
BUILTIN = Path(__file__).resolve().parent.parent / "data" / "glyph_bank.npz"


def _unit(img: np.ndarray) -> np.ndarray:
    v = img.astype(np.float32).ravel()
    v = v - v.mean()
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else v


class GlyphBank:
    def __init__(self) -> None:
        self._items: dict[str, dict[str, list[np.ndarray]]] = {"rank": {}, "suit": {}}
        self._stack_cache: dict[tuple[str, str], np.ndarray] = {}

    def add(self, kind: str, label: str, img: np.ndarray) -> None:
        self._items[kind].setdefault(label, []).append(_unit(img))
        self._stack_cache.clear()

    def labels(self, kind: str) -> list[str]:
        return list(self._items[kind])

    def count(self) -> int:
        return sum(len(v) for d in self._items.values() for v in d.values())

    def match(self, kind: str, img: np.ndarray, allowed: str) -> tuple[str | None, float, float]:
        """Return (best label, best score, runner-up score) among ``allowed`` labels."""
        q = _unit(img)
        scores: list[tuple[float, str]] = []
        for label, vecs in self._items[kind].items():
            if label not in allowed:
                continue
            key = (kind, label)
            stack = self._stack_cache.get(key)
            if stack is None:
                stack = np.stack(vecs)
                self._stack_cache[key] = stack
            scores.append((float((stack * q).sum(axis=1).max()), label))  # avoids BLAS matmul warnings on macOS
        if not scores:
            return None, 0.0, 0.0
        scores.sort(reverse=True)
        second = scores[1][0] if len(scores) > 1 else 0.0
        return scores[0][1], scores[0][0], second

    # ---- persistence -------------------------------------------------
    def save_npz(self, path: Path) -> None:
        arrays = {}
        for kind, d in self._items.items():
            for label, vecs in d.items():
                arrays[f"{kind}__{label}"] = np.stack(vecs)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load_npz(cls, path: Path, shapes: dict[str, tuple[int, int]]) -> GlyphBank:
        bank = cls()
        with np.load(path) as z:
            for name in z.files:
                kind, label = name.split("__", 1)
                for vec in z[name]:
                    bank._items[kind].setdefault(label, []).append(vec.astype(np.float32))
        return bank

    def merge_user_dir(self, folder: Path = USER_DIR) -> int:
        """Add taught glyphs saved as ``<kind>_<label>_<n>.npy``."""
        n = 0
        if not folder.exists():
            return 0
        for f in sorted(folder.glob("*.npy")):
            parts = f.stem.split("_")
            if len(parts) < 2 or parts[0] not in ("rank", "suit"):
                continue
            img = np.load(f)
            # user glyphs count extra: they come from the real deck + camera
            for _ in range(2):
                self.add(parts[0], parts[1], img)
            n += 1
        return n


@lru_cache(maxsize=1)
def default_bank() -> GlyphBank:
    return load_bank()


def load_bank(include_user: bool = True) -> GlyphBank:
    bank = GlyphBank()
    if BUILTIN.exists():
        bank = GlyphBank.load_npz(BUILTIN, {})
    if include_user:
        bank.merge_user_dir()
    return bank


def save_user_glyph(kind: str, label: str, img: np.ndarray, folder: Path = USER_DIR) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    i = len(list(folder.glob(f"{kind}_{label}_*.npy")))
    path = folder / f"{kind}_{label}_{i}.npy"
    np.save(path, img.astype(np.float32))
    return path

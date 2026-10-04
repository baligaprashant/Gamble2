"""Find card *index corners* (rank letter stacked over a suit pip) anywhere in a frame.

Whole-card outlines are unreliable when cards are fanned or overlapped - the
back card is only a strip.  Its index corner, though, is always visible.  So we
look for the index itself: a small ink blob stacked directly over a compact,
solid ink blob (the pip).  Each pair is straightened, normalised and handed to
the same glyph matcher the card-based path uses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from gamble2.vision.cards import Card
from gamble2.vision.classify import RANK_SIZE, SUIT_SIZE, _normalise, match_glyphs, IndexGlyphs
from gamble2.vision.glyph_bank import GlyphBank

_WORK_W = 1280


@dataclass(frozen=True)
class _Blob:
    x: int
    y: int
    w: int
    h: int
    area: int
    label: int

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    @property
    def bottom(self) -> int:
        return self.y + self.h

    @property
    def fill(self) -> float:
        return self.area / float(max(1, self.w * self.h))


@dataclass(frozen=True)
class IndexPair:
    """One candidate index corner, straightened and normalised, not yet matched."""

    glyphs: IndexGlyphs
    bbox: tuple[int, int, int, int]  # input-frame pixels
    upright: bool = True  # False: read upside-down (the bottom-right index)


@dataclass(frozen=True)
class IndexRead:
    card: Card
    confidence: float
    bbox: tuple[int, int, int, int]  # in input-frame pixels


def index_pairs(frame: np.ndarray) -> list[IndexPair]:
    h0, w0 = frame.shape[:2]
    scale = min(1.0, _WORK_W / float(w0))
    img = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame
    out: list[IndexPair] = []
    # strict threshold for bold print, looser one for thin / washed-out strokes
    for frac in (0.62, 0.80):
        out.extend(_pairs_at(img, scale, frac))
    return out


def _pairs_at(img: np.ndarray, scale: float, frac: float) -> list[IndexPair]:
    ink, _mc = _ink_mask(img, frac)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    blobs = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if h < 7 or w < 3 or h > 130 or w > 130 or area < 18:
            continue
        blobs.append(_Blob(x, y, w, h, area, i))

    ranks = _merge_pairs(blobs)  # allow "10" = two blobs side by side
    out: list[IndexPair] = []
    for pip in blobs:
        if not _pip_like(pip):
            continue
        for upright in (True, False):
            stacked = []
            for rk in ranks:
                angle = _stacked(rk, pip, upright)
                if angle is not None:
                    stacked.append((rk, angle))
            # a "10" must not also be read as a lone "1" (which looks like a J)
            merged_parts = {p for rk, _ in stacked if rk.label < 0 for p in _parts(rk)}
            for rk, angle in stacked:
                if rk.label >= 0 and rk.label in merged_parts:
                    continue
                rank_img = _glyph_image(labels, rk, _parts(rk), angle, upright, RANK_SIZE)
                suit_img = _glyph_image(labels, pip, _parts(pip), angle, upright, SUIT_SIZE)
                if rank_img is None or suit_img is None:
                    continue
                g = IndexGlyphs(rank_img, suit_img, _is_red(img, labels, pip), (0, 0, 0, 0), (0, 0, 0, 0))
                x = min(rk.x, pip.x)
                y = min(rk.y, pip.y)
                w = max(rk.x + rk.w, pip.x + pip.w) - x
                h = max(rk.bottom, pip.bottom) - y
                out.append(IndexPair(g, (int(x / scale), int(y / scale), int(w / scale), int(h / scale)), upright))
    return out


def find_index_reads(frame: np.ndarray, bank: GlyphBank, min_conf: float = 0.0) -> list[IndexRead]:
    reads: list[IndexRead] = []
    for pair in index_pairs(frame):
        hit = match_glyphs(pair.glyphs, bank)
        if hit is None:
            continue
        card, conf = hit
        if conf >= min_conf:
            reads.append(IndexRead(card, conf, pair.bbox))
    return reads


# ---------------------------------------------------------------- ink mask
def _ink_mask(img: np.ndarray, frac: float = 0.62) -> tuple[np.ndarray, np.ndarray]:
    b, g, r = cv2.split(img)
    mc = cv2.min(cv2.min(b, g), r)
    # local paper brightness: a grey-level *maximum* over a window bigger than a glyph
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    paper = cv2.dilate(mc, k)
    paper = cv2.GaussianBlur(paper, (0, 0), 6)
    ink = ((mc.astype(np.float32) < paper.astype(np.float32) * frac) & (paper > 110)).astype(np.uint8)
    return ink, mc


# -------------------------------------------------------------- pairing
def _pip_like(b: _Blob) -> bool:
    if not (0.55 <= b.w / b.h <= 1.5):
        return False
    return b.fill >= 0.40 and 9 <= b.h <= 70


def _merge_pairs(blobs: list[_Blob]) -> list[_Blob]:
    """Rank candidates: every blob, plus merged horizontal neighbours (the '10')."""
    out = list(blobs)
    for a in blobs:
        for b in blobs:
            if b.x <= a.x or b.label == a.label:
                continue
            gap = b.x - (a.x + a.w)
            if gap < -1 or gap > 0.55 * max(a.w, b.w):
                continue
            if abs(a.h - b.h) > 0.25 * max(a.h, b.h) or abs(a.cy - b.cy) > 0.2 * max(a.h, b.h):
                continue
            x, y = min(a.x, b.x), min(a.y, b.y)
            w = max(a.x + a.w, b.x + b.w) - x
            h = max(a.bottom, b.bottom) - y
            out.append(_Blob(x, y, w, h, a.area + b.area, -1 - a.label * 100000 - b.label))
    return out


def _stacked(rank: _Blob, pip: _Blob, upright: bool) -> float | None:
    """Return the tilt (radians) if ``pip`` sits directly below (or above, if
    not ``upright``) ``rank`` like an index corner; else None."""
    if rank.label == pip.label or _overlaps(rank, pip):
        return None
    if upright:
        gap = pip.y - rank.bottom
    else:
        gap = rank.y - pip.bottom
    if gap < -2 or gap > 0.7 * max(rank.h, pip.h):
        return None
    if not (0.55 <= rank.h / pip.h <= 2.8):
        return None
    if rank.w > 2.6 * pip.w + 4 or rank.w < 0.3 * pip.w:
        return None
    dx = pip.cx - rank.cx
    dy = (pip.cy - rank.cy) if upright else (rank.cy - pip.cy)
    if dy <= 0:
        return None
    angle = math.atan2(dx, dy)
    if abs(angle) > math.radians(34):
        return None
    return angle


def _overlaps(a: _Blob, b: _Blob) -> bool:
    ix = min(a.x + a.w, b.x + b.w) - max(a.x, b.x)
    iy = min(a.bottom, b.bottom) - max(a.y, b.y)
    return ix > 2 and iy > 2


# ------------------------------------------------------------ reading
def _blob_mask(labels: np.ndarray, blob: _Blob, img_shape) -> np.ndarray:
    if blob.label >= 0:
        sub = labels[blob.y : blob.y + blob.h, blob.x : blob.x + blob.w] == blob.label
        return sub.astype(np.uint8)
    return None  # merged blob: handled by caller


def _glyph_image(
    ink_labels: np.ndarray, blob: _Blob, parts: list[int], angle: float, upright: bool, size: tuple[int, int]
) -> np.ndarray:
    pad = 4
    x0, y0 = max(blob.x - pad, 0), max(blob.y - pad, 0)
    x1, y1 = min(blob.x + blob.w + pad, ink_labels.shape[1]), min(blob.bottom + pad, ink_labels.shape[0])
    sub = np.isin(ink_labels[y0:y1, x0:x1], parts).astype(np.uint8) * 255
    hh, ww = sub.shape
    deg = -math.degrees(angle) + (0 if upright else 180)
    M = cv2.getRotationMatrix2D((ww / 2, hh / 2), deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(hh * sin + ww * cos) + 2, int(hh * cos + ww * sin) + 2
    M[0, 2] += nw / 2 - ww / 2
    M[1, 2] += nh / 2 - hh / 2
    rot = cv2.warpAffine(sub, M, (nw, nh), flags=cv2.INTER_LINEAR)
    rot = (rot > 100).astype(np.uint8)
    ys, xs = np.nonzero(rot)
    if len(xs) < 8:
        return None
    box = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    return _normalise(rot, box, size)


def _parts(b: _Blob, blobs_by_label: dict[int, _Blob] | None = None) -> list[int]:
    if b.label >= 0:
        return [b.label]
    code = -b.label - 1
    la, lb = divmod(code, 100000)
    return [la, lb]


def _is_red(img: np.ndarray, labels: np.ndarray, pip: _Blob) -> bool:
    m = labels[pip.y : pip.bottom, pip.x : pip.x + pip.w] == pip.label
    patch = img[pip.y : pip.bottom, pip.x : pip.x + pip.w].astype(np.int32)
    if int(m.sum()) < 6:
        return False
    b, g, r = patch[..., 0][m].mean(), patch[..., 1][m].mean(), patch[..., 2][m].mean()
    # red ink has red well above green even in a dim, blue-tinted camera image
    # (where r - max(b, g) is useless); black ink has r <= g.
    return (r - g) > 8

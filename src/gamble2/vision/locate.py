"""Find playing-card candidates in a frame and warp them upright.

Segmentation is deliberately liberal: several different foreground masks each
propose rectangles, and a blob that might be several touching cards is also
proposed as 1/2/3 pieces.  The classifier (``classify.py``) is the arbiter -
a candidate that does not contain a readable card index is simply dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

WARP_W = 200
WARP_H = 280
_WORK_W = 720  # frames are analysed at about this width


@dataclass(frozen=True)
class Candidate:
    quad: np.ndarray  # 4x2 float32, frame coordinates, short side first
    bbox: tuple[int, int, int, int]  # x, y, w, h in frame coordinates


@dataclass(frozen=True)
class LocatedCard:
    bbox: tuple[int, int, int, int]
    warped: np.ndarray


def find_candidates(frame: np.ndarray, max_candidates: int = 14) -> list[Candidate]:
    h, w = frame.shape[:2]
    scale = min(1.0, _WORK_W / float(w))
    small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame
    sh, sw = small.shape[:2]
    area = sh * sw
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    rects: list[tuple[tuple[float, float], tuple[float, float], float]] = []
    for mask in _masks(blur):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            a = cv2.contourArea(cnt)
            if a < 0.006 * area or a > 0.85 * area:
                continue
            hull = cv2.convexHull(cnt)
            rect = cv2.minAreaRect(hull)
            (rw, rh) = rect[1]
            if min(rw, rh) < 20:
                continue
            fill = cv2.contourArea(hull) / max(rw * rh, 1.0)
            if fill < 0.78:
                continue
            rects.append(rect)

    quads: list[np.ndarray] = []
    for rect in rects:
        base = _normalise_quad(cv2.boxPoints(rect).astype(np.float32))
        quads.extend(_trim_to_card(base))
        short = float(np.linalg.norm(base[1] - base[0]))
        long = float(np.linalg.norm(base[2] - base[1]))
        for k in (2, 3):
            piece_aspect = short / (long / k)
            if 1.0 / piece_aspect > 0 and 0.55 <= min(piece_aspect, 1 / piece_aspect) <= 0.92:
                for piece in _split(base, k):
                    quads.extend(_trim_to_card(piece))

    out: list[Candidate] = []
    for q in sorted(quads, key=lambda q: -cv2.contourArea(q)):
        q_full = (q / scale).astype(np.float32)
        x, y, bw, bh = cv2.boundingRect(q_full)
        cand = Candidate(quad=_normalise_quad(q_full), bbox=(x, y, bw, bh))
        if any(_iou(cand.bbox, o.bbox) > 0.85 for o in out):
            continue
        out.append(cand)
        if len(out) >= max_candidates:
            break
    return out


def _masks(blur: np.ndarray) -> list[np.ndarray]:
    k5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    masks: list[np.ndarray] = []

    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks.append(otsu)
    masks.append(255 - otsu)  # card darker than its surroundings

    tiny = cv2.resize(blur, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    local = cv2.resize(cv2.GaussianBlur(tiny, (0, 0), 9), (blur.shape[1], blur.shape[0]))
    diff = blur.astype(np.int16) - local.astype(np.int16)
    masks.append(((diff > 14) * 255).astype(np.uint8))

    edges = cv2.Canny(blur, 30, 100)
    edges = cv2.dilate(edges, k5, iterations=2)
    masks.append(edges)

    cleaned = []
    for m in masks:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k3)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k5, iterations=2)
        cleaned.append(m)
    return cleaned


def _normalise_quad(pts: np.ndarray) -> np.ndarray:
    """Clockwise order (image coords) with the *short* edge first."""
    pts = pts.reshape(4, 2).astype(np.float32)
    # order around the centroid so we always have a proper polygon
    c = pts.mean(axis=0)
    order = np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))
    pts = pts[order]  # increasing angle == clockwise on screen (y down)
    e0 = np.linalg.norm(pts[1] - pts[0])
    e1 = np.linalg.norm(pts[2] - pts[1])
    if e0 > e1:
        pts = np.roll(pts, -1, axis=0)
    return pts


CARD_ASPECT = 1.4  # long / short for a standard poker-size card


def _trim_to_card(quad: np.ndarray) -> list[np.ndarray]:
    """A blob that is too long to be one card (fingers, shadow, a sliver of the
    next card) is trimmed to card proportions, anchored at either end."""
    p0, p1, p2, p3 = quad
    short = float(np.linalg.norm(p1 - p0))
    long = float(np.linalg.norm(p2 - p1))
    if long <= short * 1.5:
        return [quad]
    frac = short * CARD_ASPECT / long
    a = np.array([p0, p1, p1 + (p2 - p1) * frac, p0 + (p3 - p0) * frac], dtype=np.float32)
    b = np.array([p0 + (p3 - p0) * (1 - frac), p1 + (p2 - p1) * (1 - frac), p2, p3], dtype=np.float32)
    return [_normalise_quad(a), _normalise_quad(b)]


def _split(quad: np.ndarray, k: int) -> list[np.ndarray]:
    """Cut a (short-first) quad into k equal pieces along its long edge."""
    p0, p1, p2, p3 = quad
    out = []
    for i in range(k):
        a, b = i / k, (i + 1) / k
        q0 = p0 + (p3 - p0) * a
        q1 = p1 + (p2 - p1) * a
        q2 = p1 + (p2 - p1) * b
        q3 = p0 + (p3 - p0) * b
        out.append(_normalise_quad(np.array([q0, q1, q2, q3], dtype=np.float32)))
    return out


def warp_candidate(frame: np.ndarray, cand: Candidate) -> np.ndarray:
    dest = np.array(
        [[0, 0], [WARP_W - 1, 0], [WARP_W - 1, WARP_H - 1], [0, WARP_H - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(cand.quad, dest)
    return cv2.warpPerspective(frame, matrix, (WARP_W, WARP_H), flags=cv2.INTER_AREA)


def locate_cards(frame: np.ndarray, max_cards: int = 8) -> list[LocatedCard]:
    """Back-compat helper: every candidate, warped (no classification)."""
    return [
        LocatedCard(bbox=c.bbox, warped=warp_candidate(frame, c))
        for c in find_candidates(frame)[:max_cards]
    ]


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    return _iou(a, b)


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0

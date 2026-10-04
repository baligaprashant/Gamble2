"""Read rank + suit from the index corner of an upright, warped card.

Pipeline for one orientation:
  1. crop the top-left corner and mask "ink" (anything clearly darker than paper)
  2. keep small connected components that look like glyphs (not borders/artwork)
  3. group them into vertical bands: the first band is the rank, the next one
     (roughly centred under it) is the suit pip
  4. compare each band to a bank of reference glyphs by normalised correlation
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from gamble2.vision.cards import RANKS, SUITS, Card
from gamble2.vision.glyph_bank import GlyphBank, default_bank

RANK_SIZE = (24, 32)  # w, h
SUIT_SIZE = (28, 28)

_CROP_W = 0.34
_CROP_H = 0.45


@dataclass(frozen=True)
class IndexGlyphs:
    rank: np.ndarray  # RANK_SIZE binary-ish float image
    suit: np.ndarray
    red: bool
    rank_box: tuple[int, int, int, int]
    suit_box: tuple[int, int, int, int]
    #: ink colour family: "red", "black", or - for four-colour decks - "green" / "blue"
    tint: str = ""


def classify_warped(
    warped: np.ndarray,
    bank: GlyphBank | None = None,
    threshold: float = 0.70,
) -> tuple[Card, float] | None:
    bank = bank or default_bank()
    best: tuple[Card, float] | None = None
    warped = snap_to_paper(warped)
    for img in (warped, cv2.rotate(warped, cv2.ROTATE_180)):
        glyphs = extract_index(img, snap=False)
        if glyphs is None:
            continue
        hit = match_glyphs(glyphs, bank)
        if hit is not None and (best is None or hit[1] > best[1]):
            best = hit
    if best is None or best[1] < threshold:
        return None
    return best


def _preferred_suits(g: IndexGlyphs) -> str:
    tint = g.tint or ("red" if g.red else "black")
    return {"red": "hd", "black": "cs", "green": "c", "blue": "d"}.get(tint, "cs")


def suit_matches_tint(g: IndexGlyphs, suit: str) -> bool:
    return suit in _preferred_suits(g)


def match_glyphs(g: IndexGlyphs, bank: GlyphBank) -> tuple[Card, float] | None:
    rank, rscore, rsecond = bank.match("rank", g.rank, RANKS)
    # Suit: ink colour is a strong hint, not a hard rule.  Camera white balance
    # and dim light can make a red pip look black (or vice versa), so the shape
    # gets a vote of its own and a colour disagreement only lowers confidence.
    # Four-colour decks (green clubs, blue diamonds) point at a single suit.
    pref = _preferred_suits(g)
    bonus = 0.15 if len(pref) == 1 else 0.10
    best = None
    for st in "cdhs":
        label, score, _ = bank.match("suit", g.suit, st)
        if label is None:
            continue
        adj = score + (bonus if st in pref else 0.0)
        if best is None or adj > best[0]:
            best = (adj, st, score)
    if rank is None or best is None:
        return None
    _adj, suit, sscore = best
    if suit not in pref:
        sscore -= 0.10
    # ambiguity penalty: a rank that is barely better than the runner-up is suspect
    margin = rscore - rsecond
    conf = min(rscore, sscore) - max(0.0, 0.08 - margin)
    return Card(rank, suit), float(conf)


def snap_to_paper(warped: np.ndarray) -> np.ndarray:
    """Re-crop the warp to the bright paper area so the index sits at a
    predictable place even when the detected outline is a few pixels off."""
    h, w = warped.shape[:2]
    mc = _min_channel(warped)
    ref = float(np.percentile(mc[::3, ::3], 90))
    if ref < 60:
        return warped
    mask = (mc > ref * 0.78).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    if n < 2:
        return warped
    i = 1 + int(np.argmax(stats[1:, 4]))
    x, y, bw, bh = (int(v) for v in stats[i, :4])
    if bw < 0.55 * w or bh < 0.55 * h:
        return warped
    crop = warped[y : y + bh, x : x + bw]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)


def _min_channel(img: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(img)
    return cv2.min(cv2.min(b, g), r)


def extract_index(warped: np.ndarray, snap: bool = True) -> IndexGlyphs | None:
    if snap:
        warped = snap_to_paper(warped)
    h, w = warped.shape[:2]
    y1, x1 = int(h * _CROP_H), int(w * _CROP_W)
    corner = warped[:y1, :x1]
    mc = _min_channel(corner).astype(np.float32)  # darkest channel: black & red ink are both low
    paper = float(np.percentile(mc, 92))
    if paper < 70:  # not a light card
        return None
    ink = (mc < paper * 0.62).astype(np.uint8)
    if not (0.008 < ink.mean() < 0.60):
        return None

    ink[:, :3] = 0  # card edge outline would otherwise fuse with the glyphs
    ink[:3, :] = 0
    found = _extract_standard(ink.copy(), h, w, x1, y1) or _extract_pip_first(ink, h, w)
    if found is None:
        return None
    rb, sb, rank_mask, suit_mask = found

    # sanity: glyph sizes of a real index
    if rb[3] < h * 0.045 or sb[3] < h * 0.035 or sb[2] < w * 0.035:
        return None

    rank_img = _normalise(rank_mask, rb, RANK_SIZE)
    suit_img = _normalise(suit_mask, sb, SUIT_SIZE)
    colour = _is_red(corner, suit_mask, sb)
    return IndexGlyphs(rank_img, suit_img, colour, rb, sb)


Box = tuple[int, int, int, int]


def _extract_standard(ink: np.ndarray, h: int, w: int, x1: int, y1: int):
    """Rank = first glyph band, suit = the next band centred under it."""
    ink = _cut_tall_blobs(ink, h)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    comps = []
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if x + cw >= x1 - 1 or y + ch >= y1 - 1:
            continue
        if ch < h * 0.035 or ch > h * 0.20 or cw > w * 0.22 or area < 12:
            continue
        comps.append((x, y, cw, ch, i))
    if not comps:
        return None

    comps.sort(key=lambda c: c[1])
    bands: list[list[tuple[int, int, int, int, int]]] = []
    for c in comps:
        for b in bands:
            top = min(m[1] for m in b)
            bot = max(m[1] + m[3] for m in b)
            overlap = min(bot, c[1] + c[3]) - max(top, c[1])
            if overlap > 0.5 * min(bot - top, c[3]):
                b.append(c)
                break
        else:
            bands.append([c])

    bands.sort(key=lambda b: min(m[1] for m in b))
    rb = _band_box(bands[0])
    suit_band = None
    for b in bands[1:]:
        sb = _band_box(b)
        gap = sb[1] - (rb[1] + rb[3])
        rcx, scx = rb[0] + rb[2] / 2, sb[0] + sb[2] / 2
        if 0 <= gap < h * 0.10 and abs(rcx - scx) < max(rb[2], sb[2]) * 0.8 + 3:
            suit_band = b
            break
    if suit_band is not None:
        return rb, _band_box(suit_band), ink, ink
    found = _pip_by_opening(ink, rb, h, w)
    if found is None:
        return None
    sb, suit_mask = found
    return rb, sb, ink, suit_mask


def _extract_pip_first(ink: np.ndarray, h: int, w: int):
    """Court cards: rank, pip and the artwork frame can all be fused into one
    blob.  The pip is the thick part - find it with an opening, then the rank
    is whatever ink sits above it."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    opened = cv2.morphologyEx(ink, cv2.MORPH_OPEN, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    best = None
    for i in range(1, n):
        x, y, cw, ch, area = (int(v) for v in stats[i])
        if not (h * 0.04 <= ch <= h * 0.12 and w * 0.04 <= cw <= w * 0.14):
            continue
        if x > w * 0.20 or y > h * 0.30 or y < h * 0.06:
            continue
        if best is None or y < best[1]:
            best = (i, y)
    if best is None:
        return None
    pip = cv2.bitwise_and(cv2.dilate((labels == best[0]).astype(np.uint8), k), ink)
    ys, xs = np.nonzero(pip)
    sb = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))

    above = ink.copy()
    above[max(sb[1] - 1, 0) :, :] = 0
    above[:, int(sb[0] + sb[2] + w * 0.06) :] = 0
    n, _, st, _ = cv2.connectedComponentsWithStats(above, connectivity=8)
    comps = []
    for i in range(1, n):
        x, y, cw, ch, area = (int(v) for v in st[i])
        if ch >= h * 0.035 and ch <= h * 0.20 and cw <= w * 0.22 and area >= 12:
            comps.append((x, y, cw, ch, i))
    if not comps:
        return None
    rb = _band_box(comps)
    if abs((rb[0] + rb[2] / 2) - (sb[0] + sb[2] / 2)) > max(rb[2], sb[2]) * 0.9 + 4:
        return None
    return rb, sb, above, pip


def _pip_by_opening(
    ink: np.ndarray, rb: tuple[int, int, int, int], card_h: int, card_w: int
) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
    """The suit pip sometimes touches the artwork frame on court cards.  Look
    just below the rank and peel the thin frame lines off with a morphological
    opening, then keep the blob that sits centred under the rank."""
    rx, ry, rw, rh = rb
    top = ry + rh
    bottom = min(ink.shape[0], top + int(card_h * 0.13))
    if bottom - top < 8:
        return None
    strip = np.zeros_like(ink)
    strip[top:bottom] = ink[top:bottom]
    rcx = rx + rw / 2
    for r in (2, 3, 4):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        opened = cv2.morphologyEx(strip, cv2.MORPH_OPEN, k)
        restored = cv2.bitwise_and(cv2.dilate(opened, k), strip)
        n, labels, stats, cents = cv2.connectedComponentsWithStats(opened, connectivity=8)
        best = None
        for i in range(1, n):
            x, y, cw, ch, area = stats[i]
            if not (card_h * 0.035 <= ch <= card_h * 0.12 and card_w * 0.035 <= cw <= card_w * 0.14):
                continue
            if y - top > card_h * 0.06:
                continue
            dx = abs(x + cw / 2 - rcx)
            if dx > max(rw, cw) * 0.9 + 4:
                continue
            if best is None or dx < best[0]:
                best = (dx, i)
        if best is not None:
            comp = (labels == best[1]).astype(np.uint8)
            clean = cv2.bitwise_and(cv2.dilate(comp, k), restored)
            ys, xs = np.nonzero(clean)
            if len(xs) < 12:
                continue
            box = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
            return box, clean
    return None


def _cut_tall_blobs(ink: np.ndarray, card_h: int) -> np.ndarray:
    """Rank and suit sometimes touch (e.g. the tail of a Q); cut such a blob
    between the two.  The suit pip is the thick blob at the bottom, so we find
    it with a morphological opening and cut just above it."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    for i in range(1, n):
        x, y, cw, ch, _area = stats[i]
        if ch <= card_h * 0.145 or ch > card_h * 0.30 or x > ink.shape[1] * 0.35:
            continue
        comp = (labels[y : y + ch, x : x + cw] == i).astype(np.uint8)
        cut = None
        opened = cv2.morphologyEx(comp, cv2.MORPH_OPEN, k)
        m, olab, ost, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
        thick = [j for j in range(1, m) if ost[j, 4] >= 40]
        if thick:
            j = max(thick, key=lambda t: ost[t, 1])  # lowest thick blob
            top = int(ost[j, 1]) - 3  # restore thin tips above the body
            if ch * 0.30 <= top <= ch * 0.80:
                cut = top
        if cut is None:
            rows = comp.sum(axis=1).astype(np.float32)
            lo, hi = int(ch * 0.30), int(ch * 0.72)
            if hi > lo:
                r = lo + int(np.argmin(rows[lo:hi]))
                if rows[r] < 0.6 * rows.max():
                    cut = r
        if cut is not None:
            ink[y + cut, x : x + cw] = 0
    return ink


def _band_box(band: list[tuple[int, int, int, int, int]]) -> tuple[int, int, int, int]:
    x0 = min(c[0] for c in band)
    y0 = min(c[1] for c in band)
    x1 = max(c[0] + c[2] for c in band)
    y1 = max(c[1] + c[3] for c in band)
    return x0, y0, x1 - x0, y1 - y0


def _normalise(ink: np.ndarray, box: tuple[int, int, int, int], size: tuple[int, int]) -> np.ndarray:
    x, y, w, h = box
    crop = ink[y : y + h, x : x + w].astype(np.float32)
    crop = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(crop, (3, 3), 0)


def _is_red(corner: np.ndarray, ink: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    x, y, w, h = box
    patch = corner[y : y + h, x : x + w].astype(np.int32)
    m = ink[y : y + h, x : x + w] > 0
    if int(m.sum()) < 6:
        return False
    b, g, r = patch[..., 0][m].mean(), patch[..., 1][m].mean(), patch[..., 2][m].mean()
    return (r - max(b, g)) > 28


def all_suits() -> str:
    return SUITS

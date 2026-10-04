"""Card reading + lock-in + overlay, end to end on synthetic frames."""

import cv2
import numpy as np
import pytest

from gamble2.capture.demo import DemoCapture
from gamble2.state.lock import StableHero
from gamble2.ui.overlay import OverlayApp
from gamble2.vision.cards import parse_cards
from gamble2.vision.render import render_card
from gamble2.vision.template import TemplateDetector


def _scene(codes, angle=8.0, seed=0, noise=4.0, gain=0.8, size=(720, 1280)):
    rng = np.random.default_rng(seed)
    h, w = size
    frame = np.full((h, w, 3), (45, 52, 48), np.uint8)
    cards = parse_cards(codes)
    cw, ch = 210, 294
    x = 330
    for c in cards:
        face = render_card(c, cw, ch)
        pad = 70
        M = cv2.getRotationMatrix2D((cw / 2, ch / 2), angle, 1.0)
        M[:, 2] += pad  # keep the whole rotated card inside the canvas
        dims = (cw + 2 * pad, ch + 2 * pad)
        rot = cv2.warpAffine(face, M, dims, borderValue=(0, 0, 0))
        mask = cv2.warpAffine(np.full((ch, cw), 255, np.uint8), M, dims)
        y = 150
        roi = frame[y : y + rot.shape[0], x : x + rot.shape[1]]
        roi[mask > 0] = rot[mask > 0]
        x += cw + 30
    frame = frame.astype(np.float32) * gain + rng.normal(0, noise, frame.shape)
    frame = np.clip(frame, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(frame, (0, 0), 0.8)


@pytest.fixture(scope="module")
def detector():
    # synthetic cards are drawn with OpenCV's font, so they use the demo bank
    from gamble2.vision.demo_bank import demo_bank

    return TemplateDetector(bank=demo_bank())


def test_shipped_bank_loads():
    assert TemplateDetector().ready


def test_bank_is_loaded(detector):
    assert detector.ready


@pytest.mark.parametrize("hand", ["Ah Kh", "7c Td", "Qs 3d", "Jc 9s", "2h 2d", "Ts 6c"])
def test_demo_frame_cards_are_read(detector, hand):
    got = {d.card.code for d in detector.detect(DemoCapture(hand).read())}
    assert got == {c.code for c in parse_cards(hand)}


@pytest.mark.parametrize("angle", [-12, 0, 9, 14])
def test_rotated_noisy_frame(detector, angle):
    got = {d.card.code for d in detector.detect(_scene("Kd 8s", angle=angle, seed=abs(angle)))}
    assert got == {"Kd", "8s"}


def test_empty_frame_sees_nothing(detector):
    assert detector.detect(np.full((480, 640, 3), 90, np.uint8)) == []


def test_lock_survives_flicker():
    lock = StableHero(need_frames=4)
    ah, kh = parse_cards("Ah Kh")
    locked = None
    # a card is lost every other frame, a wrong read appears once
    for i in range(14):
        frame = [ah, kh] if i % 2 == 0 else [ah]
        if i == 5:
            frame = frame + parse_cards("2c")
        locked = lock.observe(frame)
    assert locked is not None and {c.code for c in locked} == {"Ah", "Kh"}


def test_overlay_produces_probability_from_camera_frames(detector):
    app = OverlayApp(read_frame=lambda: None, detect=detector.detect, detector=detector)
    frame = _scene("Ah Kh", angle=5)
    for _ in range(8):
        app.tick(frame.copy())
    assert [c.code for c in sorted(app.state.hero, key=str)] == ["Ah", "Kh"]
    eq = app._last_equity
    assert eq is not None and 0.6 < eq.equity < 0.7  # AKs vs a random hand ~ 66%


def test_calibration_finishes_without_crashing(tmp_path, monkeypatch):
    import numpy as np
    from gamble2.ui.overlay import OverlayApp
    from gamble2.vision import glyph_bank
    from gamble2.vision.index_finder import IndexPair
    from gamble2.vision.classify import IndexGlyphs
    from gamble2.vision.template import TemplateDetector

    monkeypatch.setattr(glyph_bank, "USER_DIR", tmp_path)
    det = TemplateDetector(bank=glyph_bank.GlyphBank())
    g = IndexGlyphs(np.ones((32, 24), np.float32), np.ones((28, 28), np.float32), False, (0, 0, 0, 0), (0, 0, 0, 0))
    det.best_pair_for = lambda fr, c, min_score=0.3, **kw: IndexPair(g, (5, 5, 20, 40))
    app = OverlayApp(lambda: None, det.detect, detector=det)
    frame = np.full((240, 320, 3), 200, np.uint8)
    app._calib_start()
    for _ in range(13 * 9):
        app.tick(frame.copy())
    assert app._calib is None


def test_region_from_retina_pixels():
    from gamble2.capture.screen import region_from_pixels

    mon = {"left": 0, "top": 0, "width": 1440, "height": 900}
    r = region_from_pixels((200, 100, 800, 600), mon, shot_width=2880)  # 2x retina
    assert (r.left, r.top, r.width, r.height) == (100, 50, 400, 300)
    r = region_from_pixels((200, 100, 800, 600), mon, shot_width=1440)  # 1x
    assert (r.left, r.top, r.width, r.height) == (200, 100, 800, 600)


def test_overlay_window_goes_to_roomier_side():
    from gamble2.capture.screen import Region, suggest_window_pos

    assert suggest_window_pos(Region(0, 0, 800, 600), 1440)[0] >= 800  # video on the left
    assert suggest_window_pos(Region(700, 0, 700, 600), 1440)[0] == 0  # video on the right

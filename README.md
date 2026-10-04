# Gamble2

Live **heads-up Texas Hold’em** equity vs the dealer from a **laptop camera** or **screen region**.

Preflop odds come from a **precomputed 169-hand lookup table** (instant). Turn/river use **exact enumeration**. Flop vs an unknown dealer uses a large Monte Carlo sample so the UI stays responsive.

## Setup

```bash
cd /Users/prashantbaliga/project/Gamble2
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Build equity tables (once):

```bash
python -m gamble2.odds.build_tables --sims 20000 --matchup-boards 0 --also-package
```

`--matchup-boards 150` (slower) also builds the 169×169 matrix for known dealer hole cards preflop.

## Run

Demo (no camera):

```bash
python -m gamble2 --demo
```

Camera:

```bash
python -m gamble2 --source camera
```

Screen region (`left,top,width,height`):

```bash
python -m gamble2 --source screen --region 100,200,800,600
```

### Overlay keys

| Key | Action |
|-----|--------|
| `H` | Enter your hole cards (`Ah Kh`) |
| `B` | Enter board (`As Td 2c` / turn / river) |
| `D` | Enter dealer cards, or blank for random |
| `C` | Clear hand |
| `Q` | Quit |

| `T` | Teach: show ONE card, type its code (e.g. `Kd`), Enter — saves that card's glyphs from *your* deck |
| `V` | Toggle debug view (yellow outlines = card candidates the finder considered) |

### How card reading works

1. `vision/locate.py` proposes card-shaped regions (several foreground masks; blobs that
   look like 2–3 touching cards are split; blobs with fingers attached are trimmed to card proportions).
2. `vision/classify.py` warps each region upright, finds the index corner (rank above suit pip),
   and matches both glyphs against a bank of reference glyphs (`data/glyph_bank.npz`, built
   from several different card designs). It tries the card both ways up.
3. `state/lock.py` votes over the last ~14 frames, so one blurry frame or a missing card
   can't flip your hand; the odds appear as soon as two cards win the vote (usually <0.5 s).

If your deck's index font differs enough that a card is misread, point the camera at that card,
press `T`, type its code. Taught glyphs are stored in `assets/templates/user/` and used from then on.

**Tips for a webcam:** hold the cards still and fairly close (index corner ≥ ~40 px tall in the
image), keep the top-left index corner uncovered by your fingers, avoid glare, plain background
helps. Press `V` to see what the finder is looking at.

## Layout

- `src/gamble2/capture/` — camera + screen (`FrameSource`)
- `src/gamble2/vision/` — `CardDetector` ABC, OpenCV `TemplateDetector`, `MLDetector` stub
- `src/gamble2/odds/` — tables, exact enum, equity router
- `src/gamble2/ui/` — live overlay
- `data/` — generated JSON tables

## Notes

- Variant: heads-up Hold’em vs one dealer/opponent (not multiway).
- Screen capture of real money clients may violate site terms of service; use for personal/educational practice.
- Swap in ML later by implementing `MLDetector.detect()` with the same `DetectedCard` return type.

## How card reading works now

The reader looks for each card's **index corner** (rank letter stacked over the suit pip) instead of
the whole card outline, so fanned or overlapped hands work. Tips: hold the cards close to the camera,
keep both index corners (top-left of each card) visible, and use even light without glare.

Keys in the camera window: `K` calibrate (show one card per prompt so it learns your deck),
`V` debug view (orange boxes + what each index was read as), `S` save the current frame to `captures/`,
`T` teach a single card, `H`/`B`/`D` set hole/board/dealer mode, `C` clear, `Q` quit.

Run `pytest -q`, `python -m gamble2 --demo`, `python -m gamble2 --source camera`.

## Reading cards from a video on your screen

    python -m gamble2 --source screen --region select

Drag a box around the video and press Enter. Then drag the Gamble2 window so it does not cover the
video (it would otherwise read itself). On macOS, allow Screen Recording for your terminal in
System Settings > Privacy & Security. Use `--monitor 2` for a second display. Make the video as large
as you can (full screen is best) and pause it on a frame where the card corners are sharp.

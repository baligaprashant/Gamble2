"""Build precomputed preflop equity tables (offline, once)."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

from gamble2.odds.evaluate import evaluate_seven
from gamble2.vision.cards import (
    Card,
    all_preflop_keys,
    canonical_preflop_key,
    full_deck,
)

# Default repo data dir
DEFAULT_OUT = Path(__file__).resolve().parents[3] / "data"
PACKAGE_OUT = Path(__file__).resolve().parents[1] / "data"


def _hole_combos_for_key(key: str) -> list[tuple[Card, Card]]:
    """Expand a 169-key into concrete suited/offsuit/pair combos."""
    deck = full_deck()
    combos: list[tuple[Card, Card]] = []
    if len(key) == 2:  # pair AA
        r = key[0]
        ranks = [c for c in deck if c.rank == r]
        for i in range(len(ranks)):
            for j in range(i + 1, len(ranks)):
                combos.append((ranks[i], ranks[j]))
        return combos

    r1, r2, flag = key[0], key[1], key[2]
    cards1 = [c for c in deck if c.rank == r1]
    cards2 = [c for c in deck if c.rank == r2]
    for a in cards1:
        for b in cards2:
            suited = a.suit == b.suit
            if flag == "s" and suited:
                combos.append((a, b))
            elif flag == "o" and not suited:
                combos.append((a, b))
    return combos


def _simulate_matchup(
    hero: tuple[Card, Card],
    dealer: tuple[Card, Card],
    rng: random.Random,
    boards: int,
) -> tuple[int, int, int]:
    dead = {hero[0], hero[1], dealer[0], dealer[1]}
    remaining = [c for c in full_deck() if c not in dead]
    wins = ties = losses = 0
    for _ in range(boards):
        board = rng.sample(remaining, 5)
        h = evaluate_seven(list(hero), board)
        d = evaluate_seven(list(dealer), board)
        if h < d:
            wins += 1
        elif h > d:
            losses += 1
        else:
            ties += 1
    return wins, ties, losses


def build_vs_random(
    sims_per_hand: int = 50_000,
    seed: int = 42,
    progress: bool = True,
) -> dict[str, dict[str, float | int]]:
    """Monte Carlo equity of each 169 hand vs a random dealer hand."""
    rng = random.Random(seed)
    keys = all_preflop_keys()
    out: dict[str, dict[str, float | int]] = {}
    t0 = time.time()

    for idx, key in enumerate(keys):
        hero_combos = _hole_combos_for_key(key)
        wins = ties = losses = 0
        per_combo = max(1, sims_per_hand // len(hero_combos))
        for hero in hero_combos:
            dead = set(hero)
            others = [c for c in full_deck() if c not in dead]
            for _ in range(per_combo):
                dealer = tuple(rng.sample(others, 2))
                w, t, l = _simulate_matchup(hero, dealer, rng, boards=1)
                wins += w
                ties += t
                losses += l
        total = wins + ties + losses
        out[key] = {
            "win": wins / total,
            "tie": ties / total,
            "lose": losses / total,
            "samples": total,
        }
        if progress and (idx + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  {idx + 1}/{len(keys)} hands ({elapsed:.1f}s)")

    return out


def build_matchups(
    boards_per_matchup: int = 200,
    seed: int = 7,
    progress: bool = True,
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Approximate 169×169 preflop matchup matrix."""
    rng = random.Random(seed)
    keys = all_preflop_keys()
    combo_cache = {k: _hole_combos_for_key(k) for k in keys}
    out: dict[str, dict[str, dict[str, float | int]]] = defaultdict(dict)
    t0 = time.time()
    total_pairs = len(keys) * len(keys)
    done = 0

    for hk in keys:
        for dk in keys:
            wins = ties = losses = 0
            h_combos = combo_cache[hk]
            d_combos = combo_cache[dk]
            for _ in range(boards_per_matchup):
                hero = rng.choice(h_combos)
                # Reject overlapping dealer combos
                for _try in range(40):
                    dealer = rng.choice(d_combos)
                    if set(hero).isdisjoint(dealer):
                        break
                else:
                    continue
                w, t, l = _simulate_matchup(hero, dealer, rng, boards=1)
                wins += w
                ties += t
                losses += l
            total = wins + ties + losses
            if total == 0:
                out[hk][dk] = {"win": 0.0, "tie": 0.0, "lose": 0.0, "samples": 0}
            else:
                out[hk][dk] = {
                    "win": wins / total,
                    "tie": ties / total,
                    "lose": losses / total,
                    "samples": total,
                }
            done += 1
            if progress and done % 500 == 0:
                print(f"  {done}/{total_pairs} matchups ({time.time() - t0:.1f}s)")

    return dict(out)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build preflop equity tables")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output directory for JSON tables",
    )
    parser.add_argument(
        "--sims",
        type=int,
        default=30_000,
        help="Approx Monte Carlo boards per 169-hand (vs random)",
    )
    parser.add_argument(
        "--matchup-boards",
        type=int,
        default=150,
        help="Boards per 169×169 cell (0 to skip matchup matrix)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--also-package",
        action="store_true",
        help="Also copy tables into src/gamble2/data for packaging",
    )
    args = parser.parse_args(argv)

    print("Building preflop vs random…")
    vs_random = build_vs_random(sims_per_hand=args.sims, seed=args.seed)
    _write_json(args.out / "preflop_vs_random.json", vs_random)

    if args.matchup_boards > 0:
        print("Building 169×169 matchup matrix…")
        matchups = build_matchups(
            boards_per_matchup=args.matchup_boards, seed=args.seed + 1
        )
        _write_json(args.out / "preflop_matchups.json", matchups)

    if args.also_package:
        PACKAGE_OUT.mkdir(parents=True, exist_ok=True)
        for name in ("preflop_vs_random.json", "preflop_matchups.json"):
            src = args.out / name
            if src.exists():
                (PACKAGE_OUT / name).write_text(src.read_text(encoding="utf-8"))
                print(f"Copied to {PACKAGE_OUT / name}")


if __name__ == "__main__":
    main()

"""CLI entrypoint for Gamble2."""

from __future__ import annotations

import argparse
import sys

from gamble2.capture.camera import CameraCapture
from gamble2.capture.demo import DemoCapture
from gamble2.capture.screen import (
    PERMISSION_HELP,
    Region,
    ScreenCapture,
    screen_capture_allowed,
    select_region,
    suggest_window_pos,
)
from gamble2.ui.overlay import OverlayApp
from gamble2.vision.template import TemplateDetector


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Live heads-up Hold'em equity: your 2 cards vs a dealer"
    )
    p.add_argument(
        "--source",
        choices=("camera", "screen"),
        default="camera",
        help="Frame source (ignored with --demo)",
    )
    p.add_argument("--device", type=int, default=0, help="Camera device index")
    p.add_argument(
        "--region",
        type=str,
        default=None,
        help="Screen region left,top,width,height - or the word \"select\" to drag a box around the video",
    )
    p.add_argument(
        "--monitor",
        type=int,
        default=1,
        help="mss monitor index when --region is omitted",
    )
    p.add_argument(
        "--ignore-permission-check",
        action="store_true",
        help="Skip the macOS Screen Recording permission check",
    )
    p.add_argument(
        "--demo",
        action="store_true",
        help="Open the overlay with two sample hole cards (no camera)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.demo:
        source = DemoCapture()
    elif args.source == "camera":
        try:
            source = CameraCapture(device=args.device)
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            print(
                "On macOS allow Camera for Terminal/Cursor, then:\n"
                "  python3 -m gamble2 --source camera",
                file=sys.stderr,
            )
            return 1
    else:
        if not args.ignore_permission_check and screen_capture_allowed(request=True) is False:
            print(PERMISSION_HELP, file=sys.stderr)
            try:
                import subprocess

                subprocess.run(
                    ["open", "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture"],
                    check=False,
                )
            except OSError:
                pass
            return 1
        if args.region and args.region.strip().lower() == "select":
            region = select_region(args.monitor)
            if region is None:
                print("No region chosen.", file=sys.stderr)
                return 1
            print(f"Capturing region {region.left},{region.top},{region.width},{region.height}")
        else:
            region = Region.parse(args.region) if args.region else None
        source = ScreenCapture(region=region, monitor=args.monitor)

    if args.demo:
        from gamble2.vision.demo_bank import demo_bank

        detector = TemplateDetector(bank=demo_bank())
    else:
        detector = TemplateDetector()
    print(f"Card reader ready: {detector.bank.count()} reference glyphs loaded.")
    print("Keys: K calibrate to your deck  H hole  B board  D dealer  T teach one card  V debug  S save frame  C clear  Q quit")
    window_pos = None
    if not args.demo and args.source == "screen" and isinstance(source, ScreenCapture) and source.region is not None:
        import mss

        with mss.mss() as sct:
            screen_w = sct.monitors[args.monitor]["width"]
        window_pos = suggest_window_pos(source.region, screen_w)
    app = OverlayApp(
        window_pos=window_pos,
        debug=(not args.demo and args.source == "screen"),
        read_frame=source.read,
        detect=detector.detect,
        source_name=source.name,
        detector=detector,
    )
    try:
        app.run()
    finally:
        source.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

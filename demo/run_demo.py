"""Drive the Morphos Ledger walkthrough with Playwright.

    python demo/run_demo.py                 # opens a visible browser and plays the whole walkthrough
    python demo/run_demo.py --pace 1.4      # slower, for a live audience
    python demo/run_demo.py --manual        # opens the page and starts the tour; you press the arrow keys
    python demo/run_demo.py --video         # also records demo/out/video/*.webm
    python demo/run_demo.py --url https://your-app.vercel.app   # demo the deployed site

The narration shown at the bottom of the page is the text to say aloud. This script prints the
same text in the terminal so you can read it from a second screen.
"""
from __future__ import annotations

import argparse
import functools
import http.server
import json
import sys
import threading
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "out"
WORDS_PER_SECOND = 2.5  # comfortable speaking pace


def serve(port: int) -> http.server.ThreadingHTTPServer:
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args, **kwargs):  # keep the terminal for the narration
            pass

    handler = functools.partial(Quiet, directory=str(ROOT / "public"))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", help="Use an already deployed site instead of starting a local server.")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--pace", type=float, default=1.0, help="Multiplier on the time spent per step (1.4 is slower).")
    ap.add_argument("--headless", action="store_true", help="No visible window (used for checks).")
    ap.add_argument("--manual", action="store_true", help="Start the tour, then wait for you to drive it.")
    ap.add_argument("--video", action="store_true", help="Record a video of the run.")
    ap.add_argument("--mobile", action="store_true", help="Use a 390 px wide phone viewport.")
    ap.add_argument("--no-shots", action="store_true", help="Do not save a screenshot per step.")
    args = ap.parse_args()

    srv = None if args.url else serve(args.port)
    url = args.url or f"http://127.0.0.1:{args.port}/"
    (OUT / "screens").mkdir(parents=True, exist_ok=True)
    viewport = {"width": 390, "height": 844} if args.mobile else {"width": 1440, "height": 900}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless, args=[] if args.headless else ["--window-size=1440,960"])
        ctx = browser.new_context(
            viewport=viewport,
            record_video_dir=str(OUT / "video") if args.video else None,
            record_video_size=viewport if args.video else None,
        )
        page = ctx.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.goto(url)
        page.wait_for_function("window.QN && window.QN.ready !== undefined", timeout=20000)
        if not page.evaluate("window.QN.ready"):
            print("The page could not load data/study.json:", page.evaluate("window.QN.error"), file=sys.stderr)
            return 1

        steps = page.evaluate("window.QN.tour.steps")
        print(f"\nMorphos Ledger walkthrough: {len(steps)} steps at {url}\n")
        page.click("#startTour")
        report = ["# Demo run\n"]
        for i, step in enumerate(steps):
            page.wait_for_selector("#tour .say p")
            say = page.inner_text("#tour .say p")
            print(f"[{i + 1}/{len(steps)}] {step['title']}\n    SAY: {say}\n")
            if step.get("terminal"):
                print("    TERMINAL: " + "  |  ".join(step["terminal"]) + "\n")
            if not args.no_shots:
                shot = OUT / "screens" / f"{i + 1:02d}-{step['id']}.png"
                page.wait_for_timeout(900)
                page.screenshot(path=str(shot))
                report.append(f"- step {i + 1}: {step['title']} ({shot.name})")
            if args.manual:
                continue
            action_extra = 2.6 if step.get("action") else 0.0
            time.sleep((len(say.split()) / WORDS_PER_SECOND + action_extra) * args.pace)
            if i == len(steps) - 1:
                page.click("#tNext")
            else:
                page.keyboard.press("ArrowRight")

        if args.manual:
            print("Manual mode: use the arrow keys or the buttons in the page. Close the window to finish.")
            try:
                page.wait_for_event("close", timeout=0)
            except Exception:
                pass
        else:
            page.wait_for_timeout(800)
        (OUT / "demo_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        ctx.close()
        browser.close()
    if srv:
        srv.shutdown()
    if errors:
        print("Browser errors:", errors, file=sys.stderr)
        return 1
    print(f"Done. Screenshots in {OUT / 'screens'}" + (f", video in {OUT / 'video'}" if args.video else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

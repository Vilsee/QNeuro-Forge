"""Capture the README screenshots from the running site with Playwright (JPEG, since *.png is git-ignored).
    python scripts/make_screenshots.py"""
from __future__ import annotations

import functools
import http.server
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets" / "screens"
OUT.mkdir(parents=True, exist_ok=True)


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a, **k):
        pass


def main() -> None:
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 8767), functools.partial(Quiet, directory=str(ROOT / "public")))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    q = dict(type="jpeg", quality=82)
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1440, "height": 900})
        pg.goto("http://127.0.0.1:8767/")
        pg.wait_for_function("window.QN && window.QN.ready === true")
        pg.screenshot(path=str(OUT / "01-answer.jpg"), **q)
        pg.click("#ledgerBody tr[data-id='E5']")
        pg.locator("#ledger").screenshot(path=str(OUT / "02-ledger.jpg"), **q)
        pg.locator("#reproduction").scroll_into_view_if_needed()
        pg.click("#holdBody button.wrong >> nth=0")
        pg.locator("#holdout").screenshot(path=str(OUT / "03-holdout-live.jpg"), **q)
        pg.locator("#fair").screenshot(path=str(OUT / "04-fair-comparison.jpg"), **q)
        pg.locator("#integrity").scroll_into_view_if_needed()
        pg.click("#btnCorrupt")
        pg.wait_for_selector("#verifyBand .band")
        pg.evaluate("document.querySelector('#verifyBand').scrollIntoView({block:'center'})")
        pg.screenshot(path=str(OUT / "05-tamper-detected.jpg"), **q)
        pg.click("#btnVerify")
        pg.wait_for_selector("#verifyBand .status-line")
        pg.evaluate("document.querySelector('#verifyBand').scrollIntoView({block:'center'})")
        pg.screenshot(path=str(OUT / "06-all-files-verified.jpg"), **q)
        pg.click("#startTour")
        for _ in range(4):
            pg.wait_for_timeout(300)
            pg.keyboard.press("ArrowRight")
        pg.wait_for_timeout(2800)
        pg.screenshot(path=str(OUT / "07-walkthrough.jpg"), **q)
        m = b.new_page(viewport={"width": 390, "height": 844})
        m.goto("http://127.0.0.1:8767/")
        m.wait_for_function("window.QN && window.QN.ready === true")
        m.screenshot(path=str(OUT / "08-phone.jpg"), **q)
        b.close()
    srv.shutdown()
    for f in sorted(OUT.glob("*.jpg")):
        print(f.name, f.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()

"""Visual check of the web client with real headless Chromium.

Pairs one browser profile with the running core (once; the session is reused),
then captures screenshots at several window sizes, in both themes and in the
main interaction states. Screenshots are written to the given output directory.

Usage: python scripts/ui_check.py OUTPUT_DIR [--url http://127.0.0.1:7777] [--only NAME]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jarvis.clients.cli.client import JarvisClient  # noqa: E402

SIZES = {
    "desktop-1440": (1440, 900),
    "desktop-1920": (1920, 1080),
    "laptop-1280": (1280, 720),
    "small-900": (900, 700),
    "phone-390": (390, 844),
}


def ensure_state(browser, url: str, state_file: Path) -> None:
    if state_file.exists():
        return
    code = JarvisClient(url).post("/auth/pairing-codes", {"kind": "web", "role": "owner", "max_risk": "DANGEROUS",
                                                          "name_hint": "UI check browser"})["code"]
    ctx = browser.new_context(bypass_csp=True)  # test-only: lets Playwright evaluate predicates
    page = ctx.new_page()
    page.goto(f"{url}/#pair={code}")
    page.wait_for_function("window.jarvis && window.jarvis.started", timeout=20000)
    ctx.storage_state(path=str(state_file))
    ctx.close()


def ready(page: Page) -> None:
    page.wait_for_function("window.jarvis && window.jarvis.started", timeout=20000)
    page.wait_for_timeout(700)


def set_presence(page: Page, facts: dict) -> None:
    page.evaluate("(f) => window.jarvis.store.update(f)", facts)
    page.wait_for_timeout(900)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--url", default="http://127.0.0.1:7777")
    ap.add_argument("--only")
    ap.add_argument("--conversation", help="conversation id to show in the thread shots")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {}
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        state = out / "storage-state.json"
        ensure_state(browser, args.url, state)
        convs = JarvisClient(args.url).get("/conversations?limit=5")
        conv_id = args.conversation or (convs[0]["id"] if convs else None)
        for theme in ("dark", "light"):
            for name, (w, h) in SIZES.items():
                if args.only and args.only not in name:
                    continue
                ctx = browser.new_context(storage_state=str(state), viewport={"width": w, "height": h}, color_scheme=theme,
                                          device_scale_factor=1, bypass_csp=True)
                page = ctx.new_page()
                errors: list[str] = []
                page.on("pageerror", lambda e, errs=errors: errs.append(str(e)))
                page.on("console", lambda m, errs=errors: errs.append(m.text) if m.type == "error" else None)
                page.add_init_script("sessionStorage.removeItem('jarvis.conversation')")
                page.goto(args.url + "/")
                ready(page)
                page.screenshot(path=str(out / f"{theme}-{name}-empty.png"))
                # composer with text
                page.fill("#input", "Compare the three best ultralight laptops under $1500 and save a summary")
                page.wait_for_timeout(250)
                page.screenshot(path=str(out / f"{theme}-{name}-typing.png"))
                page.fill("#input", "")
                if conv_id:
                    page.evaluate("(id) => window.jarvis.openConversation(id)", conv_id)
                    page.wait_for_timeout(600)
                    page.screenshot(path=str(out / f"{theme}-{name}-thread.png"))
                if name == "desktop-1440":
                    # Every orb state, large (empty state) and small (header), driven through the real state machine.
                    states = {
                        "idle": {},
                        "listening": {"micCapturing": True},
                        "thinking": {"micCapturing": False, "agentBusy": True},
                        "speaking": {"agentBusy": False, "streaming": True},
                        "success": {"streaming": False},
                        "waiting": {"streaming": False, "pendingApprovals": 1},
                        "error": {"pendingApprovals": 0, "error": "x"},
                        "offline": {"error": None, "coreReachable": False},
                    }
                    for mode in ("large", "header"):
                        page.evaluate("(m) => m === 'large' ? window.jarvis.newConversation() : null", mode)
                        if mode == "header" and conv_id:
                            page.evaluate("(id) => window.jarvis.openConversation(id)", conv_id)
                        page.wait_for_timeout(500)
                        sel = "#presence-large" if mode == "large" else ".header__presence"
                        for st, facts in states.items():
                            set_presence(page, facts)
                            if st == "success":
                                page.evaluate("() => window.jarvis.presences.forEach(p => p.acknowledge())")
                                page.wait_for_timeout(380)
                            if st in ("speaking", "listening"):
                                # exercise the level path the way real audio would
                                page.evaluate("() => { let n = 0; const t = setInterval(() => { window.jarvis.presenceLevel(0.55 + 0.35 * Math.sin(n++ / 2)); if (n > 40) clearInterval(t); }, 30); }")
                                page.wait_for_timeout(400)
                            box = page.locator(sel).bounding_box()
                            pad = 70 if mode == "large" else 16
                            page.screenshot(path=str(out / f"{theme}-face-{mode}-{st}.png"),
                                            clip={"x": max(0, box["x"] - pad - (0 if mode == "large" else 40)), "y": max(0, box["y"] - pad),
                                                  "width": box["width"] + 2 * pad + (0 if mode == "large" else 180), "height": box["height"] + 2 * pad})
                        set_presence(page, {"coreReachable": True, "error": None, "pendingApprovals": 0, "agentBusy": False,
                                            "micCapturing": False, "streaming": False})
                    page.click('.navitem[data-view="settings"]')
                    page.wait_for_timeout(700)
                    page.screenshot(path=str(out / f"{theme}-{name}-settings.png"))
                if name == "phone-390":
                    page.click("#show-sidebar")
                    page.wait_for_timeout(500)
                    page.screenshot(path=str(out / f"{theme}-{name}-drawer.png"))
                overflow = page.evaluate("document.documentElement.scrollWidth > window.innerWidth")
                report[f"{theme}-{name}"] = {"errors": errors[:5], "horizontal_overflow": overflow}
                ctx.close()
        browser.close()
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    t = time.time()
    rc = main()
    print(f"done in {time.time() - t:.1f}s")
    sys.exit(rc)

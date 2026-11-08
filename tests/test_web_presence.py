"""The presence engine and the face, in a real browser.

Most tests load the real modules on a small test page (served by Playwright
from the static folder), seed the presence's randomness, and use Playwright's
fake clock so minutes of behaviour run in seconds and every frame can be
inspected. The last tests run the full web app against a real core and drive
it with real core events.
"""

from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import Any

import pytest

sync_api = pytest.importorskip("playwright.sync_api")

STATIC = Path(__file__).resolve().parents[1] / "jarvis" / "clients" / "web" / "static"
ORIGIN = "http://jarvis.test"
FACE_PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="/static/css/tokens.css"><link rel="stylesheet" href="/static/css/presence.css">
<style>body{margin:0;height:100vh;display:grid;place-items:center} #face{--size:160px}
#btn{position:fixed;right:40px;bottom:40px} #field{position:fixed;left:40px;bottom:40px;width:300px}
.step{position:fixed;left:30px;top:30px;width:200px;height:20px}</style>
</head><body>
<jarvis-presence id="face"></jarvis-presence>
<button id="btn" type="button">A button</button><input id="field" placeholder="type">
<script type="module">
import { presence } from "/static/js/presence.js";
window.presence = presence;
window.face = document.getElementById("face");
window.facts = { coreReachable: true, link: "connected", error: null, pendingApprovals: 0, micCapturing: false,
  voiceListening: false, audioPlaying: false, streaming: false, agentBusy: false };
window.setFacts = (patch) => { Object.assign(window.facts, patch); presence.syncFacts(window.facts); };
presence.syncFacts(window.facts);
window.ready = true;
</script></body></html>"""

INSTRUMENT = """
(() => {
  const live = new Map();
  const tracked = (t) => t === window || t === document || t === document.documentElement || t instanceof MediaQueryList;
  const add = EventTarget.prototype.addEventListener, rem = EventTarget.prototype.removeEventListener;
  const key = (type, opts) => type + (typeof opts === "object" ? !!opts.capture : !!opts);
  EventTarget.prototype.addEventListener = function (type, fn, opts) {
    if (tracked(this)) { const s = live.get(key(type, opts)) || new Set(); s.add(fn); live.set(key(type, opts), s); }
    return add.call(this, type, fn, opts);
  };
  EventTarget.prototype.removeEventListener = function (type, fn, opts) {
    if (tracked(this)) live.get(key(type, opts))?.delete(fn);
    return rem.call(this, type, fn, opts);
  };
  window.__listeners = () => [...live.values()].reduce((n, s) => n + s.size, 0);
  window.__rafs = 0;
  const raf = window.requestAnimationFrame.bind(window);
  window.requestAnimationFrame = (cb) => { window.__rafs++; return raf(cb); };
})();
"""


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"Chromium is not available: {e}")
        yield b
        b.close()


def serve_static(route: Any) -> None:
    path = route.request.url.split(ORIGIN, 1)[1].split("?")[0]
    if path in ("/", "/face.html"):
        route.fulfill(status=200, content_type="text/html", body=FACE_PAGE)
        return
    file = (STATIC / path.removeprefix("/static/")).resolve()
    if STATIC not in file.parents or not file.is_file():
        route.fulfill(status=404, body="")
        return
    ctype = {".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml"}.get(file.suffix, "application/octet-stream")
    route.fulfill(status=200, content_type=ctype, body=file.read_bytes())


def open_face(browser: Any, *, reduced: bool = False, viewport: dict[str, int] | None = None, touch: bool = False,
              seed: int = 1234) -> Any:
    ctx = browser.new_context(viewport=viewport or {"width": 1000, "height": 700}, has_touch=touch, is_mobile=touch,
                              reduced_motion="reduce" if reduced else "no-preference")
    ctx.route(f"{ORIGIN}/**", serve_static)
    page = ctx.new_page()
    page.clock.install(time=0)
    page.add_init_script(f"window.__JARVIS_PRESENCE_SEED__ = {seed};")
    page.add_init_script(INSTRUMENT)  # after the fake clock, so it counts the frames the page really requests
    page.goto(f"{ORIGIN}/face.html")
    page.clock.run_for(200)
    page.wait_for_function("window.ready === true")
    page.clock.run_for(1500)
    return page


def record(page: Any, ms: int, expr: str = "face.debug()") -> list[Any]:
    """Advance the fake clock by ``ms``, recording ``expr`` on every animation frame."""
    page.evaluate(f"window.__rec = []; window.__recOn = true; (function f() {{ if (!window.__recOn) return; "
                  f"window.__rec.push({expr}); requestAnimationFrame(f); }})()")
    page.clock.run_for(ms)
    return page.evaluate("window.__recOn = false; window.__rec")


def dbg(page: Any) -> dict[str, Any]:
    return page.evaluate("presence.debug()")


def face(page: Any) -> dict[str, Any]:
    return page.evaluate("face.debug()")


def follow(page: Any) -> float:
    """How much the face is following the mouse right now (0..1)."""
    return page.evaluate("face.debug().follow")


# ------------------------------------------------------------- state
def test_primary_state_follows_real_facts_and_events(browser: Any) -> None:
    page = open_face(browser)
    assert dbg(page)["state"] == "idle"
    page.evaluate("setFacts({ agentBusy: true })")
    assert dbg(page)["state"] == "thinking"
    page.evaluate("presence.dispatch('TOOL_STARTED', { tool: 'calculate' })")
    assert dbg(page)["state"] == "tool_use"
    page.evaluate("presence.dispatch('TOOL_COMPLETED', { tool: 'calculate', ok: true })")
    assert dbg(page)["state"] == "thinking"
    page.evaluate("setFacts({ streaming: true })")
    assert dbg(page)["state"] == "responding"
    page.evaluate("setFacts({ audioPlaying: true })")
    assert dbg(page)["state"] == "speaking"
    page.evaluate("setFacts({ audioPlaying: false, streaming: false, agentBusy: false })")
    assert dbg(page)["state"] in ("idle", "attentive")
    page.evaluate("setFacts({ voiceListening: true })")
    assert dbg(page)["state"] == "listening"
    page.evaluate("setFacts({ voiceListening: false, pendingApprovals: 1 })")
    assert dbg(page)["state"] == "waiting"
    page.evaluate("setFacts({ pendingApprovals: 0, error: 'boom' })")
    assert dbg(page)["state"] == "error"
    page.evaluate("setFacts({ link: 'disconnected' })")
    assert dbg(page)["state"] == "offline"
    page.clock.run_for(50)
    assert page.get_attribute("#face", "aria-label") == "JARVIS, offline", "state is always available in words"


def test_typing_makes_jarvis_attentive_then_it_relaxes(browser: Any) -> None:
    page = open_face(browser)
    page.focus("#field")
    page.keyboard.type("hello")
    assert dbg(page)["state"] == "attentive"
    page.clock.run_for(12_000)
    assert dbg(page)["state"] == "idle"


# ------------------------------------------------------------- transitions
def beats(page: Any) -> list[str]:
    return [h["what"] for h in page.evaluate("presence.history") if h["kind"] == "beat"]


def test_transitions_are_intentional(browser: Any) -> None:
    page = open_face(browser)
    page.evaluate("presence.history.length = 0; setFacts({ voiceListening: true })")
    assert "noticed" in beats(page), "idle -> listening: alert at once"
    page.evaluate("presence.history.length = 0; setFacts({ voiceListening: false, agentBusy: true })")
    assert beats(page) == ["heard"], "listening -> thinking: 'I heard you'"
    page.clock.run_for(2000)
    page.evaluate("presence.history.length = 0; presence.dispatch('MODEL_RESPONSE_STARTED'); presence.emit({ type: 'agent.output.started' });"
                  "setFacts({ streaming: true })")
    assert beats(page) == ["ready"], "thinking -> responding: 'I have an answer', exactly once"
    page.evaluate("setFacts({ streaming: false, agentBusy: false, error: 'tool exploded' })")
    page.clock.run_for(1500)
    assert dbg(page)["emotion"] == "concerned"
    page.evaluate("presence.history.length = 0; setFacts({ error: null })")
    assert "recovered" in beats(page), "error -> recovered: visibly relaxes"
    page.clock.run_for(5000)
    assert dbg(page)["emotion"] == "neutral"
    page.evaluate("setFacts({ link: 'disconnected' })")
    page.clock.run_for(800)
    page.evaluate("presence.history.length = 0; setFacts({ link: 'connected' })")
    assert "wake" in beats(page), "connection restored: wakes up"


def test_interruption_is_obvious_and_immediate(browser: Any) -> None:
    page = open_face(browser)
    page.evaluate("setFacts({ audioPlaying: true }); window.__m = setInterval(() => presence.level('speaker', 0.7), 30)")
    page.clock.run_for(600)
    assert face(page)["mouth"] > 0.3
    page.evaluate("clearInterval(window.__m); presence.dispatch('SPEECH_INTERRUPTED', {}); setFacts({ audioPlaying: false, voiceListening: true })")
    frames = record(page, 300)
    assert frames[1]["mouth"] == 0, "the mouth stops at once (by the next frame)"
    assert all(f["mouth"] == 0 for f in frames[1:])
    d = dbg(page)
    assert d["state"] == "listening" and "interrupted" in d["beats"] and d["attention"] == "interrupt"
    peak = max(f["L"]["eyeH"] for f in frames)
    assert peak > 1.2, "eyes widen: it noticed being interrupted"


# ------------------------------------------------------------- attention
def test_eyes_follow_the_mouse_subtly_and_more_when_it_is_near(browser: Any) -> None:
    page = open_face(browser)  # the face: 160 px, centred at (500, 350)
    page.mouse.move(40, 350, steps=8)  # far away on the left
    page.clock.run_for(800)
    far_x, far_follow = face(page)["eyes"][0]["x"], follow(page)
    assert -0.45 < far_x < -0.05 and 0.1 < far_follow < 0.4, "far away: a subtle look, never a stare across the screen"
    page.mouse.move(380, 350, steps=8)  # close, on the left of the face
    page.clock.run_for(800)
    near_x, near_follow = face(page)["eyes"][0]["x"], follow(page)
    assert near_follow > far_follow + 0.4 and near_x < -0.2, "near the face: the eyes follow it properly"
    page.clock.run_for(4500)  # the mouse rests: the eyes ease back to the person (no staring)
    assert follow(page) == 0 and abs(face(page)["eyes"][0]["x"]) < 0.2
    page.mouse.move(400, 300, steps=4)
    page.evaluate("setFacts({ agentBusy: true })")  # thinking: attention turns inward, the mouse doesn't pull it out
    page.clock.run_for(1200)
    d = face(page)
    assert follow(page) == 0 and d["gaze"] == "state" and d["eyes"][0]["y"] < -0.2


def test_following_is_off_while_typing_and_on_touch_screens(browser: Any) -> None:
    page = open_face(browser)
    page.focus("#field")
    page.keyboard.type("hello")
    page.mouse.move(420, 350, steps=6)
    page.clock.run_for(300)
    assert follow(page) == 0, "while the person types, the mouse doesn't matter"
    touch = open_face(browser, viewport={"width": 375, "height": 740}, touch=True)
    touch.touchscreen.tap(200, 380)
    touch.clock.run_for(300)
    assert touch.evaluate("face.debug().follow") == 0, "a tap is a look, not something to follow"


def test_high_priority_events_win(browser: Any) -> None:
    page = open_face(browser)
    page.mouse.move(900, 100, steps=6)
    page.evaluate("presence.dispatch('WAKE_WORD_DETECTED', { score: 0.9 })")
    page.clock.run_for(100)
    assert face(page)["gaze"] == "wake"
    page.clock.run_for(2000)
    page.evaluate("const s = document.createElement('div'); s.className = 'step';"
                  "s.dataset.status = 'failed'; document.body.appendChild(s); presence.dispatch('TOOL_FAILED', { tool: 'x', ok: false })")
    page.clock.run_for(200)
    d = face(page)
    assert d["gaze"] == "concern" and d["eyes"][0]["x"] < -0.3 and d["eyes"][0]["y"] < -0.2, "a failure draws attention to it"
    page.clock.run_for(600)
    assert dbg(page)["emotion"] == "concerned"


# ------------------------------------------------------------- emotion
def test_emotion_comes_from_real_signals_and_fades(browser: Any) -> None:
    page = open_face(browser)
    page.evaluate("presence.emit({ type: 'user.submitted', text: 'Why is the sky blue?' })")
    page.clock.run_for(600)
    assert dbg(page)["emotion"] == "curious"
    page.clock.run_for(6000)
    page.evaluate("presence.emit({ type: 'agent.reply.completed', text: \"I'm not sure, I couldn't find that.\" })")
    page.clock.run_for(600)
    assert dbg(page)["emotion"] == "uncertain"
    page.clock.run_for(8000)
    assert dbg(page)["emotion"] == "neutral", "it does not stay emotional"
    page.evaluate("presence.dispatch('TOOL_COMPLETED', { tool: 'x', ok: true })")
    page.clock.run_for(400)
    assert dbg(page)["emotions"]["satisfied"] > 0.15
    peak = max(f["L"]["lidBot"] for f in record(page, 700))
    assert 0.04 < peak < 0.45, "satisfaction shows, short of a full happy squint (an idle micro-smile may add to it)"


def test_a_warm_word_earns_happy_eyes_and_a_small_hop(browser: Any) -> None:
    page = open_face(browser)
    page.evaluate("presence.history.length = 0; presence.emit({ type: 'user.submitted', text: 'What time is it?' })")
    page.clock.run_for(1500)
    assert "cheer" not in beats(page), "an ordinary question is not a reason to be happy"
    page.clock.run_for(8000)
    page.evaluate("presence.history.length = 0; presence.emit({ type: 'user.submitted', text: 'Thanks, that was perfect' })")
    frames = record(page, 1400)
    assert "cheer" in beats(page)
    assert max(f["L"]["lidBot"] for f in frames) > 0.35, "the eyes squint into happy crescents"
    assert min(f["body"]["y"] for f in frames) < -1.0, "and it hops (a small one)"
    page.clock.run_for(3000)
    d = face(page)
    assert d["L"]["lidBot"] < 0.12 and abs(d["body"]["y"]) < 0.6, "then it settles"


def test_bad_news_ends_a_happy_moment_at_once(browser: Any) -> None:
    page = open_face(browser)
    page.evaluate("presence.emit({ type: 'user.submitted', text: 'Hello!' }); setFacts({ agentBusy: true })")
    page.clock.run_for(150)
    assert "cheer" in dbg(page)["beats"]
    page.evaluate("setFacts({ agentBusy: false, error: 'no model' })")
    page.clock.run_for(50)
    assert "cheer" not in dbg(page)["beats"], "no smiling through an error"
    page.clock.run_for(400)
    d = face(page)
    assert max(e["lidBot"] for e in d["drawn"]) < 0.1 and d["L"]["lidAngle"] < -5, "the eyes turn worried instead"


def test_an_error_shows_briefly_then_the_face_returns_to_what_is_going_on(browser: Any) -> None:
    page = open_face(browser)
    page.evaluate("setFacts({ error: 'no model' })")
    assert dbg(page)["state"] == "error"
    page.clock.run_for(4200)
    assert dbg(page)["state"] in ("idle", "attentive"), "not stuck looking upset"
    page.evaluate("presence.history.length = 0; setFacts({ audioPlaying: true })")
    assert dbg(page)["state"] == "speaking", "and it follows what happens next"
    page.evaluate("setFacts({ audioPlaying: false, error: null })")
    assert "recovered" not in beats(page), "the old error being cleared later is nothing to react to"
    page.evaluate("setFacts({ error: 'again' })")
    page.clock.run_for(1000)
    page.evaluate("setFacts({ audioPlaying: true })")
    assert dbg(page)["state"] == "speaking", "JARVIS talking takes over from an error at once"
    page.evaluate("setFacts({ audioPlaying: false })")
    page.evaluate("setFacts({ error: null }); setFacts({ audioPlaying: true }); setFacts({ error: 'spoken' })")
    page.clock.run_for(400)
    assert dbg(page)["state"] == "speaking" and face(page)["L"]["lidAngle"] < -5, "saying the bad news, it looks concerned"
    page.evaluate("setFacts({ audioPlaying: false, error: null }); setFacts({ error: 'again' })")
    page.clock.run_for(1000)
    page.evaluate("presence.history.length = 0; setFacts({ error: null })")
    assert "recovered" in beats(page), "fixed while it still showed: a visible relief"


def test_expression_changes_ease_rather_than_jump(browser: Any) -> None:
    page = open_face(browser)
    page.clock.run_for(3000)
    page.evaluate("setFacts({ error: 'boom' })")  # the largest pose change: lids slant, the eyes narrow
    frames = record(page, 600, "face.debug().drawn[0]")
    angles = [f["lidAngle"] for f in frames]
    assert angles[-1] < -9, "it gets there"
    assert max(abs(b - a) for a, b in zip(angles, angles[1:])) < 4, "over several frames, not in one"


def test_something_unclear_looks_puzzled_then_passes(browser: Any) -> None:
    page = open_face(browser)
    page.evaluate("presence.history.length = 0; presence.emit({ type: 'voice.unclear' })")
    frames = record(page, 1000)
    assert "puzzled" in beats(page)
    assert max(abs(f["body"]["rot"]) for f in frames) > 2.5, "the head cocks"
    assert max(abs(f["L"]["lidTop"] - f["R"]["lidTop"]) for f in frames) > 0.1, "one eye narrows"
    page.clock.run_for(3000)
    assert abs(face(page)["body"]["rot"]) < 0.8, "and it passes"


def test_the_float_is_composited_paused_asleep_and_off_for_reduced_motion(browser: Any) -> None:
    style = "getComputedStyle(face.querySelector('.face__float'))"
    page = open_face(browser)
    assert page.evaluate(f"{style}.animationName") == "face-float"
    assert page.evaluate(f"{style}.animationPlayState") == "running"
    assert page.evaluate("getComputedStyle(face.querySelector('.face__mouth')).opacity") == "0", "no mouth at rest"
    page.evaluate("setFacts({ link: 'disconnected' })")
    page.clock.run_for(100)
    assert page.evaluate(f"{style}.animationPlayState") == "paused", "asleep: no float"
    reduced = open_face(browser, reduced=True)
    assert reduced.evaluate(f"{style}.animationName") == "none"


def test_behaviour_is_reproducible_with_a_seed(browser: Any) -> None:
    def run(seed: int) -> list[tuple[str, str]]:
        page = open_face(browser, seed=seed)
        page.clock.run_for(60_000)
        hist = page.evaluate("presence.history")
        # The page loads in real time, so compare the first 55 s after the engine started (not up to an
        # arbitrary edge where one blink more or less depends on how fast the page loaded).
        return [(h["kind"], str(h.get("what", ""))) for h in hist if h["at"] - hist[0]["at"] < 55_000]
    a, b, c = run(7), run(7), run(8)
    assert a == b, "same seed, same behaviour"
    assert a != c


# ------------------------------------------------------------- idle
def test_idle_is_alive_but_never_loops(browser: Any) -> None:
    page = open_face(browser)
    page.mouse.move(500, 690)
    page.clock.run_for(5000)
    page.evaluate("presence.history.length = 0")
    hist: list[dict[str, Any]] = []
    for _ in range(30):  # five minutes, nobody touching anything
        page.clock.run_for(10_000)
        hist += page.evaluate("presence.history.splice(0)")
    blinks = [h for h in hist if h["kind"] == "blink"]
    beats_ = [h for h in hist if h["kind"] == "idle"]
    gaps = [b["at"] - a["at"] for a, b in zip(blinks, blinks[1:])]
    assert 20 <= len(blinks) <= 150
    assert statistics.pstdev(gaps) / statistics.mean(gaps) > 0.25, "blinks are not on a timer"
    assert len({round(g / 10) for g in gaps}) > len(gaps) * 0.8
    assert any(b["double"] for b in blinks) and any(b["asym"] for b in blinks), "double and asymmetric blinks happen"
    kinds = [b["what"] for b in beats_]
    assert "still" in kinds and len(set(kinds)) >= 4, f"varied, including stillness: {sorted(set(kinds))}"
    gaps2 = [b["at"] - a["at"] for a, b in zip(beats_, beats_[1:])]
    assert statistics.pstdev(gaps2) / statistics.mean(gaps2) > 0.3
    assert not any(a == b == c and a != "still" for a, b, c in zip(kinds, kinds[1:], kinds[2:])), "never the same thing three times running"


def test_blink_is_quick_and_natural(browser: Any) -> None:
    page = open_face(browser)
    page.clock.run_for(2000)
    h0 = page.evaluate("face.querySelector('.face__eye').getBBox().height")
    page.evaluate("presence.blink.at = -1e9; presence.blinkNow(true)")
    hs = record(page, 300, "face.querySelector('.face__eye').getBBox().height")
    closed = min(hs)
    assert closed < 0.3 * h0 and 1 <= hs.index(closed) <= 8
    assert hs[-1] > 0.95 * h0


# ------------------------------------------------------------- audio
def test_mouth_follows_real_speech_audio_believably(browser: Any) -> None:
    page = open_face(browser)
    page.evaluate("setFacts({ streaming: true })")
    page.clock.run_for(800)
    assert face(page)["mouth"] == 0, "text arriving without audio: the mouth rests"
    page.evaluate("window.__m = setInterval(() => presence.level('mic', 0.9), 30)")
    page.clock.run_for(500)
    assert face(page)["mouth"] == 0, "the person's voice never moves JARVIS's mouth"
    page.evaluate("clearInterval(window.__m); setFacts({ audioPlaying: true }); window.__m = setInterval(() => presence.level('speaker', 1.0), 30)")
    page.clock.run_for(500)
    loud = face(page)["mouth"]
    assert 0.5 < loud <= 0.86, "loud audio opens the mouth, within a believable range"
    # A natural pause inside speech closes the mouth; speech resuming opens it again.
    page.evaluate("clearInterval(window.__m); window.__m = setInterval(() => presence.level('speaker', 0.0), 30)")
    page.clock.run_for(300)
    assert face(page)["mouth"] < 0.05
    page.evaluate("clearInterval(window.__m); window.__m = setInterval(() => presence.level('speaker', 0.4), 30)")
    page.clock.run_for(300)
    assert 0.2 < face(page)["mouth"] < loud
    # The audio stream stops arriving: the mouth must not stay open.
    page.evaluate("clearInterval(window.__m)")
    page.clock.run_for(400)
    assert face(page)["mouth"] < 0.02


def test_listening_reacts_to_the_persons_voice_but_not_jarvis(browser: Any) -> None:
    page = open_face(browser)
    page.evaluate("setFacts({ voiceListening: true })")
    page.clock.run_for(1500)

    def eye_h(level: float) -> float:
        page.evaluate(f"window.__m = setInterval(() => presence.level('mic', {level}), 30)")
        page.clock.run_for(500)
        h = face(page)["L"]["eyeH"]
        page.evaluate("clearInterval(window.__m)")
        page.clock.run_for(800)
        return h
    quiet, normal, loud = eye_h(0.1), eye_h(0.4), eye_h(1.0)
    assert quiet < normal < loud < quiet + 0.2, "stronger voice, stronger but bounded reaction"
    # While JARVIS itself is speaking, microphone levels are not the person.
    page.evaluate("setFacts({ voiceListening: false, audioPlaying: true }); window.__m = setInterval(() => presence.level('mic', 0.9), 30)")
    page.clock.run_for(400)
    assert not page.evaluate("presence.audio.voiced") and page.evaluate("presence.audio.voice.value") == 0
    page.evaluate("clearInterval(window.__m)")


def test_tts_start_interrupt_start_again_leaves_nothing_stale(browser: Any) -> None:
    page = open_face(browser)
    for _ in range(3):
        page.evaluate("setFacts({ audioPlaying: true }); presence.dispatch('SPEECH_STARTED'); window.__m = setInterval(() => presence.level('speaker', 0.6), 30)")
        page.clock.run_for(400)
        assert face(page)["mouth"] > 0.2
        page.evaluate("clearInterval(window.__m); presence.dispatch('SPEECH_INTERRUPTED'); setFacts({ audioPlaying: false })")
        page.clock.run_for(50)
        assert face(page)["mouth"] == 0
    page.clock.run_for(4000)
    d = dbg(page)
    assert d["state"] in ("idle", "attentive") and d["beats"] == [] and d["mouth"] == 0 and d["attention"] in ("person", "idle")


# ------------------------------------------------------------- coherence
def test_rapid_interaction_stays_coherent(browser: Any) -> None:
    page = open_face(browser)
    page.evaluate("""() => {
      presence.emit({ type: 'user.submitted', text: 'search for news' }); setFacts({ agentBusy: true });
      presence.dispatch('TOOL_STARTED', { tool: 'web_search' });
      setFacts({ audioPlaying: true }); presence.dispatch('SPEECH_STARTED'); presence.level('speaker', 0.8);
      presence.dispatch('SPEECH_INTERRUPTED'); setFacts({ audioPlaying: false, voiceListening: true });
      presence.dispatch('USER_SPEECH_STARTED'); presence.level('mic', 0.6); presence.dispatch('USER_SPEECH_ENDED');
      setFacts({ voiceListening: false }); presence.emit({ type: 'ui.panel.opened', element: document.getElementById('btn') });
      presence.dispatch('TOOL_COMPLETED', { tool: 'web_search', ok: true });
      presence.emit({ type: 'user.submitted', text: 'and again?' }); presence.dispatch('TOOL_STARTED', { tool: 'calculate' });
      presence.emit({ type: 'user.interrupted' }); presence.dispatch('TOOL_FAILED', { tool: 'calculate', ok: false });
      setFacts({ agentBusy: false });
    }""")
    page.clock.run_for(9000)
    d = dbg(page)
    assert d["state"] in ("idle", "attentive"), d
    assert d["beats"] == [] and d["mouth"] == 0 and d["attention"] in ("person", "idle")
    assert page.evaluate("presence.x.toolsRunning") == 0
    assert d["emotion"] in ("neutral", "concerned")
    page.clock.run_for(6000)
    assert dbg(page)["emotion"] == "neutral", "nothing stays stuck"


# ------------------------------------------------------------- motion
def test_eyes_move_smoothly_with_limits(browser: Any) -> None:
    page = open_face(browser)
    page.mouse.move(500, 350, steps=4)
    page.clock.run_for(600)
    page.mouse.move(330, 350, steps=2)  # the mouse jumps to the left of the face
    xs = [f["eyes"][0]["x"] for f in record(page, 1000)]
    assert xs[-1] < -0.3
    assert all(b <= a + 1e-6 for a, b in zip(xs, xs[1:])), "no overshoot, no jitter"
    assert max(abs(b - a) for a, b in zip(xs, xs[1:])) < 0.1, "no snapping"
    for x, y in [(0, 0), (999, 0), (0, 699), (999, 699)]:
        page.mouse.move(x, y, steps=3)
        page.clock.run_for(900)
        for e in face(page)["eyes"]:
            assert math.hypot(e["x"], e["y"] * 1.1) <= 1.0001
        for x0, y0, w, h in page.evaluate("[...face.querySelectorAll('.face__eye')].map((p) => { const b = p.getBBox(); return [b.x, b.y, b.width, b.height]; })"):
            assert all(math.hypot(px, py) < 44 for px, py in [(x0, y0), (x0 + w, y0), (x0, y0 + h), (x0 + w, y0 + h)])


def test_the_face_has_physical_weight(browser: Any) -> None:
    page = open_face(browser)
    page.evaluate("presence.dispatch('WAKE_WORD_DETECTED', {})")
    frames = record(page, 1500)
    scales = [f["body"]["scale"] for f in frames]
    assert max(scales) > 1.01, "it rises to attention"
    assert abs(scales[-1] - 1) < 0.003, "and settles back"
    assert max(abs(f["body"]["rot"]) for f in frames) < 4 and max(abs(f["body"]["x"]) for f in frames) < 3, "restrained"


def test_reduced_motion_keeps_personality_with_less_movement(browser: Any) -> None:
    page = open_face(browser, reduced=True)
    page.mouse.move(420, 340, steps=10)
    page.clock.run_for(2000)
    assert follow(page) == 0, "no following the mouse"
    page.evaluate("setFacts({ agentBusy: true })")
    page.clock.run_for(1500)
    d = face(page)
    assert 0.08 < abs(d["eyes"][0]["y"]) < 0.32, "thinking still looks away, with much less movement"
    assert d["L"]["lidTop"] >= 0.08 and min(d["L"]["eyeH"], d["R"]["eyeH"]) < 0.92, "expression changes remain"
    page.evaluate("presence.history.length = 0")
    page.clock.run_for(60_000)
    idle = {h["what"] for h in page.evaluate("presence.history") if h["kind"] == "idle"}
    assert idle <= {"still", "micro"}, "no drifting about"
    page.evaluate("setFacts({ agentBusy: false, audioPlaying: true }); window.__m = setInterval(() => presence.level('speaker', 0.6), 30)")
    page.clock.run_for(400)
    assert face(page)["mouth"] > 0.1, "the mouth still follows speech"


def test_touch_devices_look_where_you_tap(browser: Any) -> None:
    page = open_face(browser, viewport={"width": 375, "height": 740}, touch=True)
    hits = 0
    for _ in range(5):
        page.touchscreen.tap(30, 700)
        page.clock.run_for(250)
        e = face(page)["eyes"][0]
        hits += e["x"] < -0.2 and e["y"] > 0.3
        page.clock.run_for(2000)
    assert hits >= 3


# ------------------------------------------------------------- performance
def test_no_listener_animation_or_loop_leaks(browser: Any) -> None:
    page = open_face(browser)
    base = page.evaluate("window.__listeners()")
    page.evaluate("""() => {
      for (let i = 0; i < 60; i++) { const f = document.createElement('jarvis-presence'); document.body.appendChild(f); f.remove(); }
      face.remove();
    }""")
    assert page.evaluate("presence.listenerCount") == 0, "the last face going away removes every listener"
    page.clock.run_for(2000)
    page.evaluate("window.__rafs = 0")
    page.clock.run_for(3000)
    assert page.evaluate("window.__rafs") == 0, "no face, no animation"
    page.evaluate("document.body.prepend(face)")
    page.clock.run_for(500)
    assert page.evaluate("window.__listeners()") == base
    # One loop for the page, however often it is woken.
    page.evaluate("window.__rafs = 0; for (let i = 0; i < 50; i++) presence.wake()")
    page.clock.run_for(16 * 30)
    assert page.evaluate("window.__rafs") <= 32, "waking repeatedly never starts a second loop"
    page.evaluate("setFacts({ link: 'disconnected' })")
    page.clock.run_for(3000)
    page.evaluate("window.__rafs = 0")
    page.clock.run_for(5000)
    assert page.evaluate("window.__rafs") == 0, "still when there is nothing to do"


def test_idle_animation_is_cheap_and_touches_only_the_face(browser: Any) -> None:
    page = open_face(browser)
    page.mouse.move(500, 690)
    page.clock.run_for(5000)
    page.evaluate("""window.__outside = 0; new MutationObserver((rs) => { for (const r of rs) if (!r.target.closest?.('jarvis-presence')) window.__outside++; })
      .observe(document.body, { subtree: true, attributes: true, childList: true, characterData: true }); window.__rafs = 0""")
    page.clock.run_for(60_000)
    assert page.evaluate("window.__outside") == 0
    frames = page.evaluate("window.__rafs")
    assert frames < 0.6 * 60_000 / 16, f"mostly still: {frames} frames in an idle minute"


# ------------------------------------------------------- the real web app
@pytest.fixture
def app_page(browser: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tests.test_cli_auth import LiveCore, _free_port

    monkeypatch.setenv("JARVIS_SECRETS_BACKEND", "file")
    core = LiveCore(tmp_path / "server", _free_port())
    ctx = browser.new_context(viewport={"width": 1280, "height": 800}, bypass_csp=True)  # test predicates need eval
    page = ctx.new_page()
    page.goto(f"{core.url}/#pair={core.pairing_code()}")
    page.wait_for_function("window.jarvis?.started === true && window.jarvis.events.state === 'connected'", timeout=20000)
    page.evaluate("import('/static/js/presence/engine.js').then((m) => { window.presence = m.presence; })")
    page.wait_for_function("window.presence")
    yield core, page
    ctx.close()
    core.stop()


def emit(core: Any, type_: str, data: dict[str, Any]) -> None:
    core.rt.events.emit(type_, data, source="test")


def test_app_presence_reacts_to_real_core_events(app_page: Any) -> None:
    core, page = app_page
    size = lambda: page.evaluate("document.getElementById('presence').getBoundingClientRect().width")  # noqa: E731
    assert size() > 140, "with no conversation, JARVIS is the centre of the screen"
    emit(core, "AGENT_STATE", {"state": "thinking", "conversation_id": "c1"})
    page.wait_for_function("presence.state === 'thinking'")
    emit(core, "AGENT_STATE", {"state": "acting", "conversation_id": "c1", "tools": ["calculate"]})
    emit(core, "TOOL_STARTED", {"tool": "calculate"})
    page.wait_for_function("presence.state === 'tool_use'")
    emit(core, "TOOL_COMPLETED", {"tool": "calculate", "ok": True})
    page.wait_for_function("presence.emotion.values.satisfied > 0.05")
    emit(core, "MODEL_RESPONSE_STARTED", {})
    page.wait_for_function("presence.history.some((h) => h.what === 'ready')")
    emit(core, "AGENT_STATE", {"state": "idle", "conversation_id": "c1"})
    page.wait_for_function("presence.state === 'idle' || presence.state === 'attentive'")
    emit(core, "WAKE_WORD_DETECTED", {"score": 0.9, "interaction": "vi_test"})
    page.wait_for_function("presence.debug().attention === 'wake'")
    emit(core, "SPEECH_STARTED", {"interaction": "vi_test"})
    for _ in range(10):
        emit(core, "AUDIO_LEVEL", {"level": 0.7, "source": "speaker"})
        page.wait_for_timeout(50)
    page.wait_for_function("document.getElementById('presence').debug().mouth > 0.3")
    emit(core, "SPEECH_INTERRUPTED", {"interaction": "vi_test"})
    page.wait_for_function("document.getElementById('presence').debug().mouth === 0", timeout=2000)
    assert "interrupted" in page.evaluate("presence.debug().beats")
    # A conversation: the presence stays, docked, and yields room to it.
    page.evaluate("jarvis.conversation.addUser('hello'); jarvis.conversation._sync()")
    page.wait_for_timeout(900)
    assert 60 <= size() <= 100, "during a conversation JARVIS stays present, not a tiny icon"


def test_app_reconnect_never_duplicates_the_event_stream(app_page: Any) -> None:
    core, page = app_page
    base = core.rt.events.subscriber_count
    listeners = page.evaluate("presence.listenerCount")
    page.evaluate("Promise.all([jarvis.events.connect(), jarvis.events.connect(), jarvis.events.connect()])")
    page.wait_for_timeout(500)
    assert core.rt.events.subscriber_count == base, "connecting again while connected opens nothing new"
    for _ in range(3):
        page.evaluate("jarvis.events._ws.close()")
        page.wait_for_function("jarvis.events.state === 'connected'", timeout=20000)
    page.evaluate("jarvis.events.connect(); jarvis.events.connect()")
    page.wait_for_timeout(800)
    assert core.rt.events.subscriber_count == base, "each reconnect replaces the old subscription"
    page.evaluate("""window.__probe = 0; jarvis.events.addEventListener('event', (e) => {
      if (e.detail.type === 'NOTIFICATION' && e.detail.data.title === 'probe') window.__probe++; })""")
    emit(core, "NOTIFICATION", {"title": "probe", "body": "", "level": "info"})
    page.wait_for_function("window.__probe >= 1")
    page.wait_for_timeout(700)
    assert page.evaluate("window.__probe") == 1, "every event arrives exactly once"
    assert page.evaluate("presence.listenerCount") == listeners, "reconnecting adds no presence listeners"


# ================================================================ Presence 2.0
def history(page: Any, kind: str | None = None) -> list[dict[str, Any]]:
    h = page.evaluate("presence.history")
    return [x for x in h if kind is None or x["kind"] == kind]


def speak(page: Any, ms: int) -> None:
    page.evaluate("window.__m = setInterval(() => presence.level('speaker', 0.5 + 0.3 * Math.sin(performance.now() / 60)), 30)")
    page.clock.run_for(ms)
    page.evaluate("clearInterval(window.__m)")


def test_presence_timeline_is_coherent(browser: Any) -> None:
    """One exchange, start to finish: every change follows from a real event, in order."""
    page = open_face(browser)
    page.evaluate("presence.history.length = 0")
    page.focus("#field")
    page.keyboard.type("search for")
    page.clock.run_for(600)
    page.evaluate("presence.emit({ type: 'user.submitted', text: 'search for the weather' }); setFacts({ agentBusy: true })")
    page.clock.run_for(1200)
    page.evaluate("presence.dispatch('AGENT_STATE', { state: 'acting', tools: ['web_search'] }); presence.dispatch('TOOL_STARTED', { tool: 'web_search' })")
    page.clock.run_for(2000)
    page.evaluate("presence.dispatch('TOOL_COMPLETED', { tool: 'web_search', ok: true })")
    page.clock.run_for(900)
    page.evaluate("presence.dispatch('MODEL_RESPONSE_STARTED', {}); setFacts({ streaming: true })")
    page.clock.run_for(500)
    page.evaluate("setFacts({ audioPlaying: true }); presence.dispatch('SPEECH_STARTED', {})")
    speak(page, 1500)
    page.evaluate("presence.dispatch('SPEECH_ENDED', {}); setFacts({ audioPlaying: false, streaming: false, agentBusy: false });"
                  "presence.emit({ type: 'agent.reply.completed', text: 'It is sunny.' })")
    page.clock.run_for(3000)
    states = [h["to"] for h in history(page, "state")]
    assert states[:6] == ["attentive", "thinking", "tool_use", "thinking", "responding", "speaking"], states
    assert states[6] in ("attentive", "idle") and len(states) <= 8, states
    phases = [h["what"] for h in history(page, "phase")]
    assert phases == ["heard", "thinking", "working", "ready", "responding", "speaking", "settling", "done"], phases
    beats_ = [h["what"] for h in history(page, "beat")]
    order = [beats_.index(b) for b in ("heard", "ready", "settle")]
    assert order == sorted(order), beats_
    assert "interrupted" not in beats_ and not {"error", "waiting", "listening"} & set(states)


def test_thinking_holds_attention_against_the_pointer(browser: Any) -> None:
    page = open_face(browser, seed=5)
    page.evaluate("setFacts({ agentBusy: true })")
    page.clock.run_for(1500)
    base = page.evaluate("face.debug().eyes[0]")
    kinds = set()
    for i in range(12):  # the pointer sweeps around while JARVIS thinks
        page.mouse.move(40 + (i % 2) * 900, 60 + (i % 3) * 280, steps=6)
        page.clock.run_for(250)
        kinds.add(page.evaluate("face.debug().gaze"))
        assert follow(page) == 0
    assert kinds <= {"state", "think-back", "idle"}, kinds
    # He still half-registers it: a faint pull, never a switch.
    page.mouse.move(980, 650, steps=4)
    page.clock.run_for(300)
    pulled = page.evaluate("face.debug().eyes[0]")
    assert 0 < math.hypot(pulled["x"] - base["x"], pulled["y"] - base["y"]) < 0.35
    assert page.evaluate("face.debug().gaze") in ("state", "think-back")


def test_the_persons_actions_can_briefly_break_through(browser: Any) -> None:
    looks = 0
    for seed in (3, 5, 8):  # it's a chance, not a rule: count over a few independent runs
        page = open_face(browser, seed=seed)
        page.evaluate("setFacts({ agentBusy: true })")
        page.clock.run_for(1500)
        for _ in range(10):
            page.mouse.click(120, 600)
            page.clock.run_for(100)
            if page.evaluate("face.debug().gaze") == "check-in":
                looks += 1
                page.clock.run_for(900)
                assert page.evaluate("face.debug().gaze") in ("state", "think-back"), "a brief look, then back to the task"
            page.clock.run_for(2600)
        page.context.close()
    assert 2 <= looks <= 22, f"a click while it works sometimes earns a brief look ({looks}/30)"


def test_typing_outranks_the_pointer_and_pauses_get_a_look_back(browser: Any) -> None:
    page = open_face(browser, seed=11)
    page.focus("#field")
    looked_back = False
    for _ in range(8):
        page.keyboard.type("hello there")
        page.mouse.move(900, 80, steps=8)
        page.clock.run_for(100)
        assert follow(page) == 0, "while typing, the pointer doesn't matter"
        for _ in range(40):  # a pause while composing
            page.clock.run_for(100)
            if page.evaluate("face.debug().gaze") == "look-back":
                looked_back = True
                break
        if looked_back:
            break
        page.clock.run_for(4000)
    assert looked_back, "pausing while composing sometimes gets a look back at the person"


def test_reactions_are_selective_under_bursts(browser: Any) -> None:
    page = open_face(browser)
    page.evaluate("presence.history.length = 0; setFacts({ agentBusy: true })")
    page.evaluate("""() => { for (let i = 0; i < 10; i++) {
        presence.dispatch('TOOL_STARTED', { tool: 'search' }); presence.dispatch('TOOL_COMPLETED', { tool: 'search', ok: true }); } }""")
    page.clock.run_for(1000)
    reacts = [h["what"] for h in history(page, "react")]
    assert reacts.count("work-done") <= 1 and reacts.count("work-start") + reacts.count("work-again") <= 2, reacts
    assert len([b for b in history(page, "beat") if b["what"] == "noticed"]) <= 2
    assert page.evaluate("presence.x.toolsRunning") == 0
    page.evaluate("setFacts({ agentBusy: false }); presence.history.length = 0; presence.dispatch('TASK_STARTED', { task_id: 't1' })")
    page.evaluate("""() => { for (let i = 1; i <= 20; i++) presence.dispatch('TASK_PROGRESS', { task_id: 't1', progress: i / 20 }); }""")
    page.clock.run_for(1500)
    assert [h["what"] for h in history(page, "react")].count("work-progress") <= 1, "twenty updates, one small acknowledgement"
    page.clock.run_for(6000)  # a separate moment: the bursts above have used up the reaction budget
    page.evaluate("presence.history.length = 0")
    for _ in range(6):
        page.mouse.click(300, 650)
        page.clock.run_for(80)
    assert [h["what"] for h in history(page, "react")].count("busy-clicks") == 1


def test_emotion_rises_holds_and_fades(browser: Any) -> None:
    page = open_face(browser)
    page.evaluate("presence.dispatch('TOOL_FAILED', { tool: 'x', ok: false })")
    vals = record(page, 8000, "presence.emotion.values.concerned")
    fps = len(vals) / 8.0
    peak = max(vals)
    rise = next(i for i, v in enumerate(vals) if v >= 0.9 * peak)
    assert peak > 0.5 and rise / fps < 0.6, "concern arrives quickly"
    assert vals[int(1.3 * fps)] > 0.8 * peak, "and holds for a moment"
    fall = vals[vals.index(peak):]
    assert max(a - b for a, b in zip(fall, fall[1:])) < 0.05, "it fades; it never snaps back to neutral"
    assert vals[-1] < 0.05


def test_the_exchange_remembers_what_happened(browser: Any) -> None:
    page = open_face(browser)
    page.evaluate("presence.emit({ type: 'user.submitted', text: 'do two things' }); setFacts({ agentBusy: true }); presence.history.length = 0")
    page.evaluate("presence.dispatch('TOOL_STARTED', { tool: 'a' })")
    page.clock.run_for(1800)
    page.evaluate("presence.dispatch('TOOL_COMPLETED', { tool: 'a', ok: true })")
    page.clock.run_for(1800)
    page.evaluate("presence.dispatch('TOOL_STARTED', { tool: 'b' })")
    reacts = [h["what"] for h in history(page, "react")]
    assert "work-start" in reacts and reacts[-1] == "work-again", "the second tool is renewed focus, not a new surprise"
    page.clock.run_for(1800)
    page.evaluate("presence.dispatch('TOOL_FAILED', { tool: 'b', ok: false })")
    page.clock.run_for(1800)
    page.evaluate("presence.dispatch('TOOL_STARTED', { tool: 'b' })")
    page.clock.run_for(1200)
    page.evaluate("presence.dispatch('TOOL_COMPLETED', { tool: 'b', ok: true })")
    page.clock.run_for(400)
    assert page.evaluate("presence.emotion.values.relief") > 0.2, "success after a failure is relief"
    assert "relief" in [h["what"] for h in history(page, "react")]


def test_long_idle_rests_without_looping_and_wakes_for_the_person(browser: Any) -> None:
    page = open_face(browser)
    page.mouse.move(500, 690)
    page.clock.run_for(2000)
    active_lid = page.evaluate("face.debug().L.lidTop")
    page.evaluate("presence.history.length = 0")
    hist: list[dict[str, Any]] = []
    for _ in range(36):  # six minutes alone
        page.clock.run_for(10_000)
        hist += page.evaluate("presence.history.splice(0)")
    d = dbg(page)
    assert d["idleLevel"] == "resting" and {"relax", "rest"} <= {h["kind"] for h in hist}
    assert page.evaluate("face.debug().L.lidTop") > active_lid + 0.15, "resting: heavier lids, eyes still open"
    rest_at = next(h["at"] for h in hist if h["kind"] == "rest")
    resting = [h for h in hist if h["at"] > rest_at]
    assert any(h.get("what") == "breath" for h in resting if h["kind"] == "beat"), "he breathes while resting"
    kinds = [h["what"] for h in resting if h["kind"] == "idle"]
    assert not any(a == b == c != "still" for a, b, c in zip(kinds, kinds[1:], kinds[2:]))
    page.evaluate("presence.history.length = 0")
    page.mouse.move(200, 200, steps=5)
    page.clock.run_for(300)
    assert any(h["kind"] == "return" for h in history(page)) and "noticed" in [b["what"] for b in history(page, "beat")]
    assert dbg(page)["idleLevel"] == "active"


def test_any_work_source_speaks_the_same_language(browser: Any) -> None:
    page = open_face(browser)
    page.evaluate("presence.history.length = 0")
    page.evaluate("presence.emit({ type: 'build.started', source: 'coding-agent', id: 'b1' })")
    page.clock.run_for(600)
    assert page.evaluate("presence.emotion.values.focused") > 0.3
    page.evaluate("presence.emit({ type: 'build.succeeded', source: 'coding-agent', id: 'b1' })")
    page.clock.run_for(500)
    assert page.evaluate("presence.emotion.values.satisfied") > 0.15
    page.clock.run_for(4000)
    page.evaluate("presence.emit({ type: 'test.started', id: 't' }); presence.emit({ type: 'test.failed', source: 'coding-agent', id: 't' })")
    page.clock.run_for(600)
    assert dbg(page)["emotion"] == "concerned"
    before = len(history(page))
    page.evaluate("presence.emit({ type: 'banana.split' }); presence.emit({ type: 'nonsense' }); presence.emit({})")
    assert len(history(page)) == before, "events that mean nothing are ignored"


def test_touching_jarvis(browser: Any) -> None:
    page = open_face(browser)
    page.evaluate("presence.history.length = 0")
    page.click("#face")
    page.clock.run_for(300)
    reacts = [h["what"] for h in history(page, "react")]
    assert reacts == ["poke"] and face(page)["gaze"] == "poke"
    page.clock.run_for(1500)
    for _ in range(3):
        page.click("#face")
        page.clock.run_for(100)
    page.clock.run_for(300)
    assert "pokes" in [h["what"] for h in history(page, "react")] and page.evaluate("presence.emotion.values.amused") > 0.2


def test_long_work_settles_into_focus_instead_of_looping(browser: Any) -> None:
    page = open_face(browser)
    page.evaluate("setFacts({ agentBusy: true }); presence.dispatch('TOOL_STARTED', { tool: 'compile' }); presence.history.length = 0")
    page.clock.run_for(8000)
    assert dbg(page)["state"] == "tool_use"
    assert any(h["kind"] == "engaged" for h in history(page)), "after a while he settles into the work"
    assert page.evaluate("presence.emotion.values.focused") > 0.5
    beats_ = [h["what"] for h in history(page, "beat")]
    assert len(beats_) <= 3, f"engaged, not performing: {beats_}"


def test_app_environment_follows_presence(app_page: Any) -> None:
    core, page = app_page
    page.evaluate("jarvis.conversation.addUser('first'); jarvis.conversation.addUser('second'); jarvis.conversation.addUser('third');"
                  "jarvis.conversation._sync()")
    emit(core, "AGENT_STATE", {"state": "thinking", "conversation_id": "c1"})
    page.wait_for_function("document.getElementById('main').dataset.presence === 'thinking'")
    page.wait_for_timeout(800)
    opacity = page.evaluate("getComputedStyle(document.querySelector('.messages > .msg')).opacity")
    assert float(opacity) < 0.8, "earlier conversation recedes while he thinks"
    emit(core, "AGENT_STATE", {"state": "idle", "conversation_id": "c1"})
    page.wait_for_function("document.getElementById('main').dataset.presence !== 'thinking'")
    page.wait_for_timeout(800)
    assert float(page.evaluate("getComputedStyle(document.querySelector('.messages > .msg')).opacity")) > 0.95


def test_app_on_a_phone_keeps_jarvis_central(browser: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_cli_auth import LiveCore, _free_port

    monkeypatch.setenv("JARVIS_SECRETS_BACKEND", "file")
    core = LiveCore(tmp_path / "server", _free_port())
    try:
        ctx = browser.new_context(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True, bypass_csp=True)
        page = ctx.new_page()
        page.goto(f"{core.url}/#pair={core.pairing_code()}")
        page.wait_for_function("window.jarvis?.started === true", timeout=20000)
        page.wait_for_timeout(200)
        size = lambda: page.evaluate("document.getElementById('presence').getBoundingClientRect().width")  # noqa: E731
        assert size() >= 128, "the empty state is JARVIS"
        page.evaluate("jarvis.conversation.addUser('hello'); jarvis.conversation._sync()")
        page.wait_for_timeout(900)
        assert size() >= 56, "still present during a conversation"
        assert page.evaluate("document.documentElement.scrollWidth") <= 390, "no horizontal scrolling"
        ctx.close()
    finally:
        core.stop()


def test_app_voice_defers_audio_playing(app_page: Any) -> None:
    core, page = app_page
    page.evaluate("""
        window.__calls = 0;
        jarvis.voice._synthesize = async (ctx, chunk) => {
            window.__calls++;
            await new Promise(r => setTimeout(r, 400));
            return ctx.createBuffer(1, ctx.sampleRate * 0.5, ctx.sampleRate);
        };
    """)
    page.evaluate("jarvis.voice.ttsAvailable = true; window.__r = jarvis.voice.startReply(); window.__r.push('hello'); window.__r.end();")
    assert page.evaluate("jarvis.store.facts.audioPlaying") is False, "audioPlaying is false when startReply is called"
    page.wait_for_function("window.__calls > 0")
    assert page.evaluate("jarvis.store.facts.audioPlaying") is False, "audioPlaying is false while synthesizing"
    page.wait_for_function("jarvis.store.facts.audioPlaying === true", timeout=5000)
    page.evaluate("window.__r.cancel()")

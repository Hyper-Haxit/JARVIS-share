// The JARVIS presence engine.
//
// One engine per page turns what is really happening into behaviour, and every
// face on the page renders it. JARVIS is not animated; he is aware: he notices,
// attends, works, finishes, gets interrupted and settles, and most of the time
// he simply is there.
//
//   events      everything arrives as one vocabulary (interpret.js): core events,
//               the interface (user.submitted, ui.panel.opened, ...), and work of
//               any kind as <family>.started/progress/succeeded/failed
//   episode     memory of the current exchange (heard -> thinking -> working ->
//               ready -> speaking -> settling), so reactions follow from what came
//               before: a second tool is renewed focus, not a new surprise; a
//               success after a failure is relief
//   state       derived from facts (connection, agent, voice, audio): offline, idle,
//               attentive, listening, thinking, tool_use, responding, speaking,
//               waiting, error; transitions carry most of the meaning (beats:
//               noticed, heard, ready, interrupted, recovered, settle, nod, breath)
//   emotion     intensities that rise, hold and fade (emotion.js)
//   attention   what he looks at, by priority, with task locks (attention.js)
//   audio       the person's voice and JARVIS's own voice, kept apart (audio.js)
//   reactions   cooldowns and a budget: most events get no visible reaction (reactions.js)
//   idle        sparse, randomized small behaviour; relaxes, then rests, when alone
//   expression  all of it composed into geometry (expression.js)
//
// Nothing runs on fixed timers: intervals are randomized and the animation
// loop sleeps whenever nothing is changing.

import { AttentionController, P } from "./attention.js";
import { AudioReactivity } from "./audio.js";
import { EmotionController, isWarm, readMessage, readReply } from "./emotion.js";
import { compose, newFrame } from "./expression.js";
import { classify, fromCore } from "./interpret.js";
import { random } from "./random.js";
import { Reactions } from "./reactions.js";

const reduced = window.matchMedia("(prefers-reduced-motion: reduce)");
const now = () => performance.now();

const INTERACTIVE = "button, a[href], input, textarea, select, summary, [role='button'], [role='menuitem'], .navitem";
const ACTIVE_MS = 8000;      // input within this long: the person is here and doing things
const ERROR_HOLD_MS = 4000;  // an error shows briefly; then the face returns to what is actually going on
const RELAX_MS = 45000;      // alone this long: relaxed
const REST_MS = 240000;      // alone this long: resting
const LONG_WORK_MS = 2500;   // work running this long: JARVIS settles into it
const INWARD = new Set(["thinking", "tool_use"]);
const FREE = new Set(["idle", "attentive", "waiting", "error"]);   // states in which the interface may draw attention
// How much the eyes follow the pointer when it is near the face, by state. Thinking and
// working look inward (the pointer gets only a faint peripheral pull); listening and
// speaking stay mostly with the person.
const FOLLOW = { idle: 0.85, attentive: 0.9, waiting: 0.6, error: 0.5, listening: 0.4, responding: 0.35, speaking: 0.35 };
const FOLLOW_REST_MS = 2500;   // the pointer resting this long near the face: the eyes ease back (no staring)

// What the idle behaviour may do. "still" is a real choice: presence includes stillness.
const IDLE = {
  idle:       { still: 0.32, drift: 0.22, glance: 0.12, ui: 0.08, micro: 0.12, posture: 0.08, breath: 0.06 },
  resting:    { still: 0.45, breath: 0.3, drift: 0.15, micro: 0.1 },
  attentive:  { still: 0.42, drift: 0.28, micro: 0.18, posture: 0.06, breath: 0.06 },
  listening:  { still: 0.55, drift: 0.25, micro: 0.2 },
  responding: { still: 0.45, drift: 0.2, micro: 0.2, read: 0.15 },
  speaking:   { still: 0.45, drift: 0.3, micro: 0.25 },
  thinking:   { still: 0.25, aside: 0.35, back: 0.2, micro: 0.2 },
  tool_use:   { still: 0.35, work: 0.3, back: 0.15, micro: 0.2 },
  waiting:    { still: 0.4, back: 0.35, drift: 0.25 },
  error:      { still: 0.45, down: 0.3, micro: 0.25 },
};
const IDLE_MEDIAN_S = { idle: 3.6, attentive: 3.2, listening: 4.4, responding: 3.6, speaking: 3.8, thinking: 2.8, tool_use: 3.0,
  waiting: 3.4, error: 4.2 };
// Median seconds between blinks: fewer while concentrating, more while talking.
const BLINK_MEDIAN_S = { idle: 4.2, attentive: 3.9, listening: 4.8, thinking: 6.0, tool_use: 6.4, responding: 4.4, speaking: 3.6,
  waiting: 4.4, error: 5.2 };
// Where each state looks, when it has a direction of its own (normalized gaze; +y is down).
const STATE_GAZE = { thinking: { x: 0.6, y: -0.62 }, tool_use: { x: 0.34, y: 0.42 }, waiting: { x: 0, y: 0.5 }, error: { x: 0, y: 0.22 } };
// The exchange, in order. Phases only move forward within an episode (interruption starts over).
const PHASES = ["noticed", "listening", "heard", "thinking", "working", "ready", "responding", "speaking", "settling", "done"];

export function derivePrimary(f, x) {
  if (!f.coreReachable || f.link !== "connected") return "offline";
  // An error is about something that already happened: it shows briefly, and live voice (the person
  // speaking, or JARVIS) takes over from it at once.
  if (f.error && !x.errorShown && !(f.micCapturing || f.voiceListening || f.audioPlaying)) return "error";
  if (f.pendingApprovals > 0) return "waiting";
  if (f.micCapturing || f.voiceListening) return "listening";
  if (f.audioPlaying) return "speaking";
  if (f.streaming) return "responding";
  if (f.agentBusy) return x.toolsRunning > 0 ? "tool_use" : "thinking";
  return x.recentlyActive ? "attentive" : "idle";
}

class PresenceEngine extends EventTarget {
  constructor() {
    super();
    this.random = random;
    this.attention = new AttentionController(random);
    this.emotion = new EmotionController();
    this.audio = new AudioReactivity(random);
    this.audio.onVoiceStart = (t) => this._voiceStarted(t);
    this.audio.onVoicePause = (dur, t) => this._voicePaused(dur, t);
    this.reactions = new Reactions();
    this.facts = { coreReachable: false, link: "connecting", error: null, pendingApprovals: 0, micCapturing: false,
      voiceListening: false, audioPlaying: false, streaming: false, agentBusy: false };
    this.x = { toolsRunning: 0, recentlyActive: false, tool: null };
    this.work = new Map();          // running work units: id -> { family, at, progress, phase }
    this.episode = null;            // the current exchange
    this._episodes = 0;
    this.state = "offline";
    this.stateSince = now();
    this.idleLevel = "active";      // active | idle | relaxed | resting
    this.beats = [];
    this.micro = [];
    this.posture = { x: 0, y: 0, rot: 0 };
    this.blink = { at: -1e9, second: -1e9, dur: 140, lead: 0, leadSide: "L" };
    this.lastInputAt = -1e9;
    this.thinkSide = random.sign();
    this.frame = newFrame();
    this.peripheral = null;
    this.pointerNear = null;        // the mouse/pen position, for the faces to follow when it is near them
    this.following = 0;             // how much a face is following the pointer right now (0..1)
    this._petSince = 0;
    this.history = [];
    this.faces = new Set();
    this._timers = {};
    this._off = [];
    this._raf = 0;
    this._last = 0;
    this._replyStarted = false;
    this._hoverAt = -1e9;
    this._typingAt = -1e9;
    this._clicks = [];
    this._pokes = [];
    this._drift = { x: 0, y: 0 };
  }

  get reduced() { return reduced.matches; }
  /** Motion amplitude: reduced motion keeps the personality but moves much less. */
  get scale() { return reduced.matches ? 0.3 : 1; }
  get relaxed() { return this.idleLevel === "relaxed" || this.idleLevel === "resting"; }

  // ------------------------------------------------------------ faces
  attach(face) {
    this.faces.add(face);
    if (this.faces.size === 1) this._listen();
    this.wake();
  }

  detach(face) {
    this.faces.delete(face);
    if (this.faces.size === 0) this._unlisten();
  }

  get listenerCount() { return this._off.length; }

  _on(target, type, fn, opts = { passive: true }) {
    target.addEventListener(type, fn, opts);
    this._off.push(() => target.removeEventListener(type, fn, opts));
  }

  _listen() {
    this._on(window, "pointermove", (e) => this._pointerMoved(e));
    this._on(window, "pointerdown", (e) => this._pointerDown(e));
    this._on(document, "pointerover", (e) => this._hover(e));
    this._on(document.documentElement, "pointerleave", () => this._pointerGone());
    this._on(window, "blur", () => this._pointerGone());
    this._on(document, "focusin", (e) => this._focus(e));
    this._on(document, "input", (e) => this._typing(e));
    this._on(document, "visibilitychange", () => { this._schedule(); this.wake(); });
    const onResize = () => { for (const f of this.faces) f.invalidateRect(); };
    this._on(window, "resize", onResize);
    this._on(window, "scroll", onResize, { passive: true, capture: true });
    const onMotion = () => { this._schedule(); this.wake(); };
    reduced.addEventListener("change", onMotion);
    this._off.push(() => reduced.removeEventListener("change", onMotion));
    this._schedule();
  }

  _unlisten() {
    for (const off of this._off.splice(0)) off();
    for (const k of Object.keys(this._timers)) clearTimeout(this._timers[k]);
    this._timers = {};
    if (this._raf) cancelAnimationFrame(this._raf);
    this._raf = 0;
    this.attention.clear();
  }

  // ------------------------------------------------------------ input from the app
  /** Facts from the app's store (connection, agent, voice, audio, approvals). */
  syncFacts(facts) {
    const was = this.facts;
    this.facts = { ...facts };
    if (facts.error !== was.error) {
      // Resolved while it still showed: a visible relief. Already settled: nothing to react to.
      if (was.error && !facts.error && !this.x.errorShown) this._recovered("error cleared");
      this.x.errorShown = false;
      clearTimeout(this._timers.error);
      if (facts.error) {
        this._concern(0.8);   // shows even if JARVIS is saying it (speech takes over the state, not the feeling)
        this._timers.error = setTimeout(() => {
          this.x.errorShown = true;
          this.emotion.release("concerned", now());
          this._derive();
        }, ERROR_HOLD_MS);
      }
    }
    if (!facts.audioPlaying && was.audioPlaying) this.audio.jarvisSpeaking = false;
    if (facts.audioPlaying && !was.audioPlaying) this.audio.voice.reset();  // what the mic hears now is JARVIS
    if (facts.audioPlaying) this.audio.jarvisSpeaking = true;
    this._derive();
  }

  /** A real audio level: "voice"/"mic" (the person) or "output"/"speaker" (JARVIS's voice). */
  level(source, value) {
    this.audio.level(source === "speaker" ? "output" : source === "mic" ? "voice" : source, value, now());
    this.wake();
  }

  /** What the agent is doing while busy: the tool being run, if any. */
  setActivity(tool) {
    this.x.tool = tool || null;
  }

  /** A core event, as received over the event stream. */
  dispatch(type, data = {}) {
    const evt = fromCore(type, data);
    if (evt) this.emit({ ...evt, source: "core" });
  }

  /**
   * A presence event: { type: "family.verb", source?, id?, ... } (see interpret.js).
   * The single way anything tells JARVIS what happened.
   */
  emit(evt) {
    if (!evt || typeof evt.type !== "string") return;
    const meaning = classify(evt.type);
    if (!meaning) return;
    const t = now();
    let handled;
    if (meaning.kind === "work") handled = this._work(evt, meaning, t);
    else if (meaning.kind === "notice") handled = this._noticed(evt, t);
    else handled = this._own(evt, t);
    if (handled === false) return;
    this._note("event", { type: evt.type });
    this.wake();
  }

  // ------------------------------------------------------------ interpretation
  _own(evt, t) {
    switch (evt.type) {
      case "voice.wake":
        this._beginEpisode(t, "noticed");
        this._react("wake", 0, true, () => {
          this._beat("noticed", 600, 1);
          this.emotion.feel("attentive", 0.8, t, 1500);
          this.attention.claim("wake", P.VOICE, { person: true }, t, 1600, 18);
          this._blinkMaybe(0.4);
        });
        return true;
      case "voice.user.started":
        if (!this._active()) this._beginEpisode(t, "listening");
        this.attention.claim("user-voice", P.VOICE, { person: true }, t, 8000, 16);
        this.emotion.feel("attentive", 0.6, t, 1500);
        return true;
      case "voice.user.ended":
        this.attention.release("user-voice");
        this._phase("heard");
        this._react("heard", 800, true, () => this._beat("heard", 620, 0.8));
        return true;
      case "voice.heard":
        for (const [e, a] of readMessage(evt.text || "")) this.emotion.feel(e, a, t, 1500);
        return true;
      case "voice.unclear":
        this._react("unclear", 3000, false, () => this.emotion.feel("uncertain", 0.4, t, 900));
        this._puzzled();
        return true;
      case "user.submitted": {
        this._beginEpisode(t, "heard");
        this._replyStarted = false;
        const warm = isWarm(evt.text || "");   // "thanks!", "hello": a small happy hop instead of the nod
        this._react("submitted", 0, true, () => {
          if (!(warm && this._cheer(0.65))) this._beat("heard", 600, 0.9);
          const msg = document.querySelector(".messages > .msg--user:last-of-type");
          if (msg) this.attention.claim("submitted", P.CRITICAL, { element: msg }, t, 450, 20);
        });
        for (const [e, a] of readMessage(evt.text || "")) this.emotion.feel(e, a, t, 2000);
        return true;
      }
      case "user.cancelled":
        // The person stopped the request: a small "oh, all right", attention back to them.
        this._react("cancelled", 1000, true, () => {
          this._beat("noticed", 450, 0.5);
          this.attention.claim("cancelled", P.UI + 5, { person: true }, t, 900, 14);
        });
        this.emotion.release("focused", t);
        this._phase("settling");
        this._finishEpisodeSoon();
        return true;
      case "user.interrupted":
      case "voice.interrupted":
        // Stop speaking at once, acknowledge it, look at the person.
        this.audio.silenceOutput();
        this.audio.jarvisSpeaking = false;
        if (this.episode) this.episode.interrupted = true;
        this._endHappyMoment(t);
        this._react("interrupted", 300, true, () => {
          this._beat("interrupted", 900, 1);
          this.emotion.feel("surprised", 0.45, t, 300, 0.8);
          this.emotion.feel("attentive", 0.7, t, 1600);
          this.attention.claim("interrupt", P.CRITICAL, { person: true }, t, 1400, 22);
          this._blinkMaybe(0.35);
        });
        this._beginEpisode(t, "listening");
        return true;
      case "agent.thinking":
        if (!this._active()) this._beginEpisode(t, "thinking");
        this._phase("thinking");
        this.emotion.feel("focused", 0.35, t, 2500, 3);
        return true;
      case "agent.error":
        this._concern(0.7, null, t);
        return true;
      case "agent.output.started":
        if (this._replyStarted) return false;  // only the first one means anything
        this._replyStarted = true;
        // The answer exists: focus releases and attention comes back to the person before the words do.
        this._phase("ready");
        this._beat("ready", 520, 1);
        this.emotion.release("focused", t);
        this.attention.release("think-back");
        this.emotion.feel("confident", 0.3, t, 1200);
        return true;
      case "agent.reply.completed":
        this._replyStarted = false;
        if (evt.error) this._concern(0.6, null, t);
        else if (!evt.stopped) {
          const felt = readReply(evt.text || "");
          for (const [e, a] of felt) this.emotion.feel(e, a, t, 1400);
          if (felt.some(([e]) => e === "uncertain")) this._puzzled();   // JARVIS itself isn't sure
          const ep = this.episode;
          if (ep && ep.failures && ep.successes) this.emotion.feel("relief", 0.4, t, 600, 1.5);
          this._react("reply-done", 1500, false, () => {
            if (random.chance(0.75)) { this.emotion.feel("satisfied", random.range(0.25, 0.4), t, 600, 1.4); this._beat("nod", 600, 0.6); }
          });
        }
        if (!this.facts.audioPlaying) { this._phase("settling"); this._finishEpisodeSoon(); }
        return true;
      case "tts.started":
        this.audio.jarvisSpeaking = true;
        this.audio.voice.reset();
        if (!this._replyStarted) { this._replyStarted = true; this._phase("ready"); this._beat("ready", 420, 0.8); }
        this._phase("speaking");
        return true;
      case "tts.finished":
        this.audio.jarvisSpeaking = false;
        return true;
      case "permission.requested":
        this.emotion.feel("attentive", 0.5, t, 1500);
        return true;
      case "ui.panel.opened":
        if (!evt.element) return false;
        this._react("panel", 2500, false, () => {
          if (!random.chance(0.8)) return;
          this.attention.claim("panel", P.UI, { element: evt.element }, t, random.range(800, 1400), 13);
          this._beat("noticed", 450, 0.45);
        });
        return true;
      case "ui.notice":
        if (evt.level === "err" || evt.level === "warn") {
          this._react("notice", 3000, false, () => this._concern(evt.level === "err" ? 0.5 : 0.35, evt.element || null, t));
        }
        return true;
      case "ui.content":
        // Something substantial appeared (a code block, a table): a look, then settle.
        if (evt.element) {
          this._react("content", 2000, false, () => {
            if (random.chance(0.5)) this.attention.claim("content", P.TASK - 5, { element: evt.element }, t, random.range(500, 900), 11);
          });
        }
        return true;
      default:
        return false;
    }
  }

  /** Work of any kind: tools, background tasks, and anything else that starts, progresses and ends. */
  _work(evt, m, t) {
    const id = evt.id || `${m.family}:${evt.name || ""}`;
    const chat = m.family === "tool";            // tools run inside the current exchange; the rest is background work
    const ep = this._active() ? this.episode : null;
    const lastStep = (status) => () => (chat ? this._lastStep(status) : document.getElementById("taskstrip")?.firstElementChild || null);
    switch (m.verb) {
      case "started": {
        // Tools can run in parallel (the same tool twice): each run is its own unit.
        // Anything else with an id we already know is a repeat: nothing new happened.
        let key = id;
        if (this.work.has(key)) {
          if (!chat) return false;
          for (let n = 2; this.work.has(key); n++) key = `${id}#${n}`;
        }
        this.work.set(key, { family: m.family, at: t, progress: 0, reacted: 0, phase: null, chat });
        if (chat) this.x.toolsRunning += 1;
        if (chat && ep) this._phase("working");
        // A second piece of work in the same exchange is renewed focus, not a new surprise.
        const renewed = Boolean(ep && ep.works > 0);
        if (ep) ep.works += 1;
        this.emotion.feel("focused", renewed ? 0.5 : 0.6, t, 2500);
        this._react(renewed ? "work-again" : "work-start", 1500, false, () => {
          this._micro({ eyeH: -0.04, lidTop: renewed ? 0.04 : 0.06 }, 500, 1);
          if (!renewed && random.chance(chat ? 0.75 : 0.4)) {
            this.attention.claim("tool", P.TOOL_EVENT, { find: lastStep("running") }, t, random.range(600, 1100), 14);
          }
        });
        clearTimeout(this._timers[`long:${key}`]);
        this._timers[`long:${key}`] = setTimeout(() => this._longWork(key), LONG_WORK_MS);
        if (chat) this._derive();
        return true;
      }
      case "progress": {
        const w = this.work.get(id);
        if (!w) return false;
        const p = typeof evt.progress === "number" ? evt.progress : null;
        const meaningful = (p !== null && p - w.reacted >= 0.25) || (evt.phase && evt.phase !== w.phase);
        if (evt.phase) w.phase = evt.phase;
        if (p !== null) w.progress = p;
        // Progress updates are grouped: only a meaningful step, now and then, gets a tiny acknowledgement.
        if (meaningful) {
          this._react("work-progress", 4000, false, () => {
            if (p !== null) w.reacted = p;
            this._micro({ eyeH: 0.04 }, 450, 1);
            if (!chat && random.chance(0.4)) this.attention.claim("work-glance", P.UI, { find: lastStep() }, t, random.range(500, 900), 12);
          });
        }
        return true;
      }
      case "succeeded":
      case "failed":
      case "cancelled": {
        const key = this.work.has(id) ? id : [...this.work.keys()].find((k) => k.startsWith(`${id}#`));
        const w = key ? this.work.get(key) : undefined;
        if (key) { this.work.delete(key); clearTimeout(this._timers[`long:${key}`]); }
        if (chat || w?.chat) this.x.toolsRunning = Math.max(0, this.x.toolsRunning - 1);
        const long = Boolean(w && t - w.at > LONG_WORK_MS);
        const instant = Boolean(w && t - w.at < 400);   // over before it could be seen: a quiet release only
        if (m.verb === "succeeded") {
          if (ep) ep.successes += 1;
          this.emotion.release("focused", t);
          if (instant && !(ep && ep.failures > 0 && !ep.relieved)) {
            this.emotion.feel("satisfied", 0.2, t, 300, 1.2);
          } else if (ep && ep.failures > 0 && !ep.relieved) {
            // It worked after something failed: visible relief, not just satisfaction.
            ep.relieved = true;
            this.emotion.release("concerned", t);
            this.emotion.feel("relief", 0.55, t, 700, 1.8);
            this._react("relief", 1500, true, () => this._beat("recovered", 1100, 0.6));
          } else if (long || (!chat && !w?.chat)) {
            // Long work that finally came through, or a background task done: a real moment of joy.
            this.emotion.feel("satisfied", 0.5, t, 900, 1.6);
            this._cheer(long ? 1 : 0.8);
            this._react("work-done", 1500, false, () => {
              if (random.chance(0.5)) this.attention.claim("tool-result", P.TOOL_EVENT, { find: lastStep("ok") }, t, random.range(450, 800), 14);
            });
          } else {
            this._react("work-done", 1500, false, () => {
              // The result is in: focus lifts ("got it").
              this._beat("noticed", 420, 0.4);
              this.emotion.feel("satisfied", random.range(0.3, 0.45), t, 700, 1.6);
              if (random.chance(0.5)) this.attention.claim("tool-result", P.TOOL_EVENT, { find: lastStep("ok") }, t, random.range(450, 800), 14);
              if (random.chance(0.4)) this._beat("nod", 520, 0.45);
            });
          }
        } else if (m.verb === "failed") {
          if (ep) ep.failures += 1;
          // Repeated failures in one exchange weigh more.
          const amount = Math.min(0.85, 0.6 + 0.12 * ((ep?.failures || 1) - 1));
          this._react("work-failed", 600, true, () => this._concern(amount, { find: lastStep("failed") }, t));
        } else {
          this.emotion.release("focused", t);
          this._react("work-cancelled", 1500, false, () => this._beat("settle", 900, 0.5));
        }
        if (chat) this._derive();
        return true;
      }
      case "retried":
        this.emotion.feel("focused", 0.6, t, 2000);
        this._react("work-again", 1500, false, () => this._micro({ lidTop: 0.05 }, 500, 1));
        return true;
      default:
        return false;
    }
  }

  /** Something happened nearby that isn't work (a file opened, a change detected): rarely worth a look. */
  _noticed(evt, t) {
    if (!FREE.has(this.state) || reduced.matches) return true;
    this._react("environment", 6000, false, () => {
      if (!random.chance(0.35)) return;
      this._micro({ eyeH: 0.04 }, 400, 1);
      if (evt.element) this.attention.claim("environment", P.UI - 5, { element: evt.element }, t, random.range(500, 900), 12);
    });
    return true;
  }

  /** Work that keeps running: JARVIS settles into it rather than looping an animation. */
  _longWork(id) {
    const w = this.work.get(id);
    if (!w) return;
    const t = now();
    // Held for as long as the work runs (released when it ends or the answer is ready).
    this.emotion.feel("focused", 0.75, t, 600000, 2);
    this._note("engaged", { what: w.family });
    this.wake();
  }

  // ------------------------------------------------------------ the exchange (episode memory)
  _active() { return Boolean(this.episode && this.episode.phase !== "done"); }

  _beginEpisode(t, phase) {
    if (this._active() && !(phase === "heard" || phase === "noticed" || phase === "listening")) return;
    clearTimeout(this._timers.episodeEnd);
    this.episode = { id: ++this._episodes, at: t, phase: null, works: 0, failures: 0, successes: 0, interrupted: false, relieved: false };
    this._note("episode", { id: this.episode.id });
    this._phase(phase);
  }

  _phase(name) {
    const ep = this.episode;
    if (!ep || ep.phase === name) return;
    // Phases move forward; an exchange never goes back to "thinking" once JARVIS is answering.
    if (ep.phase && PHASES.indexOf(name) < PHASES.indexOf(ep.phase) && name !== "listening") return;
    ep.phase = name;
    this._note("phase", { what: name, episode: ep.id });
  }

  _finishEpisodeSoon() {
    clearTimeout(this._timers.episodeEnd);
    this._timers.episodeEnd = setTimeout(() => this._phase("done"), 1300);
  }

  // ------------------------------------------------------------ state
  _derive() {
    const t = now();
    this.x.recentlyActive = t - this.lastInputAt < ACTIVE_MS || t - this._typingAt < 3000;
    const next = derivePrimary(this.facts, this.x);
    if (next === this.state) return;
    const prev = this.state;
    this.state = next;
    this.stateSince = t;
    this._transition(prev, next, t);
    this._note("state", { from: prev, to: next });
    this._schedule();
    this.dispatchEvent(new CustomEvent("state", { detail: { state: next, previous: prev } }));
    this.wake();
  }

  /** Transitions carry most of the feeling: each one is a small, intentional change. */
  _transition(prev, next, t) {
    const A = this.attention;
    A.release("state");
    if (STATE_GAZE[next]) {
      const g = STATE_GAZE[next];
      if (INWARD.has(next) && !INWARD.has(prev)) this.thinkSide = random.sign();
      const gaze = { x: g.x * (INWARD.has(next) ? this.thinkSide : 1), y: g.y };
      const element = next === "waiting" ? () => document.querySelector("#requests > *") : null;
      A.claim("state", P.TASK, element ? { find: element, gaze } : { gaze, state: next }, t, Infinity, next === "error" ? 9 : 6);
      if (element) A.claim("state-fallback", P.TASK - 1, { gaze, state: next }, t, Infinity, 6);
    } else if (next === "listening") {
      A.claim("state", P.VOICE, { person: true }, t, Infinity, 16);
    } else if (next === "speaking" || next === "responding") {
      A.claim("state", P.TASK, { person: true }, t, Infinity, 10);
    }
    if (next !== "waiting") A.release("state-fallback");
    if (next === "offline") { A.clear(); this.emotion.release("attentive", t); return; }
    if (prev === "offline") {
      // Back online: he wakes up (a stretch, eyes coming open, a double blink).
      this._beat("wake", 1300, 1);
      this.blinkNow(true);
      this.blink.second = t + random.range(260, 340);
      return;
    }
    // Phases of the exchange follow the state.
    const phase = { listening: "listening", thinking: "thinking", tool_use: "working", responding: "responding", speaking: "speaking" }[next];
    if (phase) {
      if (!this._active() && (next === "thinking" || next === "listening")) this._beginEpisode(t, phase);
      else this._phase(phase);
    }
    if (next === "listening") {
      this._beat("noticed", 450, prev === "idle" || prev === "attentive" ? 1 : 0.6);
      this._blinkMaybe(0.4);
    }
    if (prev === "listening" && INWARD.has(next)) this._beat("heard", 620, 0.8);
    if (prev === "thinking" && next === "tool_use") this._blinkMaybe(0.5);   // attention shifts to the task
    if (INWARD.has(prev) && (next === "responding" || next === "speaking") && !this._replyStarted) {
      this._replyStarted = true;
      this._beat("ready", 480, 0.9);
      this._blinkMaybe(0.3);
    }
    if (next === "tool_use") this.emotion.feel("focused", 0.6, t, 2000);
    if (INWARD.has(prev) && !INWARD.has(next)) this.emotion.release("focused", t);   // the work is over, whatever the reason
    if ((prev === "speaking" || prev === "responding") && (next === "idle" || next === "attentive")) {
      // Settle rather than snap: a small exhale as the reply ends (not when he was just cut off).
      if (!this.beats.some((b) => b.kind === "interrupted")) {
        this._beat("settle", 1100, 0.8);
        this._blinkMaybe(0.35);
      }
      this._phase("settling");
      this._finishEpisodeSoon();
    }
    if (next === "error") this._concern(0.8, null, t);
  }

  _recovered(why) {
    const t = now();
    this.emotion.release("concerned", t);
    this.emotion.release("uncertain", t);
    this._beat("recovered", 1300, 1);
    this.emotion.feel("relief", 0.4, t, 500, 1.6);
    this._blinkMaybe(0.6, true);
    this._note("recovered", { why });
  }

  /** Bad news or being cut off ends a happy moment at once (no smiling through an error). */
  _endHappyMoment(t) {
    this.beats = this.beats.filter((b) => b.kind !== "cheer");
    this.emotion.drop("amused", t);
    this.emotion.drop("satisfied", t);
  }

  _concern(amount, target, t = now()) {
    this._endHappyMoment(t);
    this.emotion.feel("concerned", amount, t, 1500, 2.2);
    if (target) this.attention.claim("concern", P.TOOL_EVENT + 5, target instanceof Element ? { element: target } : target, t, random.range(700, 1100), 14);
  }

  _voiceStarted(t) {
    if (this.state === "listening") this.attention.claim("user-voice", P.VOICE, { person: true }, t, 2500, 16);
  }

  _voicePaused(dur, t) {
    // A small nod when the person pauses after saying something: "go on, I'm following".
    if (this.state === "listening" && dur > 900 && random.chance(0.5)) this._react("listen-nod", 2600, false, () => this._beat("nod", 560, 0.5));
  }

  _lastStep(status) {
    const steps = status ? document.querySelectorAll(`.step[data-status="${status}"]`) : [];
    return steps.length ? steps[steps.length - 1] : null;
  }

  /** Run a visible reaction if its cooldown and the shared budget allow; returns whether it ran. */
  _react(kind, cooldownMs, essential, fn) {
    if (this.state === "offline") return false;
    if (!this.reactions.allow(kind, now(), cooldownMs, essential)) return false;
    fn();
    this._note("react", { what: kind });
    return true;
  }

  _beat(kind, dur, amount, extra = null) {
    if (this.state === "offline") return;
    const t = now();
    this.beats = this.beats.filter((b) => t - b.at < b.dur && b.kind !== kind);
    this.beats.push({ ...extra, kind, at: t, dur: reduced.matches ? dur * 0.7 : dur, amount: amount * random.range(0.85, 1.1) });
    this._note("beat", { what: kind });
  }

  /** A real success worth a moment of joy (not every ordinary answer): a happy squint and a small hop. */
  _cheer(amount = 1) {
    return this._react("cheer", 6000, false, () => { this._beat("cheer", 1300, amount); this._blinkMaybe(0.3); });
  }

  /** Something didn't make sense: the head cocks and one eye narrows. */
  _puzzled() {
    this._react("puzzled", 5000, false, () => this._beat("puzzled", 1500, 1, { side: random.sign() }));
  }

  _micro(spec, dur, amount, side) {
    const t = now();
    this.micro = this.micro.filter((m) => t - m.at < m.dur);
    this.micro.push({ ...spec, at: t, dur, amount, side });
  }

  _note(kind, extra) {
    this.history.push({ kind, at: Math.round(now()), state: this.state, ...extra });
    if (this.history.length > 160) this.history.shift();
  }

  // ------------------------------------------------------------ the person, in the interface
  _input() {
    const t = now();
    const was = this.idleLevel;
    if (was === "relaxed" || was === "resting") {
      // The person is back: notice them (after a long rest, wake up properly).
      this.emotion.release("calm", t);
      if (this.state !== "offline") {
        this._beat("noticed", 650, was === "resting" ? 1 : 0.7);
        this.emotion.feel("attentive", was === "resting" ? 0.6 : 0.4, t, 900);
        if (was === "resting") this.blinkNow(true);
        this._note("return", { from: was });
      }
    }
    this.idleLevel = "active";
    const wasActive = this.x.recentlyActive;
    this.lastInputAt = t;
    clearTimeout(this._timers.relax);
    clearTimeout(this._timers.rest);
    this._timers.relax = setTimeout(() => this._settleInto("relaxed"), RELAX_MS);
    this._timers.rest = setTimeout(() => this._settleInto("resting"), REST_MS);
    clearTimeout(this._timers.active);
    this._timers.active = setTimeout(() => { if (this.idleLevel === "active") this.idleLevel = "idle"; this._derive(); }, ACTIVE_MS + 100);
    if (was !== "active" && was !== "idle") this._schedule();
    if (!wasActive) this._derive();
  }

  _settleInto(level) {
    this.idleLevel = level;
    if (level === "resting") this.emotion.feel("calm", 0.6, now(), 600000, 3);
    this._note(level === "resting" ? "rest" : "relax");
    this._schedule();
    this.wake();
  }

  /** Whether JARVIS is free to let the interface (a hovered control) draw his attention. */
  _pointerOpen(t) {
    return FREE.has(this.state) && t - this._typingAt > 2000 && !reduced.matches;
  }

  /**
   * How much the eyes follow the pointer when it is near a face (each face applies its own
   * distance): by state, never while typing or under reduced motion, less while something
   * important holds attention, and easing back to the person once the pointer rests.
   */
  followGain(t) {
    const pn = this.pointerNear;
    if (!pn || reduced.matches || t - this._typingAt < 2000) return 0;
    let g = FOLLOW[this.state] || 0;
    if (this.target && this.target.priority >= P.TOOL_EVENT) g *= 0.2;
    const still = t - pn.at;
    if (still > FOLLOW_REST_MS) g *= Math.max(0, 1 - (still - FOLLOW_REST_MS) / 1200);
    return g;
  }

  _pointerMoved(e) {
    if (e.pointerType === "touch" && !e.buttons) return;
    this._input();
    if (this.state === "offline") return;
    const t = now();
    this.attention.pointerMoved(e.clientX, e.clientY, t);
    if (e.pointerType !== "touch") {
      this.pointerNear = { x: e.clientX, y: e.clientY, at: t };
      // Wake again when the pointer has rested long enough for the eyes to ease back.
      clearTimeout(this._timers.pointerRest);
      this._timers.pointerRest = setTimeout(() => this.wake(), FOLLOW_REST_MS + 20);
      this._pet(e, t);
    }
    this.wake();
  }

  _pointerGone() {
    this.pointerNear = null;
    this._petSince = 0;
    this.attention.pointerLeft();
    this.wake();
  }

  /** The pointer resting on JARVIS himself: after a moment he is pleased (a smile in the eyes, warm cheeks). */
  _pet(e, t) {
    const on = e.target instanceof Element && e.target.closest("jarvis-presence");
    if (!on || reduced.matches || !FREE.has(this.state)) { this._petSince = 0; return; }
    if (!this._petSince) this._petSince = t;
    if (t - this._petSince > 700) {
      this.emotion.feel("satisfied", 0.55, t, 700, 1.2);
      this._react("pet", 5000, false, () => this._beat("noticed", 420, 0.35));
    }
  }

  _pointerDown(e) {
    this._input();
    if (this.state === "offline" || reduced.matches) return;
    const t = now();
    if (e.target instanceof Element && e.target.closest("jarvis-presence")) { this._poke(e, t); this.wake(); return; }
    // Clicking around quickly: momentarily attentive (once, not per click).
    this._clicks = this._clicks.filter((x) => t - x < 1500);
    this._clicks.push(t);
    if (this._clicks.length >= 3) {
      this._react("busy-clicks", 5000, false, () => { this.emotion.feel("attentive", 0.5, t, 900); this._beat("noticed", 400, 0.5); });
    }
    const point = { point: { x: e.clientX, y: e.clientY } };
    if (INWARD.has(this.state)) {
      // Busy, but the person did something: sometimes a brief look, then back to work.
      if (random.chance(0.35)) this._react("check-in", 2500, false, () => this.attention.claim("check-in", P.TASK + 2, point, t, random.range(350, 650), 18));
    } else if (random.chance(e.pointerType === "touch" ? 0.9 : 0.55)) {
      this.attention.claim("click", P.UI, point, t, random.range(450, 800), 18);
      this._micro({ eyeH: 0.05 }, 320, 1);
    }
    this.wake();
  }

  /** The person touched JARVIS himself. */
  _poke(e, t) {
    this._pokes = this._pokes.filter((x) => t - x < 1500);
    this._pokes.push(t);
    if (this._pokes.length >= 3) {
      this._react("pokes", 6000, false, () => {
        // Several in a row: mild amusement, a small tilt.
        this.emotion.feel("amused", 0.5, t, 900, 1.5);
        this._micro({ head: random.sign() * 2.5, lidTop: 0.04 }, 1200, 1);
        this.blinkNow(true);
      });
      return;
    }
    this._react("poke", 1200, false, () => {
      this.blinkNow(true);
      this._beat("noticed", 450, 0.6);
      this.attention.claim("poke", P.UI, { person: true }, t, 900, 16);
    });
  }

  _hover(e) {
    if (e.pointerType === "touch" || this.state === "offline" || reduced.matches) return;
    const t = now();
    if (e.target instanceof Element && e.target.closest("jarvis-presence")) {
      // The pointer comes close: sometimes he looks at it.
      if (random.chance(0.5)) this._react("approach", 4000, false, () => {
        this.emotion.feel("attentive", 0.3, t, 600);
        this.attention.claim("approach", P.UI, { point: { x: e.clientX, y: e.clientY } }, t, 700, 14);
      });
      return;
    }
    const el = e.target.closest?.(INTERACTIVE);
    if (!el || !this._pointerOpen(t)) return;
    if (t - this._hoverAt < 1600 || !random.chance(0.35)) return;
    this._hoverAt = t;
    this.attention.claim("hover", P.UI, { element: el }, t, random.range(500, 850), 12);
    this.wake();
  }

  _focus(e) {
    const el = e.target;
    if (!(el instanceof Element) || el.closest("jarvis-presence") || reduced.matches) return;
    let keyboard = false;
    try { keyboard = el.matches(":focus-visible"); } catch { /* older engines */ }
    if (keyboard && !el.matches("input, textarea, [contenteditable]")) {
      this.attention.claim("focus", P.UI, { element: el }, now(), random.range(600, 1000), 12);
      this.wake();
    }
  }

  _typing(e) {
    const el = e.target;
    if (!(el instanceof Element) || !el.matches("input, textarea, [contenteditable]")) return;
    const t = now();
    const newBurst = t - this._typingAt > 2500;
    this._typingAt = t;
    this._input();
    this.attention.pointerLeft();   // while the person types, the pointer doesn't matter
    if (newBurst && this.state !== "offline" && !reduced.matches) {
      this.emotion.feel("attentive", 0.25, t, 800);
      if (INWARD.has(this.state)) {
        if (random.chance(0.3)) this._react("check-in", 2500, false, () => this.attention.claim("check-in", P.TASK + 2, { element: el, typing: true }, t, random.range(400, 700), 14));
      } else if (random.chance(0.6)) {
        // Sometimes a glance at what is being written; never a stare.
        this.attention.claim("typing", P.INPUT, { element: el, typing: true }, t, random.range(900, 2200), 9);
      }
    }
    // A pause while composing: now and then, a look back at the person.
    clearTimeout(this._timers.composePause);
    this._timers.composePause = setTimeout(() => {
      const hasText = Boolean((el.value ?? el.textContent ?? "").trim());
      if (!hasText || !el.isConnected || !FREE.has(this.state) || now() - this._typingAt < 1500) return;
      if (random.chance(0.5)) {
        this._react("compose-pause", 4000, false, () => this.attention.claim("look-back", P.INPUT + 2, { person: true }, now(), random.range(700, 1300), 10));
        this.wake();
      }
    }, random.range(1600, 3200));
    this._derive();
    this.wake();
  }

  // ------------------------------------------------------------ idle behaviour and blinks
  _idleTable() {
    if (this.state === "idle" && this.idleLevel === "resting") return IDLE.resting;
    const table = IDLE[this.state];
    if (!table) return null;
    // Background work running: now and then he checks on it.
    if ((this.state === "idle" || this.state === "attentive") && [...this.work.values()].some((w) => !w.chat)) return { ...table, check: 0.15 };
    return table;
  }

  _schedule() {
    clearTimeout(this._timers.blink);
    clearTimeout(this._timers.idle);
    if (!this.faces.size || document.hidden || this.state === "offline") return;
    const rest = this.idleLevel === "resting" ? 1.6 : this.idleLevel === "relaxed" ? 1.2 : 1;
    const bm = (BLINK_MEDIAN_S[this.state] || 4.2) * (this.emotion.values.focused > 0.4 ? 1.3 : 1) * rest;
    this._timers.blink = setTimeout(() => { this.blinkNow(); this._schedule(); }, random.lognormal(bm, 0.45, 1.2, 15) * 1000);
    this._scheduleIdle();
  }

  _scheduleIdle() {
    clearTimeout(this._timers.idle);
    if (!this.faces.size || document.hidden || this.state === "offline") return;
    const level = this.idleLevel === "resting" ? 1.8 : this.idleLevel === "relaxed" ? 1.5 : 1;
    const im = (IDLE_MEDIAN_S[this.state] || 3.6) * level * (reduced.matches ? 2.5 : 1);
    this._timers.idle = setTimeout(() => { this._idleBeat(); this._scheduleIdle(); }, random.lognormal(im, 0.6, 1.0, 20) * 1000);
  }

  /** Close the eyes briefly: sometimes twice, sometimes one eye a moment ahead of the other. */
  blinkNow(force = false) {
    const t = now();
    if (this.state === "offline" || (!force && t - this.blink.at < 600)) return;
    const b = this.blink;
    b.dur = random.range(115, 170) * (this.idleLevel === "resting" ? 1.35 : 1);
    b.at = t;
    b.second = random.chance(0.12) ? t + random.range(230, 330) : -1e9;
    b.lead = random.chance(0.3) ? random.range(18, 42) : 0;
    b.leadSide = random.chance(0.5) ? "L" : "R";
    this._note("blink", { double: b.second > 0, asym: b.lead > 0 });
    this.wake();
  }

  _blinkMaybe(p, force = false) { if (random.chance(p)) this.blinkNow(force); }

  /** Eyelid closure 0..1 for one eye at time t (closing is quicker than opening). */
  closure(side, t) {
    const b = this.blink;
    const off = b.lead && side !== b.leadSide ? b.lead : 0;
    const one = (at) => {
      const p = (t - at - off) / b.dur;
      if (p < 0 || p >= 1) return 0;
      return p < 0.38 ? Math.sin((p / 0.38) * Math.PI / 2) : Math.cos(((p - 0.38) / 0.62) * Math.PI / 2);
    };
    return Math.max(one(b.at), one(b.second));
  }

  _idleBeat() {
    const table = this._idleTable();
    if (!table) return;
    const t = now();
    // Something more important holds attention, or the eyes are following the pointer near the face: let it be.
    const cur = this.attention.current(t);
    if (cur && cur.priority > P.IDLE && cur.kind !== "state" && cur.kind !== "state-fallback") return;
    if (this.following > 0.2) return;
    let options = table;
    if (reduced.matches) options = { still: 0.6, micro: 0.4 };
    // Memory against visible patterns: nothing (but stillness) twice in the last three beats.
    const recent = this._idleRecent || [];
    let pick = random.pick(options, recent.filter((k) => k !== "still").slice(-2));
    if (pick !== "still" && recent.slice(-3).filter((k) => k === pick).length >= 1) pick = random.pick(options, [pick]);
    this._idleRecent = [...recent, pick].slice(-4);
    this._note("idle", { what: pick });
    const A = this.attention;
    switch (pick) {
      case "still": break;
      case "drift": {
        // A small shift of fixation: the eyes are never pinned to one exact point.
        this._drift = { x: Math.max(-0.2, Math.min(0.2, this._drift.x * 0.4 + random.range(-0.13, 0.13))),
          y: Math.max(-0.14, Math.min(0.14, this._drift.y * 0.4 + random.range(-0.09, 0.09))) };
        break;
      }
      case "glance": {
        const a = random.range(0, Math.PI * 2), m = random.range(0.35, 0.7);
        A.claim("idle", P.IDLE, { gaze: { x: Math.cos(a) * m, y: Math.sin(a) * m * 0.7 } }, t, random.range(450, 1500), 16);
        this._blinkMaybe(0.3);
        break;
      }
      case "ui": {
        // Look over the interface for a moment: the latest message, the composer, the sidebar.
        const choices = [document.querySelector(".messages > .msg:last-child"), document.getElementById("input"),
          document.querySelector(".sidebar__list")].filter((el) => el && el.getBoundingClientRect().width);
        if (choices.length) A.claim("idle", P.IDLE, { element: choices[Math.floor(random.next() * choices.length)] }, t, random.range(700, 1500), 11);
        break;
      }
      case "read": {
        const msg = document.querySelector(".messages > .msg--jarvis:last-child");
        if (msg) A.claim("idle", P.IDLE, { element: msg }, t, random.range(500, 1100), 10);
        break;
      }
      case "check": {
        const strip = document.getElementById("taskstrip");
        if (strip && strip.getBoundingClientRect().height) A.claim("idle", P.IDLE, { element: strip }, t, random.range(600, 1200), 11);
        break;
      }
      case "micro": {
        const kinds = [
          { eyeH: 0.07 }, { lidTop: 0.08 }, { eyeH: 0.06, lift: 0.6 }, { head: random.range(-3.5, 3.5) },
          { lidTop: 0.09, side: random.chance(0.5) ? "L" : "R" }, { lidBot: 0.14 },
        ];
        const k = kinds[Math.floor(random.next() * kinds.length)];
        this._micro(k, random.range(500, 1400), random.range(0.6, 1), k.side);
        break;
      }
      case "posture": {
        this.posture.x = random.range(-0.6, 0.6);
        this.posture.rot = random.range(-1.2, 1.2);
        break;
      }
      case "breath": this._beat("breath", random.range(2600, 3800), random.range(0.7, 1)); break;
      case "aside": {
        this.thinkSide *= -1;
        const g = STATE_GAZE[this.state];
        if (g) A.claim("state", P.TASK, { gaze: { x: g.x * this.thinkSide, y: g.y + random.range(-0.08, 0.08) }, state: this.state }, t, Infinity, 6);
        this._blinkMaybe(0.25);
        break;
      }
      case "back": A.claim("think-back", P.TASK + 1, { person: true }, t, random.range(500, 1200), 12); break;
      case "work": {
        const step = this._lastStep("running");
        if (step) A.claim("tool", P.TOOL_EVENT, { element: step }, t, random.range(600, 1300), 12);
        break;
      }
      case "down": A.claim("idle-down", P.TASK + 1, { gaze: { x: random.range(-0.3, 0.3), y: 0.4 } }, t, random.range(600, 1300), 10); break;
      default:
    }
    this.wake();
  }

  // ------------------------------------------------------------ the loop
  wake() {
    if (this._raf || document.hidden || !this.faces.size) return;
    this._last = 0;
    this._raf = requestAnimationFrame((t) => this._frame(t));
  }

  _frame(t) {
    this._raf = 0;
    // The first frame after waking advances one ordinary frame, never zero.
    const dt = this._last ? Math.min(0.05, (t - this._last) / 1000) : 1 / 60;
    this._last = t;
    const tt = now();
    let active = this.audio.step(dt, tt);
    active = this.emotion.step(dt, tt) || active;
    this.beats = this.beats.filter((b) => tt - b.at < b.dur);
    this.micro = this.micro.filter((m) => tt - m.at < m.dur);
    if (this.beats.length || this.micro.length) active = true;
    const b = this.blink;
    if (tt < Math.max(b.at, b.second) + b.dur + b.lead + 20) active = true;
    compose(this.frame, this, tt, this.scale);
    this.target = this.attention.current(tt);
    // Peripheral awareness: while his attention is on the task he still half-registers
    // the pointer moving (a faint pull, never a switch).
    const p = this.attention.pointer;
    this.peripheral = this.target && this.target.priority >= P.TASK && (this.state === "thinking" || this.state === "responding"
      || this.state === "speaking") && tt - (p.movedAt || -1e9) < 1200 && !reduced.matches ? { x: p.lastX, y: p.lastY, w: 0.12 } : null;
    if (this.peripheral) active = true;
    let following = 0;
    for (const f of this.faces) {
      active = f.render(this, dt, tt) || active;
      following = Math.max(following, f._follow || 0);
    }
    this.following = following;
    if (active && !document.hidden) this._raf = requestAnimationFrame((x) => this._frame(x));
    else this._sleepUntilNextChange(tt);
  }

  /** Nothing is moving: wake again when the next attention claim expires (its target changes then). */
  _sleepUntilNextChange(t) {
    clearTimeout(this._timers.expire);
    const next = this.attention.busyUntil;
    if (next > t) this._timers.expire = setTimeout(() => this.wake(), next - t + 5);
  }

  // ------------------------------------------------------------ for tests and diagnostics
  debug() {
    const c = this.attention.current(now());
    return { state: this.state, phase: this.episode?.phase || null, emotion: this.emotion.dominant(),
      emotions: { ...this.emotion.values }, attention: c ? c.kind : "person", beats: this.beats.map((x) => x.kind),
      relaxed: this.relaxed, idleLevel: this.idleLevel, work: this.work.size, seed: random.seed,
      listeners: this._off.length, faces: this.faces.size, mouth: this.audio.mouth };
  }
}

export const presence = new PresenceEngine();
export { reduced };

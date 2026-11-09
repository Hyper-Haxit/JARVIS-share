// JARVIS web client: wires the API, live events, presence, conversation,
// composer, approvals, tasks, sidebar and sheets together.

import { presence } from "./presence.js";
import { api } from "./api.js";
import { Composer } from "./composer.js";
import { Conversation } from "./conversation.js";
import { EventStream } from "./events.js";
import { Palette } from "./palette.js";
import { Requests } from "./requests.js";
import { Sheets } from "./sheets.js";
import { Sidebar } from "./sidebar.js";
import { PRESENCE, Store, derivePresence } from "./state.js";
import { TaskStrip } from "./tasks.js";
import { applyTheme, h, prefs, relTime, setPref, toast, tone } from "./ui.js";
import { Voice } from "./voice.js";

const $ = (id) => document.getElementById(id);
const NARROW = matchMedia("(max-width: 1080px)");

function greeting(name) {
  const hr = new Date().getHours();
  const part = hr < 5 ? "Good evening" : hr < 12 ? "Good morning" : hr < 18 ? "Good afternoon" : "Good evening";
  return name ? `${part}, ${name}` : part;
}

class App {
  constructor() {
    this.store = new Store();
    this.events = new EventStream();
    this.conversation = new Conversation({ thread: $("thread"), list: $("messages"), empty: $("empty"), main: $("main"), jump: $("jump") });
    this.composer = new Composer({ form: $("composer"), input: $("input"), send: $("send"), mic: $("mic"), levels: $("levels"),
      hint: $("hint"), hintLeft: $("hint-left"), hintRight: $("hint-right") });
    this.requests = new Requests($("requests"));
    this.taskStrip = new TaskStrip($("taskstrip"), this);
    this.sheets = new Sheets($("sheet-layer"), this);
    this.sidebar = new Sidebar(this);
    this.palette = new Palette($("palette-layer"), () => this.commands());
    this.voice = new Voice(this);
    this.recentEvents = [];
    this.systemStatus = null;
    this.conversationId = sessionStorage.getItem("jarvis.conversation") || null;
    this.runs = new Map();          // conversation_id -> agent state, from AGENT_STATE events
    this.activity = null;           // e.g. "Running get_time…"
    this.localRun = null;
    this.lastModel = null;
    this.started = false;
  }

  // -------------------------------------------------------------- startup
  async start() {
    applyTheme();
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => applyTheme());
    this._bind();
    this.store.addEventListener("change", (e) => this._onState(e.detail));
    presence.addEventListener("state", (e) => { $("main").dataset.presence = e.detail.state; this._renderChrome(); });
    this.events.addEventListener("state", (e) => { this.store.update({ link: e.detail }); this._renderChrome(); });
    this.events.addEventListener("event", (e) => this._onEvent(e.detail));
    this.events.addEventListener("autherror", (e) => { if (e.detail?.status === 401) this._showPairing(e.detail.detail); });
    this.requests.addEventListener("change", () => { this.store.update({ pendingApprovals: this.requests.count }); this._renderCounts(); });
    this.taskStrip.addEventListener("change", () => this._renderCounts());

    const code = new URLSearchParams(location.hash.slice(1)).get("pair");
    if (code) {
      history.replaceState(null, "", location.pathname);
      try { api.forget(); await api.pair(code); }
      catch (e) { this._showPairing(e.detail || "That pairing code didn't work."); return; }
    }
    if (!api.paired) { this._showPairing(); return; }

    // Startup: each step is a real check; the outcome shows only if something fails.
    const health = await api.health();
    this.store.update({ coreReachable: Boolean(health) });
    if (!health) { this._startupFailed("Can't reach JARVIS on this computer.", "Start it with “jarvis start”, then reload."); this._watchHealth(); return; }
    try { await api.ensureSession(); }
    catch (e) { if (e.status === 401) { this._showPairing("This browser's access was revoked or has expired."); return; } }
    this.events.connect();
    await Promise.all([this.refreshStatus(), this.voice.probe(), this.sidebar.refresh(), this.taskStrip.refresh(), this.requests.loadPending()]);
    this.setVoiceArmed(this.voice.status?.pipeline?.running ? this.voice.status.pipeline.state : "off");
    this.started = true;
    if (this.conversationId) await this.openConversation(this.conversationId, true);
    this._renderEmpty();
    this._watchHealth();
    setInterval(() => this.refreshStatus(), 20000);
    if (!NARROW.matches) this.composer.focus();
  }

  _watchHealth() {
    clearInterval(this._healthTimer);
    this._healthTimer = setInterval(async () => {
      const ok = Boolean(await api.health());
      const was = this.store.facts.coreReachable;
      this.store.update({ coreReachable: ok });
      if (ok && !was) { if (!this.started) location.reload(); else { this.events.connect(); this.refreshStatus(); } }
      if (ok && this.events.state === "disconnected") this.events.connect();
      this._renderChrome();
    }, 4000);
  }

  async refreshStatus() {
    try {
      this.systemStatus = await api.get("/system/status");
      this.store.update({ coreReachable: true });
    } catch (e) {
      if (e.status === 401) { this._showPairing(e.detail); return; }
      if (!e.status) this.store.update({ coreReachable: false });
    }
    this._renderChrome();
    this._renderEmpty();
  }

  /**
   * This browser has no credential. On the computer JARVIS runs on it can be set
   * up directly (the person at the computer confirms); anywhere else it is
   * paired with a code from a device that is already set up.
   */
  async _showPairing(message) {
    // A later call without news (e.g. the event stream noticing too) keeps what is already explained.
    if ((!message || message === "not paired") && !$("pairing").hidden) message = $("pair-note").textContent;
    $("pairing").hidden = false;
    // Plain language, never internal error strings.
    if (!message || message === "not paired") message = "";
    else if (/revoked|invalid|expired|authentication required/i.test(message)) message = "This browser's access was removed or has expired.";
    const note = $("pair-note");
    note.hidden = !message;
    note.textContent = message;
    $("pair-error").textContent = "";
    let setup = null;
    try { setup = await api.setupStatus(); } catch { /* unreachable: the code form still works once it's back */ }
    const local = Boolean(setup?.this_computer);
    $("pair-title").textContent = !local ? "Pair this device" : setup.first_run ? "Set up this device" : "Set up this browser";
    $("pair-text").textContent = !local
      ? "JARVIS runs on another computer. On a device where JARVIS is already set up, open Devices and choose “Pair another device”, then enter the code it shows here."
      : setup.first_run
        ? "This is the computer JARVIS runs on. Make this browser your first device: you confirm it on this computer, no code needed."
        : "This browser isn't set up yet. You can approve it on this computer, no code needed.";
    $("setup-start").textContent = setup?.first_run ? "Set up this device" : "Set up this browser";
    $("setup-start-box").hidden = !local;
    $("setup-wait").hidden = true;
    $("pair-code-box").open = !local;
    $("pair-code-summary").hidden = !local;
    const waiting = local ? api.pendingSetup : null;
    if (waiting) this._waitForSetup(waiting);   // a reload while waiting carries on waiting
    else setTimeout(() => (local ? $("setup-start") : $("pair-code")).focus(), 0);
  }

  async _startSetup() {
    $("pair-error").textContent = "";
    const btn = $("setup-start");
    btn.dataset.loading = "true";
    try { this._waitForSetup(await api.requestSetup()); }
    catch (e) {
      $("pair-error").textContent = e.status === 429 ? "Too many attempts. Wait a minute, then try again."
        : e.status === 403 ? `${e.detail}.` : "Couldn't reach JARVIS on this computer.";
    }
    btn.dataset.loading = "false";
  }

  async _waitForSetup(req) {
    this._setupWaiting = req.id;
    $("setup-start-box").hidden = true;
    $("setup-wait").hidden = false;
    $("setup-code").textContent = req.code;
    $("setup-wait-text").textContent = req.approval === "window"
      ? "Confirm on this computer: JARVIS opened a small window asking whether to allow this browser. Check it shows the same code."
      : "Confirm on this computer: run the command below in a terminal and check it shows the same code.";
    while (this._setupWaiting === req.id) {
      let out;
      try { out = await api.claimSetup(req); }
      catch (e) {
        if (e.status === 404 || e.status === 403) out = { status: "expired" };
        else { await new Promise((r) => setTimeout(r, 3000)); continue; }   // connection blip: keep waiting
      }
      if (this._setupWaiting !== req.id) return;
      if (out.status === "approved") { location.reload(); return; }
      if (out.status === "pending") continue;
      this._setupWaiting = null;
      api.clearPendingSetup();
      $("setup-wait").hidden = true;
      $("setup-start-box").hidden = false;
      $("pair-error").textContent = { denied: "Setup was declined on this computer.", expired: "The request expired. Try again.",
        cancelled: "Setup was cancelled." }[out.status] || "Setup didn't complete. Try again.";
    }
  }

  _startupFailed(title, detail) {
    $("empty-line").textContent = title;
    $("startup").hidden = false;
    $("startup").replaceChildren(h("div", { class: "startup__row", dataset: { state: "fail" } }, detail));
    this._renderChrome();
  }

  // ------------------------------------------------------------- rendering
  _activeModel() {
    const roles = this.systemStatus?.roles;
    const ref = this.lastModel || roles?.fast?.active;
    return ref ? ref.split("/").slice(1).join("/") : null;
  }

  _renderEmpty() {
    if (!this.started) return;
    const s = this.systemStatus;
    $("greeting").textContent = greeting(s?.user_name);
    const model = s?.roles?.fast?.active;
    const provider = s?.providers?.find((p) => model && p.provider === model.split("/")[0]);
    const line = $("empty-line");
    line.textContent = "";
    if (!s) line.textContent = "";
    else if (!model) line.append("No model is available right now.");
    else line.append(provider?.local ? "Running on this computer with " : "Using ", h("b", { text: model.split("/").slice(1).join("/") }), ".");
    $("startup").hidden = true;
    const recent = this.sidebar.items.filter((c) => c.id !== this.conversationId).slice(0, 3);
    const box = $("empty-recent");
    box.hidden = !recent.length;
    box.replaceChildren(h("span", { class: "label", text: "Continue" }),
      ...recent.map((c) => h("button", { class: "recent-link", type: "button", onclick: () => this.openConversation(c.id) },
        h("span", { text: c.title || "Untitled" }), h("span", { text: relTime(c.updated_at) }))));
  }

  _renderChrome() {
    const f = this.store.facts;
    const conn = $("conn");
    const state = !f.coreReachable ? "disconnected" : f.link;
    conn.dataset.state = state;
    conn.title = { connected: "Connected", connecting: "Connecting", disconnected: "Offline" }[state];
    conn.querySelector(".dot").dataset.level = state === "connected" ? "ok" : state === "connecting" ? "busy" : "err";
    $("conn-text").textContent = state === "connected" ? "" : state === "connecting" ? "Connecting…" : "Offline";

    const s = this.systemStatus;
    // The orb carries state; words appear only when they add something.
    // An error shows briefly, like the face; after that the words follow what is actually going on
    // (the error itself stays in the conversation).
    const p = this.store.presence === PRESENCE.ERROR && presence.state !== "error" ? derivePresence({ ...f, error: null }) : this.store.presence;
    let text = "", tone_ = "";
    if (!f.coreReachable) text = this.started ? "Offline" : "Starting";
    else if (f.link !== "connected") text = "Connecting…";
    else if (p === PRESENCE.ERROR) { text = "Something went wrong"; tone_ = "err"; }
    else if (p === PRESENCE.WAITING) { text = "Waiting for your OK"; tone_ = "warn"; }
    else if (p === PRESENCE.LISTENING) text = "Listening";
    else if (p === PRESENCE.SPEAKING && f.audioPlaying) text = "Speaking";
    else if (p === PRESENCE.THINKING && this.activity) text = this.activity;
    const caption = $("caption");
    caption.textContent = text;
    caption.dataset.tone = tone_;
    const model = this._activeModel();
    const provider = s?.providers?.find((x) => this.lastModel ? x.provider === this.lastModel.split("/")[0] : x.provider === s?.roles?.fast?.active?.split("/")[0]);
    this.composer.setOffline(!f.coreReachable || f.link !== "connected");
    this.composer.setRouteInfo(model ? `${model}${provider ? (provider.local ? " · on this computer" : " · cloud") : ""}${s?.routing_mode === "local_only" ? " · local only" : ""}` : "");
    this._renderNotice();
  }

  _renderNotice() {
    const n = $("notice");
    const f = this.store.facts;
    const s = this.systemStatus;
    let content = null, level = "warn";
    if (!f.coreReachable && this.started) {
      content = [h("b", { text: "Can't reach JARVIS." }), " Retrying automatically."];
      level = "err";
    } else if (s && f.coreReachable && !s.roles?.fast?.active) {
      const down = (s.providers || []).filter((x) => !x.ok).map((x) => x.provider);
      content = [h("b", { text: "No model is available." }), ` ${down.includes("ollama") ? "Start Ollama, then try again." : "Check your model providers."}`,
        h("button", { class: "link", type: "button", onclick: () => this.sheets.open("settings", { section: "diagnostics" }) }, "Details")];
    }
    const shown = !n.hidden;
    n.hidden = !content;
    if (content) { n.dataset.level = level; n.replaceChildren(...content); }
    if (content && !shown) presence.emit({ type: "ui.notice", level, element: n });
  }

  _renderCounts() {
    const tasks = this.taskStrip.activeCount;
    const tc = $("tasks-count");
    tc.textContent = tasks ? String(tasks) : "";
    tc.dataset.level = tasks ? "busy" : "";
    const sc = $("settings-count");
    sc.textContent = this.requests.count ? `${this.requests.count} waiting` : "";
    sc.dataset.level = this.requests.count ? "warn" : "";
  }

  _onState({ changed, facts }) {
    presence.syncFacts(facts);
    if (changed) this._renderChrome();
  }

  /** Real audio level for the presence: "mic" (someone speaking to JARVIS) or "speaker" (JARVIS speaking). */
  presenceLevel(v, source = "mic") { presence.level(source, v); }

  setVoiceArmed(state) {
    const btn = $("mic-armed");
    const on = state && state !== "off" && state !== "error";
    btn.hidden = !on;
    btn.dataset.state = state || "off";
    if (this.voice.status?.pipeline) this.voice.status.pipeline.state = state;
    if (this.voice.status?.pipeline) this.voice.status.pipeline.running = on;
  }

  /** A reply with substantial content (code, a table) just appeared: the presence may look at it. */
  _showContent() {
    const block = document.querySelector(".messages > .msg--jarvis:last-child :is(pre, table)");
    if (block) presence.emit({ type: "ui.content", element: block });
  }

  _setTitle(title) {
    $("conv-title").textContent = title || "";
    document.title = title ? `${title} · JARVIS` : "JARVIS";
  }

  // ---------------------------------------------------------------- events
  _onEvent(evt) {
    this.recentEvents.push(evt);
    if (this.recentEvents.length > 400) this.recentEvents.shift();
    this.requests.onEvent(evt);
    this.taskStrip.onEvent(evt);
    this.sheets.onEvent(evt);
    presence.dispatch(evt.type, evt.data || {});
    const d = evt.data || {};
    switch (evt.type) {
      case "AGENT_STATE": {
        const busy = d.state !== "idle" && d.state !== "error";
        if (busy) this.runs.set(d.conversation_id, d.state); else this.runs.delete(d.conversation_id);
        this.sidebar.setBusy(d.conversation_id, busy);
        this.activity = d.state === "acting" && d.tools?.length ? `Running ${d.tools.join(", ")}…` : null;
        presence.setActivity(d.state === "acting" ? d.tools?.[0] : null);
        // Busy comes from the core's own run events (the request's local lifetime lags them slightly).
        this.store.update({ agentBusy: this.runs.size > 0, streaming: false });
        this._renderChrome();
        break;
      }
      case "MODEL_RESPONSE_DELTA":
        this.store.update({ streaming: true });
        break;
      case "MODEL_RESPONSE_COMPLETED":
        this.lastModel = `${d.provider}/${d.model}`;
        break;
      case "MIC_STATE":
        // standby = wake word armed (mic open, nothing captured yet); recording/followup = taking a command
        this.store.update({ voiceListening: d.state === "recording" || d.state === "followup" || d.state === "listening" });
        this.setVoiceArmed(d.state);
        break;
      case "TRANSCRIPT":
        if (d.text && this.voice.pipelineRunning) this.sidebar.scheduleRefresh();
        break;
      case "AUDIO_LEVEL":   // real levels from the core's voice pipeline
        this.presenceLevel(d.level || 0, d.source === "speaker" ? "speaker" : "mic");
        break;
      case "SPEECH_STARTED": this.store.update({ audioPlaying: true }); break;
      case "SPEECH_ENDED": case "SPEECH_INTERRUPTED": this.store.update({ audioPlaying: false }); break;
      case "CHAT_MESSAGE": this.sidebar.scheduleRefresh(); break;
      case "PROVIDER_STATUS": case "MEMORY_CREATED": case "MEMORY_DELETED": this.refreshStatus(); break;
      case "DEVICE_REVOKED":
        if (d.device_id === api.session?.device_id) { api.forget(); this._showPairing("This browser was removed from JARVIS."); }
        break;
      case "NOTIFICATION":
        if (!d.task_id) toast(d.title || "JARVIS", d.body || "", d.level || "info");
        break;
      default:
    }
  }

  // ------------------------------------------------------------------ chat
  async send(text, { voice = false } = {}) {
    if (this.localRun) return;
    this.store.update({ error: null });
    if (this.conversation.isEmpty) this._setTitle(text.length > 60 ? text.slice(0, 60) + "…" : text);
    this.conversation.addUser(text);
    presence.emit({ type: "user.submitted", text });
    tone("send");
    const controller = new AbortController();
    this.localRun = controller;
    this.composer.setBusy(true);
    this.store.update({ agentBusy: true });
    this.conversation.startReply();
    let finished = false;
    // A spoken reply starts speaking while it is still being written (not after it is done).
    let spoken = null;
    try {
      await api.chat({ message: text, conversation_id: this.conversationId,
        device: { capabilities: api.browserCapabilities(), voice } }, (event, data) => {
        switch (event) {
          case "start":
            if (this.conversationId !== data.conversation_id) {
              this.conversationId = data.conversation_id;
              sessionStorage.setItem("jarvis.conversation", data.conversation_id);
              this.sidebar.scheduleRefresh();
            }
            break;
          case "delta":
            this.conversation.appendDelta(data.text);
            presence.emit({ type: "agent.output.started" });
            this.store.update({ streaming: true });
            if (this.voice.speakNextReply && this.voice.ttsAvailable) (spoken ||= this.voice.startReply()).push(data.text);
            break;
          case "tool_start": this.conversation.toolStart(data); break;
          case "tool_end": this.conversation.toolEnd(data); break;
          case "approval": this.requests.onEvent({ type: "PERMISSION_REQUESTED", data }); break;
          case "approval_decided": this.requests.onEvent({ type: data.approved ? "PERMISSION_GRANTED" : "PERMISSION_DENIED", data }); break;
          case "done":
            finished = true;
            this.runs.delete(data.conversation_id);   // the run is over; don't wait for the event echo
            this.conversation.finish(data);
            if (data.route?.model) this.lastModel = `${data.route.provider}/${data.route.model}`;
            presence.emit({ type: "agent.reply.completed", text: data.text, error: data.error, stopped: data.stop_reason === "cancelled" });
            this._showContent();
            if (data.error || data.stop_reason === "cancelled") spoken?.cancel();
            if (data.error) this.store.update({ error: data.error });
            else if (spoken) spoken.end();
            else if (this.voice.speakNextReply) this.voice.speak(data.text);  // nothing was streamed
            this.voice.speakNextReply = false;
            tone("done");
            break;
          case "error":
            finished = true;
            spoken?.cancel();
            this.voice.speakNextReply = false;
            this.conversation.fail(`Something went wrong: ${data.error}`);
            presence.emit({ type: "agent.reply.completed", error: data.error });
            this.store.update({ error: data.error });
            break;
          default:
        }
      }, controller.signal);
      if (!finished) {
        spoken?.cancel();
        this.conversation.fail("The connection closed before the reply finished.");
        this.store.update({ error: "connection closed" });
      }
    } catch (e) {
      spoken?.cancel();
      if (e.name === "AbortError") this.conversation.finish({ text: "", stop_reason: "cancelled" });
      else {
        const msg = e.status ? `JARVIS couldn't take that request (${e.detail}).` : "Couldn't reach JARVIS. Your message wasn't sent.";
        this.conversation.fail(msg);
        this.store.update({ error: msg });
      }
    } finally {
      this.localRun = null;
      this.composer.setBusy(false);
      this.store.update({ agentBusy: this.runs.size > 0, streaming: false });
      this._renderChrome();
    }
  }

  async stop() {
    if (this.voice.speaking || this.store.facts.audioPlaying) presence.emit({ type: "user.interrupted" });
    else if (this.localRun) presence.emit({ type: "user.cancelled" });
    if (this.voice.speaking) this.voice.stopSpeaking();
    if (this.conversationId && this.localRun) {
      try { await api.post(`/chat/${this.conversationId}/cancel`); } catch { this.localRun?.abort(); }
    }
  }

  newConversation() {
    this.conversationId = null;
    sessionStorage.removeItem("jarvis.conversation");
    this.conversation.clear();
    this.store.update({ error: null });
    this.sidebar.markCurrent(null);
    this._setTitle("");
    this._renderEmpty();
    this._closeDrawer();
    this.composer.focus();
  }

  async openConversation(id, quiet = false) {
    try {
      const conv = await api.get(`/conversations/${id}`);
      this.conversationId = id;
      sessionStorage.setItem("jarvis.conversation", id);
      this.conversation.load(conv);
      this._setTitle(conv.title);
      this.sidebar.markCurrent(id);
      this.store.update({ error: null });
      this._closeDrawer();
    } catch (e) {
      if (!quiet) toast("Couldn't open that conversation", e.detail || "", "warn");
      if (e.status === 404) this.newConversation();
    }
  }

  // -------------------------------------------------------------- commands
  commands() {
    const p = prefs();
    return [
      { label: "New conversation", run: () => this.newConversation() },
      { label: "Stop", disabled: !this.localRun && !this.voice.speaking, why: "Nothing to stop", run: () => this.stop() },
      { label: this.voice.capturing ? "Stop listening" : "Talk to JARVIS", hint: "Ctrl+Shift+Space", disabled: !this.voice.canCapture, why: this.voice.reason, run: () => this.voice.toggle() },
      { label: this.voice.pipelineRunning ? "Stop listening for “Hey Jarvis”" : "Listen for “Hey Jarvis”", keywords: "voice wake word microphone",
        run: () => this.toggleWakeWord() },
      { label: "Voice settings", keywords: "speech microphone wake", run: () => this.sheets.open("settings", { section: "voice" }) },
      { label: "Tasks", run: () => this.sheets.open("tasks") },
      { label: "Memory", keywords: "remember forget", run: () => this.sheets.open("memory") },
      { label: "Devices", keywords: "phone pair revoke", run: () => this.sheets.open("devices") },
      { label: "Settings", run: () => this.sheets.open("settings") },
      { label: "Personality", keywords: "name tone identity", run: () => this.sheets.open("settings", { section: "personality" }) },
      { label: "Models", keywords: "ollama routing local cloud", run: () => this.sheets.open("settings", { section: "models" }) },
      { label: "Permissions", keywords: "approvals grants", run: () => this.sheets.open("settings", { section: "permissions" }) },
      { label: "Activity", keywords: "events log", run: () => this.sheets.open("settings", { section: "activity" }) },
      { label: "Diagnostics", keywords: "doctor health", run: () => this.sheets.open("settings", { section: "diagnostics" }) },
      { label: `Switch to ${document.documentElement.dataset.theme === "dark" ? "light" : "dark"} theme`, keywords: "appearance",
        run: () => { const t = document.documentElement.dataset.theme === "dark" ? "light" : "dark"; setPref("theme", t); applyTheme(t); } },
      { label: p.sound ? "Turn sounds off" : "Turn sounds on", run: () => setPref("sound", !p.sound) },
      { label: "Toggle sidebar", hint: "Ctrl+B", run: () => this.toggleSidebar() },
      { label: "Open compact view", keywords: "hud", run: () => window.open("/hud", "jarvis-hud", "width=420,height=140") },
    ];
  }

  async toggleWakeWord() {
    const on = this.voice.pipelineRunning;
    try {
      const st = await api.post(on ? "/voice/stop" : "/voice/start");
      this.voice.status = st;
      this.setVoiceArmed(st.pipeline.running ? st.pipeline.state : "off");
      toast(on ? "Voice is off" : "Listening for “Hey Jarvis”", on ? "The microphone is closed." : "Audio stays on this computer.");
    } catch (e) { toast("Couldn't change voice", e.detail || String(e), "warn"); }
  }

  toggleSidebar() {
    const app = $("app");
    if (NARROW.matches) {
      if (app.dataset.drawer === "open") this._closeDrawer(); else this._openDrawer();
    } else {
      const open = app.dataset.sidebar !== "open";
      app.dataset.sidebar = open ? "open" : "closed";
      setPref("sidebar", open ? "open" : "closed");
    }
  }

  _openDrawer() {
    const app = $("app");
    app.dataset.drawer = "open";
    const scrim = h("div", { class: "scrim", onclick: () => this._closeDrawer() });
    scrim.id = "drawer-scrim";
    document.body.appendChild(scrim);
    $("new-chat").focus();
  }

  _closeDrawer() {
    $("app").dataset.drawer = "closed";
    $("drawer-scrim")?.remove();
  }

  // ------------------------------------------------------------------ bind
  _bind() {
    $("app").dataset.sidebar = prefs().sidebar === "closed" ? "closed" : "open";
    this.composer.addEventListener("submit", (e) => this.send(e.detail));
    this.composer.addEventListener("stop", () => this.stop());
    this.composer.addEventListener("voice", () => this.voice.toggle());
    this.composer.addEventListener("focus", () => { if (this.store.facts.error) this.store.update({ error: null }); });
    $("new-chat").addEventListener("click", () => this.newConversation());
    $("hide-sidebar").addEventListener("click", () => this.toggleSidebar());
    $("show-sidebar").addEventListener("click", () => this.toggleSidebar());
    $("open-palette").addEventListener("click", () => this.palette.toggle());
    $("mic-armed").addEventListener("click", async () => {
      try { await api.post("/voice/stop"); this.setVoiceArmed("off"); toast("Voice is off", "The microphone is closed."); }
      catch (e) { toast("Couldn't turn voice off", e.detail, "warn"); }
    });
    document.querySelectorAll(".navitem").forEach((b) => b.addEventListener("click", () => { this._closeDrawer(); this.sheets.open(b.dataset.view); }));
    $("setup-start").addEventListener("click", () => this._startSetup());
    $("setup-cancel").addEventListener("click", async () => {
      const req = api.pendingSetup;
      this._setupWaiting = null;
      $("setup-wait").hidden = true;
      $("setup-start-box").hidden = false;
      if (req) await api.cancelSetup(req);
    });
    $("pair-form").addEventListener("submit", async (e) => {
      e.preventDefault();
      const btn = e.submitter;
      if (btn) btn.dataset.loading = "true";
      try { await api.pair($("pair-code").value.trim()); location.reload(); }
      catch (err) { $("pair-error").textContent = err.status === 401 ? "That code is invalid or has expired." : err.detail || "Couldn't connect."; }
      if (btn) btn.dataset.loading = "false";
    });
    window.addEventListener("keydown", (e) => {
      const k = e.key.toLowerCase();
      if ((e.ctrlKey || e.metaKey) && k === "k") { e.preventDefault(); this.palette.toggle(); }
      else if ((e.ctrlKey || e.metaKey) && k === "b") { e.preventDefault(); this.toggleSidebar(); }
      else if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.code === "Space") { e.preventDefault(); this.voice.toggle(); }
      else if (e.key === "Escape") {
        if (this.voice.capturing) this.voice.cancelCapture();
        else if (this.voice.speaking) { presence.emit({ type: "user.interrupted" }); this.voice.stopSpeaking(); }
        else if ($("app").dataset.drawer === "open") this._closeDrawer();
      } else if (e.key === "/" && !/INPUT|TEXTAREA|SELECT/.test(document.activeElement?.tagName) && !this.sheets.isOpen) {
        e.preventDefault(); this.composer.focus();
      }
    });
    NARROW.addEventListener("change", () => this._closeDrawer());
    // A panel or the command palette opening draws a brief glance.
    const glanceAtNew = (records) => {
      for (const r of records) for (const n of r.addedNodes) {
        const panel = n instanceof Element ? (n.matches(".sheet, .palette") ? n : n.querySelector(".sheet, .palette")) : null;
        if (panel) presence.emit({ type: "ui.panel.opened", element: panel });
      }
    };
    for (const id of ["sheet-layer", "palette-layer"]) new MutationObserver(glanceAtNew).observe($(id), { childList: true });
    window.addEventListener("hashchange", () => { if (new URLSearchParams(location.hash.slice(1)).get("pair")) location.reload(); });
    if ("serviceWorker" in navigator) navigator.serviceWorker.register("/sw.js").catch(() => {});
  }
}

window.jarvis = new App();
window.jarvis.start();

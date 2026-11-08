// Side sheets for secondary work: Tasks, Memory, Devices, Settings.
// Every value comes from the API; unavailable data is described, never invented.

import { api } from "./api.js";
import { ICONS, applyTheme, clock, h, prefs, relTime, setPref, svgIcon, toast } from "./ui.js";

const TITLES = { tasks: "Tasks", memory: "Memory", devices: "Devices", settings: "Settings" };
const SETTINGS = [
  ["general", "General"], ["voice", "Voice"], ["personality", "Personality"], ["models", "Models"], ["permissions", "Permissions"],
  ["activity", "Activity"], ["diagnostics", "Diagnostics"], ["about", "About"],
];
const STATUS_LEVEL = { completed: "ok", running: "busy", planning: "busy", pending: "busy", failed: "err", cancelled: "", interrupted: "warn" };

function item(title, sub, side, onclick) {
  return h("div", { class: `item${onclick ? " item--click" : ""}`, onclick, tabindex: onclick ? "0" : null,
    onkeydown: onclick ? (e) => { if (e.key === "Enter") onclick(); } : null },
  h("div", { class: "item__main" }, h("div", { class: "item__title", text: title }), sub ? h("div", { class: "item__sub", text: sub }) : null),
  h("div", { class: "item__side" }, side));
}

function tag(text, level) { return h("span", { class: "tag", dataset: { level: level || "" } }, text); }

export class Sheets {
  constructor(layer, app) {
    this.layer = layer;
    this.app = app;
    this.view = null;
  }

  get isOpen() { return Boolean(this.view); }

  open(view, opts = {}) {
    this.view = view;
    this.opts = opts;
    this._restore = this._restore || document.activeElement;
    const isSettings = view === "settings";
    const close = h("button", { class: "btn btn--ghost btn--icon", type: "button", "aria-label": "Close", onclick: () => this.close() }, svgIcon(ICONS.close));
    const body = h("div", { class: "sheet__body" });
    const sheet = h("aside", { class: `sheet${isSettings ? " sheet--settings" : ""}`, role: "dialog", "aria-modal": "true", "aria-label": TITLES[view] },
      h("div", { class: "sheet__head" }, h("h2", { class: "sheet__title", text: TITLES[view] }), close), body);
    sheet.addEventListener("keydown", (e) => { if (e.key === "Escape") { e.stopPropagation(); this.close(); } });
    this.layer.replaceChildren(h("div", { class: "sheet-scrim", onclick: () => this.close() }), sheet);
    document.querySelectorAll(".navitem").forEach((b) => b.setAttribute("aria-current", String(b.dataset.view === view)));
    if (isSettings) {
      const content = h("div", { class: "sheet__content" });
      const nav = h("nav", { class: "sheet__nav", "aria-label": "Settings sections" });
      const section = opts.section || this._section || "general";
      for (const [id, label] of SETTINGS) {
        nav.appendChild(h("button", { type: "button", "aria-current": String(id === section), onclick: () => this.open("settings", { section: id }) }, label));
      }
      this._section = section;
      body.append(nav, content);
      this._fill(content, () => this._settings(section, content));
      nav.querySelector('[aria-current="true"]')?.focus();
    } else {
      this._fill(body, () => this[`_${view}`](body));
      close.focus();
    }
  }

  close() {
    this.view = null;
    this.layer.textContent = "";
    document.querySelectorAll(".navitem").forEach((b) => b.setAttribute("aria-current", "false"));
    this._restore?.focus?.();
    this._restore = null;
  }

  refresh() { if (this.view) this.open(this.view, { ...this.opts, section: this._section }); }

  onEvent(evt) {
    if (!this.view) return;
    if (this.view === "tasks" && evt.type.startsWith("TASK_")) { clearTimeout(this._rt); this._rt = setTimeout(() => this.refresh(), 300); }
    if (this.view === "settings" && this._section === "activity") this._activityDraw?.();
    if (this.view === "devices" && evt.type.startsWith("DEVICE_")) this.refresh();
  }

  async _fill(target, fn) {
    target.replaceChildren(h("p", { class: "empty-note", text: "Loading…" }));
    try {
      const nodes = await fn();
      if (nodes) target.replaceChildren(...[nodes].flat().filter(Boolean));
    } catch (e) {
      target.replaceChildren(h("p", { class: "empty-note", text: e.status === 404 ? "Not available in this version." : `Couldn't load this: ${e.detail || e.message}` }));
    }
  }

  // ------------------------------------------------------------------ tasks
  async _tasks() {
    if (this.opts.taskId) return this._taskDetail(this.opts.taskId);
    const tasks = await api.get("/tasks?limit=60");
    const goal = h("input", { class: "input", placeholder: "Describe a goal with several steps…", "aria-label": "New task goal" });
    const start = h("button", { class: "btn btn--primary", type: "submit" }, "Start");
    const form = h("form", { class: "row-form", onsubmit: async (e) => {
      e.preventDefault();
      if (!goal.value.trim()) return;
      start.dataset.loading = "true";
      try { const t = await api.post("/tasks", { goal: goal.value.trim() }); this.open("tasks", { taskId: t.id }); }
      catch (err) { toast("Couldn't start the task", err.detail, "warn"); start.dataset.loading = "false"; }
    } }, goal, start);
    return [
      h("section", { class: "section" }, h("p", { class: "section__intro", text: "Background work that JARVIS plans and carries out step by step." }), form),
      h("section", { class: "section" }, h("span", { class: "label", text: "Recent" }),
        tasks.length ? h("div", { class: "list" }, tasks.map((t) => item(t.goal, `${relTime(t.created_at)}${t.error ? ` · ${t.error.slice(0, 80)}` : ""}`,
          tag(t.status === "running" ? `${Math.round(t.progress * 100)}%` : t.status, STATUS_LEVEL[t.status]),
          () => this.open("tasks", { taskId: t.id }))))
          : h("p", { class: "empty-note", text: "No tasks yet. Ask JARVIS for something with several steps, or start one here." })),
    ];
  }

  async _taskDetail(id) {
    const t = await api.get(`/tasks/${id}`);
    const back = h("button", { class: "btn btn--ghost", type: "button", onclick: () => this.open("tasks") }, "← All tasks");
    const actions = h("div", { class: "row-form" });
    if (t.running) actions.appendChild(h("button", { class: "btn", type: "button", onclick: async () => { await api.post(`/tasks/${id}/cancel`); this.refresh(); } }, "Cancel"));
    else if (["failed", "cancelled", "interrupted"].includes(t.status)) actions.appendChild(h("button", { class: "btn", type: "button", onclick: async () => {
      try { await api.post(`/tasks/${id}/resume`); this.refresh(); } catch (e) { toast("Couldn't resume", e.detail, "warn"); }
    } }, "Resume"));
    const ver = t.verification;
    return [
      h("div", { class: "row-form" }, back),
      h("section", { class: "section" },
        h("div", { class: "item__title", style: "font-size:15px;font-weight:600", text: t.goal }),
        h("div", { class: "item__sub", text: `${t.status}${t.running ? " · running now" : ""} · started ${relTime(t.created_at)}` }),
        h("div", { class: "progress", dataset: { level: t.status === "completed" ? "ok" : t.status === "failed" ? "err" : "" } },
          h("i", { style: `width:${Math.round((t.progress || 0) * 100)}%` }))),
      actions.children.length ? actions : null,
      t.result ? h("section", { class: "section" }, h("span", { class: "label", text: "Result" }), h("p", { style: "white-space:pre-wrap;margin:0", text: t.result })) : null,
      ver ? h("section", { class: "section" }, h("span", { class: "label", text: "Verification" }),
        h("dl", { class: "kv" },
          h("dt", { text: "Goal achieved" }), h("dd", { text: ver.achieved === true ? "Yes" : ver.achieved === false ? "No" : "Couldn't verify" }),
          h("dt", { text: "Actions run" }), h("dd", { text: `${ver.actions_executed ?? "?"} (${ver.actions_failed ?? 0} failed)` }),
          ver.issues?.length ? [h("dt", { text: "Issues" }), h("dd", { text: ver.issues.join("; ") })] : null)) : null,
      t.error && t.status !== "completed" ? h("section", { class: "section" }, h("span", { class: "label", text: "Problem" }), h("p", { style: "margin:0;color:var(--err)", text: t.error })) : null,
      h("section", { class: "section" }, h("span", { class: "label", text: "Steps" }),
        t.nodes.length ? h("ol", { class: "nodes" }, t.nodes.map((n) => h("li", { class: "node", dataset: { status: n.status } },
          h("span", { class: "step__mark" }),
          h("div", {}, h("div", { text: `${n.title}${n.attempts > 1 ? ` (attempt ${n.attempts})` : ""}` }),
            h("div", { class: "node__out", text: n.error || n.output || (n.status === "pending" ? "Waiting" : n.status) })))))
          : h("p", { class: "empty-note", text: t.status === "planning" ? "Planning…" : "No steps yet." })),
    ];
  }

  // ----------------------------------------------------------------- memory
  async _memory(body) {
    const search = h("input", { class: "search", type: "search", placeholder: "Search memory", "aria-label": "Search memory" });
    const list = h("div", { class: "list" });
    const status = h("p", { class: "section__intro" });
    const add = h("input", { class: "input", placeholder: "Something JARVIS should remember…", "aria-label": "New memory" });
    const load = async () => {
      const q = search.value.trim();
      const data = await api.get(`/memory?limit=150${q ? `&q=${encodeURIComponent(q)}` : ""}`);
      const c = data.counts;
      status.textContent = `${c.long_term || 0} facts · ${c.episodic || 0} conversation summaries · ${c.semantic || 0} document passages. ` +
        (data.embedding.startsWith("ok") ? "Search uses meaning and keywords." : data.embedding === "unknown" ? "" : "Meaning-based search is unavailable; using keywords.");
      list.replaceChildren(...(data.items.length ? data.items.map((m) => item(m.content,
        `${m.layer === "long_term" ? m.kind : m.layer.replace("_", " ")} · ${relTime(m.created_at)}${m.score ? ` · match ${m.score}` : ""}`,
        h("button", { class: "btn btn--ghost", type: "button", title: "Delete permanently", onclick: async (e) => {
          e.currentTarget.dataset.loading = "true";
          await api.del(`/memory/${m.id}`);
          load();
        } }, "Forget"))) : [h("p", { class: "empty-note", text: q ? "Nothing matches." : "Nothing remembered yet. Tell JARVIS “remember that…”." })]));
    };
    let t;
    search.addEventListener("input", () => { clearTimeout(t); t = setTimeout(load, 220); });
    const exportBtn = h("button", { class: "btn", type: "button", onclick: async () => {
      const data = await api.get("/memory/export");
      const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }));
      h("a", { href: url, download: "jarvis-memory.json" }).click();
      URL.revokeObjectURL(url);
    } }, "Export");
    await load();
    return [
      h("div", { class: "row-form" }, search, exportBtn),
      h("form", { class: "row-form", onsubmit: async (e) => {
        e.preventDefault();
        if (!add.value.trim()) return;
        await api.post("/memory", { content: add.value.trim() });
        add.value = "";
        load();
      } }, add, h("button", { class: "btn", type: "submit" }, "Remember")),
      status, list,
    ];
  }

  // ---------------------------------------------------------------- devices
  async _devices() {
    const devices = await api.get("/devices");
    const pair = h("div");
    return [
      h("section", { class: "section" }, h("p", { class: "section__intro", text: "Every browser, phone, terminal and piece of hardware that can reach JARVIS. Revoking a device disconnects it immediately." }),
        h("div", { class: "list" }, devices.map((d) => item(
          `${d.name}${d.current ? " (this browser)" : ""}`,
          `${d.kind} · ${d.role === "owner" ? "full access" : d.role === "client" ? `up to ${d.max_risk.toLowerCase().replace("_", " ")}` : "hardware"} · ${d.last_seen ? `seen ${relTime(d.last_seen)}` : "never seen"}`,
          [d.revoked ? tag("Revoked", "err") : tag(d.online ? "Online" : "Offline", d.online ? "ok" : ""),
            !d.revoked && !d.current ? h("button", { class: "btn btn--ghost", type: "button", onclick: async () => {
              if (!confirm(`Revoke ${d.name}? It will be disconnected immediately.`)) return;
              try { await api.post(`/devices/${d.id}/revoke`); this.refresh(); } catch (e) { toast("Couldn't revoke", e.detail, "warn"); }
            } }, "Revoke") : null])))),
      h("section", { class: "section" }, h("span", { class: "label", text: "Pair another device" }),
        h("p", { class: "section__intro", text: "For a phone, tablet or another computer: this creates a one-time code (valid 10 minutes). Open the link on that device, or enter the code there. A browser on this computer can simply choose “Set up this browser” instead." }),
        h("button", { class: "btn", type: "button", onclick: async (e) => {
          const btn = e.currentTarget;
          btn.dataset.loading = "true";
          try {
            const out = await api.post("/auth/pairing-codes", { kind: "mobile", role: "client", max_risk: "SENSITIVE", name_hint: "New device" });
            pair.replaceChildren(h("dl", { class: "kv", style: "margin-top:12px" },
              h("dt", { text: "Code" }), h("dd", { class: "mono", text: out.code }),
              h("dt", { text: "Link" }), h("dd", { class: "mono", text: out.url }),
              h("dt", { text: "Expires" }), h("dd", { text: clock(out.expires_at) })));
          } catch (err) { toast("Couldn't create a code", err.status === 403 ? "Only owner devices can add devices." : err.detail, "warn"); }
          btn.dataset.loading = "false";
        } }, "Pair another device"), pair),
    ];
  }

  // --------------------------------------------------------------- settings
  async _settings(section) {
    switch (section) {
      case "general": return this._general();
      case "voice": return this._voiceSettings();
      case "personality": return this._personality();
      case "models": return this._models();
      case "permissions": return this._permissions();
      case "activity": return this._activity();
      case "diagnostics": return this._diagnostics();
      case "about": return this._about();
      default: return h("p", { class: "empty-note", text: "Unknown section." });
    }
  }

  _seg(options, current, onPick) {
    const seg = h("div", { class: "seg", role: "group" });
    for (const [value, label] of options) {
      seg.appendChild(h("button", { type: "button", "aria-pressed": String(value === current), onclick: (e) => {
        seg.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b === e.currentTarget)));
        onPick(value);
      } }, label));
    }
    return seg;
  }

  _general() {
    const p = prefs();
    return [
      h("section", { class: "section" }, h("span", { class: "label", text: "Appearance" }),
        h("div", { class: "field" }, h("span", { text: "Theme" }),
          this._seg([["system", "System"], ["light", "Light"], ["dark", "Dark"]], p.theme || "system", (v) => { setPref("theme", v); applyTheme(v); })),
        ),
      h("section", { class: "section" }, h("span", { class: "label", text: "Sound" }),
        h("div", { class: "field" }, h("span", { text: "Quiet tones for send, finish and approvals" }),
          this._seg([["off", "Off"], ["on", "On"]], p.sound ? "on" : "off", (v) => setPref("sound", v === "on")))),
    ];
  }

  async _voiceSettings() {
    const st = await api.get("/voice/status");
    this.app.voice.status = st;
    const p = st.pipeline;
    const ready = (x) => tag(x.available ? "Ready" : "Unavailable", x.available ? "ok" : "warn");
    const toggle = h("button", { class: p.running ? "btn" : "btn btn--primary", type: "button", onclick: async (e) => {
      e.currentTarget.dataset.loading = "true";
      await this.app.toggleWakeWord();
      this.refresh();
    } }, p.running ? "Stop listening" : "Listen for “Hey Jarvis”");
    const test = h("button", { class: "btn", type: "button", disabled: !st.tts.available || null, onclick: async (e) => {
      e.currentTarget.dataset.loading = "true";
      await this.app.voice.speak("This is how I sound.");
      e.currentTarget.dataset.loading = "false";
    } }, "Play a sample");
    return [
      h("section", { class: "section" },
        h("p", { class: "section__intro", text: "Everything runs on this computer: the wake word, speech recognition and the voice. Audio is never sent anywhere." }),
        h("div", { class: "row-form" }, toggle),
        h("p", { class: "section__intro", text: p.running
          ? `Listening on ${st.microphone.name || "the default microphone"}. Say “Hey Jarvis”, then your request. You can talk over JARVIS to interrupt.`
          : "When on, the microphone stays open and only reacts after “Hey Jarvis”. The mic icon at the top shows when it's on." }),
        p.error ? h("p", { class: "section__intro", style: "color:var(--err)", text: p.error }) : null),
      h("section", { class: "section" }, h("span", { class: "label", text: "Components" }),
        h("div", { class: "list" },
          item("Speech recognition", `${st.stt.engine} · ${st.stt.model}${st.stt.loaded ? " · loaded" : ""}${st.stt.reason ? ` · ${st.stt.reason}` : ""}`, ready(st.stt)),
          item("Voice", `${st.tts.engine || "none"}${st.tts.voice ? ` · ${st.tts.voice}` : ""}${st.tts.reason ? ` · ${st.tts.reason}` : ""}`, [test, ready(st.tts)]),
          item("Wake word", `${st.wake.word.replace("_", " ")}${st.wake.reason ? ` · ${st.wake.reason}` : ""}`, ready(st.wake)),
          item("Microphone", st.microphone.available ? st.microphone.name : st.microphone.reason, tag(st.microphone.available ? "Found" : "None", st.microphone.available ? "ok" : "err")))),
      p.stats ? h("section", { class: "section" }, h("span", { class: "label", text: "This session" }),
        h("dl", { class: "kv" },
          h("dt", { text: "Wake words heard" }), h("dd", { text: String(p.stats.wakes) }),
          h("dt", { text: "Requests" }), h("dd", { text: String(p.stats.utterances) }),
          h("dt", { text: "Interruptions" }), h("dd", { text: String(p.stats.interruptions) }),
          p.stats.last_transcript ? [h("dt", { text: "Last heard" }), h("dd", { text: `“${p.stats.last_transcript}”` })] : null)) : null,
      h("section", { class: "section" }, h("span", { class: "label", text: "Voice in this browser" }),
        h("p", { class: "section__intro", text: "The microphone button next to the message box records here and transcribes on the computer running JARVIS. Replies to spoken messages are read aloud." })),
    ];
  }

  async _personality() {
    const ident = await api.get("/system/identity");
    const fields = [["name", "Name"], ["user_name", "What JARVIS calls you"], ["tone", "Tone"], ["formality", "Formality"],
      ["verbosity", "Length of answers"], ["humor", "Humour"], ["response_style", "Style"]];
    const inputs = {};
    const behaviors = h("textarea", { rows: "4" });
    behaviors.value = (ident.behaviors || []).join("\n");
    const save = h("button", { class: "btn btn--primary", type: "submit" }, "Save");
    return h("form", { onsubmit: async (e) => {
      e.preventDefault();
      save.dataset.loading = "true";
      const patch = { behaviors: behaviors.value.split("\n").map((s) => s.trim()).filter(Boolean) };
      for (const [f] of fields) patch[f] = inputs[f].value.trim() || (f === "name" ? "JARVIS" : null);
      try { await api.put("/system/identity", patch); toast("Saved", "JARVIS will use this on every device.", "ok"); this.app.refreshStatus(); }
      catch (err) { toast("Couldn't save", err.status === 403 ? "Only owner devices can change this." : err.detail, "warn"); }
      save.dataset.loading = "false";
    } },
    h("p", { class: "section__intro", text: "This is part of JARVIS itself, so it follows you to every device." }),
    fields.map(([f, label]) => { inputs[f] = h("input", { value: ident[f] ?? "" }); return h("label", { class: "field" }, h("span", { text: label }), inputs[f]); }),
    h("label", { class: "field" }, h("span", { text: "Standing preferences (one per line)" }), behaviors), save);
  }

  async _models() {
    const data = await api.get("/models");
    const roleNames = { fast: "Everyday", reasoning: "Reasoning", coding: "Code", vision: "Images", embedding: "Memory search", fallback: "Fallback" };
    return [
      h("section", { class: "section" }, h("span", { class: "label", text: "Where JARVIS may think" }),
        h("div", { class: "field" }, this._seg([["auto", "Automatic"], ["local_only", "This computer only"], ["cloud_preferred", "Prefer cloud"]], data.routing.mode,
          async (v) => { try { await api.put("/models/routing", { mode: v }); this.app.refreshStatus(); } catch (e) { toast("Couldn't change routing", e.detail, "warn"); } })),
        h("p", { class: "section__intro", text: "“This computer only” never sends your conversations to a cloud provider." })),
      h("section", { class: "section" }, h("span", { class: "label", text: "Providers" }),
        h("div", { class: "list" }, data.providers.map((p) => item(p.name,
          `${p.local ? "on this computer" : "cloud"} · ${p.models.length} model${p.models.length === 1 ? "" : "s"}${p.health.error ? ` · ${p.health.error}` : ""}`,
          tag(p.health.ok ? (p.health.latency_ms != null ? `${Math.round(p.health.latency_ms)} ms` : "Available") : "Unavailable", p.health.ok ? "ok" : "err"))))),
      h("section", { class: "section" }, h("span", { class: "label", text: "Which model does what" }),
        h("dl", { class: "kv" }, Object.entries(data.routing.roles).flatMap(([role, info]) => [
          h("dt", { text: roleNames[role] || role }), h("dd", { class: "mono", style: "font-size:12.5px", text: info.candidates.join(" → ") })]))),
    ];
  }

  async _permissions() {
    const [policy, history, grants] = await Promise.all([api.get("/permissions/policy"), api.get("/permissions/approvals?state=all"), api.get("/permissions/grants")]);
    const label = { allow: "Automatic", ask: "Ask me", deny: "Never" };
    const levels = { SAFE: "Reading and calculating", LOW_RISK: "Ordinary actions", SENSITIVE: "Sensitive actions", DANGEROUS: "Destructive or physical actions" };
    return [
      h("section", { class: "section" }, h("span", { class: "label", text: "When JARVIS needs your OK" }),
        h("dl", { class: "kv" }, Object.entries(policy.defaults).flatMap(([lvl, d]) => [h("dt", { text: levels[lvl] || lvl }), h("dd", { text: label[d] || d })])),
        h("p", { class: "section__intro", style: "margin-top:12px", text: "Always enforced: destructive actions need a fresh OK every time, and devices can't exceed their limits." })),
      h("section", { class: "section" }, h("span", { class: "label", text: "Standing permissions" }),
        grants.length ? h("div", { class: "list" }, grants.map((g) => item(g.tool, `${g.scope_type === "conversation" ? "in one conversation" : g.scope_type} · ${relTime(g.created_at)}`,
          h("button", { class: "btn btn--ghost", type: "button", onclick: async () => { await api.del(`/permissions/grants/${g.id}`); this.refresh(); } }, "Remove"))))
          : h("p", { class: "empty-note", text: "None." })),
      h("section", { class: "section" }, h("span", { class: "label", text: "Recent requests" }),
        history.length ? h("div", { class: "list" }, history.slice(0, 30).map((a) => item(a.summary, relTime(a.requested_at),
          tag(a.status === "approved" ? "Allowed" : a.status === "denied" ? "Denied" : a.status === "pending" ? "Waiting" : "Expired",
            a.status === "approved" ? "ok" : a.status === "pending" ? "busy" : "")))) : h("p", { class: "empty-note", text: "JARVIS hasn't asked for anything yet." })),
    ];
  }

  _activity() {
    const log = h("div", { class: "log", role: "log" });
    const draw = () => {
      const evts = this.app.recentEvents.slice(-120).reverse();
      log.replaceChildren(...(evts.length ? evts.map((e) => h("div", {}, h("span", { text: clock(e.ts) }), h("span", { text: e.type }),
        h("span", { text: JSON.stringify(e.data) }))) : [h("p", { class: "empty-note", text: "Nothing has happened since this page opened." })]));
    };
    this._activityDraw = draw;
    draw();
    return [h("p", { class: "section__intro", text: "Live events from JARVIS, newest first." }), log];
  }

  async _diagnostics() {
    const checks = await api.get("/system/diagnostics");
    const problems = checks.filter((c) => c.status !== "ok");
    const word = { ok: "OK", warn: "Check", fail: "Problem", skip: "Skipped" };
    const lvl = { ok: "ok", warn: "warn", fail: "err" };
    return [
      h("p", { class: "section__intro", text: problems.length ? `${problems.length} thing${problems.length === 1 ? "" : "s"} to look at.` : "Everything checks out." }),
      h("div", { class: "list" }, [...problems, ...checks.filter((c) => c.status === "ok")].map((c) =>
        item(c.name.replace(/^[a-z]+:/, (m) => m.slice(0, -1) + " · "), c.detail + (c.fix && c.status !== "ok" ? ` — ${c.fix}` : ""), tag(word[c.status] || c.status, lvl[c.status])))),
    ];
  }

  async _about() {
    const s = this.app.systemStatus || await api.get("/system/status");
    return [
      h("dl", { class: "kv" },
        h("dt", { text: "Version" }), h("dd", { text: s.version }),
        h("dt", { text: "Running on" }), h("dd", { text: `${s.node_name} (${s.platform})` }),
        h("dt", { text: "Node id" }), h("dd", { class: "mono", text: s.node_id }),
        h("dt", { text: "Up for" }), h("dd", { text: s.uptime_s != null ? `${Math.round(s.uptime_s / 60)} min` : "unknown" }),
        h("dt", { text: "Secrets stored in" }), h("dd", { text: s.secret_backend })),
      h("section", { class: "section", style: "margin-top:24px" }, h("span", { class: "label", text: "This browser" }),
        h("p", { class: "section__intro", text: "Signing out removes this browser's credential. You'll need a new pairing code to come back." }),
        h("button", { class: "btn btn--danger", type: "button", onclick: () => { api.forget(); location.reload(); } }, "Sign out this browser")),
    ];
  }
}

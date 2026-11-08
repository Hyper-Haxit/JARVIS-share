// Conversation transcript. JARVIS replies are plain text in the column; your
// messages sit in a quiet block on the right. Tool use is a compact, collapsible
// list of what actually ran.

import { renderMarkdown } from "./markdown.js";
import { ICONS, clock, fmtMs, h, svgIcon } from "./ui.js";

const BLOCKED = new Set(["policy_denied", "denied", "expired"]);

function argSummary(args) {
  if (!args) return "";
  const s = Object.values(args).map((v) => (typeof v === "string" ? v : JSON.stringify(v))).join(" · ");
  return s.length > 120 ? s.slice(0, 120) + "…" : s;
}

export class Conversation {
  constructor({ thread, list, empty, main, jump }) {
    this.thread = thread;
    this.list = list;
    this.empty = empty;
    this.main = main;
    this.jump = jump;
    this.current = null;
    this.thread.addEventListener("scroll", () => this._onScroll(), { passive: true });
    this.jump.addEventListener("click", () => this.scrollToEnd(true));
    this._sync();
  }

  get isEmpty() { return !this.list.children.length; }

  _sync() {
    this.empty.hidden = !this.isEmpty;
    this.main.dataset.empty = String(this.isEmpty);
  }

  _onScroll() {
    const t = this.thread;
    this.main.dataset.scrolled = String(t.scrollTop > 4);
    const atEnd = t.scrollHeight - t.scrollTop - t.clientHeight < 80;
    this._follow = atEnd;
    this.jump.hidden = atEnd || this.isEmpty;
    // Reading back through the conversation: the presence yields some room (it stays visible).
    this.main.dataset.reading = String(!atEnd && !this.isEmpty && t.scrollTop > 120);
  }

  scrollToEnd(force = false) {
    if (force || this._follow !== false) {
      this.thread.scrollTop = this.thread.scrollHeight;
      this._follow = true;
      this.jump.hidden = true;
      this.main.dataset.reading = "false";
    }
  }

  clear() {
    this.list.textContent = "";
    this.current = null;
    this._sync();
    this.main.dataset.scrolled = "false";
    this.main.dataset.reading = "false";
  }

  addUser(text, ts = Date.now() / 1000) {
    const msg = h("article", { class: "msg msg--user" },
      h("div", { class: "msg__text", text }),
      h("div", { class: "msg__meta" }, h("span", { text: clock(ts) })));
    this.list.appendChild(msg);
    this._sync();
    this.scrollToEnd(true);
  }

  startReply() {
    const text = h("div", { class: "msg__text" }, h("span", { class: "msg__waiting", "aria-label": "Working" }, h("i"), h("i"), h("i")));
    const meta = h("div", { class: "msg__meta" });
    const msg = h("article", { class: "msg msg--jarvis", dataset: { status: "pending" }, "aria-label": "JARVIS" }, text, meta);
    this.list.appendChild(msg);
    this.current = { msg, text, meta, steps: null, stepsById: new Map(), body: "", started: performance.now() };
    this._sync();
    this.scrollToEnd();
    return this.current;
  }

  _ensure() { return this.current || this.startReply(); }

  _renderText(final) {
    const c = this.current;
    if (!c.body) {
      if (final) c.text.textContent = "";
      return;
    }
    renderMarkdown(c.body, c.text);
    if (!final) {
      const tail = c.text.lastElementChild && c.text.lastElementChild.tagName !== "PRE" ? c.text.lastElementChild : c.text;
      tail.appendChild(h("span", { class: "caret", "aria-hidden": "true" }));
    }
  }

  appendDelta(text) {
    const c = this._ensure();
    c.msg.dataset.status = "streaming";
    c.body += text;
    if (!this._raf) {
      this._raf = requestAnimationFrame(() => { this._raf = 0; if (this.current) { this._renderText(false); this.scrollToEnd(); } });
    }
  }

  _stepsBlock() {
    const c = this._ensure();
    if (c.steps) return c.steps;
    const label = h("span", { text: "Working…" });
    const summary = h("button", { class: "steps__summary", type: "button", "aria-expanded": "true" }, svgIcon(ICONS.chevron), label);
    const list = h("ol", { class: "steps__list" });
    const block = h("div", { class: "steps", dataset: { open: "true" } }, summary, list);
    summary.addEventListener("click", () => {
      const open = block.dataset.open !== "true";
      block.dataset.open = String(open);
      summary.setAttribute("aria-expanded", String(open));
    });
    c.msg.insertBefore(block, c.text);
    c.steps = { block, list, label, summary };
    // While tools run, the waiting dots move below the steps.
    return c.steps;
  }

  toolStart({ id, tool, args }) {
    const s = this._stepsBlock();
    const row = h("li", { class: "step", dataset: { status: "running" } },
      h("span", { class: "step__mark", "aria-hidden": "true" }), h("span", { class: "step__name", text: tool }),
      h("span", { class: "step__detail", text: argSummary(args) }), h("span", { class: "step__time" }));
    s.list.appendChild(row);
    this.current.stepsById.set(id, row);
    s.label.textContent = `Running ${tool}…`;
    this.scrollToEnd();
  }

  toolEnd(data) {
    const row = this.current?.stepsById.get(data.id);
    if (!row) return;
    const blocked = BLOCKED.has(data.decision);
    row.dataset.status = data.ok ? "ok" : blocked ? "blocked" : "failed";
    const detail = row.querySelector(".step__detail");
    if (!data.ok) detail.textContent = blocked ? `Not allowed — ${data.error || data.decision}` : data.error || "Failed";
    else if (data.summary) detail.textContent = data.summary;
    detail.title = detail.textContent;
    row.querySelector(".step__time").textContent = fmtMs(data.duration_ms);
    this.current.steps.label.textContent = "Working…";
  }

  finish(result) {
    const c = this._ensure();
    if (this._raf) { cancelAnimationFrame(this._raf); this._raf = 0; }
    if (result.text && result.text.trim() !== c.body.trim()) c.body = result.text;
    this._renderText(true);
    c.msg.dataset.status = result.error ? "error" : "done";
    if (c.steps) {
      const n = c.stepsById.size;
      const failed = [...c.stepsById.values()].filter((r) => r.dataset.status !== "ok").length;
      const names = [...new Set([...c.stepsById.values()].map((r) => r.querySelector(".step__name").textContent))];
      c.steps.label.textContent = `Used ${names.slice(0, 3).join(", ")}${names.length > 3 ? ` +${names.length - 3}` : ""}` +
        (failed ? ` · ${failed} of ${n} didn't complete` : "");
      if (!failed) { c.steps.block.dataset.open = "false"; c.steps.summary.setAttribute("aria-expanded", "false"); }
    }
    const r = result.route || {};
    c.meta.textContent = "";
    const parts = [];
    if (r.model) parts.push(h("span", { text: r.model, title: `${r.provider}/${r.model} · ${r.role || ""}` }));
    if (result.duration_ms) parts.push(h("span", { text: fmtMs(result.duration_ms) }));
    if (r.degraded) parts.push(h("span", { class: "warn", text: "used a fallback model",
      title: (r.attempts || []).map((a) => `${a.provider}/${a.model}: ${a.outcome} ${a.detail || ""}`).join("\n") }));
    if (result.stop_reason === "cancelled") parts.push(h("span", { class: "warn", text: "stopped" }));
    c.meta.append(...parts);
    this.current = null;
    this.scrollToEnd();
  }

  fail(message) {
    this._ensure();
    this.finish({ text: message, error: message });
  }

  /** Render a stored conversation (GET /conversations/{id}). */
  load(conv) {
    this.clear();
    let pending = [];
    for (const m of conv.messages) {
      if (m.role === "user") { this.addUser(m.content, m.created_at); pending = []; }
      else if (m.role === "assistant" && m.tool_calls?.length) pending.push(...m.tool_calls.map((t) => ({ id: t.id, tool: t.name, args: t.arguments })));
      else if (m.role === "tool") {
        const a = pending.find((x) => x.id === m.tool_call_id);
        if (a) { a.ok = !m.meta?.is_error; a.summary = (m.content || "").split("\n")[0]; }
      } else if (m.role === "assistant") {
        this.startReply();
        for (const a of pending) {
          this.toolStart(a);
          this.toolEnd({ id: a.id, ok: a.ok !== false, decision: a.ok === false ? "failed" : "allow", summary: a.summary, error: a.ok === false ? a.summary : null });
        }
        pending = [];
        this.current.body = m.content;
        this.finish({ text: m.content, route: m.meta?.route || {}, error: m.meta?.error ? m.content : null,
          stop_reason: m.meta?.cancelled ? "cancelled" : "end" });
      }
    }
    requestAnimationFrame(() => this.scrollToEnd(true));
  }
}

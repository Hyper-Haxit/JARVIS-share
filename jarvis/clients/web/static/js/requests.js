// Approval requests, docked above the composer. A request says what JARVIS
// wants to do, why, and what will happen, and records the human decision.

import { api } from "./api.js";
import { h, toast, tone } from "./ui.js";

const RISK_TEXT = { SAFE: "Safe", LOW_RISK: "Low risk", SENSITIVE: "Needs your OK", DANGEROUS: "Potentially destructive" };

export class Requests extends EventTarget {
  constructor(host) {
    super();
    this.host = host;
    this.queue = new Map();
  }

  get count() { return this.queue.size; }

  async loadPending() {
    try {
      for (const a of await api.get("/permissions/approvals")) {
        if (!this.queue.has(a.id)) {
          this.queue.set(a.id, { approval_id: a.id, tool: a.tool, capability: a.capability, risk: a.risk, summary: a.summary,
            effects: a.effects, reason: a.reason, args: a.args, conversation_id: a.conversation_id, task_id: a.task_id, expires_at: null });
        }
      }
    } catch { /* not allowed to read approvals, or offline */ }
    this.render();
  }

  onEvent(evt) {
    const id = evt.data?.approval_id;
    if (!id) return;
    if (evt.type === "PERMISSION_REQUESTED") {
      if (!this.queue.has(id)) tone("alert");
      this.queue.set(id, evt.data);
      this.render();
    } else if ((evt.type === "PERMISSION_GRANTED" || evt.type === "PERMISSION_DENIED") && this.queue.delete(id)) {
      this.render();
    }
  }

  async decide(req, approve, scope, button) {
    if (button) button.dataset.loading = "true";
    try {
      await api.post(`/permissions/approvals/${req.approval_id}`, { approve, scope });
    } catch (e) {
      toast("Couldn't record that", e.status === 409 ? "This request was already answered or expired." : e.detail || String(e), "warn");
    }
    this.queue.delete(req.approval_id);
    this.render();
  }

  render() {
    clearInterval(this._timer);
    this.host.textContent = "";
    this.dispatchEvent(new Event("change"));
    const req = this.queue.values().next().value;
    if (!req) return;
    const dangerous = req.risk === "DANGEROUS";
    const timer = h("span", { class: "request__timer" });
    const allow = h("button", { class: "btn btn--primary", type: "button", onclick: (e) => this.decide(req, true, "once", e.currentTarget) }, "Allow");
    const allowConv = !dangerous && req.conversation_id
      ? h("button", { class: "btn", type: "button", onclick: (e) => this.decide(req, true, "conversation", e.currentTarget) }, "Allow for this conversation")
      : null;
    const deny = h("button", { class: "btn", type: "button", onclick: (e) => this.decide(req, false, "once", e.currentTarget) }, "Deny");
    const card = h("section", { class: "request", dataset: { risk: req.risk }, role: "alertdialog", "aria-labelledby": "req-title", "aria-describedby": "req-text" },
      this.queue.size > 1 ? h("p", { class: "request__more", text: `${this.queue.size} requests waiting` }) : null,
      h("div", { class: "request__head" },
        h("h2", { class: "request__title", id: "req-title", text: `JARVIS wants to use ${req.tool}` }),
        h("span", { class: "request__risk", text: RISK_TEXT[req.risk] || req.risk })),
      h("p", { class: "request__text", id: "req-text", text: req.effects || req.summary }),
      req.reason ? h("p", { class: "request__why", text: `Because: ${req.reason}` }) : null,
      h("details", {}, h("summary", { text: "Details" }),
        h("pre", { text: `capability: ${req.capability || "-"}\n${JSON.stringify(req.args || {}, null, 2)}` })),
      h("div", { class: "request__foot" }, timer, deny, allowConv, allow));
    card.addEventListener("keydown", (e) => { if (e.key === "Escape") { e.preventDefault(); this.decide(req, false, "once"); } });
    this.host.appendChild(card);
    if (req.expires_at) {
      const tick = () => {
        const left = Math.max(0, Math.round(req.expires_at - Date.now() / 1000));
        timer.textContent = left > 0 ? `Expires in ${Math.floor(left / 60)}:${String(left % 60).padStart(2, "0")}` : "Expired";
        if (left <= 0) { clearInterval(this._timer); this.queue.delete(req.approval_id); setTimeout(() => this.render(), 600); }
      };
      tick();
      this._timer = setInterval(tick, 1000);
    }
  }
}

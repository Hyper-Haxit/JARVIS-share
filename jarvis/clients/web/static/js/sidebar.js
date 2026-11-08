// Sidebar: recent conversations and quiet navigation.

import { api } from "./api.js";
import { ICONS, h, relTime, svgIcon } from "./ui.js";

export class Sidebar {
  constructor(app) {
    this.app = app;
    this.list = document.getElementById("conv-list");
    this.items = [];
    this.busy = new Set();
  }

  async refresh() {
    try {
      this.items = await api.get("/conversations?limit=40");
    } catch {
      return;
    }
    this.render();
  }

  scheduleRefresh() {
    clearTimeout(this._t);
    this._t = setTimeout(() => this.refresh(), 400);
  }

  setBusy(conversationId, busy) {
    if (busy) this.busy.add(conversationId); else this.busy.delete(conversationId);
    const li = this.list.querySelector(`[data-id="${CSS.escape(conversationId || "")}"]`);
    if (li) li.dataset.busy = String(busy);
  }

  render() {
    this.list.textContent = "";
    if (!this.items.length) {
      this.list.appendChild(h("li", { class: "sidebar__empty", text: "No conversations yet." }));
      return;
    }
    for (const c of this.items) {
      const del = h("button", { class: "btn btn--ghost btn--icon conv__del", type: "button", "aria-label": `Delete ${c.title || "conversation"}`,
        title: "Delete", onclick: async (e) => {
          e.stopPropagation();
          try { await api.del(`/conversations/${c.id}`); } catch { return; }
          if (this.app.conversationId === c.id) this.app.newConversation();
          this.refresh();
        } }, svgIcon(ICONS.trash));
      const li = h("li", { class: "conv", tabindex: "0", role: "button", dataset: { id: c.id, busy: String(this.busy.has(c.id)) },
        "aria-current": String(c.id === this.app.conversationId), title: `${c.title || "Untitled"} · ${relTime(c.updated_at)}`,
        onclick: () => this.app.openConversation(c.id),
        onkeydown: (e) => { if (e.key === "Enter") this.app.openConversation(c.id); } },
      h("span", { class: "conv__title", text: c.title || "Untitled" }), del);
      this.list.appendChild(li);
    }
  }

  markCurrent(id) {
    for (const li of this.list.querySelectorAll(".conv")) li.setAttribute("aria-current", String(li.dataset.id === id));
  }
}

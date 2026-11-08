// Command palette (Ctrl/Cmd + K): filter, arrow keys, Enter, Escape.

import { ICONS, h, svgIcon } from "./ui.js";

function score(query, text) {
  if (!query) return 1;
  const q = query.toLowerCase(), t = text.toLowerCase();
  if (t.startsWith(q)) return 3;
  if (t.includes(q)) return 2 - t.indexOf(q) / 100;
  let i = 0;
  for (const ch of t) if (ch === q[i]) i++;
  return i === q.length ? 0.5 : 0;
}

export class Palette {
  constructor(layer, getCommands) {
    this.layer = layer;
    this.getCommands = getCommands;
    this.open = false;
  }

  toggle() { this.open ? this.close() : this.show(); }

  close() {
    this.open = false;
    this.layer.textContent = "";
    this._restore?.focus?.();
  }

  show() {
    this.open = true;
    this._restore = document.activeElement;
    const input = h("input", { type: "text", placeholder: "Type a command", "aria-label": "Search commands", role: "combobox",
      "aria-expanded": "true", "aria-controls": "palette-list", "aria-autocomplete": "list" });
    const list = h("ul", { class: "palette__list", id: "palette-list", role: "listbox" });
    let items = [], sel = 0;
    const mark = () => {
      [...list.children].forEach((li, i) => li.setAttribute("aria-selected", String(i === sel)));
      list.children[sel]?.scrollIntoView({ block: "nearest" });
      input.setAttribute("aria-activedescendant", `pal-${sel}`);
    };
    const run = (i) => {
      const c = items[i];
      if (!c || c.disabled) return;
      this.close();
      c.run();
    };
    const render = () => {
      const q = input.value.trim();
      items = this.getCommands().map((c) => ({ c, s: score(q, `${c.label} ${c.keywords || ""}`) }))
        .filter((x) => x.s > 0).sort((a, b) => b.s - a.s).map((x) => x.c);
      sel = Math.min(sel, Math.max(0, items.length - 1));
      list.textContent = "";
      if (!items.length) { list.appendChild(h("li", { class: "palette__empty", text: "No matching command" })); return; }
      items.forEach((c, i) => list.appendChild(h("li", { class: "palette__item", role: "option", id: `pal-${i}`, "aria-selected": String(i === sel),
        "aria-disabled": c.disabled ? "true" : null, onclick: () => run(i), onmousemove: () => { if (sel !== i) { sel = i; mark(); } } },
      h("span", { text: c.label }), h("small", { text: c.disabled ? c.why || "Unavailable" : c.hint || "" }))));
      input.setAttribute("aria-activedescendant", `pal-${sel}`);
    };
    input.addEventListener("input", () => { sel = 0; render(); });
    input.addEventListener("keydown", (e) => {
      if (e.key === "ArrowDown") { e.preventDefault(); sel = (sel + 1) % Math.max(1, items.length); mark(); }
      else if (e.key === "ArrowUp") { e.preventDefault(); sel = (sel - 1 + items.length) % Math.max(1, items.length); mark(); }
      else if (e.key === "Enter") { e.preventDefault(); run(sel); }
      else if (e.key === "Escape") { e.preventDefault(); this.close(); }
    });
    const scrim = h("div", { class: "palette-scrim", onmousedown: (e) => { if (e.target === scrim) this.close(); } },
      h("div", { class: "palette", role: "dialog", "aria-label": "Commands" },
        h("div", { class: "palette__search" }, svgIcon(ICONS.search), input), list));
    this.layer.replaceChildren(scrim);
    render();
    input.focus();
  }
}

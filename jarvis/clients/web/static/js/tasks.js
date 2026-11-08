// Running-task strip above the composer. Shows only tasks that are actually
// active, with their real step progress; finished tasks leave quietly.

import { api } from "./api.js";
import { h, toast } from "./ui.js";

const ACTIVE = new Set(["pending", "planning", "running"]);

export class TaskStrip extends EventTarget {
  constructor(host, app) {
    super();
    this.host = host;
    this.app = app;
    this.tasks = new Map();
  }

  get activeCount() { return [...this.tasks.values()].filter((t) => ACTIVE.has(t.status)).length; }

  async refresh() {
    try {
      const list = await api.get("/tasks?limit=20");
      this.tasks = new Map(list.map((t) => [t.id, t]));
    } catch { return; }
    this.render();
  }

  async onEvent(evt) {
    const id = evt.data?.task_id;
    if (!id || !evt.type.startsWith("TASK_")) return;
    try {
      const t = await api.get(`/tasks/${id}`);
      this.tasks.set(id, t);
      if (evt.type === "TASK_COMPLETED") toast("Task finished", t.goal.slice(0, 90), "ok");
      else if (evt.type === "TASK_FAILED") toast("Task didn't finish", t.error ? t.error.slice(0, 120) : t.goal.slice(0, 90), "warn");
    } catch { return; }
    this.render();
  }

  render() {
    this.host.textContent = "";
    for (const t of this.tasks.values()) {
      if (!ACTIVE.has(t.status)) continue;
      const nodes = t.nodes || [];
      const done = nodes.filter((n) => n.status === "completed").length;
      const meta = t.status === "planning" || !nodes.length ? "Planning…" : `${done} of ${nodes.length} steps`;
      this.host.appendChild(h("button", { class: "taskline", type: "button", title: t.goal, onclick: () => this.app.sheets.open("tasks", { taskId: t.id }) },
        h("span", { class: "dot", dataset: { level: "busy" } }),
        h("span", { class: "taskline__goal", text: t.goal }),
        h("span", { class: "taskline__meta", text: meta }),
        h("span", { class: "taskline__bar", "aria-hidden": "true" }, h("i", { style: `width:${Math.round((t.progress || 0) * 100)}%` }))));
    }
    this.dispatchEvent(new Event("change"));
  }
}

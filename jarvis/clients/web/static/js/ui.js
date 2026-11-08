// Small DOM helpers shared by components.

export function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "dataset") Object.assign(node.dataset, v);
    else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    node.appendChild(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

export function svgIcon(d, cls = "icon") {
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("class", cls);
  svg.setAttribute("aria-hidden", "true");
  const p = document.createElementNS(ns, "path");
  p.setAttribute("d", d);
  svg.appendChild(p);
  return svg;
}

export const ICONS = {
  chevron: "M9 6l6 6-6 6",
  close: "M6 6l12 12M18 6L6 18",
  search: "M11 18a7 7 0 1 1 0-14 7 7 0 0 1 0 14zM20 20l-4-4",
  trash: "M5 7h14M10 7V5h4v2M7 7l1 12h8l1-12",
};

export function toast(title, body = "", level = "info", ms = 5000) {
  const host = document.getElementById("toasts");
  const t = h("div", { class: "toast", dataset: { level } }, h("b", { text: title }), body ? h("span", { text: body }) : null);
  host.appendChild(t);
  setTimeout(() => t.remove(), ms);
}

export function relTime(ts) {
  if (!ts) return "";
  const s = Date.now() / 1000 - ts;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  if (s < 86400 * 7) return `${Math.floor(s / 86400)}d ago`;
  return new Date(ts * 1000).toLocaleDateString([], { month: "short", day: "numeric" });
}

export function clock(ts) {
  return ts ? new Date(ts * 1000).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }) : "";
}

export function fmtMs(ms) {
  if (ms === null || ms === undefined) return "";
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`;
}

export function prefs() {
  try { return JSON.parse(localStorage.getItem("jarvis.prefs") || "{}"); } catch { return {}; }
}

export function setPref(key, value) {
  const p = prefs();
  p[key] = value;
  try { localStorage.setItem("jarvis.prefs", JSON.stringify(p)); } catch { /* storage blocked */ }
}

export function applyTheme(pref = prefs().theme || "system") {
  const dark = pref === "dark" || (pref === "system" && matchMedia("(prefers-color-scheme: dark)").matches);
  document.documentElement.setAttribute("data-theme", dark ? "dark" : "light");
  document.querySelector('meta[name="theme-color"]')?.setAttribute("content", dark ? "#0d0f12" : "#fbfbfa");
}

// Optional, quiet system tones (off by default).
let audioCtx = null;
export function tone(kind) {
  if (!prefs().sound) return;
  try {
    audioCtx = audioCtx || new AudioContext();
    const now = audioCtx.currentTime;
    const [f, d] = { send: [740, 0.04], done: [880, 0.06], alert: [520, 0.12] }[kind] || [600, 0.05];
    const osc = audioCtx.createOscillator(), gain = audioCtx.createGain();
    osc.frequency.setValueAtTime(f, now);
    gain.gain.setValueAtTime(0.0001, now);
    gain.gain.exponentialRampToValueAtTime(0.03, now + 0.01);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + d + 0.06);
    osc.connect(gain).connect(audioCtx.destination);
    osc.start(now);
    osc.stop(now + d + 0.08);
  } catch { /* audio unavailable */ }
}

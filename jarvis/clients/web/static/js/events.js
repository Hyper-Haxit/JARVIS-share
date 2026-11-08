// Live event stream over WebSocket with authentication-by-first-message and
// exponential-backoff reconnect. Connection state is reported, never guessed.

import { api } from "./api.js";

export class EventStream extends EventTarget {
  constructor() {
    super();
    this.state = "disconnected"; // disconnected | connecting | connected
    this._attempt = 0;
    this._closed = false;
    this.lastEventAt = null;
  }

  _set(state) {
    if (state === this.state) return;
    this.state = state;
    this.dispatchEvent(new CustomEvent("state", { detail: state }));
  }

  /** Open the stream. Safe to call any number of times: there is never more than one socket. */
  async connect() {
    if (this._connecting || (this._ws && this._ws.readyState <= WebSocket.OPEN)) return;
    this._connecting = true;
    this._closed = false;
    clearTimeout(this._timer);
    this._set("connecting");
    let token;
    try {
      token = (await api.ensureSession()).session_token;
    } catch (e) {
      this._connecting = false;
      this._set("disconnected");
      this.dispatchEvent(new CustomEvent("autherror", { detail: e }));
      if (e.status !== 401) this._retry();
      return;
    }
    this._connecting = false;
    if (this._closed) { this._set("disconnected"); return; }
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/api/v1/events`);
    this._ws = ws;
    ws.onopen = () => ws.send(JSON.stringify({ token, patterns: ["*"] }));
    ws.onmessage = (m) => {
      if (ws !== this._ws) { ws.close(); return; } // a replaced socket never delivers events
      let msg;
      try { msg = JSON.parse(m.data); } catch { return; }
      this.lastEventAt = Date.now();
      if (msg.type === "ready") { this._attempt = 0; this._set("connected"); }
      else if (msg.type === "event") this.dispatchEvent(new CustomEvent("event", { detail: msg.event }));
      else if (msg.type === "error") this.dispatchEvent(new CustomEvent("autherror", { detail: msg }));
    };
    ws.onclose = () => {
      if (ws !== this._ws) return;
      this._ws = null;
      this._set("disconnected");
      if (!this._closed) this._retry();
    };
    ws.onerror = () => { /* onclose follows */ };
  }

  _retry() {
    const delay = Math.min(15000, 600 * 2 ** this._attempt++) + Math.random() * 400;
    clearTimeout(this._timer);
    this._timer = setTimeout(() => this.connect(), delay);
  }

  close() {
    this._closed = true;
    clearTimeout(this._timer);
    const ws = this._ws;
    this._ws = null;
    ws?.close();
    this._set("disconnected");
  }
}

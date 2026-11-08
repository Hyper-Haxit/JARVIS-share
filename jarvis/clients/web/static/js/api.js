// API client: device pairing, sessions, JSON calls and streamed chat (SSE over fetch).

const DEVICE_KEY = "jarvis.device_token";
const SESSION_KEY = "jarvis.session";
const SETUP_KEY = "jarvis.setup";   // a setup request waiting for approval (this tab only)

function safeGet(store, key) { try { return store.getItem(key); } catch { return null; } }
function safeSet(store, key, value) { try { value === null ? store.removeItem(key) : store.setItem(key, value); } catch { /* storage blocked */ } }

export class ApiError extends Error {
  constructor(status, detail) { super(`${status}: ${detail}`); this.status = status; this.detail = detail; }
}

class Api {
  constructor() {
    this.deviceToken = safeGet(localStorage, DEVICE_KEY);
    const s = safeGet(sessionStorage, SESSION_KEY);
    this.session = s ? JSON.parse(s) : null;
  }

  get paired() { return Boolean(this.deviceToken); }
  get token() { return this.session?.session_token || null; }

  deviceKind() {
    return /Android|iPhone|iPad|Mobile/i.test(navigator.userAgent) ? "mobile" : "web";
  }

  deviceName() {
    const ua = navigator.userAgent;
    const browser = /Edg\//.test(ua) ? "Edge" : /Chrome\//.test(ua) ? "Chrome" : /Firefox\//.test(ua) ? "Firefox" : /Safari\//.test(ua) ? "Safari" : "Browser";
    const os = /Windows/.test(ua) ? "Windows" : /Android/.test(ua) ? "Android" : /iPhone|iPad/.test(ua) ? "iOS" : /Mac OS/.test(ua) ? "macOS" : /Linux/.test(ua) ? "Linux" : "";
    return `${browser}${os ? " on " + os : ""}`;
  }

  async pair(code) {
    const res = await fetch("/api/v1/auth/pair", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code, name: this.deviceName(), kind: this.deviceKind(), platform: navigator.platform || null,
        capabilities: this.browserCapabilities() }),
    });
    if (!res.ok) throw new ApiError(res.status, (await res.json().catch(() => ({}))).detail || res.statusText);
    const data = await res.json();
    this.deviceToken = data.device_token;
    safeSet(localStorage, DEVICE_KEY, data.device_token);
    this._storeSession({ session_token: data.session_token, expires_at: data.expires_at, device_id: data.device_id });
    return data;
  }

  // ------------------------------------------------- set up this device (no code)
  async setupStatus() {
    const res = await fetch("/api/v1/auth/setup", { cache: "no-store" });
    if (!res.ok) throw new ApiError(res.status, res.statusText);
    return res.json();
  }

  /** Ask the computer JARVIS runs on to set up this browser; the person there approves it. */
  async requestSetup() {
    const res = await fetch("/api/v1/auth/setup/requests", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: this.deviceName(), kind: this.deviceKind(), platform: navigator.platform || null,
        capabilities: this.browserCapabilities() }),
    });
    if (!res.ok) throw new ApiError(res.status, (await res.json().catch(() => ({}))).detail || res.statusText);
    const req = await res.json();
    safeSet(sessionStorage, SETUP_KEY, JSON.stringify(req));
    return req;
  }

  get pendingSetup() {
    try {
      const req = JSON.parse(safeGet(sessionStorage, SETUP_KEY) || "null");
      return req && req.expires_at > Date.now() / 1000 ? req : null;
    } catch { return null; }
  }

  clearPendingSetup() { safeSet(sessionStorage, SETUP_KEY, null); }

  /** Waits (the server holds the request up to 25 s) for the decision; stores the credential once approved. */
  async claimSetup(req) {
    const res = await fetch(`/api/v1/auth/setup/requests/${encodeURIComponent(req.id)}/claim`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ secret: req.secret }),
    });
    if (!res.ok) throw new ApiError(res.status, (await res.json().catch(() => ({}))).detail || res.statusText);
    const data = await res.json();
    if (data.status === "approved") {
      this.deviceToken = data.device_token;
      safeSet(localStorage, DEVICE_KEY, data.device_token);
      this._storeSession({ session_token: data.session_token, expires_at: data.expires_at, device_id: data.device_id });
      this.clearPendingSetup();
    }
    return { status: data.status };
  }

  async cancelSetup(req) {
    this.clearPendingSetup();
    await fetch(`/api/v1/auth/setup/requests/${encodeURIComponent(req.id)}/cancel`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ secret: req.secret }),
    }).catch(() => {});
  }

  browserCapabilities() {
    const caps = ["display", "text_input"];
    if (navigator.mediaDevices?.getUserMedia) caps.push("microphone", "camera");
    if ("speechSynthesis" in window || "AudioContext" in window) caps.push("speaker");
    if ("geolocation" in navigator) caps.push("location");
    if ("Notification" in window) caps.push("notifications");
    return caps;
  }

  _storeSession(s) {
    this.session = s;
    safeSet(sessionStorage, SESSION_KEY, s ? JSON.stringify(s) : null);
  }

  async ensureSession() {
    if (this.session && this.session.expires_at - 60 > Date.now() / 1000) return this.session;
    if (!this.deviceToken) throw new ApiError(401, "not paired");
    const res = await fetch("/api/v1/auth/session", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ device_token: this.deviceToken }),
    });
    if (res.status === 401) { this.forget(); throw new ApiError(401, "this device was revoked or its credential is invalid"); }
    if (!res.ok) throw new ApiError(res.status, res.statusText);
    const s = await res.json();
    this._storeSession(s);
    return s;
  }

  forget() {
    this.deviceToken = null;
    safeSet(localStorage, DEVICE_KEY, null);
    this._storeSession(null);
  }

  async request(method, path, body, { raw = false, retry = true } = {}) {
    await this.ensureSession();
    const res = await fetch(`/api/v1${path}`, {
      method, headers: { Authorization: `Bearer ${this.token}`, ...(body !== undefined ? { "Content-Type": "application/json" } : {}) },
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
    if (res.status === 401 && retry) { this._storeSession(null); return this.request(method, path, body, { raw, retry: false }); }
    if (!res.ok) {
      const detail = await res.json().then((j) => j.detail, () => res.statusText);
      throw new ApiError(res.status, typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    if (raw) return res;
    return res.headers.get("content-type")?.includes("application/json") ? res.json() : res.text();
  }

  get(path) { return this.request("GET", path); }
  post(path, body = {}) { return this.request("POST", path, body); }
  put(path, body = {}) { return this.request("PUT", path, body); }
  patch(path, body = {}) { return this.request("PATCH", path, body); }
  del(path) { return this.request("DELETE", path); }

  async health() {
    try {
      const res = await fetch("/api/v1/health", { cache: "no-store" });
      return res.ok ? res.json() : null;
    } catch { return null; }
  }

  /** Stream a chat turn. onEvent(name, data) is called for every SSE event. */
  async chat(body, onEvent, signal) {
    await this.ensureSession();
    const res = await fetch("/api/v1/chat", {
      method: "POST", signal,
      headers: { Authorization: `Bearer ${this.token}`, "Content-Type": "application/json" },
      body: JSON.stringify({ ...body, stream: true }),
    });
    if (!res.ok) {
      const detail = await res.json().then((j) => j.detail, () => res.statusText);
      throw new ApiError(res.status, typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf("\n\n")) >= 0) {
        const chunk = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        let event = "message", data = "";
        for (const line of chunk.split("\n")) {
          if (line.startsWith("event:")) event = line.slice(6).trim();
          else if (line.startsWith("data:")) data += line.slice(5).trim();
        }
        if (data) onEvent(event, JSON.parse(data));
      }
    }
  }
}

export const api = new Api();

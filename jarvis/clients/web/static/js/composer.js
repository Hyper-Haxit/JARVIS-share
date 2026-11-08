// Composer: a plain text box with honest states.
//
//   empty      quiet; send unavailable
//   ready      text present; send becomes the primary action
//   busy       a reply is in progress; send becomes Stop, Enter is held back
//   listening  the microphone is capturing; bars show the real input level
//   offline    the core is unreachable; typing is allowed, sending is not

export class Composer extends EventTarget {
  constructor({ form, input, send, mic, levels, hintLeft, hintRight, hint }) {
    super();
    Object.assign(this, { form, input, send, mic, levels, hintLeft, hintRight, hint });
    this.busy = false;
    this.listening = false;
    this.offline = true;
    this.bars = [...levels.children];
    this._bind();
    this._refresh();
  }

  _bind() {
    this.input.addEventListener("input", () => { this._autosize(); this._refresh(); });
    this.input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        if (!this.busy) this.form.requestSubmit();
      } else if (e.key === "Escape" && this.busy) {
        e.preventDefault();
        this.dispatchEvent(new Event("stop"));
      }
    });
    this.input.addEventListener("focus", () => { this.dispatchEvent(new Event("focus")); this._refresh(); });
    this.input.addEventListener("blur", () => this._refresh());
    this.form.addEventListener("submit", (e) => {
      e.preventDefault();
      if (this.busy) { this.dispatchEvent(new Event("stop")); return; }
      const text = this.input.value.trim();
      if (!text || this.offline) return;
      this.input.value = "";
      this._autosize();
      this.dispatchEvent(new CustomEvent("submit", { detail: text }));
      this._refresh();
    });
    this.mic.addEventListener("click", () => this.dispatchEvent(new Event("voice")));
  }

  _autosize() {
    this.input.style.height = "auto";
    this.input.style.height = Math.min(220, this.input.scrollHeight) + "px";
  }

  setBusy(v) { this.busy = v; this._refresh(); }
  setOffline(v) { this.offline = v; this._refresh(); }

  setListening(v) {
    this.listening = v;
    this.mic.setAttribute("aria-pressed", String(v));
    if (!v) this.setLevel(0);
    this._refresh();
  }

  setVoiceAvailable(ok, reason) {
    this.mic.disabled = !ok;
    this.mic.title = ok ? "Voice input (Ctrl+Shift+Space)" : `Voice input unavailable: ${reason}`;
  }

  /** Real microphone level 0..1 -> bar heights. */
  setLevel(level, spectrum) {
    const n = this.bars.length;
    for (let i = 0; i < n; i++) {
      const v = spectrum ? spectrum[i] : level * (0.55 + 0.45 * Math.sin((i / (n - 1)) * Math.PI));
      this.bars[i].style.height = `${Math.round(4 + Math.min(1, v) * 14)}px`;
    }
  }

  setRouteInfo(text) { this.hintRight.textContent = text || ""; }

  _refresh() {
    const has = this.input.value.trim().length > 0;
    const state = this.listening ? "listening" : this.busy ? "busy" : this.offline ? "offline" : has ? "ready" : "empty";
    this.form.dataset.state = state;
    this.send.disabled = !this.busy && (!has || this.offline);
    // SVG elements have no .hidden property; toggle the attribute instead.
    this.send.querySelector(".send-icon").toggleAttribute("hidden", this.busy);
    this.send.querySelector(".stop-icon").toggleAttribute("hidden", !this.busy);
    this.send.setAttribute("aria-label", this.busy ? "Stop" : "Send");
    this.send.title = this.busy ? "Stop (Esc)" : "Send (Enter)";
    this.input.placeholder = this.listening ? "" : this.offline ? "Waiting for JARVIS to come back…" : this.busy ? "Write a follow-up…" : "Message JARVIS";
    const focused = document.activeElement === this.input;
    let left = "";
    this.hint.dataset.tone = "";
    if (this.listening) left = "Listening. Click the microphone again when you're done.";
    else if (this.busy) left = focused ? "Esc to stop" : "";
    else if (focused) left = has ? "Enter to send · Shift+Enter for a new line" : "";
    this.hintLeft.textContent = left;
  }

  focus() { this.input.focus(); }
}

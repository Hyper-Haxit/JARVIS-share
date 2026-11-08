// Browser voice: capture from this device's microphone and play JARVIS's
// speech. Levels sent to the presence and the composer come from real
// AnalyserNodes on the actual audio streams. If the core reports speech
// recognition unavailable, voice input is disabled and says why.

import { api } from "./api.js";
import { SpeechChunker } from "./speech-chunks.js";
import { toast } from "./ui.js";

function rms(analyser, buf) {
  analyser.getFloatTimeDomainData(buf);
  let sum = 0;
  for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
  return Math.sqrt(sum / buf.length);
}

export class Voice {
  constructor(app) {
    this.app = app;
    this.status = null;
    this.sttAvailable = false;
    this.ttsAvailable = false;
    this.reason = "not checked yet";
    this.capturing = false;
    this.speaking = false;
    this.speakNextReply = false;
  }

  get canCapture() { return this.sttAvailable && Boolean(navigator.mediaDevices?.getUserMedia) && window.isSecureContext; }

  async probe() {
    try {
      this.status = await api.get("/voice/status");
      this.sttAvailable = Boolean(this.status.stt?.available);
      this.ttsAvailable = Boolean(this.status.tts?.available);
      this.reason = !this.sttAvailable ? (this.status.stt?.reason || "speech recognition is unavailable")
        : !window.isSecureContext ? "browsers only allow microphones on https or localhost"
          : !navigator.mediaDevices?.getUserMedia ? "this browser cannot capture audio" : "";
    } catch (e) {
      this.status = null;
      this.sttAvailable = this.ttsAvailable = false;
      this.reason = e.status === 404 ? "voice isn't set up on the core yet" : "voice status unavailable";
    }
    this.app.composer.setVoiceAvailable(this.canCapture, this.reason);
    return this.status;
  }

  get pipelineRunning() { return Boolean(this.status?.pipeline?.running); }

  async toggle() {
    if (this.capturing) { this._stopCapture(true); return; }
    if (this.speaking) this.stopSpeaking();
    if (!this.canCapture) { toast("Voice input isn't available", this.reason, "warn"); return; }
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 } });
    } catch (e) {
      toast("Microphone not available", e.name === "NotAllowedError" ? "Permission was denied. Allow microphone access for this site." :
        e.name === "NotFoundError" ? "No microphone was found." : e.message, "warn");
      return;
    }
    const ctx = this._ctx || (this._ctx = new AudioContext());
    if (ctx.state === "suspended") await ctx.resume();
    const source = ctx.createMediaStreamSource(stream);
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 1024;
    source.connect(analyser);
    const buf = new Float32Array(analyser.fftSize);
    const chunks = [];
    const recorder = new MediaRecorder(stream);
    this._cap = { stream, source, analyser, recorder, chunks, send: true };
    recorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
    recorder.onstop = () => this._onRecorded(recorder, chunks);
    recorder.start(250);
    this.capturing = true;
    this.app.composer.setListening(true);
    this.app.store.update({ micCapturing: true });
    const loop = () => {
      if (!this.capturing) return;
      const level = Math.min(1, rms(analyser, buf) * 6);
      this.app.presenceLevel(level, "mic");
      this.app.composer.setLevel(level);
      this._raf = requestAnimationFrame(loop);
    };
    loop();
  }

  cancelCapture() { if (this.capturing) this._stopCapture(false); }

  _stopCapture(send) {
    const cap = this._cap;
    if (!cap) return;
    cap.send = send;
    cancelAnimationFrame(this._raf);
    this.capturing = false;
    try { cap.recorder.stop(); } catch { /* already stopped */ }
    cap.stream.getTracks().forEach((t) => t.stop());
    cap.source.disconnect();
    this.app.composer.setListening(false);
    // When the recording will be sent, go straight from listening to thinking.
    this.app.store.update(send ? { micCapturing: false, agentBusy: true } : { micCapturing: false });
  }

  async _onRecorded(recorder, chunks) {
    const send = this._cap?.send;
    this._cap = null;
    if (!send) return;
    const blob = new Blob(chunks, { type: recorder.mimeType || "audio/webm" });
    if (blob.size < 1500) { this.app.store.update({ agentBusy: false }); toast("Didn't catch that", "The recording was too short.", "warn"); return; }
    this.app.store.update({ agentBusy: true });
    try {
      await api.ensureSession();
      const res = await fetch("/api/v1/voice/transcribe", {
        method: "POST", headers: { Authorization: `Bearer ${api.token}`, "Content-Type": blob.type }, body: blob,
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || res.statusText);
      this.app.store.update({ agentBusy: false });
      const text = (data.text || "").trim();
      if (!text) { toast("Didn't catch that", "No speech was recognised.", "warn"); return; }
      this.speakNextReply = this.ttsAvailable;
      this.app.send(text, { voice: true });
    } catch (e) {
      this.app.store.update({ agentBusy: false });
      toast("Couldn't transcribe", e.message, "warn");
    }
  }

  /** Speak a complete text (a sample, or a reply that wasn't streamed). */
  speak(text) {
    if (!this.ttsAvailable || !text.trim()) return;
    const reply = this.startReply();
    reply.push(text);
    reply.end();
  }

  /**
   * Speak a reply while it is still being written: push() its text as it arrives,
   * end() when it is complete. Chunks (speech-chunks.js) are synthesized by the core
   * one after another, the next while the previous plays, and scheduled back to back
   * on one audio graph: one voice, never overlapping. Levels come from the real playback.
   */
  startReply() {
    this.stopSpeaking();
    const chunker = new SpeechChunker();
    const run = { stopped: false, ending: false, busy: false, pending: [], sources: [], at: 0, ctx: null, analyser: null };
    this._run = run;
    // Committed to speaking: stay in the speaking state while the audio is prepared.
    this.speaking = true;
    return {
      push: (text) => { if (!run.stopped) this._queue(run, chunker.feed(text)); },
      end: () => {
        if (run.stopped) return;
        run.ending = true;
        this._queue(run, chunker.flush());
      },
      cancel: () => { if (this._run === run) this.stopSpeaking(); },
    };
  }

  _queue(run, chunks) {
    run.pending.push(...chunks);
    this._pump(run);
  }

  async _pump(run) {
    if (run.busy) return;  // the running pump picks up what was queued
    run.busy = true;
    try {
      await api.ensureSession();
      if (!run.ctx) {
        const ctx = this._ctx || (this._ctx = new AudioContext());
        if (ctx.state === "suspended") await ctx.resume();
        run.ctx = ctx;
        run.analyser = ctx.createAnalyser();
        run.analyser.fftSize = 1024;
        run.analyser.connect(ctx.destination);
      }
      while (!run.stopped && run.pending.length) {
        const audio = await this._synthesize(run.ctx, run.pending.shift());
        if (run.stopped) return;
        const src = run.ctx.createBufferSource();
        src.buffer = audio;
        src.connect(run.analyser);
        run.at = Math.max(run.at, run.ctx.currentTime + 0.02);
        src.start(run.at);
        run.at += audio.duration;
        run.sources.push(src);
        if (run.sources.length === 1) {
          this.app.store.update({ audioPlaying: true });
          this._levels(run);
        }
      }
    } catch (e) {
      if (run.stopped) return;
      toast("Couldn't speak the reply", e.message, "warn");
      run.pending.length = 0;
      run.ending = true;  // what was already scheduled still plays
    } finally {
      run.busy = false;
    }
    if (run.stopped || !run.ending || run.pending.length) return;
    // The reply is complete and scheduled: done once the last chunk has played.
    const finish = () => { if (this._run === run) this._finishSpeaking(); };
    const left = run.ctx ? run.at - run.ctx.currentTime : 0;
    if (left > 0) setTimeout(finish, left * 1000 + 50);
    else finish();
  }

  async _synthesize(ctx, chunk) {
    const res = await fetch("/api/v1/voice/speak", {
      method: "POST", headers: { Authorization: `Bearer ${api.token}`, "Content-Type": "application/json" }, body: JSON.stringify({ text: chunk }),
    });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
    return ctx.decodeAudioData(await res.arrayBuffer());
  }

  _levels(run) {
    const buf = new Float32Array(run.analyser.fftSize);
    const loop = () => {
      if (this._run !== run) return;
      this.app.presenceLevel(Math.min(1, rms(run.analyser, buf) * 5), "speaker");
      this._sraf = requestAnimationFrame(loop);
    };
    loop();
  }

  _finishSpeaking() {
    this._run = null;
    this.speaking = false;
    cancelAnimationFrame(this._sraf);
    this.app.store.update({ audioPlaying: false });
  }

  stopSpeaking() {
    const run = this._run;
    if (!run) return;
    run.stopped = true;
    for (const src of run.sources) {
      try { src.stop(); } catch { /* not started yet, or already stopped */ }
    }
    this._finishSpeaking();
  }
}

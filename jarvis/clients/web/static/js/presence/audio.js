// Audio reactivity: two strictly separate channels.
//
//   voice   the person speaking to JARVIS (the core's microphone while it is
//           recording a request, or this browser's microphone while capturing)
//   output  JARVIS's own voice (the core's speaker playback level, or the
//           browser's playback analyser)
//
// Levels only ever come from real audio. Each channel has an attack/release
// envelope; the output channel is shaped into a believable mouth opening
// (compressed, never simply "loud = 100% open"), with natural pauses closing
// the mouth. JARVIS's own voice never counts as the person speaking: voice
// levels that arrive while JARVIS is talking are ignored unless it has been
// interrupted.

export class Envelope {
  constructor(attackS, releaseS) {
    this.attack = attackS;
    this.release = releaseS;
    this.value = 0;
    this.input = 0;
    this.at = -1e9;
  }

  push(v, t) {
    v = Math.max(0, Math.min(1, v || 0));
    // Several levels within one frame (two sources): keep the loudest. Otherwise the newest wins,
    // so a pause in speech registers at once.
    this.input = t - this.at < 8 ? Math.max(this.input, v) : v;
    this.at = t;
  }

  /** Fresh input decays to zero if no new level arrives (a stalled stream must not hold the mouth open). */
  step(dt, t, staleMs = 200) {
    if (t - this.at > staleMs) this.input = 0;
    const tau = this.input > this.value ? this.attack : this.release;
    this.value += (this.input - this.value) * (1 - Math.exp(-dt / tau));
    if (this.value < 0.002 && this.input === 0) this.value = 0;
    return this.value;
  }

  reset() { this.value = this.input = 0; this.at = -1e9; }
  get active() { return this.value > 0 || this.input > 0; }
}

export class AudioReactivity {
  constructor(random) {
    this.random = random;
    this.voice = new Envelope(0.04, 0.22);
    this.output = new Envelope(0.025, 0.05);
    this.outputSlow = 0;     // slow envelope of JARVIS's voice: emphasis, not syllables
    this.mouth = 0;          // 0..1 opening
    this.mouthShape = 0;     // -1..1: narrower/wider variation between syllables
    this._shapeTarget = 0;
    this._lastPeakAt = -1e9;
    this.voiced = false;     // the person is speaking (hysteresis)
    this.voicedSince = 0;
    this.quietSince = 0;
    this.jarvisSpeaking = false;
    this.onVoiceStart = null;  // callbacks set by the engine
    this.onVoicePause = null;
  }

  level(source, v, t) {
    if (source === "output" || source === "speaker") this.output.push(v, t);
    else if (!this.jarvisSpeaking) this.voice.push(v, t);
  }

  /** JARVIS stopped talking abruptly (interrupted, stopped): the mouth closes now. */
  silenceOutput() {
    this.output.reset();
    this.outputSlow = 0;
    this.mouth = 0;
  }

  step(dt, t) {
    const v = this.voice.step(dt, t, 260);
    const o = this.output.step(dt, t, 160);
    this.outputSlow += (o - this.outputSlow) * (1 - Math.exp(-dt / 0.6));
    if (this.outputSlow < 0.002) this.outputSlow = 0;

    // Mouth: compress the level into a believable range and let pauses close it.
    const shaped = o < 0.035 ? 0 : Math.min(1, Math.pow((o - 0.035) / 0.55, 0.65)) * 0.86;
    // Syllable-like variation: when the level rises quickly, pick a new mouth shape.
    if (shaped - this.mouth > 0.12 && t - this._lastPeakAt > 90) {
      this._lastPeakAt = t;
      this._shapeTarget = this.random.range(-1, 1);
    }
    this.mouth += (shaped - this.mouth) * (1 - Math.exp(-dt / (shaped > this.mouth ? 0.03 : 0.045)));
    if (this.mouth < 0.004 && shaped === 0) this.mouth = 0;
    this.mouthShape += (this._shapeTarget * Math.min(1, this.mouth * 3) - this.mouthShape) * (1 - Math.exp(-dt / 0.08));

    // The person's voice activity, with hysteresis so it doesn't flicker.
    if (!this.voiced && v > 0.16) {
      this.voiced = true;
      this.voicedSince = t;
      this.onVoiceStart?.(t);
    } else if (this.voiced && v < 0.07) {
      this.voiced = false;
      this.quietSince = t;
      this.onVoicePause?.(t - this.voicedSince, t);
    }
    return this.active;
  }

  get active() {
    return this.voice.active || this.output.active || this.mouth > 0 || this.outputSlow > 0;
  }
}

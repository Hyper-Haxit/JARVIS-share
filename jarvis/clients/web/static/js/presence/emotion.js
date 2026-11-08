// Emotion: a few intensities that rise when something happens, hold briefly
// and fade back to neutral. Several can be present at once (focused and a
// little uncertain); expression blends them. The sources are real and
// deterministic: tool results, errors, interruptions, what the person said,
// what JARVIS answered. Nothing here is random except reaction strength.
//
// Each emotion maps to a readable change in the eyes (lids, lid angle, size)
// and the head. Intensities stay below 1, so even a strong feeling is a clear
// expression, never a caricature.

export const EMOTIONS = ["curious", "focused", "attentive", "amused", "confident", "concerned", "uncertain",
  "satisfied", "surprised", "empathetic", "relief", "calm"];

// How fast each rises (seconds) and how slowly it lets go (multiplier):
// surprise and concern arrive at once; calm arrives slowly and leaves slowly.
const DYNAMICS = { surprised: [0.05, 0.5], concerned: [0.12, 1], attentive: [0.1, 1], uncertain: [0.2, 1], satisfied: [0.2, 1],
  relief: [0.3, 1.3], calm: [1.4, 1.6], focused: [0.4, 1.1], curious: [0.2, 1], amused: [0.2, 1], confident: [0.25, 1],
  empathetic: [0.3, 1.2] };

// Offsets per unit intensity. eyeH/eyeW: size; lidTop/lidBot: lids (lidBot is a
// smile in the eyes); lidAngle: deg, + determined, - worried; tilt: the whole eye
// (deg); asym: right eye relative to left (skeptical/curious asymmetry); head:
// head tilt (deg); lift: eyes raised (units); lookY: gaze bias.
export const EMOTION_FACE = {
  curious:    { eyeH: 0.12, lidTop: -0.05, asym: 0.1, head: 6, lift: 0.8 },
  focused:    { eyeH: -0.05, lidTop: 0.16, lidAngle: 2 },
  attentive:  { eyeH: 0.12, lidTop: -0.06, lift: 1.0 },
  amused:     { eyeH: -0.06, lidBot: 0.42, head: -3 },
  confident:  { lidTop: 0.05, lift: 0.4 },
  concerned:  { lidAngle: -11, tilt: 3, lidTop: 0.1, lookY: 0.14, eyeH: -0.03 },
  uncertain:  { asym: -0.16, head: -5, lidTop: 0.06 },
  satisfied:  { lidBot: 0.42, lidTop: 0.05 },
  surprised:  { eyeH: 0.28, eyeW: 0.06, lidTop: -0.1, lift: 0.9 },
  empathetic: { lidAngle: -8, tilt: 2, lidTop: 0.1, eyeH: -0.04, head: 3 },
  relief:     { lidTop: 0.12, lidBot: 0.2, lift: -0.4 },
  calm:       { lidTop: 0.16, eyeH: -0.03 },
};

export class EmotionController {
  constructor() {
    this.values = Object.fromEntries(EMOTIONS.map((e) => [e, 0]));
    this._targets = Object.fromEntries(EMOTIONS.map((e) => [e, { amount: 0, until: 0, fade: 2.5 }]));
  }

  /** Feel ``name`` at ``amount`` (0..1) for at least ``holdMs``, then fade over ``fadeS``. */
  feel(name, amount, t, holdMs = 1200, fadeS = 2.5) {
    const g = this._targets[name];
    if (!g) return;
    g.amount = Math.max(amount, g.until > t ? g.amount : 0);
    g.until = Math.max(g.until, t + holdMs);
    g.fade = fadeS;
  }

  /** Let something go immediately (e.g. concern once the problem is resolved). */
  release(name, t) {
    const g = this._targets[name];
    if (g) { g.until = Math.min(g.until, t); g.fade = 0.8; }
  }

  /** Let something go entirely: no lingering hold or fade, it eases out at its own release speed. */
  drop(name, t) {
    const g = this._targets[name];
    if (g) { g.amount = 0; g.until = Math.min(g.until, t); }
  }

  step(dt, t) {
    let active = false;
    for (const e of EMOTIONS) {
      const g = this._targets[e];
      if (t > g.until) g.amount *= Math.exp(-dt / g.fade);
      if (g.amount < 0.01) g.amount = 0;
      const cur = this.values[e];
      const [attack, release] = DYNAMICS[e] || [0.18, 1];
      const tau = g.amount > cur ? attack : 0.45 * release;
      const next = cur + (g.amount - cur) * (1 - Math.exp(-dt / tau));
      this.values[e] = Math.abs(next) < 0.004 && g.amount === 0 ? 0 : next;
      if (this.values[e] || g.amount) active = true;
    }
    return active;
  }

  dominant() {
    let best = "neutral", v = 0.12;
    for (const e of EMOTIONS) if (this.values[e] > v) { best = e; v = this.values[e]; }
    return best;
  }
}

// What the person said, and what JARVIS answered, colour the reaction a little.
const GREETING = /^\s*(hi|hello|hey|good (morning|afternoon|evening)|thanks|thank you|cheers)\b/i;
const QUESTION = /\?\s*$|^\s*(what|why|how|when|where|who|which|can|could|would|should|is|are|do|does)\b/i;
const URGENT = /\b(urgent|asap|immediately|emergency|broken|not working|help)\b/i;
const NEGATIVE = /\b(sad|upset|frustrat\w*|angry|worried|stressed|sorry to|lost my|died|sick)\b/i;
const UNSURE = /\b(i('| a)m not sure|i don't know|i do not know|i couldn't|i could not|unable to|unclear|i can't (find|tell))\b/i;
const APOLOGY = /^\s*(sorry|apologies|unfortunately)\b/i;
const POSITIVE = /\b(done|great|all set|success(fully)?|here you go|perfect|ready)\b/i;
const FUNNY = /\b(haha|ha ha|lol|joke|funny|😄|😂)\b/i;

/** A greeting or thanks: worth a small happy moment. */
export function isWarm(text) {
  return GREETING.test(text);
}

/** Emotions suggested by the person's message: [name, amount][] */
export function readMessage(text) {
  const out = [];
  if (GREETING.test(text)) out.push(["amused", 0.35]);
  if (QUESTION.test(text)) out.push(["curious", 0.5]);
  if (URGENT.test(text)) out.push(["attentive", 0.6]);
  if (NEGATIVE.test(text)) out.push(["empathetic", 0.6]);
  if (FUNNY.test(text)) out.push(["amused", 0.5]);
  return out;
}

/** Emotions suggested by JARVIS's own reply. */
export function readReply(text) {
  const out = [];
  if (UNSURE.test(text)) out.push(["uncertain", 0.55]);
  else if (APOLOGY.test(text)) out.push(["empathetic", 0.4]);
  else if (POSITIVE.test(text)) out.push(["satisfied", 0.45]);
  if (FUNNY.test(text)) out.push(["amused", 0.45]);
  return out;
}

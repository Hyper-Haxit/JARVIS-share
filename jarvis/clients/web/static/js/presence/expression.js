// Expression: turns the engine's state into face geometry.
//
// The face is two eyes on a disc, so the eyes carry the expression and the
// body carries the motion. Layers, in order: the primary state's pose,
// transition beats (noticed, heard, ready, interrupted, recovered, settle,
// nod, breath, cheer, puzzled, wake), emotions, micro-expressions. Each eye is
// composed separately so asymmetry (a skeptical lid, one eye squinting while
// thinking) is possible. The result is plain numbers for the renderer;
// nothing here touches the DOM.
//
//   eye   eyeH, eyeW (size); lidTop (upper lid down); lidBot (lower lid up: a
//         smile in the eyes, a full squint is a happy crescent); lidAngle (deg;
//         + inner corner lowered: determined, - raised: worried); tilt (deg)
//   body  x, y, rot, scale; sx/sy: squash and stretch, anchored at the bottom

import { EMOTION_FACE } from "./emotion.js";

// Viewbox units: eyes at x = ±14, y = -2.5; eye 13 x 23.
export const POSES = {
  offline:    { eyeH: 0.1, lidTop: 0, lift: -0.8, dim: 1, tone: "off" },
  idle:       { eyeH: 1.0, lidTop: 0.05, lift: 0, tone: "rest" },
  attentive:  { eyeH: 1.05, lidTop: 0.02, lift: 0.5, tone: "rest" },
  listening:  { eyeH: 1.16, eyeW: 1.04, lidTop: 0, lift: 1.3, tone: "active", sy: 1.015 },
  thinking:   { eyeH: 0.96, lidTop: 0.08, lift: 0.6, tone: "active", think: true },
  tool_use:   { eyeH: 0.96, lidTop: 0.15, lidAngle: 2, lift: 0, tone: "active" },
  responding: { eyeH: 1.03, lidTop: 0.03, lift: 0.4, tone: "active" },
  speaking:   { eyeH: 1.04, lidTop: 0.02, lift: 0.5, tone: "active" },
  waiting:    { eyeH: 1.06, lidTop: 0.03, lidAngle: -6, lift: 0.3, tone: "warn" },
  error:      { eyeH: 0.92, lidTop: 0.1, lidAngle: -10, lift: -0.3, tone: "err" },
};

const ease = (p) => (p <= 0 ? 0 : p >= 1 ? 1 : p * p * (3 - 2 * p));

/** Envelope of a beat at progress p (0..1): quick onset, hold, release. */
function envelope(p, onset = 0.15, release = 0.45) {
  if (p < 0 || p >= 1) return 0;
  if (p < onset) return ease(p / onset);
  if (p < 1 - release) return 1;
  return 1 - ease((p - (1 - release)) / release);
}

/** A damped bounce: n hops, each smaller than the last (0 at both ends). */
function bounce(p, hops = 2) {
  if (p <= 0 || p >= 1) return 0;
  return Math.abs(Math.sin(p * Math.PI * hops)) * (1 - p) ** 1.6;
}

/** Compose one frame. ``out`` is reused to avoid allocating every frame. */
export function compose(out, s, t, scale) {
  const pose = POSES[s.state] || POSES.idle;
  const L = out.L, R = out.R, body = out.bodyTarget, both = out.eyes;
  for (const eye of both) {
    eye.eyeH = pose.eyeH; eye.eyeW = pose.eyeW || 1; eye.lidTop = pose.lidTop; eye.lidBot = pose.lidBot || 0;
    eye.tilt = 0; eye.lidAngle = pose.lidAngle || 0;
  }
  let lift = pose.lift, head = 0, lookY = 0, bodyY = 0, bodyScale = 1, sx = 1, sy = pose.sy || 1;
  out.dim = pose.dim || 0;
  out.tone = pose.tone;

  // Thinking: the eye on the side it looks toward is a little smaller (a curious asymmetry) and the
  // head tilts that way.
  if (pose.think) {
    const side = s.thinkSide || 1;
    const near = side > 0 ? R : L;
    near.eyeH -= 0.07;
    head += 4.5 * side;
  }

  // Transition beats.
  for (const b of s.beats) {
    const p = (t - b.at) / b.dur;
    if (p < 0 || p >= 1) continue;
    const a = b.amount;
    switch (b.kind) {
      case "noticed": {
        // eyes pop open and the face rises: a little stretch upward
        const e = envelope(p, 0.12, 0.6) * a;
        for (const eye of both) { eye.eyeH += 0.22 * e; eye.lidTop -= 0.08 * e; }
        bodyScale += 0.025 * e; bodyY -= 2.4 * e; sy += 0.04 * e; sx -= 0.022 * e;
        break;
      }
      case "heard":
      case "nod": {
        // down and back up once, the second half softer; squashes a little at the bottom
        const d = Math.sin(p * Math.PI) * (1 - 0.35 * p) * a;
        bodyY += 3.8 * d; lift -= 1.2 * d; sy -= 0.03 * d; sx += 0.022 * d;
        if (b.kind === "heard") for (const eye of both) { eye.lidTop += 0.1 * envelope(p); eye.lidBot += 0.12 * envelope(p); }
        break;
      }
      case "ready": {
        // anticipation: a dip with a squash, then up and open ("I have it")
        const dip = p < 0.35 ? Math.sin((p / 0.35) * Math.PI) : 0;
        const rise = envelope((p - 0.25) / 0.75, 0.3, 0.5);
        bodyY += (2 * dip - 1.8 * rise) * a;
        sy += (-0.045 * dip + 0.03 * rise) * a; sx += (0.03 * dip - 0.018 * rise) * a;
        for (const eye of both) { eye.eyeH += 0.12 * rise * a; eye.lidTop -= 0.05 * rise * a; }
        break;
      }
      case "interrupted": {
        // a startle: eyes wide, a jolt back
        const e = envelope(p, 0.05, 0.55) * a;
        for (const eye of both) { eye.eyeH += 0.32 * e; eye.lidTop -= 0.1 * e; eye.lidBot = Math.max(0, eye.lidBot - 0.1 * e); }
        bodyScale -= 0.022 * e; bodyY -= 2 * e; sy += 0.045 * e; sx -= 0.03 * e;
        break;
      }
      case "recovered": {
        // a visible relaxation: lids soften into a smile, a small settle downward
        const e = envelope(p, 0.25, 0.5) * a;
        for (const eye of both) { eye.lidTop += 0.1 * e; eye.lidBot += 0.2 * e; }
        bodyY += 1.3 * e; sy -= 0.02 * e;
        break;
      }
      case "settle": {
        // a soft release after speaking or finishing: a small exhale, lids ease
        const e = envelope(p, 0.25, 0.55) * a;
        for (const eye of both) { eye.lidTop += 0.08 * e; eye.lidBot += 0.06 * e; }
        bodyY += 1.2 * e * (1 - p * 0.5); sy -= 0.018 * e; sx += 0.01 * e;
        break;
      }
      case "breath": {
        // one slow breath: visible, and never on a fixed rhythm
        const e = Math.sin(p * Math.PI) * a;
        bodyScale += 0.01 * e; bodyY -= 0.8 * e; sy += 0.008 * e;
        break;
      }
      case "cheer": {
        // a real success: a happy squint (crescent eyes) and a little double hop
        const e = envelope(p, 0.12, 0.5) * a;
        const hop = bounce(p, 2) * a;
        for (const eye of both) { eye.lidBot += 0.45 * e; eye.lidTop += 0.04 * e; }
        bodyY -= 4 * hop; sy += 0.045 * hop - 0.025 * e * (1 - hop); sx -= 0.028 * hop;
        break;
      }
      case "puzzled": {
        // something didn't make sense: the head cocks and one eye narrows
        const e = envelope(p, 0.15, 0.4) * a;
        const side = b.side || 1;
        head += 8 * e * side;
        const sq = side > 0 ? L : R;
        sq.lidTop += 0.24 * e; sq.lidBot += 0.14 * e;
        (sq === L ? R : L).eyeH += 0.1 * e;
        break;
      }
      case "wake": {
        // waking up: a long stretch, eyes coming open
        const e = envelope(p, 0.3, 0.5) * a;
        sy += 0.055 * e; sx -= 0.03 * e; bodyY -= 2.8 * e;
        for (const eye of both) eye.lidTop += 0.25 * (1 - ease(p / 0.6));
        break;
      }
      default:
    }
  }

  // Emotions.
  const em = s.emotion.values;
  for (const name in EMOTION_FACE) {
    const v = em[name];
    if (!v) continue;
    const f = EMOTION_FACE[name];
    for (const eye of both) {
      if (f.eyeH) eye.eyeH += f.eyeH * v;
      if (f.eyeW) eye.eyeW += f.eyeW * v;
      if (f.lidTop) eye.lidTop += f.lidTop * v;
      if (f.lidBot) eye.lidBot += f.lidBot * v;
      if (f.tilt) eye.tilt += f.tilt * v;
      if (f.lidAngle) eye.lidAngle += f.lidAngle * v;
    }
    if (f.asym) { R.eyeH += f.asym * v; R.lidTop -= f.asym * 0.6 * v; }
    if (f.head) head += f.head * v;
    if (f.lift) lift += f.lift * v;
    if (f.lookY) lookY += f.lookY * v;
  }

  // Micro-expressions: brief, small, often one-sided.
  for (const mi of s.micro) {
    const p = (t - mi.at) / mi.dur;
    if (p < 0 || p >= 1) continue;
    const e = envelope(p, 0.2, 0.5) * mi.amount;
    for (const eye of both) {
      if ((mi.side === "L" && eye === R) || (mi.side === "R" && eye === L)) continue;
      if (mi.eyeH) eye.eyeH += mi.eyeH * e;
      if (mi.lidTop) eye.lidTop += mi.lidTop * e;
      if (mi.lidBot) eye.lidBot += mi.lidBot * e;
    }
    if (mi.head) head += mi.head * e;
    if (mi.lift) lift += mi.lift * e;
  }

  // The person's voice makes the eyes a little more open while listening (real microphone level).
  if (s.state === "listening") for (const eye of both) eye.eyeH += 0.12 * s.audio.voice.value;
  // JARVIS's own voice: emphasis lifts the face and opens the eyes slightly (real audio only).
  if (s.state === "speaking") {
    const v = s.audio.outputSlow;
    for (const eye of both) eye.eyeH += 0.06 * v;
    bodyY -= 1.4 * v; lift += 0.5 * v;
  }
  // After a while alone JARVIS relaxes, and after longer it rests: heavier lids, never closed.
  if (s.state === "idle" || s.state === "attentive") {
    const heavy = s.idleLevel === "resting" ? 0.22 : s.idleLevel === "relaxed" ? 0.1 : 0;
    if (heavy) for (const eye of both) eye.lidTop += heavy;
  }

  for (const eye of both) {
    eye.lidTop = Math.max(0, Math.min(0.75, eye.lidTop));
    eye.lidBot = Math.max(0, Math.min(0.6, eye.lidBot));
    eye.eyeH = Math.max(0.05, eye.eyeH);
    eye.lidAngle = Math.max(-18, Math.min(18, eye.lidAngle));
  }
  out.lift = lift;
  out.lookY = lookY;
  // Body: physical presence. Scaled down under reduced motion.
  body.y = (bodyY - lift * 0.35) * scale;
  body.rot = (head + s.posture.rot) * scale;
  body.x = s.posture.x * scale;
  body.scale = 1 + (bodyScale - 1 + (s.state === "listening" ? 0.012 : 0)) * scale;
  body.sx = 1 + (sx - 1) * scale;
  body.sy = 1 + (sy - 1) * scale;
  return out;
}

export function newFrame() {
  const eye = () => ({ eyeH: 1, eyeW: 1, lidTop: 0, lidBot: 0, tilt: 0, lidAngle: 0 });
  const L = eye(), R = eye();
  return { L, R, eyes: [L, R], lift: 0, lookY: 0, dim: 1, tone: "off", bodyTarget: { x: 0, y: 0, rot: 0, scale: 1, sx: 1, sy: 1 } };
}

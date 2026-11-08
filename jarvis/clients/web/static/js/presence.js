// <jarvis-presence> — draws JARVIS.
//
// A flat disc with two pill eyes, and a small mouth only while JARVIS's voice
// is actually playing. Deliberately minimal: the personality is in how it
// moves, not in more parts. This element decides nothing: the presence engine
// (presence/engine.js) interprets what is happening and every face on the page
// renders the same engine, so they are one presence. What is particular to a
// face is its position on screen: each face turns the engine's attention
// target (and the mouse: subtly anywhere, more when it is near this face)
// into a gaze for each eye, moves the eyes on critically damped springs (no
// snapping, no overshoot; the right eye a touch slower than the left), and
// gives the whole face a little physical weight: it leans toward what it
// looks at, tilts with curiosity or doubt, squashes and stretches with
// anticipation and surprise, and settles after moving.
//
// The gentle float is a CSS animation on an HTML layer (composited, no work per
// frame); everything else is drawn here, only while something is changing.

import { presence } from "./presence/engine.js";

const SVG = "http://www.w3.org/2000/svg";
const WORDS = { offline: "offline", idle: "ready", attentive: "ready", listening: "listening", thinking: "thinking",
  tool_use: "working", responding: "responding", speaking: "speaking", waiting: "waiting for your approval",
  error: "something went wrong" };
const R_DISC = 46;
const EYE_X = 14, EYE_Y = -2.5, EYE_W = 13, EYE_H = 23;
const RANGE_X = 7, RANGE_Y = 5;     // how far the eyes move (viewBox units)
const DEPTH_PX = 520;               // virtual distance to the screen: turns offsets on screen into gaze angles
const MOUTH_Y = 19;
// Eye shape eases toward the expression (time constant, s): a change of state reads as a change of
// expression rather than a jump, while beats and blinks keep their own timing.
const SHAPE_TAU = 0.075;
const SHAPE = ["eyeH", "eyeW", "lidTop", "lidBot", "lidAngle", "tilt"];
let uid = 0;

const f2 = (v) => (Math.round(v * 100) / 100).toString();
const smooth = (a, b, x) => { const p = Math.max(0, Math.min(1, (x - a) / (b - a))); return p * p * (3 - 2 * p); };

function capsule(cx, cy, w, h) {
  const r = Math.min(w, h) / 2, hw = w / 2, hh = h / 2;
  return h <= w
    ? `M${f2(cx - hw + r)},${f2(cy - hh)} H${f2(cx + hw - r)} A${f2(r)},${f2(r)} 0 0 1 ${f2(cx + hw - r)},${f2(cy + hh)} H${f2(cx - hw + r)} A${f2(r)},${f2(r)} 0 0 1 ${f2(cx - hw + r)},${f2(cy - hh)} Z`
    : `M${f2(cx - hw)},${f2(cy - hh + r)} A${f2(r)},${f2(r)} 0 0 1 ${f2(cx + hw)},${f2(cy - hh + r)} V${f2(cy + hh - r)} A${f2(r)},${f2(r)} 0 0 1 ${f2(cx - hw)},${f2(cy + hh - r)} Z`;
}

class JarvisPresence extends HTMLElement {
  connectedCallback() {
    if (!this._built) this._build();
    presence.attach(this);
  }

  disconnectedCallback() { presence.detach(this); }

  invalidateRect() { this._rect = null; }

  /** Current geometry, for tests and diagnostics. ``mouth`` is the opening from JARVIS's real voice. */
  debug() {
    return { state: presence.state, eyes: this._eyes.map((e) => ({ x: e.x, y: e.y })), body: { ...this._body },
      mouth: this._mouthOpen, follow: this._follow, gaze: presence.target ? presence.target.kind : "person",
      L: { ...presence.frame.L }, R: { ...presence.frame.R }, drawn: this._shape.map((e) => ({ ...e })) };
  }

  _build() {
    this._built = true;
    this.setAttribute("role", "img");
    const el = (tag, cls) => { const n = document.createElementNS(SVG, tag); if (cls) n.setAttribute("class", cls); return n; };
    const id = ++uid;
    const float = document.createElement("div");
    float.className = "face__float";
    const svg = el("svg");
    svg.setAttribute("viewBox", "-50 -50 100 100");
    svg.setAttribute("aria-hidden", "true");
    const defs = el("defs");
    this._bodyG = el("g", "face__body");
    const disc = el("circle", "face__disc");
    disc.setAttribute("r", String(R_DISC));
    this._bodyG.appendChild(disc);
    this._parts = [];
    for (let i = 0; i < 2; i++) {
      const clip = el("clipPath");
      clip.setAttribute("id", `face-lids-${id}-${i}`);
      const lids = el("path");
      clip.appendChild(lids);
      defs.appendChild(clip);
      const g = el("g");
      const eye = el("path", "face__eye");
      eye.setAttribute("clip-path", `url(#face-lids-${id}-${i})`);
      g.appendChild(eye);
      this._bodyG.appendChild(g);
      this._parts.push({ g, eye, lids });
    }
    this._mouth = el("path", "face__mouth");
    this._bodyG.appendChild(this._mouth);
    svg.append(defs, this._bodyG);
    float.appendChild(svg);
    this.appendChild(float);
    this._eyes = [{ x: 0, y: 0.15, vx: 0, vy: 0 }, { x: 0, y: 0.15, vx: 0, vy: 0 }];
    this._body = { x: 0, y: 0, rot: 0, scale: 1, sx: 1, sy: 1, vx: 0, vy: 0, vr: 0, vs: 0, vsx: 0, vsy: 0 };
    this._mouthOpen = 0;
    this._follow = 0;
    this._rect = null;
    this._rectAt = 0;
    this._tone = "";
    this._state = "";
    this._dim = -1;
    this._label = "";
    this._tgt = [{ x: 0, y: 0 }, { x: 0, y: 0 }];
    this._shape = [{ ...presence.frame.L }, { ...presence.frame.R }];
    this._lift = presence.frame.lift;
    this._tmp = { x: 0, y: 0 };
  }

  _box(t) {
    if (!this._rect || t - this._rectAt > 1000) {
      this._rect = this.getBoundingClientRect();
      this._rectAt = t;
    }
    return this._rect;
  }

  /** Gaze (normalized, -1..1) of eye ``i`` toward a point on screen. */
  _gazeTo(point, box, i, out) {
    const scale = box.width / 100;
    const ex = box.left + box.width / 2 + (i === 0 ? -EYE_X : EYE_X) * scale;
    const ey = box.top + box.height / 2 + EYE_Y * scale;
    out.x = Math.atan2(point.x - ex, DEPTH_PX) / 0.62;
    out.y = Math.atan2(point.y - ey, DEPTH_PX) / 0.55;
    return out;
  }

  /** How much the eyes follow the mouse: a little wherever it is, more when it comes near this face. */
  _followWeight(engine, t) {
    const pn = engine.pointerNear;
    if (!pn) return 0;
    const gain = engine.followGain(t);
    if (gain <= 0) return 0;
    const box = this._box(t);
    if (!box.width) return 0;
    const d = Math.hypot(pn.x - (box.left + box.width / 2), pn.y - (box.top + box.height / 2)) / box.width;
    return (0.35 + 0.65 * (1 - smooth(0.8, 2.6, d))) * gain;
  }

  /** Where each eye should look (normalized, -1..1) given the engine's attention target. */
  _targets(engine, t) {
    const c = engine.target;
    const frame = engine.frame;
    const drift = engine._drift || { x: 0, y: 0 };
    let point = null;
    if (c && c.point) point = c.point;
    else if (c && c.element && c.element.isConnected) {
      const r = c.element.getBoundingClientRect();
      // Typing: look near where the text is being written, not the middle of the field.
      point = c.typing ? { x: r.left + Math.min(r.width * 0.4, 220), y: r.top + r.height / 2 }
        : { x: r.left + Math.min(r.width / 2, 220), y: r.top + Math.min(r.height / 2, 120) };
    }
    const follow = this._followWeight(engine, t);
    this._follow = follow;
    const box = point || engine.peripheral || follow ? this._box(t) : null;
    const tmp = this._tmp;
    for (let i = 0; i < 2; i++) {
      const tg = this._tgt[i];
      if (point && box && box.width) {
        this._gazeTo(point, box, i, tg);
      } else if (c && c.gaze) {
        tg.x = c.gaze.x; tg.y = c.gaze.y;
      } else {
        // The person in front of the screen, with the small fixation drift of the idle behaviour.
        tg.x = drift.x; tg.y = drift.y;
      }
      // The mouse: the eyes follow it, smoothly and only so far.
      if (follow && box && box.width) {
        this._gazeTo(engine.pointerNear, box, i, tmp);
        tg.x += (tmp.x - tg.x) * follow;
        tg.y += (tmp.y - tg.y) * follow;
      }
      // Peripheral awareness while looking inward: a faint pull toward the moving pointer, never a switch.
      const pe = engine.peripheral;
      if (pe && box && box.width) {
        this._gazeTo(pe, box, i, tmp);
        tg.x += (tmp.x - tg.x) * pe.w;
        tg.y += (tmp.y - tg.y) * pe.w;
      }
      tg.y += frame.lookY;
      const m = Math.hypot(tg.x, tg.y * 1.1);
      if (m > 1) { tg.x /= m; tg.y /= m; }
      if (engine.reduced) { tg.x *= 0.45; tg.y *= 0.45; }
      if (engine.state === "offline") { tg.x = 0; tg.y = 0.3; }
    }
    const omega = c ? c.omega : 6;
    return follow > 0.05 ? Math.max(omega, 8) : omega;
  }

  render(engine, dt, t) {
    // Not displayed (e.g. on a hidden screen): nothing to draw.
    if (!this.isConnected || !this.getClientRects().length) return false;
    let moving = false;
    const followWas = this._follow;
    const omega = this._targets(engine, t);
    if (Math.abs(this._follow - followWas) > 0.002) moving = true;
    // Eyes: critically damped springs toward their targets.
    for (let i = 0; i < 2; i++) {
      const e = this._eyes[i], tg = this._tgt[i];
      const w = omega * (i === 0 ? 1 : 0.93);
      let rem = dt;
      while (rem > 1e-6) {
        const h = Math.min(rem, 1 / 120);
        rem -= h;
        e.vx += (w * w * (tg.x - e.x) - 2 * w * e.vx) * h;
        e.vy += (w * w * (tg.y - e.y) - 2 * w * e.vy) * h;
        e.x += e.vx * h;
        e.y += e.vy * h;
      }
      if (Math.abs(tg.x - e.x) > 0.003 || Math.abs(tg.y - e.y) > 0.003 || Math.abs(e.vx) + Math.abs(e.vy) > 0.02) moving = true;
      else { e.x = tg.x; e.y = tg.y; e.vx = e.vy = 0; }
    }
    // Body: leans toward what it looks at; squash and stretch settle with a slight give.
    const bt = engine.frame.bodyTarget, b = this._body, s = engine.scale;
    const lean = ((this._eyes[0].x + this._eyes[1].x) / 2) * 1.6 * s;
    const goals = [bt.x + lean, bt.y, bt.rot + lean, bt.scale, bt.sx, bt.sy];
    const keys = ["x", "y", "rot", "scale", "sx", "sy"], vel = ["vx", "vy", "vr", "vs", "vsx", "vsy"];
    for (let k = 0; k < 6; k++) {
      const w = k >= 4 ? 13 : 9, z = k >= 4 ? 0.55 : 0.72;  // squash/stretch springs back a little bouncier
      let rem = dt;
      while (rem > 1e-6) {
        const h = Math.min(rem, 1 / 120);
        rem -= h;
        b[vel[k]] += (w * w * (goals[k] - b[keys[k]]) - 2 * z * w * b[vel[k]]) * h;
        b[keys[k]] += b[vel[k]] * h;
      }
      const eps = k >= 3 ? 0.0005 : 0.01;
      if (Math.abs(goals[k] - b[keys[k]]) > eps || Math.abs(b[vel[k]]) > eps * 2) moving = true;
      else { b[keys[k]] = goals[k]; b[vel[k]] = 0; }
    }
    // Eye shape and lift: a short ease toward the expression.
    const fr = engine.frame, k = 1 - Math.exp(-dt / SHAPE_TAU);
    for (let i = 0; i < 2; i++) {
      const src = i === 0 ? fr.L : fr.R, dst = this._shape[i];
      for (const c of SHAPE) {
        const d = src[c] - dst[c];
        if (Math.abs(d) > (c === "lidAngle" || c === "tilt" ? 0.02 : 0.001)) { dst[c] += d * k; moving = true; } else dst[c] = src[c];
      }
    }
    if (Math.abs(fr.lift - this._lift) > 0.005) { this._lift += (fr.lift - this._lift) * k; moving = true; } else this._lift = fr.lift;
    // Mouth: JARVIS's real voice only.
    const mouth = engine.audio.mouth * (engine.reduced ? 0.6 : 1);
    if (Math.abs(mouth - this._mouthOpen) > 0.001) moving = true;
    this._mouthOpen = mouth;
    this._draw(engine, t);
    return moving;
  }

  _draw(engine, t) {
    const fr = engine.frame;
    const state = engine.state;
    const label = `JARVIS, ${WORDS[state] || state}`;
    if (label !== this._label) { this._label = label; this.setAttribute("aria-label", label); }
    if (fr.tone !== this._tone) { this._tone = fr.tone; this.dataset.tone = fr.tone; }
    if (state !== this._state) { this._state = state; this.dataset.state = state; }
    const b = this._body;
    // Squash and stretch are anchored at the bottom of the disc, like a soft body resting on something.
    this._bodyG.setAttribute("transform", `translate(${f2(b.x)} ${f2(b.y)}) rotate(${f2(b.rot)}) translate(0 ${R_DISC}) `
      + `scale(${(Math.round(b.scale * b.sx * 1000) / 1000)} ${(Math.round(b.scale * b.sy * 1000) / 1000)}) translate(0 ${-R_DISC})`);
    for (let i = 0; i < 2; i++) {
      const side = i === 0 ? -1 : 1;
      const p = this._shape[i];
      const e = this._eyes[i];
      const blink = engine.closure(i === 0 ? "L" : "R", t);
      const gx = e.x * RANGE_X, gy = e.y * RANGE_Y;
      // Looking aside foreshortens the eyes slightly.
      const w = EYE_W * p.eyeW * (1 - 0.1 * Math.min(1, Math.abs(e.x)));
      const fullH = EYE_H * p.eyeH;
      const cx = side * EYE_X + gx;
      const cy = EYE_Y + gy - this._lift;
      // The upper lid lowers the top of the eye and keeps it round: heavy, relaxed or focused
      // eyes are shorter pills, never cut flat.
      const top0 = cy - fullH / 2 + Math.max(0, Math.min(0.8, p.lidTop)) * fullH, bottom0 = cy + fullH / 2;
      const open = bottom0 - top0;
      // A blink closes the eye toward a line a little below its middle.
      const h = Math.max(1.3, open * (1 - 0.92 * blink));
      const ey = (top0 + bottom0) / 2 + 0.16 * open * blink;
      const { g, eye, lids } = this._parts[i];
      eye.setAttribute("d", capsule(cx, ey, w, h));
      g.setAttribute("transform", `rotate(${f2(-side * p.tilt)} ${f2(cx)} ${f2(cy)})`);
      // Clipped only for what a round pill can't show: a slanted upper lid (inner corner down:
      // determined; up: worried), and the lower lid rising as a round arc (a smile in the eyes;
      // all the way up, a thick happy arch).
      const far = w * 2, r = w / 2, bw = r + 1.4;
      const slope = Math.tan((p.lidAngle * Math.PI) / 180) * -side;   // + lowers the lid toward the inner corner
      const y0 = top0 - 0.3 + 1.2 * r * Math.abs(slope);              // level with the top when not slanted
      const yt = (x) => y0 + slope * (x - cx);
      const sq = Math.max(0, Math.min(1, p.lidBot / 0.45));
      const ys0 = bottom0 + 1.5, ys1 = Math.min(top0 + r + 4.5, bottom0 - 1);
      const ys = ys0 + (ys1 - ys0) * sq;                                // where the lid meets the sides of the eye
      const ym = ys - Math.sqrt(sq) * Math.max(0, ys1 - (top0 + 5.5));   // its peak in the middle: round from the start
      const yc = (ym - 0.25 * ys) / 0.75;                               // cubic control height giving that peak
      const yb = Math.max(ys, bottom0) + 4;
      lids.setAttribute("d", `M${f2(cx - far)},${f2(yt(cx - far))} L${f2(cx + far)},${f2(yt(cx + far))} `
        + `L${f2(cx + far)},${f2(yb)} L${f2(cx + bw)},${f2(ys)} `
        + `C${f2(cx + bw * 0.55)},${f2(yc)} ${f2(cx - bw * 0.55)},${f2(yc)} ${f2(cx - bw)},${f2(ys)} `
        + `L${f2(cx - far)},${f2(yb)} Z`);
    }
    // Mouth: a small ellipse that opens with JARVIS's voice while it plays; its width varies
    // a little between syllables. At rest there is no mouth at all.
    const open = this._mouthOpen;
    const avgY = (this._eyes[0].y + this._eyes[1].y) / 2 * RANGE_Y;
    const y = MOUTH_Y + avgY * 0.3 - this._lift * 0.15;
    const rx = (8 - open * 1.4 + engine.audio.mouthShape * 1.1 * open) / 2, ry = (1 + open * 6) / 2;
    this._mouth.setAttribute("d", `M${f2(-rx)},${f2(y)} a${f2(rx)},${f2(ry)} 0 1 0 ${f2(2 * rx)},0 a${f2(rx)},${f2(ry)} 0 1 0 ${f2(-2 * rx)},0 Z`);
    this._mouth.style.opacity = open > 0.01 ? Math.min(1, open * 4).toFixed(2) : "0";
    if (fr.dim !== this._dim) { this._dim = fr.dim; this.style.setProperty("--dim", fr.dim.toFixed(2)); }
  }
}

customElements.define("jarvis-presence", JarvisPresence);
export { presence };

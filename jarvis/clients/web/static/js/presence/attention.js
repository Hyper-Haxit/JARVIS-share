// Attention: what JARVIS is attending to, chosen by priority.
//
//   CRITICAL    being interrupted, a message just submitted, an error
//   VOICE       the wake word, the person speaking, listening
//   TOOL_EVENT  a tool starting, finishing or failing (brief looks at the work)
//   TASK        the current task: thinking (attention turns inward), working,
//               speaking to the person, an approval waiting
//   UI          a panel opening, a click, a control being hovered or focused
//   INPUT       the field the person is typing in
//   IDLE        the idle behaviour's glances
//   (none)      the person in front of the screen
//
// Each claim has a target (a point on screen, an element, an abstract gaze
// direction, or "the person") and an expiry. The highest-priority live claim
// wins; ties go to the most recent. A task claim is a lock: while thinking or
// working, nothing below it (hovering, typing) takes attention away; only the
// person's own actions get a brief check-in look, and only sometimes.
//
// The pointer is not a claim: the faces follow it a little wherever it is and
// more when it is near them (presence.js, engine.followGain); while JARVIS
// looks inward it is at most a faint peripheral pull. Here it is only recorded.

export const P = Object.freeze({ CRITICAL: 95, VOICE: 85, TOOL_EVENT: 75, TASK: 70, UI: 50, INPUT: 40, IDLE: 10 });

export class AttentionController {
  constructor(random) {
    this.random = random;
    this.claims = new Map();     // kind -> claim
    this.pointer = { lastX: 0, lastY: 0, movedAt: -1e9 };
  }

  /**
   * Claim attention. ``target``: { point: {x, y} } | { element } | { find: () => element } | { gaze: {x, y} } | { person: true }.
   * ``find`` looks the element up when it is needed (it may not be on screen yet when the event arrives).
   * ``omega``: how quickly the eyes move there (rad/s of the gaze spring).
   */
  claim(kind, priority, target, t, ms, omega = 14) {
    this.claims.set(kind, { kind, priority, ...target, at: t, until: ms === Infinity ? Infinity : t + ms, omega });
  }

  release(kind) { this.claims.delete(kind); }

  has(kind, t) {
    const c = this.claims.get(kind);
    return Boolean(c && c.until > t);
  }

  /** The pointer moved (for the peripheral pull while JARVIS looks inward). */
  pointerMoved(x, y, t) {
    const p = this.pointer;
    p.lastX = x; p.lastY = y; p.movedAt = t;
  }

  pointerLeft() { this.pointer.movedAt = -1e9; }

  /** The highest-priority live claim, or null (look at the person). */
  current(t) {
    let best = null;
    for (const [kind, c] of this.claims) {
      if (c.until <= t) { this.claims.delete(kind); continue; }
      if (c.find) c.element = c.find() || null;
      if (c.element && !c.element.isConnected) { this.claims.delete(kind); continue; }
      if (c.find && !c.element) continue;  // nothing to look at yet
      if (!best || c.priority > best.priority || (c.priority === best.priority && c.at > best.at)) best = c;
    }
    return best;
  }

  get busyUntil() {
    let u = 0;
    for (const c of this.claims.values()) if (c.until !== Infinity) u = Math.max(u, c.until);
    return u;
  }

  clear() { this.claims.clear(); }
}

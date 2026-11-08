// Reaction selectivity: what makes a reaction mean something is that most
// events don't get one.
//
// Every visible reaction goes through allow(): each kind has a cooldown, and
// there is an overall budget, so a burst of events (five pointer clicks, twenty
// progress updates, ten tools finishing at once) produces at most one or two
// reactions instead of a flurry.

const BUDGET = 4;          // visible reactions ...
const WINDOW_MS = 5000;    // ... per this long, across all kinds

export class Reactions {
  constructor() {
    this.last = new Map();
    this.recent = [];
  }

  /**
   * May a reaction of ``kind`` happen at ``t``? Records it if so.
   * ``essential`` reactions (being interrupted, an error) ignore the shared budget.
   */
  allow(kind, t, cooldownMs, essential = false) {
    if (t - (this.last.get(kind) ?? -1e9) < cooldownMs) return false;
    this.recent = this.recent.filter((x) => t - x < WINDOW_MS);
    if (!essential && this.recent.length >= BUDGET) return false;
    this.last.set(kind, t);
    this.recent.push(t);
    return true;
  }

  clear() {
    this.last.clear();
    this.recent = [];
  }
}

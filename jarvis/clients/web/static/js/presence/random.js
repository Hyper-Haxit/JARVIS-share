// Seedable randomness for the presence.
//
// Everything that varies (idle beats, blink timing, reaction strength, whether
// JARVIS bothers to look at something) draws from one generator. Production
// seeds it from the platform's random source; tests set
// window.__JARVIS_PRESENCE_SEED__ before the modules load so behaviour is
// reproducible.

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function initialSeed() {
  const fixed = globalThis.__JARVIS_PRESENCE_SEED__;
  if (Number.isFinite(fixed)) return fixed;
  try { return crypto.getRandomValues(new Uint32Array(1))[0]; } catch { return Math.floor(Math.random() * 2 ** 32); }
}

export class Random {
  constructor(seed = initialSeed()) { this.reseed(seed); }

  reseed(seed) {
    this.seed = seed >>> 0;
    this._next = mulberry32(this.seed);
  }

  next() { return this._next(); }
  range(a, b) { return a + this._next() * (b - a); }
  chance(p) { return this._next() < p; }
  sign() { return this._next() < 0.5 ? -1 : 1; }

  /** Standard normal sample. */
  gauss() {
    const u = 1 - this._next(), v = this._next();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /** Log-normal sample clamped to [lo, hi]: irregular like natural timing, never a fixed period. */
  lognormal(median, sigma, lo, hi) {
    return Math.max(lo, Math.min(hi, median * Math.exp(sigma * this.gauss())));
  }

  /** Weighted choice from { option: weight }; ``avoid`` options are left out when possible. */
  pick(table, avoid = []) {
    let entries = Object.entries(table).filter(([k, w]) => w > 0 && !avoid.includes(k));
    if (!entries.length) entries = Object.entries(table).filter(([, w]) => w > 0);
    let r = this._next() * entries.reduce((n, [, w]) => n + w, 0);
    for (const [k, w] of entries) if ((r -= w) <= 0) return k;
    return entries[entries.length - 1][0];
  }
}

export const random = new Random();

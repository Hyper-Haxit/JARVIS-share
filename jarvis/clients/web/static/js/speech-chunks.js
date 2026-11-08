// How a reply is cut into chunks to speak. The same rules as the core's
// jarvis/voice/tts.py (split_sentences, SpeechChunker); tests/test_tts.py checks
// that both give identical chunks. No imports, so it can be tested on its own.
//
//  - whole sentences; abbreviations ("e.g.", "Dr.", "vs.") and list numbers
//    ("1. Set a goal.") don't end one
//  - while little has been spoken, the sentence being written may go out early:
//    as its first clause (>= 4 words), or, with no comma yet, before a word that
//    starts a clause ("because", "which", ...) once 7 words come before it (a comma
//    is added so the voice keeps the "more is coming" intonation)
//  - a sentence longer than 28 words is cut at its clauses (pieces >= 4 words)

const ABBREVIATIONS = new Set(["e.g.", "i.e.", "vs.", "etc.", "mr.", "mrs.", "ms.", "dr.", "prof.", "st.", "jr.", "sr.",
  "approx.", "fig.", "inc.", "ltd.", "co.", "dept.", "est.", "jan.", "feb.", "mar.", "apr.",
  "jun.", "jul.", "aug.", "sep.", "sept.", "oct.", "nov.", "dec."]);
const BOUNDARY = /[.!?;:]+["')\]]*(?=\s|$)/g;
const LIST_MARKER = /^\s*(?:[-*•]\s*)?\d{1,3}$/;
const CLAUSE = /(?:,|\s[-–—]|[–—])(?=\s+\S)/g;
const PHRASE = /\s(?=(?:because|but|which|while|when|whereas|although|though|since|unless|until|if|where)\s+\S)/gi;
const FIRST_CLAUSE_MIN_WORDS = 4;
const PHRASE_MIN_WORDS = 7;
const MIN_PIECE_WORDS = 4;
const EARLY_WORDS = 8;
const LONG_SENTENCE_WORDS = 28;

const words = (s) => (s.trim() ? s.trim().split(/\s+/).length : 0);

function abbreviationBefore(text, dot) {
  const m = text.slice(0, dot + 1).match(/([^\s("']+)$/);
  const word = m ? m[1].toLowerCase() : "";
  return ABBREVIATIONS.has(word) || /^(?:[a-z]\.){2,}$/.test(word);
}

/** Complete sentences off the front of streaming text: [sentences, remainder]. */
export function splitSentences(text) {
  const out = [];
  const re = new RegExp(BOUNDARY.source, "g");
  let pos = 0;
  let m;
  while ((m = re.exec(text))) {
    const end = m.index + m[0].length;
    if (end >= text.length) break;
    if (m[0].replace(/["')\]]+$/, "") === "." && (abbreviationBefore(text, m.index) || LIST_MARKER.test(text.slice(pos, m.index)))) continue;
    const sentence = text.slice(pos, end).trim();
    if (sentence.length > 1) out.push(sentence);
    pos = end + text.slice(end).match(/^\s*/)[0].length;
    re.lastIndex = pos;
  }
  return [out, text.slice(pos)];
}

export class SpeechChunker {
  constructor() {
    this.buf = "";
    this.chunks = 0;
    this.words = 0;
  }

  /** Text as it arrives; returns the chunks that are ready to speak. */
  feed(text) {
    this.buf += text;
    const [done, rest] = splitSentences(this.buf);
    this.buf = rest;
    let out = done.flatMap((s) => this._splitLong(s));
    if (!out.length && this.words < EARLY_WORDS) {
      out = this._takeClause(FIRST_CLAUSE_MIN_WORDS, false);
      if (!out.length) out = this._takePhrase();
    } else if (words(this.buf) > LONG_SENTENCE_WORDS) {
      out = out.concat(this._takeClause(MIN_PIECE_WORDS, true));
    }
    return this._counted(out);
  }

  /** The reply is complete: whatever is left. */
  flush() {
    const rest = this.buf.trim();
    this.buf = "";
    return this._counted(rest ? this._splitLong(rest) : []);
  }

  _counted(out) {
    this.chunks += out.length;
    this.words += out.reduce((n, c) => n + words(c), 0);
    return out;
  }

  _takeClause(minWords, last) {
    const cuts = [];
    for (const m of this.buf.matchAll(CLAUSE)) {
      const end = m.index + m[0].length;
      if (words(this.buf.slice(0, end)) >= minWords) cuts.push(end);
    }
    if (!cuts.length) return [];
    const cut = last ? cuts[cuts.length - 1] : cuts[0];
    const head = this.buf.slice(0, cut).trim();
    this.buf = this.buf.slice(cut).replace(/^\s+/, "");
    return [head];
  }

  _takePhrase() {
    for (const m of this.buf.matchAll(PHRASE)) {
      const head = this.buf.slice(0, m.index).trim();
      if (words(head) >= PHRASE_MIN_WORDS) {
        this.buf = this.buf.slice(m.index + m[0].length);
        return [/[,;:]$/.test(head) ? head : head + ","];
      }
    }
    return [];
  }

  _splitLong(sentence) {
    if (words(sentence) <= LONG_SENTENCE_WORDS) return [sentence];
    const pieces = [];
    let start = 0;
    for (const m of sentence.matchAll(CLAUSE)) {
      const end = m.index + m[0].length;
      if (words(sentence.slice(start, end)) >= MIN_PIECE_WORDS && words(sentence.slice(end)) >= MIN_PIECE_WORDS) {
        pieces.push(sentence.slice(start, end).trim());
        start = end;
      }
    }
    pieces.push(sentence.slice(start).trim());
    return pieces.filter(Boolean);
  }
}

/** All the chunks of a complete text. */
export function speechChunks(text) {
  const c = new SpeechChunker();
  return c.feed(text).concat(c.flush());
}

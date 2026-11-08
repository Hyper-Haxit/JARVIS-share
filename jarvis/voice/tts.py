"""Text-to-speech: JARVIS's voice.

  VoiceChain   what JARVIS speaks with: a primary engine and a fallback engine
               (make_voice builds it from the [voice] config).
  Kokoro       the primary voice: a local neural voice (jarvis/voice/kokoro.py).
  OS voice     the fallback, and the voice when tts_engine = "system":
    Windows      SAPI directly through COM (SpVoice -> SpMemoryStream, 16 kHz PCM).
                 pyttsx3's SAPI driver deadlocks on repeated runAndWait() calls, so it
                 is not used on Windows.
    macOS/Linux  pyttsx3 (NSSpeechSynthesizer / eSpeak) on one dedicated thread.

Engines are not thread-safe, so each backend owns one worker thread.

Text goes through prepare_text() (markdown removed, a few pronunciations fixed)
and streaming replies are cut into chunks by SpeechChunker, so speech can start
before the whole reply has been written.
"""

from __future__ import annotations

import importlib.util
import io
import os
import queue
import re
import sys
import tempfile
import threading
import time
import wave
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from jarvis.core.observability import get_logger

log = get_logger("voice.tts")

_MD = [
    (re.compile(r"```.*?```", re.S), " (code omitted) "),
    (re.compile(r"`([^`]*)`"), r"\1"),
    (re.compile(r"\*\*([^*]+)\*\*"), r"\1"),
    (re.compile(r"\*([^*]+)\*"), r"\1"),
    (re.compile(r"^\s*#+\s*", re.M), ""),
    (re.compile(r"^\s*[-*•]\s+", re.M), ""),
    (re.compile(r"\[([^\]]+)\]\([^)]+\)"), r"\1"),
    (re.compile(r"https?://\S+"), "a link"),
]


def speakable(text: str) -> str:
    for pat, rep in _MD:
        text = pat.sub(rep, text)
    return re.sub(r"\s+", " ", text).strip()


# Words the pronunciation front ends (espeak-ng for Kokoro, SAPI) get wrong, found by
# checking their phonemes (.jarvis-dev/kokoro-bench/results/pronunciation.json):
# "pytest" came out as "PIE-dist", "CLI" as "cly", "$12.50" as "dollar twelve point
# five zero", "Oct 6" as "ock six", "vs." as "V S".
_MONTHS = {"jan": "January", "feb": "February", "mar": "March", "apr": "April", "jun": "June", "jul": "July",
           "aug": "August", "sep": "September", "sept": "September", "oct": "October", "nov": "November", "dec": "December"}
_SAY_AS: list[tuple[re.Pattern[str], Any]] = [
    (re.compile(r"\bpytest\b", re.I), "pie test"),
    (re.compile(r"\bPostgreSQL\b", re.I), "Postgres Q L"),
    (re.compile(r"\bCLI\b"), "C L I"),
    (re.compile(r"\be\.g\.(?=[\s,]|$)", re.I), "for example"),
    (re.compile(r"\bi\.e\.(?=[\s,]|$)", re.I), "that is"),
    (re.compile(r"\bvs\.?(?=\s)", re.I), "versus"),
    (re.compile(r"\s?°\s?C\b"), " degrees Celsius"),
    (re.compile(r"\s?°\s?F\b"), " degrees Fahrenheit"),
    (re.compile(r"\b(\d+(?:\.\d+)?)x\b"), r"\1 times"),
    (re.compile(r"\$(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{2}))?\b"),
     lambda m: f"{m[1]} dollar{'' if m[1] == '1' else 's'}" + (f" {m[2]}" if m[2] and m[2] != "00" else "")),
    (re.compile(r"\b(Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec)\.?\s+(\d{1,2})\b(?![\d:])"),
     lambda m: f"{_MONTHS[m[1].lower()]} {_ordinal(int(m[2]))}"),
]


def _ordinal(n: int) -> str:
    suffix = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def prepare_text(text: str) -> str:
    """What a voice engine is given: speakable text with the known mispronunciations fixed."""
    text = speakable(text)
    for pat, rep in _SAY_AS:
        text = pat.sub(rep, text)
    return text


# A "." after one of these doesn't end a sentence ("e.g. this", "Dr. Smith", "vs. that").
ABBREVIATIONS = frozenset({"e.g.", "i.e.", "vs.", "etc.", "mr.", "mrs.", "ms.", "dr.", "prof.", "st.", "jr.", "sr.",
                           "approx.", "fig.", "inc.", "ltd.", "co.", "dept.", "est.", "jan.", "feb.", "mar.", "apr.",
                           "jun.", "jul.", "aug.", "sep.", "sept.", "oct.", "nov.", "dec."})
_BOUNDARY = re.compile(r"[.!?;:]+[\"')\]]*(?=\s|$)")


def _abbreviation_before(text: str, dot: int) -> bool:
    """Is the "." at index ``dot`` the end of an abbreviation rather than of a sentence?"""
    m = re.search(r"([^\s(\"']+)$", text[:dot + 1])
    word = m.group(1).lower() if m else ""
    return word in ABBREVIATIONS or bool(re.fullmatch(r"(?:[a-z]\.){2,}", word))  # "a.m.", "u.s."


_LIST_MARKER = re.compile(r"\s*(?:[-*•]\s*)?\d{1,3}")


def split_sentences(text: str) -> tuple[list[str], str]:
    """Split complete sentences off the front of streaming text. Returns (sentences, remainder).

    A sentence is complete once whitespace follows its end mark (a "." at the very end
    may still become "2.5"). Abbreviations don't end a sentence, and neither does the
    number of a list item ("1. Set a goal." is one sentence, not "1." and "Set a goal.")."""
    out: list[str] = []
    pos = search = 0
    while (m := _BOUNDARY.search(text, search)) is not None:
        end = m.end()
        if end >= len(text):
            break
        if m.group(0).rstrip("\"')]") == "." and (_abbreviation_before(text, m.start())
                                                   or _LIST_MARKER.fullmatch(text[pos:m.start()])):
            search = end
            continue
        sentence = text[pos:end].strip()
        if len(sentence) > 1:
            out.append(sentence)
        pos = search = end + len(re.match(r"\s*", text[end:]).group(0))
    return out, text[pos:]


class SpeechChunker:
    """Cuts a reply that is still being written into chunks to speak, as early as sounds natural.

    - Whole sentences (split_sentences: abbreviations and list numbers don't end one).
    - While little has been handed over to speak (under EARLY_WORDS words, e.g. nothing
      yet, or just "Sure."), the sentence being written may go out early:
        * as its first clause ("You have three meetings tomorrow,") once that clause has
          FIRST_CLAUSE_MIN_WORDS words. On replies with such a clause the benchmark
          measured first speech roughly halving (~1.1 s -> ~0.5 s);
        * with no comma yet, before a word that starts a new clause ("because", "which",
          "when", ...) once at least PHRASE_MIN_WORDS words come before it. A comma is
          added so the voice keeps the "more is coming" intonation. Measured on the real
          path: a 29-word opening sentence without an early comma held first audio for 3.5 s.
      Speech then starts sooner, and a short opening isn't followed by a long silence
      while a long sentence is synthesized.
    - A sentence longer than LONG_SENTENCE_WORDS words is cut at its clauses.
    Clause pieces always have at least MIN_PIECE_WORDS words; only whole sentences
    ("Sure.") can be shorter, so there are no unnatural fragments.
    The web client's js/speech-chunks.js follows the same rules (kept in step by tests/test_tts.py).
    """

    FIRST_CLAUSE_MIN_WORDS = 4
    PHRASE_MIN_WORDS = 7
    MIN_PIECE_WORDS = 4
    EARLY_WORDS = 8
    LONG_SENTENCE_WORDS = 28
    _CLAUSE = re.compile(r"(?:,|\s[-–—]|[–—])(?=\s+\S)")  # a clause that the text continues after
    # Words that start a clause; "and"/"or"/"that" are left out (they as often join two nouns).
    _PHRASE = re.compile(r"\s(?=(?:because|but|which|while|when|whereas|although|though|since|unless|until|if|where)\s+\S)", re.I)

    def __init__(self) -> None:
        self._buf = ""
        self.chunks = 0
        self.words = 0

    def feed(self, text: str) -> list[str]:
        self._buf += text
        done, self._buf = split_sentences(self._buf)
        out = [piece for s in done for piece in self._split_long(s)]
        if not out and self.words < self.EARLY_WORDS:
            out = self._take_clause(self.FIRST_CLAUSE_MIN_WORDS, last=False) or self._take_phrase()
        elif len(self._buf.split()) > self.LONG_SENTENCE_WORDS:
            out += self._take_clause(self.MIN_PIECE_WORDS, last=True)
        return self._counted(out)

    def _take_phrase(self) -> list[str]:
        for m in self._PHRASE.finditer(self._buf):
            head = self._buf[:m.start()].strip()
            if len(head.split()) >= self.PHRASE_MIN_WORDS:
                self._buf = self._buf[m.end():]
                return [head if head[-1] in ",;:" else head + ","]
        return []

    def flush(self) -> list[str]:
        rest, self._buf = self._buf.strip(), ""
        return self._counted(self._split_long(rest) if rest else [])

    def _counted(self, out: list[str]) -> list[str]:
        self.chunks += len(out)
        self.words += sum(len(c.split()) for c in out)
        return out

    def _take_clause(self, min_words: int, last: bool) -> list[str]:
        cuts = [m.end() for m in self._CLAUSE.finditer(self._buf) if len(self._buf[:m.end()].split()) >= min_words]
        if not cuts:
            return []
        cut = cuts[-1] if last else cuts[0]
        head, self._buf = self._buf[:cut].strip(), self._buf[cut:].lstrip()
        return [head]

    def _split_long(self, sentence: str) -> list[str]:
        if len(sentence.split()) <= self.LONG_SENTENCE_WORDS:
            return [sentence]
        pieces, start = [], 0
        for m in self._CLAUSE.finditer(sentence):
            if len(sentence[start:m.end()].split()) >= self.MIN_PIECE_WORDS and len(sentence[m.end():].split()) >= self.MIN_PIECE_WORDS:
                pieces.append(sentence[start:m.end()].strip())
                start = m.end()
        pieces.append(sentence[start:].strip())
        return [p for p in pieces if p]


class TTSUnavailable(RuntimeError):
    """No voice could produce this utterance."""


class TTSCancelled(Exception):
    """The utterance was cancelled (interrupted, stopped) before it was produced."""


@dataclass
class Speech:
    pcm: np.ndarray       # float32 mono
    sample_rate: int
    text: str
    latency_ms: float
    provider: str = ""    # the engine that actually produced it ("kokoro", "sapi", ...)

    def wav_bytes(self) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            w.writeframes((np.clip(self.pcm, -1, 1) * 32767).astype(np.int16).tobytes())
        return buf.getvalue()


def read_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        width = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768
    elif width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128
    else:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2**31
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data, sr


class _Pyttsx3Worker:
    """The single pyttsx3 engine of this process (pyttsx3 caches one engine per
    process and it must stay on the thread that created it)."""

    _instance: _Pyttsx3Worker | None = None
    _lock = threading.Lock()

    @classmethod
    def get(cls) -> _Pyttsx3Worker:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self) -> None:
        self._q: queue.Queue[tuple] = queue.Queue()
        self.voices: list[tuple[str, str]] = []
        self.error: str | None = None
        self._ready = threading.Event()
        threading.Thread(target=self._run, name="jarvis-tts", daemon=True).start()
        self._ready.wait(20)

    def _run(self) -> None:
        if sys.platform == "win32":
            try:
                import pythoncom  # noqa: PLC0415

                pythoncom.CoInitialize()
            except Exception:  # noqa: BLE001
                pass
        engine = None
        try:
            import pyttsx3  # noqa: PLC0415

            engine = pyttsx3.init()
            self.voices = [(v.id, v.name) for v in engine.getProperty("voices")]
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
        self._ready.set()
        while True:
            text, voice, rate, fut = self._q.get()
            if engine is None:
                fut.set_exception(RuntimeError(f"text-to-speech unavailable: {self.error}"))
                continue
            started = time.perf_counter()
            fd, path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            try:
                if voice:
                    for vid, vname in self.voices:
                        if voice.lower() in vname.lower() or voice == vid:
                            engine.setProperty("voice", vid)
                            break
                engine.setProperty("rate", rate)
                engine.save_to_file(text, path)
                engine.runAndWait()
                pcm, sr = read_wav(path)
                fut.set_result(Speech(pcm=pcm, sample_rate=sr, text=text, latency_ms=round((time.perf_counter() - started) * 1000, 1)))
            except Exception as e:  # noqa: BLE001
                fut.set_exception(e)
            finally:
                try:
                    os.unlink(path)
                except OSError:
                    pass

    def submit(self, text: str, voice: str | None, rate: int) -> Future[Speech]:
        fut: Future[Speech] = Future()
        self._q.put((text, voice, rate, fut))
        return fut


class Pyttsx3TTS:
    name = "pyttsx3"

    def __init__(self, voice: str | None = None, rate: int = 185) -> None:
        self.voice = voice
        self.rate = rate

    @staticmethod
    def dependency_ok() -> tuple[bool, str]:
        # Checked without importing (status reads run on the server's event loop).
        if importlib.util.find_spec("pyttsx3") is None:
            return False, "pyttsx3 is not installed (pip install 'jarvis-core[voice]')"
        return True, ""

    @property
    def voices(self) -> list[str]:
        return [n for _, n in _Pyttsx3Worker.get().voices]

    @property
    def error(self) -> str | None:
        """The engine's start-up error, once it has started. Reading it never starts (or waits
        for) the engine: status reads run on the server's event loop."""
        worker = _Pyttsx3Worker._instance
        return worker.error if worker is not None else None

    def start(self) -> None:
        _Pyttsx3Worker.get()

    def synthesize(self, text: str, timeout: float = 60) -> Speech:
        worker = _Pyttsx3Worker.get()
        if worker.error:
            raise RuntimeError(f"text-to-speech unavailable: {worker.error}")
        text = prepare_text(text)
        if not text:
            return Speech(pcm=np.zeros(0, dtype=np.float32), sample_rate=22050, text="", latency_ms=0, provider=self.name)
        speech = worker.submit(text, self.voice, self.rate).result(timeout=timeout)
        speech.provider = self.name
        return speech

    def stop(self) -> None:
        """The shared engine lives for the life of the process."""


class _SapiWorker:
    """One SAPI SpVoice on one COM-initialised thread for the whole process."""

    _instance: _SapiWorker | None = None
    _lock = threading.Lock()

    @classmethod
    def get(cls) -> _SapiWorker:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self) -> None:
        self._q: queue.Queue[tuple] = queue.Queue()
        self.voices: list[str] = []
        self.error: str | None = None
        self._ready = threading.Event()
        threading.Thread(target=self._run, name="jarvis-sapi", daemon=True).start()
        self._ready.wait(20)

    def _run(self) -> None:
        voice = None
        try:
            import pythoncom  # noqa: PLC0415
            import win32com.client  # noqa: PLC0415

            pythoncom.CoInitialize()
            voice = win32com.client.Dispatch("SAPI.SpVoice")
            tokens = voice.GetVoices()
            self._tokens = [tokens.Item(i) for i in range(tokens.Count)]
            self.voices = [t.GetDescription() for t in self._tokens]
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
        self._ready.set()
        while True:
            text, voice_name, rate, fut = self._q.get()
            if voice is None:
                fut.set_exception(RuntimeError(f"text-to-speech unavailable: {self.error}"))
                continue
            started = time.perf_counter()
            try:
                import win32com.client  # noqa: PLC0415

                if voice_name:
                    for tok, name in zip(self._tokens, self.voices, strict=True):
                        if voice_name.lower() in name.lower():
                            voice.Voice = tok
                            break
                # SAPI rate is -10..10; map words-per-minute around a 180 wpm default.
                voice.Rate = max(-10, min(10, round((rate - 180) / 18)))
                stream = win32com.client.Dispatch("SAPI.SpMemoryStream")
                fmt = win32com.client.Dispatch("SAPI.SpAudioFormat")
                fmt.Type = 18  # SAFT16kHz16BitMono
                stream.Format = fmt
                voice.AudioOutputStream = stream
                voice.Speak(text)
                pcm = np.frombuffer(bytes(stream.GetData()), dtype=np.int16).astype(np.float32) / 32768
                fut.set_result(Speech(pcm=pcm, sample_rate=16000, text=text, latency_ms=round((time.perf_counter() - started) * 1000, 1)))
            except Exception as e:  # noqa: BLE001
                fut.set_exception(e)

    def submit(self, text: str, voice: str | None, rate: int) -> Future[Speech]:
        fut: Future[Speech] = Future()
        self._q.put((text, voice, rate, fut))
        return fut


class SapiTTS(Pyttsx3TTS):
    name = "sapi"

    @staticmethod
    def dependency_ok() -> tuple[bool, str]:
        if importlib.util.find_spec("win32com") is None:
            return False, "pywin32 is not installed"
        return True, ""

    @property
    def voices(self) -> list[str]:
        return _SapiWorker.get().voices

    @property
    def error(self) -> str | None:
        """See Pyttsx3TTS.error: never starts or waits for the engine."""
        worker = _SapiWorker._instance
        return worker.error if worker is not None else None

    def start(self) -> None:
        _SapiWorker.get()

    def synthesize(self, text: str, timeout: float = 60) -> Speech:
        worker = _SapiWorker.get()
        if worker.error:
            raise RuntimeError(f"text-to-speech unavailable: {worker.error}")
        text = prepare_text(text)
        if not text:
            return Speech(pcm=np.zeros(0, dtype=np.float32), sample_rate=16000, text="", latency_ms=0, provider=self.name)
        speech = worker.submit(text, self.voice, self.rate).result(timeout=timeout)
        speech.provider = self.name
        return speech


def make_tts(engine: str, voice: str | None, rate: int) -> Pyttsx3TTS | None:
    """The operating system's voice (SAPI on Windows, pyttsx3 elsewhere); None when TTS is disabled."""
    if engine == "none":
        return None
    if sys.platform == "win32" and engine in ("pyttsx3", "sapi", "system"):
        return SapiTTS(voice, rate)
    return Pyttsx3TTS(voice, rate)


# ------------------------------------------------------------------ the voice
class VoiceChain:
    """What JARVIS speaks with: a primary engine, and a fallback for anything it can't say.

    An utterance the primary can't produce (not installed, failed to load, still
    loading after the allowed wait, or a synthesis error) is spoken by the fallback.
    The primary is tried again for the next reply, so one transient error never
    disables it. Within one reply, once the fallback has spoken, it speaks the rest
    of that reply too: a reply never switches voices halfway.

    Every Speech carries the provider that produced it; ``stats`` counts them, so
    which voice actually spoke is always visible.
    """

    def __init__(self, primary: Any | None, fallback: Any | None,
                 on_fallback: Callable[[str, str], None] | None = None) -> None:
        if primary is None and fallback is None:
            raise ValueError("a voice needs at least one engine")
        self.primary = primary
        self.fallback = fallback
        self.on_fallback = on_fallback  # (engine that failed, reason) -> None; notifies the person
        self.name = (primary or fallback).name
        self.stats: dict[str, Any] = {"spoken_by": {}, "fallbacks": 0, "failures": 0, "last_provider": None,
                                      "last_fallback_reason": None}
        self._fallback_replies: deque[str] = deque(maxlen=32)
        self._last_reason: str | None = None

    @property
    def voice(self) -> str | None:
        return getattr(self.primary or self.fallback, "voice", None)

    @property
    def error(self) -> str | None:
        """Set only when no engine can speak at all."""
        reasons = [r for r in (self._engine_problem(self.primary), self._engine_problem(self.fallback)) if r]
        usable = (self.primary is not None and not self._engine_problem(self.primary)) or \
                 (self.fallback is not None and not self._engine_problem(self.fallback))
        return None if usable else "; ".join(reasons) or "no voice configured"

    @staticmethod
    def _engine_problem(engine: Any | None) -> str | None:
        if engine is None:
            return None
        ok, why = engine.dependency_ok()
        if not ok:
            return f"{engine.name}: {why}"
        state = getattr(engine, "state", None)
        if state in ("unavailable", "failed"):
            return f"{engine.name}: {getattr(engine, 'error', None) or state}"
        err = getattr(engine, "error", None)
        return f"{engine.name}: {err}" if err and state is None else None

    def dependency_ok(self) -> tuple[bool, str]:
        err = self.error
        return (err is None, err or "")

    def start(self) -> None:
        """Begin loading the engines. Kokoro loads on its own thread, so this returns quickly.

        The fallback starts first: its start-up (SAPI's COM set-up) is quick on its own
        but took ~3 s when it had to compete with Kokoro's loading."""
        for engine in (self.fallback, self.primary):
            if engine is not None:
                try:
                    engine.start()
                except Exception as e:  # noqa: BLE001 - a fallback that can't start is reported, not fatal
                    log.warning("voice engine %s could not start: %s", engine.name, e)

    def stop(self) -> None:
        for engine in (self.primary, self.fallback):
            if engine is not None:
                engine.stop()

    def cancel(self) -> None:
        """Stop the utterance being synthesized right now (interrupted, stopped)."""
        cancel = getattr(self.primary, "cancel", None)
        if cancel is not None:
            cancel()

    def synthesize(self, text: str, timeout: float = 60, *, reply: str | None = None,
                   should_stop: Callable[[], bool] | None = None) -> Speech:
        """Speak ``text`` with the primary, or the fallback when the primary can't.

        ``reply`` groups the chunks of one reply (no switching voices midway);
        ``should_stop`` is polled so a cancelled utterance stops early (TTSCancelled)."""
        reasons: list[str] = []
        if self.primary is not None and (reply is None or reply not in self._fallback_replies):
            try:
                return self._spoken(self._call(self.primary, text, timeout, should_stop))
            except TTSCancelled:
                raise
            except Exception as e:  # noqa: BLE001 - any failure of the primary means: use the fallback
                reason = str(e) or type(e).__name__
                reasons.append(f"{self.primary.name}: {reason}")
                if self.fallback is not None:
                    self._fell_back(reason, reply)
        elif self.primary is not None:
            reasons.append(f"{self.primary.name}: not used for the rest of this reply")
        if should_stop is not None and should_stop():
            raise TTSCancelled("cancelled")
        if self.fallback is not None:
            try:
                return self._spoken(self._call(self.fallback, text, timeout, None))
            except Exception as e:  # noqa: BLE001
                reasons.append(f"{self.fallback.name}: {e}")
        self.stats["failures"] += 1
        raise TTSUnavailable("no voice could speak: " + "; ".join(reasons))

    @staticmethod
    def _call(engine: Any, text: str, timeout: float, should_stop: Callable[[], bool] | None) -> Speech:
        if getattr(engine, "cancellable", False):
            return engine.synthesize(text, timeout, should_stop=should_stop)
        return engine.synthesize(text, timeout)

    def _spoken(self, speech: Speech) -> Speech:
        speech.provider = speech.provider or "unknown"
        if speech.pcm.size:
            by = self.stats["spoken_by"]
            by[speech.provider] = by.get(speech.provider, 0) + 1
            self.stats["last_provider"] = speech.provider
        return speech

    def _fell_back(self, reason: str, reply: str | None) -> None:
        self.stats["fallbacks"] += 1
        self.stats["last_fallback_reason"] = reason
        if reply is not None:
            self._fallback_replies.append(reply)
        if reason != self._last_reason:  # say it once per distinct problem, not for every sentence
            self._last_reason = reason
            log.warning("voice: %s could not speak (%s); using %s", self.primary.name, reason, self.fallback.name)
            if self.on_fallback is not None:
                try:
                    self.on_fallback(self.primary.name, reason)
                except Exception:  # noqa: BLE001
                    log.exception("fallback notification failed")
        else:
            log.info("voice: %s could not speak (%s); using %s", self.primary.name, reason, self.fallback.name)

    def status(self) -> dict[str, Any]:
        primary = self.primary.status() if hasattr(self.primary, "status") else (
            {"engine": self.primary.name, "voice": getattr(self.primary, "voice", None)} if self.primary else None)
        fallback = {"engine": self.fallback.name, "voice": getattr(self.fallback, "voice", None) or "system default",
                    "available": self.fallback.dependency_ok()[0] and not getattr(self.fallback, "error", None)} if self.fallback else None
        return {"primary": primary, "fallback": fallback, **self.stats}


def make_voice(cfg: Any, model_dir: Path, voice_override: str | None = None,
               on_fallback: Callable[[str, str], None] | None = None) -> VoiceChain | None:
    """Build JARVIS's voice from the [voice] config (the single place it is chosen)."""
    engine = cfg.tts_engine
    if engine == "none":
        return None
    system_voice = cfg.tts_fallback_voice
    if engine in ("system", "pyttsx3"):  # "pyttsx3": the old name for the OS voice
        legacy = cfg.tts_voice if cfg.tts_voice and not re.fullmatch(r"[a-z]{2}_[a-z]+", cfg.tts_voice) else None
        return VoiceChain(None, make_tts("system", voice_override or system_voice or legacy, cfg.tts_rate))
    from jarvis.voice.kokoro import KokoroTTS  # noqa: PLC0415

    primary = KokoroTTS(model_dir, voice=voice_override or cfg.tts_voice, speed=cfg.tts_speed,
                        boost_priority=cfg.tts_boost_priority, init_wait_s=cfg.tts_init_wait_s)
    fallback = make_tts("system", system_voice, cfg.tts_rate) if cfg.tts_fallback == "system" else None
    return VoiceChain(primary, fallback, on_fallback=on_fallback)

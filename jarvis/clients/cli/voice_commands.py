"""`jarvis voice ...` commands."""

from __future__ import annotations

import argparse
import json

from jarvis.clients.cli import extra_commands
from jarvis.clients.cli.client import JarvisClient


def _install(args: argparse.Namespace) -> int:
    """Put the Kokoro voice model files in place (verified), then tell a running JARVIS to load them."""
    from pathlib import Path  # noqa: PLC0415

    from jarvis.config.paths import JarvisPaths  # noqa: PLC0415
    from jarvis.config.settings import load_config  # noqa: PLC0415
    from jarvis.voice.kokoro import ASSETS, install_assets  # noqa: PLC0415

    paths = JarvisPaths()
    cfg = load_config(paths).voice
    target = Path(cfg.tts_model_dir).expanduser() if cfg.tts_model_dir else paths.models / "tts" / "kokoro"
    total = sum(a.size for a in ASSETS) / 1e6
    print(f"Kokoro voice model -> {target} ({total:.0f} MB" + (f", from {args.source})" if args.source else ", downloading)"))
    try:
        install_assets(target, Path(args.source) if args.source else None, progress=lambda m: print(f"  {m}"))
    except Exception as e:  # noqa: BLE001
        print(f"Install failed: {e}")
        return 1
    try:
        st = JarvisClient(timeout=30).post("/voice/tts/reload")
        print(f"JARVIS is loading the voice: speaking with {st['speaking_with']}.")
    except Exception:  # noqa: BLE001 - JARVIS not running: it loads the voice when it starts
        print("JARVIS will use the Kokoro voice the next time it starts.")
    return 0


def cmd_voice(args: argparse.Namespace) -> int:
    a = args.action
    if a == "install":
        return _install(args)
    client = JarvisClient(timeout=180)
    if a == "status":
        st = client.get("/voice/status")
        if args.json:
            print(json.dumps(st, indent=2))
            return 0
        p = st["pipeline"]
        print(f"pipeline     {'on' if p['running'] else 'off'} ({p['state']})" + (f" - {p['error']}" if p["error"] else ""))
        for key, label in (("stt", "recognition"), ("wake", "wake word")):
            item = st[key]
            detail = item.get("model") or item.get("word") or ""
            print(f"{label:<12} {'ready' if item['available'] else 'unavailable'}  {item.get('engine') or ''} {detail}"
                  + ("" if item["available"] else f" ({item['reason']})"))
        t = st["tts"]
        print(f"voice        {t.get('engine') or 'none'} {t.get('voice') or ''} - speaking with {t.get('speaking_with') or 'nothing'}"
              + (f" ({t['reason']})" if t.get("reason") else ""))
        d = t.get("details") or {}
        if d.get("spoken_by"):
            print("spoken by    " + ", ".join(f"{k}: {v} chunk(s)" for k, v in d["spoken_by"].items())
                  + (f"; {d['fallbacks']} fallback(s)" if d.get("fallbacks") else ""))
        mic = st["microphone"]
        print(f"microphone   {mic.get('name') if mic['available'] else 'none: ' + mic.get('reason', '')}")
        sig = p.get("signal") or {}
        if p["running"]:
            print(f"mic signal   {sig.get('state')}  ({sig.get('frames', 0)} frames, level {sig.get('level_db', -120)} dBFS)")
            print(f"listeners    {p.get('listeners', '?')} voice listener thread(s)")
        if p["running"]:
            barge = p.get("barge_in", "off")
            print("interrupting " + {"wake_word": "say the wake word while JARVIS speaks",
                                     "voice": "speak over JARVIS (headset mode)"}.get(barge, "off")
                  + (f"; follow-up window {p['followup_window_s']:.0f} s" if p.get("followup_window_s") else ""))
        if p.get("stats"):
            s = p["stats"]
            print(f"session      {s['wakes']} wake word(s), {s['utterances']} request(s), {s['interruptions']} interruption(s), "
                  f"{s.get('ignored', 0)} ignored" + (f", last heard: \"{s['last_transcript']}\"" if s.get("last_transcript") else ""))
    elif a == "start":
        st = client.post("/voice/start")
        print(f"Listening for \"{st['wake']['word'].replace('_', ' ')}\" on {st['microphone'].get('name')}. "
              "Audio stays on this computer. Stop with `jarvis voice stop`.")
    elif a == "stop":
        client.post("/voice/stop")
        print("Voice is off; the microphone is closed.")
    elif a == "say":
        out = client.post("/voice/say", {"text": " ".join(args.text)})
        if out.get("interrupted"):
            print(f"interrupted (speaking with {out.get('provider') or 'the voice'})")
        else:
            print(f"spoke {out['spoken_s']} s with {out.get('provider') or 'the voice'}")
    elif a == "interrupt":
        client.post("/voice/interrupt")
    return 0


def _register(sub: argparse._SubParsersAction) -> None:
    s = sub.add_parser("voice", help="voice: wake word, listening, speaking; `install` gets the Kokoro voice model")
    s.add_argument("action", choices=["status", "start", "stop", "say", "interrupt", "install"])
    s.add_argument("text", nargs="*")
    s.add_argument("--json", action="store_true")
    s.add_argument("--from", dest="source", help="install: take the model files from this folder instead of downloading")
    s.set_defaults(func=cmd_voice)


extra_commands.add(_register)

# JARVIS

A personal AI assistant that runs on your own computer. It talks to local models
through [Ollama](https://ollama.com) by default, has an optional voice (wake word,
speech-to-text, a local neural text-to-speech voice), a set of tools (files,
shell, web, screen, browser) behind a permission system, a long-term memory, and
a small web interface with an animated "presence" face.

Everything JARVIS stores lives in one folder on your machine. Nothing is sent to
the cloud unless you add a cloud model provider yourself.

## Requirements

- **Python 3.11 or newer**
- **[Ollama](https://ollama.com)** running locally, with at least one model pulled
  (the defaults expect `qwen3:4b-instruct`, `qwen2.5:7b`, `llama3.2:3b`; any of
  them is enough to start, the rest fall back).
- Windows, macOS or Linux. About 2 GB of free RAM for a small model, more for
  the voice.
- Optional: a microphone and speakers for the voice, Node.js for the web-client
  syntax check script.

## Setup in 5 steps

1. **Clone and create a virtual environment**

   ```bash
   git clone <this repository URL>
   cd JARVIS
   python -m venv .venv
   ```

   Activate it: `.venv\Scripts\activate` on Windows, `source .venv/bin/activate`
   on macOS/Linux.

2. **Install JARVIS**

   ```bash
   pip install -e ".[dev]"
   ```

   For the voice (wake word, speech recognition, Kokoro voice) install the
   voice extra too: `pip install -e ".[voice]"`. Other extras: `vision`,
   `control`, `browser`, `desktop`, `tls`, or `all`.

3. **Pull a model and check the installation**

   ```bash
   ollama pull qwen3:4b-instruct
   jarvis doctor
   ```

   `jarvis doctor` tells you where JARVIS keeps its data (its *home* folder) and
   what is missing.

4. **Start the core**

   ```bash
   jarvis start
   ```

   The server listens on `http://127.0.0.1:7777`. Open that address in a browser
   and choose **Set up this device**, or run `jarvis open` from another terminal.
   The first browser becomes the owner device; later devices need a pairing code
   (`jarvis devices`).

5. **Talk to it**

   ```bash
   jarvis chat
   jarvis ask "what can you do?"
   ```

   Optional voice: `jarvis voice install` downloads the Kokoro model files, then
   set `enabled = true` under `[voice]` in your config (see below) and restart.

## Configuration and where your keys go

JARVIS runs with no configuration file. To change anything, copy
[`config.example.toml`](config.example.toml) to `<JARVIS home>/config.toml`
(the home folder is shown by `jarvis doctor`; by default `~/.jarvis` on Windows,
`~/Library/Application Support/JARVIS` on macOS, `~/.local/share/jarvis` on
Linux). Any setting can also be given as an environment variable
`JARVIS__SECTION__KEY`; see [`.env.example`](.env.example).

**API keys and tokens are never written into config files or the repository.**
Store them in your operating system's credential store with

```bash
jarvis secret set anthropic_api_key
```

and refer to them by name in the config (`api_key_secret = "anthropic_api_key"`).
If you cannot use a credential store, set `JARVIS_SECRET_<NAME>` in the
environment or `JARVIS_SECRETS_BACKEND=file`. Every secret JARVIS reads is
registered with the log redactor, so it cannot appear in logs or events.

## Useful commands

| Command | What it does |
| --- | --- |
| `jarvis start` | start the core server (`--open` also opens the web UI) |
| `jarvis open` | open the web interface and pair this browser |
| `jarvis status`, `jarvis doctor` | health and diagnostics |
| `jarvis chat`, `jarvis ask "..."` | talk from the terminal |
| `jarvis voice install` / `status` / `say "..."` | the voice |
| `jarvis secret set <name>` | store an API key or token |
| `jarvis devices`, `jarvis permissions` | devices and what tools may do |
| `jarvis memory`, `jarvis logs`, `jarvis identity` | memory, event log, personality |

## Development

```bash
pip install -e ".[dev]"
pytest
```

Tests that need a running Ollama server, the voice extras or the Kokoro model
skip themselves when those are not available. The web-client tests use
Playwright (`pip install -e ".[browser]"` then `playwright install chromium`).
`python scripts/check_web.py` syntax-checks the web client's JavaScript with
Node.

Layout:

- `jarvis/core` – runtime, agent loop, memory, permissions, auth, events
- `jarvis/models` – model providers (Ollama, Anthropic, OpenAI-compatible) and routing
- `jarvis/capabilities` – tools (files, shell, web, vision, browser, computer control)
- `jarvis/voice` – wake word, VAD, speech-to-text, text-to-speech, voice pipeline
- `jarvis/api` – FastAPI server; `jarvis/clients` – CLI, web UI, desktop wrapper
- `jarvis/platforms` – OS adapters; `tests/` – test suite

## Licence

MIT, see [LICENSE](LICENSE). The optional Kokoro voice pulls in `phonemizer`
and `espeakng-loader`, which are GPL-3.0: fine for personal use, something to
keep in mind if you redistribute a build that includes them.

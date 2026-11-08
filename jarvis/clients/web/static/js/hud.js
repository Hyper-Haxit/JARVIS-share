// Compact view: the same presence state machine as the main interface, fed by
// the same live events. Presentation only.

import { presence as engine } from "./presence.js";
import { api } from "./api.js";
import { EventStream } from "./events.js";
import { PRESENCE, Store } from "./state.js";

const presence = document.getElementById("presence");
const line = document.getElementById("line");
const store = new Store();
const runs = new Map();
let activity = null;
let pending = new Set();

const TEXT = {
  [PRESENCE.OFFLINE]: "Offline", [PRESENCE.ERROR]: "Something went wrong", [PRESENCE.WAITING]: "Waiting for your OK",
  [PRESENCE.LISTENING]: "Listening", [PRESENCE.SPEAKING]: "", [PRESENCE.THINKING]: "Thinking…", [PRESENCE.IDLE]: "",
};

function render() {
  engine.syncFacts(store.facts);
  line.textContent = store.presence === PRESENCE.THINKING && activity ? activity : TEXT[store.presence] ?? "";
}
store.addEventListener("change", render);

if (!api.paired) {
  line.textContent = "Open JARVIS in this browser first to connect it.";
} else {
  const events = new EventStream();
  events.addEventListener("state", (e) => store.update({ link: e.detail }));
  events.addEventListener("event", ({ detail: evt }) => {
    const d = evt.data || {};
    engine.dispatch(evt.type, d);
    switch (evt.type) {
      case "AGENT_STATE":
        if (d.state === "idle" || d.state === "error") runs.delete(d.conversation_id); else runs.set(d.conversation_id, d.state);
        activity = d.state === "acting" && d.tools?.length ? `Running ${d.tools.join(", ")}…` : null;
        engine.setActivity(d.state === "acting" ? d.tools?.[0] : null);
        store.update({ agentBusy: runs.size > 0, streaming: false, error: d.state === "error" ? d.error || "error" : null });
        render();
        break;
      case "MODEL_RESPONSE_DELTA": store.update({ streaming: true }); break;
      case "MIC_STATE": store.update({ voiceListening: d.state === "listening" || d.state === "recording" }); break;
      case "AUDIO_LEVEL": engine.level(d.source === "speaker" ? "speaker" : "mic", d.level || 0); break;
      case "SPEECH_STARTED": store.update({ audioPlaying: true }); break;
      case "SPEECH_ENDED": case "SPEECH_INTERRUPTED": store.update({ audioPlaying: false }); break;
      case "PERMISSION_REQUESTED": pending.add(d.approval_id); store.update({ pendingApprovals: pending.size }); break;
      case "PERMISSION_GRANTED": case "PERMISSION_DENIED": pending.delete(d.approval_id); store.update({ pendingApprovals: pending.size }); break;
      default:
    }
  });
  api.health().then((h) => { store.update({ coreReachable: Boolean(h) }); events.connect(); });
  setInterval(async () => store.update({ coreReachable: Boolean(await api.health()) }), 5000);
}
render();

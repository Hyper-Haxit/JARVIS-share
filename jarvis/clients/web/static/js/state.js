// Presence state machine.
//
// Presence is *derived* from facts reported by the core and the connection,
// in strict priority order. Nothing here is driven by timers.
//
//   offline    core unreachable or the live event link is down
//   error      the last request ended in an error (cleared by the next user action)
//   waiting    an action is waiting for the user's approval
//   listening  a microphone is capturing (this browser or the core's voice pipeline)
//   speaking   response text is streaming, or speech audio is playing
//   thinking   the agent is working (model or tools)
//   idle       otherwise

export const PRESENCE = Object.freeze({
  OFFLINE: "offline", ERROR: "error", WAITING: "waiting", LISTENING: "listening",
  SPEAKING: "speaking", THINKING: "thinking", IDLE: "idle",
});

export function derivePresence(f) {
  if (!f.coreReachable || f.link !== "connected") return PRESENCE.OFFLINE;
  if (f.error) return PRESENCE.ERROR;
  if (f.pendingApprovals > 0) return PRESENCE.WAITING;
  if (f.micCapturing || f.voiceListening) return PRESENCE.LISTENING;
  if (f.audioPlaying || f.streaming) return PRESENCE.SPEAKING;
  if (f.agentBusy) return PRESENCE.THINKING;
  return PRESENCE.IDLE;
}

export class Store extends EventTarget {
  constructor() {
    super();
    this.facts = {
      coreReachable: false, link: "connecting", error: null, pendingApprovals: 0, micCapturing: false,
      voiceListening: false, audioPlaying: false, streaming: false, agentBusy: false,
    };
    this.presence = PRESENCE.OFFLINE;
  }

  update(patch) {
    let dirty = false;
    for (const [k, v] of Object.entries(patch)) if (this.facts[k] !== v) { this.facts[k] = v; dirty = true; }
    if (!dirty) return;
    const next = derivePresence(this.facts);
    const changed = next !== this.presence;
    this.presence = next;
    this.dispatchEvent(new CustomEvent("change", { detail: { presence: next, changed, facts: this.facts } }));
  }
}

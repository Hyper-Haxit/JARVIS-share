// Interpreting what happens into presence events.
//
// Everything the presence reacts to arrives as one small vocabulary of
// events: { type: "family.verb", source?, id?, ... }. Core events (as delivered
// over the event stream) are translated here; the interface emits its own
// directly (user.submitted, ui.panel.opened, ...).
//
// Work of any kind shares one shape, so new sources need no presence code:
//
//     <family>.started | .progress | .succeeded | .failed | .cancelled
//
// Tools and background tasks use it today. A coding agent can later emit
// test.failed, build.succeeded, command.started, search.completed or
// file.edit.started and JARVIS will react as to any other work. Events that
// fit nothing are ignored: JARVIS only knows what it is actually told.

const VERB = { started: "started", began: "started", progress: "progress", succeeded: "succeeded", completed: "succeeded",
  passed: "succeeded", finished: "succeeded", done: "succeeded", failed: "failed", errored: "failed", cancelled: "cancelled",
  canceled: "cancelled", retried: "retried", retrying: "retried" };
// Things that happened nearby that JARVIS may notice (rarely), without it being work.
const NOTICE_VERBS = new Set(["opened", "changed", "detected", "appeared", "created"]);
// Families that are not work.
const NOT_WORK = new Set(["voice", "tts", "agent", "user", "ui", "connection", "permission"]);

/** Core event type -> presence event (or null: nothing the presence should react to). */
const CORE = {
  WAKE_WORD_DETECTED: () => ({ type: "voice.wake" }),
  USER_SPEECH_STARTED: () => ({ type: "voice.user.started" }),
  USER_SPEECH_ENDED: () => ({ type: "voice.user.ended" }),
  TRANSCRIPT: (d) => (d.ignored || d.reason || d.error ? { type: "voice.unclear" } : d.text ? { type: "voice.heard", text: d.text } : null),
  SPEECH_STARTED: () => ({ type: "tts.started" }),
  SPEECH_ENDED: () => ({ type: "tts.finished" }),
  SPEECH_INTERRUPTED: () => ({ type: "voice.interrupted" }),
  AGENT_STATE: (d) => (d.state === "thinking" ? { type: "agent.thinking" } : d.state === "error" ? { type: "agent.error" } : null),
  MODEL_RESPONSE_STARTED: () => ({ type: "agent.output.started" }),
  TOOL_STARTED: (d) => ({ type: "tool.started", id: `tool:${d.tool}`, name: d.tool }),
  TOOL_COMPLETED: (d) => ({ type: d.ok === false ? "tool.failed" : "tool.succeeded", id: `tool:${d.tool}`, name: d.tool }),
  TOOL_FAILED: (d) => ({ type: "tool.failed", id: `tool:${d.tool}`, name: d.tool }),
  TASK_STARTED: (d) => ({ type: "task.started", id: `task:${d.task_id}` }),
  TASK_PROGRESS: (d) => ({ type: "task.progress", id: `task:${d.task_id}`, progress: d.progress, phase: d.phase }),
  TASK_COMPLETED: (d) => ({ type: d.achieved === false ? "task.failed" : "task.succeeded", id: `task:${d.task_id}` }),
  TASK_FAILED: (d) => ({ type: "task.failed", id: `task:${d.task_id}` }),
  TASK_CANCELLED: (d) => ({ type: "task.cancelled", id: `task:${d.task_id}` }),
  PERMISSION_REQUESTED: () => ({ type: "permission.requested" }),
};

export function fromCore(type, data = {}) {
  const f = CORE[type];
  return f ? f(data || {}) : null;
}

/**
 * What an event means: { kind: "work" | "notice" | "own", family, verb }.
 * "own" events (voice, agent, user, ui, ...) are handled one by one by the engine.
 */
export function classify(type) {
  const dot = type.lastIndexOf(".");
  if (dot <= 0) return null;
  const family = type.slice(0, dot), raw = type.slice(dot + 1);
  const root = family.split(".")[0];
  if (NOT_WORK.has(root)) return { kind: "own", family, verb: raw };
  if (VERB[raw]) return { kind: "work", family, verb: VERB[raw] };
  if (NOTICE_VERBS.has(raw)) return { kind: "notice", family, verb: raw };
  return null;
}

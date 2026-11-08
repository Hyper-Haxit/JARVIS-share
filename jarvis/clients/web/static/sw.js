// Service worker: caches the interface shell so the PWA opens instantly and can
// show its OFFLINE state when the core is unreachable. API calls are never cached.
const CACHE = "jarvis-shell-v10";
const SHELL = [
  "/",
  "/hud",
  "/static/js/theme.js",
  "/static/css/tokens.css",
  "/static/css/presence.css",
  "/static/css/app.css",
  "/static/css/overlays.css",
  "/static/css/hud.css",
  "/static/js/app.js",
  "/static/js/hud.js",
  "/static/js/api.js",
  "/static/js/events.js",
  "/static/js/state.js",
  "/static/js/presence.js",
  "/static/js/presence/engine.js",
  "/static/js/presence/attention.js",
  "/static/js/presence/emotion.js",
  "/static/js/presence/audio.js",
  "/static/js/presence/expression.js",
  "/static/js/presence/random.js",
  "/static/js/presence/interpret.js",
  "/static/js/presence/reactions.js",
  "/static/js/ui.js",
  "/static/js/markdown.js",
  "/static/js/conversation.js",
  "/static/js/composer.js",
  "/static/js/requests.js",
  "/static/js/tasks.js",
  "/static/js/sheets.js",
  "/static/js/sidebar.js",
  "/static/js/palette.js",
  "/static/js/voice.js",
  "/static/js/speech-chunks.js",
  "/static/icon.svg",
];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))).then(() => self.clients.claim()));
});

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.origin !== location.origin || url.pathname.startsWith("/api/")) return;
  // Network first (always fresh when the core is up), cache as fallback.
  e.respondWith(
    fetch(e.request).then((res) => {
      if (res.ok) { const copy = res.clone(); caches.open(CACHE).then((c) => c.put(e.request, copy)); }
      return res;
    }).catch(() => caches.match(e.request).then((r) => r || caches.match("/"))),
  );
});

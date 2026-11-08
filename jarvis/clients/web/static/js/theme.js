// Applies the saved theme before first paint (classic script, not a module).
(function () {
  var pref = "system";
  try { pref = (JSON.parse(localStorage.getItem("jarvis.prefs") || "{}").theme) || "system"; } catch (e) { /* storage blocked */ }
  var dark = pref === "dark" || (pref === "system" && window.matchMedia("(prefers-color-scheme: dark)").matches);
  document.documentElement.setAttribute("data-theme", dark ? "dark" : "light");
})();

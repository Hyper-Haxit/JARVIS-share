// Minimal, safe Markdown renderer. Builds DOM nodes directly; model output is
// never inserted as HTML, so it cannot inject markup or scripts.

function inline(text, parent) {
  // tokens: `code`, **bold**, *em*, [label](http...)
  const re = /(`[^`]+`)|(\*\*[^*]+\*\*)|(\*[^*\s][^*]*\*)|(\[[^\]]+\]\((https?:\/\/[^\s)]+)\))/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) parent.appendChild(document.createTextNode(text.slice(last, m.index)));
    if (m[1]) { const c = document.createElement("code"); c.textContent = m[1].slice(1, -1); parent.appendChild(c); }
    else if (m[2]) { const b = document.createElement("strong"); b.textContent = m[2].slice(2, -2); parent.appendChild(b); }
    else if (m[3]) { const e = document.createElement("em"); e.textContent = m[3].slice(1, -1); parent.appendChild(e); }
    else if (m[4]) {
      const a = document.createElement("a");
      a.textContent = m[4].slice(1, m[4].indexOf("]("));
      a.href = m[5]; a.target = "_blank"; a.rel = "noopener noreferrer";
      parent.appendChild(a);
    }
    last = re.lastIndex;
  }
  if (last < text.length) parent.appendChild(document.createTextNode(text.slice(last)));
}

export function renderMarkdown(src, target) {
  target.textContent = "";
  const lines = src.replace(/\r\n/g, "\n").split("\n");
  let i = 0, para = [], list = null;
  const flushPara = () => {
    if (!para.length) return;
    const p = document.createElement("p");
    para.forEach((l, idx) => { if (idx) p.appendChild(document.createElement("br")); inline(l, p); });
    target.appendChild(p);
    para = [];
  };
  const flushList = () => { if (list) { target.appendChild(list); list = null; } };
  while (i < lines.length) {
    const line = lines[i];
    const fence = line.match(/^```\s*([\w+-]*)/);
    if (fence) {
      flushPara(); flushList();
      const pre = document.createElement("pre");
      const code = document.createElement("code");
      const body = [];
      i++;
      while (i < lines.length && !lines[i].startsWith("```")) body.push(lines[i++]);
      code.textContent = body.join("\n");
      if (fence[1]) code.dataset.lang = fence[1];
      pre.appendChild(code);
      target.appendChild(pre);
      i++;
      continue;
    }
    const h = line.match(/^(#{1,3})\s+(.*)/);
    const ul = line.match(/^\s*[-*•]\s+(.*)/);
    const ol = line.match(/^\s*\d+[.)]\s+(.*)/);
    if (h) {
      flushPara(); flushList();
      const el = document.createElement(`h${h[1].length}`);
      inline(h[2], el);
      target.appendChild(el);
    } else if (ul || ol) {
      flushPara();
      const tag = ul ? "UL" : "OL";
      if (!list || list.tagName !== tag) { flushList(); list = document.createElement(tag.toLowerCase()); }
      const li = document.createElement("li");
      inline((ul || ol)[1], li);
      list.appendChild(li);
    } else if (!line.trim()) {
      flushPara(); flushList();
    } else {
      flushList();
      para.push(line);
    }
    i++;
  }
  flushPara(); flushList();
}

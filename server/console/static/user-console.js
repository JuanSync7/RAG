// src/dom.ts
var byId = (id) => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Missing required element #${id}`);
  return el;
};
function escHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;").replace(/\//g, "&#x2F;");
}
function fmtTime(ms) {
  return new Date(ms).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}
function fmtRelative(ms) {
  const now = Date.now();
  const diff = now - ms;
  if (diff < 864e5) return "Today";
  if (diff < 1728e5) return "Yesterday";
  return new Date(ms).toLocaleDateString([], { month: "short", day: "numeric" });
}

// src/refs.ts
var _refs = {};
var refs = _refs;
function populateRefs() {
  _refs.sidebar = byId("sidebar");
  _refs.backdrop = byId("sidebarBackdrop");
  _refs.settingsOverlay = byId("settingsOverlay");
  _refs.settingsPanel = byId("settingsPanel");
  _refs.thread = byId("thread");
  _refs.fab = byId("scrollFab");
  _refs.dropdown = byId("slashDropdown");
  _refs.ta = byId("inputArea");
  _refs.attachPopover = byId("attachPopover");
  _refs.webInputPanel = byId("webInputPanel");
  _refs.kbPanel = byId("kbPanel");
  _refs.cmdPicker = byId("cmdPicker");
  _refs.attachBtn = byId("attachBtn");
  _refs.cmdBtn = byId("cmdBtn");
}

// src/state.ts
var state = {
  activeConversationId: localStorage.getItem("nc_active_conv") || null,
  isStreaming: false,
  dynamicCmds: [],
  allSlashItems: [],
  slashSelIdx: 0,
  allPickerItems: [],
  pickerIdx: 0,
  attachments: [],
  userScrolledUp: false,
  streamAbortCtrl: null,
  pendingUserGroup: null,
  pendingAssistantGroup: null,
  pendingQueryText: "",
  convMenuTargetId: null,
  convMenuTargetTitle: "",
  renameTargetId: null
};
function setActiveConversation(id) {
  state.activeConversationId = id;
  if (id) localStorage.setItem("nc_active_conv", id);
  else localStorage.removeItem("nc_active_conv");
  document.dispatchEvent(new CustomEvent("conversation-changed", { detail: { id } }));
}

// src/toast.ts
function showToast(msg) {
  const t = byId("toast");
  t.textContent = msg;
  t.classList.add("show");
  setTimeout(() => t.classList.remove("show"), 2200);
}
function copyMsg(btn, text) {
  navigator.clipboard.writeText(text).then(() => {
    btn.classList.add("copied");
    btn.textContent = "\u2713 Copied";
    setTimeout(() => {
      btn.classList.remove("copied");
      btn.innerHTML = "&#128203; Copy";
    }, 2e3);
  });
  showToast("Copied to clipboard");
}
function copyBubble(btn, id) {
  copyMsg(btn, document.getElementById(id)?.innerText || "");
}
function initToast() {
  window["copyMsg"] = copyMsg;
  window["copyBubble"] = copyBubble;
  window["showToast"] = showToast;
  refs.thread.addEventListener("click", (e) => {
    const btn = e.target.closest(".copy-code-btn");
    if (!btn) return;
    const codeDiv = btn.closest(".code-block-wrap")?.querySelector(".code-block");
    if (codeDiv) {
      navigator.clipboard.writeText(codeDiv.textContent ?? "");
      showToast("Code copied");
    }
  });
}

// src/markdown.ts
import { marked } from "marked";
import DOMPurify from "dompurify";
marked.use({
  gfm: true,
  breaks: false,
  renderer: {
    code({ text, lang }) {
      const langLabel = escHtml(lang ?? "code");
      const escaped = text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
      return [
        `<div class="code-block-wrap">`,
        `<div class="code-block-header">`,
        `<span>${langLabel}</span>`,
        `<button class="copy-code-btn">&#128203; Copy</button>`,
        `</div>`,
        `<div class="code-block">${escaped}</div>`,
        `</div>`
      ].join("");
    }
  }
});
function isListMarker(word) {
  if (!word.endsWith(".")) return false;
  const n = Number(word.slice(0, -1));
  return Number.isInteger(n) && n > 0;
}
function splitInlineList(line) {
  const words = line.split(" ");
  if (words.length < 3) return line;
  const lineStartsWithBullet = words[0] === "-" || words[0] === "*";
  const subLines = [];
  let current = [];
  for (let i = 0; i < words.length; i++) {
    const w = words[i];
    const isBulletCont = lineStartsWithBullet && i > 0 && (w === "-" || w === "*");
    const isNumberedCont = i > 0 && isListMarker(w);
    if (current.length > 0 && (isBulletCont || isNumberedCont)) {
      subLines.push(current.join(" "));
      current = [w];
    } else {
      current.push(w);
    }
  }
  if (current.length) subLines.push(current.join(" "));
  return subLines.length > 1 ? subLines.join("\n") : line;
}
var PIPE_ROW_RE = /^\s*\|.*\|\s*$/;
var PIPE_SEP_RE = /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$/;
function countPipeColumns(line) {
  const trimmed = line.trim().replace(/^\|/, "").replace(/\|$/, "");
  return trimmed.split("|").length;
}
function fixOrphanPipeTables(text) {
  const lines = text.split("\n");
  const out = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (PIPE_ROW_RE.test(line)) {
      let j = i;
      while (j < lines.length && PIPE_ROW_RE.test(lines[j])) j++;
      const block = lines.slice(i, j);
      const hasSep = block.length >= 2 && PIPE_SEP_RE.test(block[1]);
      if (!hasSep && block.length >= 2) {
        const cols = countPipeColumns(block[0]);
        const sep = "|" + " --- |".repeat(cols);
        out.push(block[0]);
        out.push(sep);
        for (let k = 1; k < block.length; k++) out.push(block[k]);
      } else {
        out.push(...block);
      }
      i = j;
      continue;
    }
    out.push(line);
    i++;
  }
  return out.join("\n");
}
function normalizeMarkdown(raw) {
  const segments = raw.split(/(```[\s\S]*?```)/);
  return segments.map((seg, i) => {
    if (i % 2 === 1) return seg;
    const listFixed = seg.split("\n").map(splitInlineList).join("\n");
    return fixOrphanPipeTables(listFixed);
  }).join("");
}
function parseMarkdown(raw) {
  return DOMPurify.sanitize(marked.parse(normalizeMarkdown(raw)));
}
function humanizeKey(key) {
  return key.replace(/[_-]+/g, " ").replace(/\s+/g, " ").trim().replace(/\b\w/g, (c) => c.toUpperCase());
}
function isPrimitive(v) {
  return v == null || typeof v === "string" || typeof v === "number" || typeof v === "boolean";
}
function formatPrimitive(v) {
  if (v == null) return "_\u2014_";
  if (typeof v === "string") return v.trim() || "_(empty)_";
  return String(v);
}
function formatValue(value, depth) {
  if (depth > 4) return "_\u2026_";
  if (isPrimitive(value)) return formatPrimitive(value);
  if (Array.isArray(value)) {
    if (!value.length) return "_(empty list)_";
    if (value.every(isPrimitive)) {
      return "\n" + value.map((v) => `- ${formatPrimitive(v)}`).join("\n");
    }
    return "\n" + value.map((v) => {
      const inner = formatValue(v, depth + 1).replace(/\n/g, "\n  ");
      return `- ${inner.startsWith("\n") ? inner.trimStart() : inner}`;
    }).join("\n");
  }
  if (typeof value === "object") {
    const entries = Object.entries(value);
    if (!entries.length) return "_(empty)_";
    const lines = entries.map(([k, v]) => {
      const label = `**${humanizeKey(k)}**`;
      if (isPrimitive(v)) return `- ${label}: ${formatPrimitive(v)}`;
      const inner = formatValue(v, depth + 1).replace(/\n/g, "\n  ");
      return `- ${label}:${inner.startsWith("\n") ? inner : "\n  " + inner}`;
    });
    return (depth === 0 ? "" : "\n") + lines.join("\n");
  }
  return String(value);
}
function formatApiPayload(value) {
  if (value == null) return "";
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  return formatValue(value, 0);
}

// src/api.ts
function getSettings() {
  const raw = localStorage.getItem("nc_settings");
  return raw ? JSON.parse(raw) : {};
}
function authHeaders() {
  return { "Content-Type": "application/json" };
}
function apiBase() {
  return "";
}
async function api(method, path, body) {
  const url = apiBase() + path;
  const opts = { method, headers: authHeaders() };
  if (body !== void 0) opts.body = JSON.stringify(body);
  const res = await fetch(url, opts);
  const json = await res.json();
  if (!res.ok || !json.ok) {
    throw new Error(json.error?.message || `HTTP ${res.status}`);
  }
  return json.data;
}

// src/chatMode.ts
var STORAGE_KEY = "rw_chat_mode";
var SUBMODE_STORAGE_KEY = "rw_retrieval_submode";
var RAIL_COLLAPSED_KEY = "rw_chat_rail_collapsed";
var TOPK_STORAGE_KEY = "rw_sources_top_k";
var DEFAULT_TOPK = 5;
var _topK = (() => {
  const raw = parseInt(localStorage.getItem(TOPK_STORAGE_KEY) || "", 10);
  return Number.isFinite(raw) && raw > 0 ? Math.min(50, raw) : DEFAULT_TOPK;
})();
function getSourcesTopK() {
  return _topK;
}
function initTopKInput() {
  const input = document.getElementById("chatTopkInput");
  if (!input) return;
  input.value = String(_topK);
  input.addEventListener("change", () => {
    const v = parseInt(input.value, 10);
    if (!Number.isFinite(v) || v < 1) {
      input.value = String(_topK);
      return;
    }
    _topK = Math.min(50, Math.max(1, v));
    input.value = String(_topK);
    localStorage.setItem(TOPK_STORAGE_KEY, String(_topK));
  });
}
var _mode = localStorage.getItem(STORAGE_KEY) === "sources" ? "sources" : "answer";
var _subMode = localStorage.getItem(SUBMODE_STORAGE_KEY) === "auto" ? "auto" : "hard";
function getRetrievalSubMode() {
  return _subMode;
}
function setRetrievalSubMode(sub) {
  _subMode = sub;
  localStorage.setItem(SUBMODE_STORAGE_KEY, sub);
  syncSubmodeUI();
}
function syncSubmodeUI() {
  const hardBtn = document.getElementById("chatSubmodeHard");
  const autoBtn = document.getElementById("chatSubmodeAuto");
  if (hardBtn) {
    hardBtn.classList.toggle("active", _subMode === "hard");
    hardBtn.setAttribute("aria-selected", _subMode === "hard" ? "true" : "false");
  }
  if (autoBtn) {
    autoBtn.classList.toggle("active", _subMode === "auto");
    autoBtn.setAttribute("aria-selected", _subMode === "auto" ? "true" : "false");
  }
}
var _docCache = /* @__PURE__ */ new Map();
var _ignoredDocIds = /* @__PURE__ */ new Set();
var _relevantDocIds = /* @__PURE__ */ new Set();
function getChatMode() {
  return _mode;
}
function setChatMode(mode) {
  _mode = mode;
  localStorage.setItem(STORAGE_KEY, mode);
  syncToggleUI();
  applyModeToView();
  if (mode === "sources" && state.activeConversationId) {
    void fetchAndRenderDocState(state.activeConversationId);
  }
  document.dispatchEvent(new CustomEvent("chat-mode-changed", { detail: { mode } }));
}
function syncToggleUI() {
  const answerBtn = document.getElementById("chatModeAnswer");
  const sourcesBtn = document.getElementById("chatModeSources");
  if (answerBtn) {
    answerBtn.classList.toggle("active", _mode === "answer");
    answerBtn.setAttribute("aria-selected", _mode === "answer" ? "true" : "false");
  }
  if (sourcesBtn) {
    sourcesBtn.classList.toggle("active", _mode === "sources");
    sourcesBtn.setAttribute("aria-selected", _mode === "sources" ? "true" : "false");
  }
}
function applyModeToView() {
  const view = document.getElementById("view-chat");
  if (view) view.dataset.chatMode = _mode;
}
function initChatMode() {
  const toggle = document.getElementById("chatModeToggle");
  if (toggle) {
    toggle.querySelectorAll("[data-mode]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const mode = btn.dataset.mode;
        if (mode === "answer" || mode === "sources") setChatMode(mode);
      });
    });
  }
  const subToggle = document.getElementById("chatSubmodeToggle");
  if (subToggle) {
    subToggle.querySelectorAll("[data-submode]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const sub = btn.dataset.submode;
        if (sub === "hard" || sub === "auto") setRetrievalSubMode(sub);
      });
    });
  }
  initRailCollapse();
  initTopKInput();
  syncToggleUI();
  syncSubmodeUI();
  applyModeToView();
  renderRail();
  document.addEventListener("conversation-changed", () => {
    _ignoredDocIds.clear();
    _relevantDocIds.clear();
    renderRail();
    if (_mode === "sources" && state.activeConversationId) {
      void fetchAndRenderDocState(state.activeConversationId);
    }
  });
}
function docKeyOf(ref) {
  return (ref.document_id || ref.source_key || ref.source_uri || ref.source || "").trim();
}
function docKeyFromMeta(meta) {
  if (!meta) return "";
  const pick = (k) => {
    const v = meta[k];
    return typeof v === "string" ? v : v == null ? "" : String(v);
  };
  return (pick("document_id") || pick("source_key") || pick("source_uri") || pick("source") || "").trim();
}
function nameOf(ref) {
  const raw = ref.source || ref.source_uri || ref.source_key || ref.document_id || "Unknown source";
  const stripped = String(raw).split("/").pop() || String(raw);
  return stripped;
}
function groupSources(sources) {
  const groups = /* @__PURE__ */ new Map();
  let synthCounter = 0;
  for (const ref of sources) {
    let key = docKeyOf(ref);
    if (!key) key = `__synth_${synthCounter++}`;
    const score = ref.score ?? 0;
    const text = ref.text ?? "";
    const existing = groups.get(key);
    if (existing) {
      existing.refs.push(ref);
      if (score > existing.bestScore) {
        existing.bestScore = score;
        existing.excerpt = text;
      }
    } else {
      groups.set(key, {
        docKey: key,
        name: nameOf(ref),
        bestScore: score,
        excerpt: text,
        refs: [ref],
        sourceUri: ref.source_uri ?? "",
        sourceKey: ref.source_key ?? "",
        source: ref.source ?? ""
      });
    }
  }
  return [...groups.values()].sort((a, b) => b.bestScore - a.bestScore);
}
function cacheDocs(groups) {
  for (const g of groups) {
    if (!g.docKey || g.docKey.startsWith("__synth_")) continue;
    _docCache.set(g.docKey, {
      name: g.name,
      sourceUri: g.sourceUri,
      sourceKey: g.sourceKey,
      source: g.source
    });
  }
}
function cacheDocsFromSources(sources) {
  cacheDocs(groupSources(sources));
}
function appendSourcesTurn(thread, sources) {
  const group = document.createElement("div");
  group.className = "msg-group";
  const ts = fmtTime(Date.now());
  const docs = groupSources(sources);
  cacheDocs(docs);
  const cardsHtml = docs.length ? docs.map((d) => renderCardHtml(d)).join("") : `<div class="sources-empty">No sources matched.</div>`;
  const lead = docs.length ? `Here are the documents I found:` : `I couldn't find any relevant documents.`;
  group.innerHTML = `
        <div class="msg-row assistant">
          <div class="avatar ai-av">AI</div>
          <div class="bubble-wrap">
            <div class="bubble assistant sources-lead">${escHtml(lead)}</div>
            <div class="sources-turn"><div class="sources-cards">${cardsHtml}</div></div>
            <div class="msg-meta">Sources \xB7 ${ts}</div>
          </div>
        </div>`;
  thread.appendChild(group);
  wireCardActions(group, docs);
  return group;
}
function renderCardHtml(d) {
  const score = Math.round(d.bestScore * 100);
  const chunkCount = d.refs.length;
  const synthetic = d.docKey.startsWith("__synth_");
  const viewBtn = !synthetic ? `<a href="#" class="sources-card-view" data-doc-key="${escHtml(d.docKey)}">[view]</a>` : "";
  const minimizeBtn = `<button class="sources-card-minimize" data-doc-key="${escHtml(d.docKey)}" title="Minimize" aria-expanded="true">&#9650;</button>`;
  const isRelevant = _relevantDocIds.has(d.docKey);
  const isHidden = _ignoredDocIds.has(d.docKey);
  const actionsDisabled = synthetic ? "disabled" : "";
  const relevantLabel = isRelevant ? "Relevant \u2713" : "Mark relevant";
  const hideLabel = isHidden ? "Hidden" : "Hide";
  const orderedRefs = [...d.refs].sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
  const chunksHtml = orderedRefs.map((ref, idx) => {
    const refScore = Math.round((ref.score ?? 0) * 100);
    const text = ref.text ?? "";
    const sectionLabel = ref.section ? escHtml(String(ref.section)) : "";
    const sectionHtml = sectionLabel ? `<span class="sources-chunk-section">${sectionLabel}</span>` : "";
    return `
              <div class="sources-chunk">
                <div class="sources-chunk-meta">
                  <span class="sources-chunk-idx">#${idx + 1}</span>
                  ${sectionHtml}
                  <span class="sources-chunk-score">${refScore}%</span>
                </div>
                <div class="sources-chunk-text markdown-body">${parseMarkdown(text)}</div>
              </div>`;
  }).join("");
  const chunkLabel = chunkCount > 1 ? `${chunkCount} chunks matched` : `1 chunk matched`;
  return `
      <div class="sources-card" data-doc-key="${escHtml(d.docKey)}">
        <div class="sources-card-header">
          <span class="sources-card-name" title="${escHtml(d.name)}">${escHtml(d.name)}</span>
          <span class="sources-card-score">${score}%</span>
          ${viewBtn}
          ${minimizeBtn}
        </div>
        <div class="sources-card-meta">${chunkLabel}</div>
        <div class="sources-card-chunks">${chunksHtml}</div>
        <div class="sources-card-actions">
          <button class="sources-card-action" data-action="relevant" data-doc-key="${escHtml(d.docKey)}" ${actionsDisabled || (isRelevant ? "disabled" : "")}>${relevantLabel}</button>
          <button class="sources-card-action" data-action="hide" data-doc-key="${escHtml(d.docKey)}" ${actionsDisabled || (isHidden ? "disabled" : "")}>${hideLabel}</button>
          <button class="sources-card-action" data-action="reset" data-doc-key="${escHtml(d.docKey)}" ${actionsDisabled || (!isRelevant && !isHidden ? "disabled" : "")}>Reset</button>
        </div>
      </div>`;
}
function wireCardActions(scope, docs) {
  const docMap = new Map(docs.map((d) => [d.docKey, d]));
  scope.querySelectorAll(".sources-card-view").forEach((link) => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      const key = link.dataset.docKey;
      if (!key) return;
      const doc = docMap.get(key);
      if (!doc) return;
      const top = doc.refs[0];
      void openSourceDocument({
        source: doc.source || void 0,
        source_uri: doc.sourceUri || void 0,
        source_key: doc.sourceKey || void 0,
        chunk_text: top?.text || void 0,
        original_start: top?.original_char_start,
        original_end: top?.original_char_end
      });
    });
  });
  scope.querySelectorAll(".sources-card-minimize").forEach((btn) => {
    btn.addEventListener("click", () => {
      const card = btn.closest(".sources-card");
      if (!card) return;
      const collapsed = card.classList.toggle("collapsed");
      btn.setAttribute("aria-expanded", collapsed ? "false" : "true");
      btn.setAttribute("title", collapsed ? "Expand" : "Minimize");
      btn.innerHTML = collapsed ? "&#9660;" : "&#9650;";
    });
  });
  scope.querySelectorAll(".sources-card-action").forEach((btn) => {
    btn.addEventListener("click", () => {
      const action = btn.dataset.action;
      const key = btn.dataset.docKey;
      if (!key) return;
      const doc = docMap.get(key);
      if (!doc) return;
      if (action === "relevant") void markRelevant(doc);
      else if (action === "hide") void hideDoc(doc);
      else if (action === "reset") void clearDocState(doc);
    });
  });
}
async function markRelevant(doc) {
  const cid = state.activeConversationId;
  if (!cid) {
    showToast("Select a conversation first");
    return;
  }
  _relevantDocIds.add(doc.docKey);
  _ignoredDocIds.delete(doc.docKey);
  refreshCardActions(doc.docKey);
  renderRail();
  try {
    await api(
      "DELETE",
      `/console/conversations/${encodeURIComponent(cid)}/ignore/${encodeURIComponent(doc.docKey)}`
    );
    showToast(`Marked relevant: ${doc.name}`);
  } catch (err) {
    _relevantDocIds.delete(doc.docKey);
    renderRail();
    refreshCardActions(doc.docKey);
    showToast("Failed to mark relevant: " + String(err));
  }
}
async function clearDocState(doc) {
  const cid = state.activeConversationId;
  if (!cid) {
    showToast("Select a conversation first");
    return;
  }
  const wasRelevant = _relevantDocIds.has(doc.docKey);
  const wasIgnored = _ignoredDocIds.has(doc.docKey);
  _relevantDocIds.delete(doc.docKey);
  _ignoredDocIds.delete(doc.docKey);
  refreshCardActions(doc.docKey);
  renderRail();
  try {
    await api(
      "DELETE",
      `/console/conversations/${encodeURIComponent(cid)}/doc-state/${encodeURIComponent(doc.docKey)}`
    );
    showToast(`Reset: ${doc.name}`);
  } catch (err) {
    if (wasRelevant) _relevantDocIds.add(doc.docKey);
    if (wasIgnored) _ignoredDocIds.add(doc.docKey);
    renderRail();
    refreshCardActions(doc.docKey);
    showToast("Failed to reset: " + String(err));
  }
}
async function hideDoc(doc) {
  const cid = state.activeConversationId;
  if (!cid) {
    showToast("Select a conversation first");
    return;
  }
  _ignoredDocIds.add(doc.docKey);
  _relevantDocIds.delete(doc.docKey);
  refreshCardActions(doc.docKey);
  renderRail();
  try {
    await api(
      "POST",
      `/console/conversations/${encodeURIComponent(cid)}/ignore`,
      { doc_id: doc.docKey }
    );
    showToast(`Hidden: ${doc.name}`);
  } catch (err) {
    _ignoredDocIds.delete(doc.docKey);
    renderRail();
    refreshCardActions(doc.docKey);
    showToast("Failed to hide document: " + String(err));
  }
}
function refreshCardActions(docKey) {
  const selector = `.sources-card[data-doc-key="${CSS.escape(docKey)}"], .citation-card[data-doc-key="${CSS.escape(docKey)}"]`;
  document.querySelectorAll(selector).forEach((card) => {
    const isRelevant = _relevantDocIds.has(docKey);
    const isHidden = _ignoredDocIds.has(docKey);
    const relBtn = card.querySelector('[data-action="relevant"]');
    const hideBtn = card.querySelector('[data-action="hide"]');
    const resetBtn = card.querySelector('[data-action="reset"]');
    if (relBtn) {
      relBtn.textContent = isRelevant ? "Relevant \u2713" : "Mark relevant";
      relBtn.disabled = isRelevant;
    }
    if (hideBtn) {
      hideBtn.textContent = isHidden ? "Hidden" : "Hide";
      hideBtn.disabled = isHidden;
    }
    if (resetBtn) {
      resetBtn.disabled = !isRelevant && !isHidden;
    }
  });
}
function wireCitationActions(scope) {
  scope.querySelectorAll(".citation-card[data-doc-key]").forEach((card) => {
    const docKey = card.getAttribute("data-doc-key") || "";
    if (!docKey) return;
    const docName = card.getAttribute("data-doc-name") || card.querySelector(".citation-name")?.textContent?.trim() || docKey;
    const cached = _docCache.get(docKey);
    const stub = {
      docKey,
      name: cached?.name || docName,
      bestScore: 0,
      excerpt: "",
      refs: [],
      sourceUri: cached?.sourceUri || card.getAttribute("data-source-uri") || "",
      sourceKey: cached?.sourceKey || card.getAttribute("data-source-key") || "",
      source: cached?.source || card.getAttribute("data-source") || ""
    };
    card.querySelectorAll(".citation-card-action").forEach((btn) => {
      btn.addEventListener("click", (ev) => {
        ev.stopPropagation();
        const action = btn.dataset.action;
        if (action === "relevant") void markRelevant(stub);
        else if (action === "hide") void hideDoc(stub);
        else if (action === "reset") void clearDocState(stub);
      });
    });
    refreshCardActions(docKey);
  });
}
function initRailCollapse() {
  const rail = document.getElementById("chatRail");
  const btn = document.getElementById("chatRailToggle");
  if (!rail || !btn) return;
  const collapsed = localStorage.getItem(RAIL_COLLAPSED_KEY) === "1";
  applyRailCollapsed(rail, btn, collapsed);
  btn.addEventListener("click", () => {
    const next = !rail.classList.contains("collapsed");
    applyRailCollapsed(rail, btn, next);
    localStorage.setItem(RAIL_COLLAPSED_KEY, next ? "1" : "0");
  });
}
function applyRailCollapsed(rail, btn, collapsed) {
  rail.classList.toggle("collapsed", collapsed);
  btn.setAttribute("aria-expanded", collapsed ? "false" : "true");
  btn.setAttribute("title", collapsed ? "Expand" : "Collapse");
  btn.innerHTML = collapsed ? "&#9664;" : "&#9654;";
}
function renderRail() {
  renderRailList("chatRailRelevant", "chatRailRelevantCount", _relevantDocIds, "relevant");
  renderRailList("chatRailHidden", "chatRailHiddenCount", _ignoredDocIds, "hidden");
}
function renderRailList(listId, countId, ids, kind) {
  const list = document.getElementById(listId);
  const count = document.getElementById(countId);
  if (!list || !count) return;
  count.textContent = String(ids.size);
  if (!ids.size) {
    list.innerHTML = `<div class="chat-rail-list-empty">${kind === "relevant" ? "No relevant docs yet." : "Nothing hidden."}</div>`;
    return;
  }
  let html = "";
  for (const docKey of ids) {
    const cached = _docCache.get(docKey);
    const name = escHtml(cached?.name || docKey);
    const synthetic = !cached;
    const viewBtn = !synthetic ? `<a href="#" class="chat-rail-item-view" data-doc-key="${escHtml(docKey)}">view</a>` : "";
    const actionLabel = kind === "relevant" ? "hide" : "restore";
    const actionAttr = kind === "relevant" ? "hide" : "relevant";
    html += `
          <div class="chat-rail-item" data-doc-key="${escHtml(docKey)}">
            <span class="chat-rail-item-name" title="${name}">${name}</span>
            ${viewBtn}
            <button class="chat-rail-item-action" data-action="${actionAttr}" data-doc-key="${escHtml(docKey)}">${actionLabel}</button>
            <button class="chat-rail-item-action" data-action="reset" data-doc-key="${escHtml(docKey)}" title="Remove from list">reset</button>
          </div>`;
  }
  list.innerHTML = html;
  wireRailActions(list);
}
function wireRailActions(scope) {
  scope.querySelectorAll(".chat-rail-item-view").forEach((link) => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      const key = link.dataset.docKey;
      const cached = key ? _docCache.get(key) : void 0;
      if (!cached) return;
      void openSourceDocument({
        source: cached.source || void 0,
        source_uri: cached.sourceUri || void 0,
        source_key: cached.sourceKey || void 0
      });
    });
  });
  scope.querySelectorAll(".chat-rail-item-action").forEach((btn) => {
    btn.addEventListener("click", () => {
      const key = btn.dataset.docKey;
      const action = btn.dataset.action;
      if (!key) return;
      const cached = _docCache.get(key);
      const docStub = {
        docKey: key,
        name: cached?.name || key,
        bestScore: 0,
        excerpt: "",
        refs: [],
        sourceUri: cached?.sourceUri || "",
        sourceKey: cached?.sourceKey || "",
        source: cached?.source || ""
      };
      if (action === "hide") void hideDoc(docStub);
      else if (action === "relevant") void markRelevant(docStub);
      else if (action === "reset") void clearDocState(docStub);
    });
  });
}
function applyDocState(relevant, ignored) {
  _relevantDocIds.clear();
  _ignoredDocIds.clear();
  relevant.forEach((id) => _relevantDocIds.add(id));
  ignored.forEach((id) => _ignoredDocIds.add(id));
  renderRail();
  [..._relevantDocIds, ..._ignoredDocIds].forEach(refreshCardActions);
}
async function fetchAndRenderDocState(conversationId) {
  try {
    const data = await api(
      "GET",
      `/console/conversations/${encodeURIComponent(conversationId)}/doc-state`
    );
    applyDocState(data.relevant_doc_ids ?? [], data.ignored_doc_ids ?? []);
  } catch {
  }
}
function isSourcesTurn(turn) {
  return !turn.content.trim() && !!turn.sources && turn.sources.length > 0;
}

// src/citations.ts
async function openSourceDocument(payload) {
  const url = apiBase() + "/console/source-document/view";
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify(payload)
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const contentType = (res.headers.get("Content-Type") ?? "text/html").split(";")[0].trim();
    const buf = await res.arrayBuffer();
    const blob = new Blob([buf], { type: contentType });
    const blobUrl = URL.createObjectURL(blob);
    const win = window.open(blobUrl, "_blank");
    if (!win) {
      showToast("Pop-up blocked. Allow pop-ups to view sources.");
      URL.revokeObjectURL(blobUrl);
      return;
    }
    setTimeout(() => URL.revokeObjectURL(blobUrl), 6e4);
  } catch (err) {
    showToast("Could not open source: " + String(err));
  }
}
var _viewPayloads = /* @__PURE__ */ new Map();
var _viewCounter = 0;
function buildCitationsHtml(results) {
  if (!results.length) return "";
  let html = `<div class="citation-label">&#128206; ${results.length} source${results.length > 1 ? "s" : ""} cited</div>`;
  results.forEach((r, i) => {
    const meta = r.metadata || {};
    const filenameRaw = String(meta.source ?? meta.filename ?? "Unknown source");
    const filename = escHtml(filenameRaw);
    const section = escHtml(String(meta.section ?? meta.heading ?? ""));
    const score = Math.round(r.score * 100);
    const scoreClass = score >= 80 ? "high" : score >= 50 ? "mid" : "low";
    const chunkHtml = parseMarkdown(r.text || "");
    const sourceUri = String(meta.source_uri ?? "").trim();
    const source = String(meta.source ?? "").trim();
    const sourceKey = String(meta.source_key ?? "").trim();
    const docKey = docKeyFromMeta(meta);
    const cardAttrs = docKey ? ` data-doc-key="${escHtml(docKey)}" data-doc-name="${escHtml(filenameRaw)}"` + (source ? ` data-source="${escHtml(source)}"` : "") + (sourceUri ? ` data-source-uri="${escHtml(sourceUri)}"` : "") + (sourceKey ? ` data-source-key="${escHtml(sourceKey)}"` : "") : "";
    const actionsHtml = docKey ? `<div class="citation-card-actions" onclick="event.stopPropagation()"><button class="citation-card-action" data-action="relevant">Mark relevant</button><button class="citation-card-action" data-action="hide">Hide</button><button class="citation-card-action" data-action="reset" disabled>Reset</button></div>` : "";
    let viewKey = "";
    if (sourceKey || sourceUri || source) {
      viewKey = `view-${++_viewCounter}`;
      _viewPayloads.set(viewKey, {
        source: source || void 0,
        source_uri: sourceUri || void 0,
        source_key: sourceKey || void 0,
        chunk_text: r.text || void 0,
        original_start: numOrUndef(meta.original_char_start),
        original_end: numOrUndef(meta.original_char_end),
        refactored_start: numOrUndef(meta.refactored_char_start),
        refactored_end: numOrUndef(meta.refactored_char_end),
        provenance_confidence: numOrUndef(meta.provenance_confidence)
      });
    }
    html += `
          <div class="citation-card"${cardAttrs} onclick="toggleCitation(this)">
            <div class="citation-header">
              <span class="citation-icon">&#128196;</span>
              <div class="citation-info">
                <div class="citation-filename"><span class="citation-name">${filename}</span>${viewKey ? `<a href="#" class="citation-view" onclick="event.stopPropagation();openSourceView(event,'${viewKey}')">[view]</a>` : ""}</div>
                ${section ? `<div class="citation-section">${section}</div>` : ""}
              </div>
              <div class="relevance-bar-wrap">
                <span class="relevance-pct ${scoreClass}">${score}%</span>
                <div class="relevance-bar"><div class="relevance-fill ${scoreClass}" style="width:${score}%"></div></div>
              </div>
              <span class="citation-chevron">&#8964;</span>
            </div>
            <div class="citation-body">
              <div class="citation-chunk markdown-body">${chunkHtml}</div>
              ${actionsHtml}
            </div>
          </div>`;
  });
  return html;
}
function numOrUndef(v) {
  const n = Number(v);
  return Number.isFinite(n) ? n : void 0;
}
async function openSourceView(e, viewKey) {
  e.preventDefault();
  const payload = _viewPayloads.get(viewKey);
  if (!payload) {
    showToast("Citation context lost \u2014 try re-running the query.");
    return;
  }
  await openSourceDocument(payload);
}
function toggleCitation(card) {
  card.classList.toggle("expanded");
}
function revealCitations(citationsEl) {
  citationsEl.style.display = "block";
  citationsEl.classList.remove("reveal");
  void citationsEl.offsetWidth;
  citationsEl.classList.add("reveal");
}
function initCitations() {
  window["toggleCitation"] = toggleCitation;
  window["openSourceView"] = openSourceView;
}

// src/contextWindow.ts
function fmtTokens(n) {
  if (!n) return "0";
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(n >= 1e4 ? 0 : 1) + "k";
  return String(n);
}
function fmtCost(usd) {
  if (usd <= 0) return "$0";
  if (usd < 0.01) return "$" + usd.toFixed(4);
  return "$" + usd.toFixed(3);
}
function updateContextIndicator(tb, stats) {
  const chip = byId("ctxChip");
  const used = Number(tb?.input_tokens ?? 0);
  const total = Number(tb?.context_length ?? 0);
  const reserved = Number(tb?.output_reservation ?? 0);
  const pctRaw = Number(tb?.usage_percent ?? 0);
  const pct = pctRaw > 1.5 ? pctRaw : pctRaw * 100;
  byId("ctxBarFill").style.width = Math.min(pct, 100) + "%";
  if (total > 0) {
    byId("ctxPct").textContent = `${fmtTokens(used)} / ${fmtTokens(total)}`;
  } else {
    byId("ctxPct").textContent = pct > 0 ? "~" + Math.round(pct) + "%" : "\u2014";
  }
  chip.classList.remove("warn", "crit");
  if (pct >= 85) chip.classList.add("crit");
  else if (pct >= 60) chip.classList.add("warn");
  byId("ttModel").textContent = String(tb?.model_name || "\u2014");
  byId("ttUsed").textContent = total > 0 ? `${fmtTokens(used)} / ${fmtTokens(total)} tok` : `${fmtTokens(used)} tok`;
  byId("ttReserved").textContent = reserved > 0 ? `${fmtTokens(reserved)} tok` : "\u2014";
  if (stats) {
    byId("ttPrompt").textContent = stats.promptTokens > 0 ? `${fmtTokens(stats.promptTokens)} tok` : "\u2014";
    byId("ttCompletion").textContent = stats.completionTokens > 0 ? `${fmtTokens(stats.completionTokens)} tok` : "\u2014";
    byId("ttSpeed").textContent = stats.tokensPerSecond > 0 ? `${stats.tokensPerSecond.toFixed(1)} tok/s` : "\u2014";
    const costRow = byId("ttCostRow");
    if (stats.costUsd > 0) {
      costRow.style.display = "";
      byId("ttCost").textContent = fmtCost(stats.costUsd);
    } else {
      costRow.style.display = "none";
    }
  }
  byId("ctxCompactBtn").style.display = pct >= 60 ? "block" : "none";
}
function clearLastTurnStats() {
  byId("ttPrompt").textContent = "\u2014";
  byId("ttCompletion").textContent = "\u2014";
  byId("ttSpeed").textContent = "\u2014";
  byId("ttCostRow").style.display = "none";
}
function initContextIndicator() {
  byId("ctxCompactBtn").addEventListener("click", async () => {
    if (!state.activeConversationId) return;
    try {
      await api("POST", `/console/conversations/${state.activeConversationId}/compact`);
      showToast("Conversation compacted");
      updateContextIndicator(null);
    } catch (err) {
      showToast("Compact failed: " + String(err));
    }
  });
  byId("ctxChip").addEventListener("click", () => {
    byId("ctxChip").classList.toggle("tooltip-open");
  });
}

// src/scrollFab.ts
function scrollToBottom() {
  if (!state.userScrolledUp) refs.thread.scrollTop = refs.thread.scrollHeight;
}
function initScrollFab() {
  refs.thread.addEventListener("scroll", () => {
    const atBottom = refs.thread.scrollHeight - refs.thread.scrollTop - refs.thread.clientHeight < 80;
    state.userScrolledUp = !atBottom;
    refs.fab.classList.toggle("visible", state.userScrolledUp);
  });
  refs.fab.addEventListener("click", () => {
    refs.thread.scrollTop = refs.thread.scrollHeight;
  });
}

// src/sidebar.ts
function isDesktop() {
  return window.innerWidth > 1024;
}
function showPanel(panelId) {
  document.querySelectorAll(".sidebar-panel").forEach(
    (p) => p.classList.remove("active")
  );
  if (panelId) document.getElementById(panelId)?.classList.add("active");
}
function setNavActive(el) {
  document.querySelectorAll(".sidebar-nav-item").forEach(
    (n) => n.classList.remove("active")
  );
  el.classList.add("active");
  if (!refs.sidebar.classList.contains("collapsed")) showPanel(el.dataset.panel);
  if (!isDesktop()) closeSidebar();
}
function toggleSidebarCollapse() {
  refs.sidebar.classList.toggle("collapsed");
  byId("sidebarCollapseBtn").innerHTML = refs.sidebar.classList.contains("collapsed") ? "&#8250;" : "&#8249;";
  if (refs.sidebar.classList.contains("collapsed")) {
    document.querySelectorAll(".sidebar-panel").forEach(
      (p) => p.classList.remove("active")
    );
  } else {
    const activeNav = refs.sidebar.querySelector(".sidebar-nav-item.active");
    if (activeNav) showPanel(activeNav.dataset.panel);
  }
}
function openSidebar() {
  if (isDesktop()) {
    refs.sidebar.classList.remove("collapsed");
    byId("sidebarCollapseBtn").innerHTML = "&#8249;";
    const activeNav = refs.sidebar.querySelector(".sidebar-nav-item.active");
    if (activeNav) showPanel(activeNav.dataset.panel);
  } else {
    refs.sidebar.classList.add("open");
    refs.backdrop.classList.add("active");
  }
}
function closeSidebar() {
  if (isDesktop()) {
    refs.sidebar.classList.add("collapsed");
    byId("sidebarCollapseBtn").innerHTML = "&#8250;";
  } else {
    refs.sidebar.classList.remove("open");
    refs.backdrop.classList.remove("active");
  }
}
function initSidebar() {
  byId("toggleBtn").addEventListener(
    "click",
    () => refs.sidebar.classList.contains("open") ? closeSidebar() : openSidebar()
  );
  document.querySelectorAll(".sidebar-nav-item").forEach(
    (item) => item.addEventListener("click", () => setNavActive(item))
  );
  document.getElementById("sidebarCollapseBtn")?.addEventListener("click", toggleSidebarCollapse);
  window.addEventListener("resize", () => {
    if (isDesktop()) {
      refs.sidebar.classList.remove("open");
      refs.backdrop.classList.remove("active");
    }
  });
  let touchStartX = 0;
  document.addEventListener(
    "touchstart",
    (e) => {
      touchStartX = e.touches[0].clientX;
    },
    { passive: true }
  );
  document.addEventListener(
    "touchend",
    (e) => {
      const dx = e.changedTouches[0].clientX - touchStartX;
      if (!isDesktop()) {
        if (dx > 60 && touchStartX < 30) openSidebar();
        if (dx < -60 && refs.sidebar.classList.contains("open")) closeSidebar();
      }
    },
    { passive: true }
  );
}

// src/settings.ts
var mq = window.matchMedia("(prefers-color-scheme: light)");
function applyThemeToDOM(val) {
  const resolved = val === "system" ? mq.matches ? "light" : "dark" : val;
  document.documentElement.dataset.theme = resolved;
  document.querySelectorAll(".theme-opt").forEach((el) => {
    el.classList.toggle("active", el.dataset.themeVal === val);
  });
}
function setTheme(val) {
  applyThemeToDOM(val);
  localStorage.setItem("nc_theme", val);
}
var PRESETS = {
  balanced: { searchLimit: 10, rerankTopK: 5 },
  precise: { searchLimit: 8, rerankTopK: 3 },
  broad: { searchLimit: 25, rerankTopK: 10 },
  fast: { searchLimit: 5, rerankTopK: 2 }
};
function applyPreset(name) {
  const p = PRESETS[name];
  if (!p) return;
  byId("searchLimit").value = String(p.searchLimit);
  byId("searchLimitVal").textContent = String(p.searchLimit);
  byId("rerankTopK").value = String(p.rerankTopK);
  byId("rerankVal").textContent = String(p.rerankTopK);
}
function openSettings() {
  refs.settingsOverlay.classList.add("open");
  refs.settingsPanel.classList.add("open");
  loadSettings();
}
function closeSettings() {
  refs.settingsOverlay.classList.remove("open");
  refs.settingsPanel.classList.remove("open");
}
function saveSettings() {
  const s = {
    theme: localStorage.getItem("nc_theme") || "dark",
    preset: byId("presetSelect").value,
    searchLimit: byId("searchLimit").value,
    rerankTopK: byId("rerankTopK").value,
    streaming: byId("streamingToggle").checked,
    memory_enabled: byId("memoryToggle").checked,
    citations: byId("citationsToggle").checked
  };
  localStorage.setItem("nc_settings", JSON.stringify(s));
  closeSettings();
  showToast("Settings saved");
}
function loadSettings() {
  const s = getSettings();
  const theme = localStorage.getItem("nc_theme") || s.theme || "dark";
  applyThemeToDOM(theme);
  if (s.preset) byId("presetSelect").value = s.preset;
  if (s.searchLimit) {
    byId("searchLimit").value = s.searchLimit;
    byId("searchLimitVal").textContent = s.searchLimit;
  }
  if (s.rerankTopK) {
    byId("rerankTopK").value = s.rerankTopK;
    byId("rerankVal").textContent = s.rerankTopK;
  }
  if (s.streaming !== void 0) byId("streamingToggle").checked = s.streaming;
  if (s.memory_enabled !== void 0) byId("memoryToggle").checked = s.memory_enabled;
  if (s.citations !== void 0) byId("citationsToggle").checked = s.citations;
}
function resetSettings() {
  localStorage.removeItem("nc_settings");
  localStorage.removeItem("nc_theme");
  applyThemeToDOM("dark");
  byId("presetSelect").value = "balanced";
  applyPreset("balanced");
  byId("streamingToggle").checked = true;
  byId("memoryToggle").checked = true;
  byId("citationsToggle").checked = true;
  showToast("Settings reset to defaults");
}
function initSettings() {
  mq.addEventListener("change", () => {
    if (localStorage.getItem("nc_theme") === "system") applyThemeToDOM("system");
  });
  document.querySelectorAll(".theme-opt").forEach((el) => {
    el.addEventListener("click", () => setTheme(el.dataset.themeVal || "dark"));
  });
  document.getElementById("presetSelect")?.addEventListener("change", (e) => {
    applyPreset(e.target.value);
  });
  document.getElementById("searchLimit")?.addEventListener("input", (e) => {
    byId("searchLimitVal").textContent = e.target.value;
  });
  document.getElementById("rerankTopK")?.addEventListener("input", (e) => {
    byId("rerankVal").textContent = e.target.value;
  });
  document.getElementById("settingsBtn")?.addEventListener("click", openSettings);
  document.getElementById("customizeOpenSettings")?.addEventListener("click", openSettings);
  refs.settingsOverlay.addEventListener("click", closeSettings);
  document.getElementById("settingsClose")?.addEventListener("click", closeSettings);
  document.getElementById("settingsSaveBtn")?.addEventListener("click", saveSettings);
  document.getElementById("settingsResetBtn")?.addEventListener("click", resetSettings);
  applyThemeToDOM(localStorage.getItem("nc_theme") || "dark");
}

// src/thread.ts
function setEmptyState(visible) {
  const el = document.getElementById("threadEmpty");
  if (el) el.style.display = visible ? "" : "none";
}
function appendUserMsg(text) {
  setEmptyState(false);
  const ts = fmtTime(Date.now());
  const group = document.createElement("div");
  group.className = "msg-group";
  group.innerHTML = `
        <div class="msg-row user">
          <div class="avatar user-av">U</div>
          <div class="bubble-wrap">
            <div class="bubble">${escHtml(text)}</div>
            <div class="msg-actions">
              <button class="msg-action-btn" onclick="copyMsg(this,'${escHtml(text)}')" >&#128203; Copy</button>
            </div>
            <div class="msg-meta">${ts}</div>
          </div>
        </div>`;
  refs.thread.appendChild(group);
  scrollToBottom();
  return group;
}
function appendPendingAssistant() {
  setEmptyState(false);
  const group = document.createElement("div");
  group.className = "msg-group";
  group.innerHTML = `
        <div class="msg-row assistant">
          <div class="avatar ai-av">AI</div>
          <div class="bubble-wrap">
            <div class="typing-indicator">
              <div class="typing-dot"></div>
              <div class="typing-dot"></div>
              <div class="typing-dot"></div>
            </div>
            <div class="bubble" style="display:none"></div>
            <div class="citations" style="display:none"></div>
            <div class="msg-actions" style="display:none">
              <button class="msg-action-btn">&#128203; Copy</button>
              <button class="msg-action-btn">&#128257; Regenerate</button>
              <button class="msg-action-btn fb-btn fb-up" title="Helpful" data-rating="up">&#128077;</button>
              <button class="msg-action-btn fb-btn fb-down" title="Not helpful" data-rating="down">&#128078;</button>
            </div>
            <div class="msg-meta" style="display:none"></div>
          </div>
        </div>`;
  refs.thread.appendChild(group);
  scrollToBottom();
  const bw = group.querySelector(".bubble-wrap");
  const bubbleEl = bw.querySelector(".bubble");
  const typingEl = bw.querySelector(".typing-indicator");
  const citationsEl = bw.querySelector(".citations");
  const actionsEl = bw.querySelector(".msg-actions");
  const metaEl = bw.querySelector(".msg-meta");
  const copyBtn = actionsEl.querySelector("button");
  if (copyBtn) {
    copyBtn.addEventListener("click", () => copyMsg(copyBtn, bubbleEl.innerText));
  }
  const fbUpBtn = actionsEl.querySelector(".fb-up");
  const fbDownBtn = actionsEl.querySelector(".fb-down");
  return { group, bubbleEl, typingEl, citationsEl, actionsEl, metaEl, fbUpBtn, fbDownBtn };
}
function appendSystemMsg(text) {
  const div = document.createElement("div");
  div.className = "msg-group";
  div.innerHTML = `<div class="msg-row assistant"><div class="avatar ai-av">&#9432;</div><div class="bubble-wrap"><div class="bubble">${parseMarkdown(text)}</div></div></div>`;
  refs.thread.appendChild(div);
  scrollToBottom();
}
function appendErrorMsg(text) {
  const div = document.createElement("div");
  div.className = "msg-group";
  div.innerHTML = `<div class="msg-row assistant"><div class="avatar ai-av">!</div><div class="bubble-wrap"><div class="bubble error-bubble">&#9888; ${escHtml(text)}</div></div></div>`;
  refs.thread.appendChild(div);
  scrollToBottom();
}

// src/user-types.ts
function sourceRefToChunkResult(ref) {
  return {
    text: ref.text ?? "",
    score: ref.score ?? 0,
    metadata: {
      source: ref.source ?? "",
      source_uri: ref.source_uri ?? "",
      source_key: ref.source_key ?? "",
      document_id: ref.document_id ?? "",
      section: ref.section ?? "",
      original_char_start: ref.original_char_start,
      original_char_end: ref.original_char_end
    }
  };
}

// src/conversations.ts
function renderConversationList(convs) {
  const container = byId("convList");
  if (!convs.length) {
    container.innerHTML = `<div class="conv-list-empty">No conversations yet.<br>Start one below!</div>`;
    return;
  }
  const groups = {};
  convs.forEach((c) => {
    const label = fmtRelative(c.updated_at_ms ?? Date.now());
    if (!groups[label]) groups[label] = [];
    groups[label].push(c);
  });
  let html = "";
  for (const [label, items] of Object.entries(groups)) {
    html += `<div class="conv-section-label">${escHtml(label)}</div>`;
    items.forEach((c) => {
      const isActive = c.conversation_id === state.activeConversationId;
      const title = escHtml(c.title || c.conversation_id);
      html += `
              <div class="conv-item-wrap">
                <div class="conv-item${isActive ? " active" : ""}" data-conv-id="${escHtml(c.conversation_id)}" title="${title}">
                  <span class="dot"></span>${title}
                </div>
                <button class="conv-item-del" data-conv-id="${escHtml(c.conversation_id)}" title="Delete">&#10005;</button>
              </div>`;
    });
  }
  container.innerHTML = html;
  container.querySelectorAll(".conv-item").forEach((el) => {
    el.addEventListener("click", () => {
      const id = el.dataset.convId;
      if (id) void selectConversation(id);
    });
    el.addEventListener("contextmenu", (e) => {
      e.preventDefault();
      const id = el.dataset.convId;
      if (!id) return;
      const conv = convs.find((c) => c.conversation_id === id);
      showConvCtxMenu(e.clientX, e.clientY, id, conv?.title || "");
    });
  });
  container.querySelectorAll(".conv-item-del").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const id = btn.dataset.convId;
      if (id) void deleteConversation(id);
    });
  });
}
async function loadConversations() {
  try {
    const data = await api(
      "GET",
      "/console/conversations?limit=50"
    );
    renderConversationList(data.conversations || []);
  } catch {
  }
}
async function selectConversation(id) {
  if (state.isStreaming) {
    state.streamAbortCtrl?.abort();
    state.isStreaming = false;
  }
  setActiveConversation(id);
  byId("convList").querySelectorAll(".conv-item").forEach((el) => {
    el.classList.toggle("active", el.dataset.convId === id);
  });
  await loadConversationHistory(id);
}
async function loadConversationHistory(id) {
  const convId = id ?? state.activeConversationId;
  if (!convId) return;
  try {
    const data = await api("GET", `/console/conversations/${convId}/history?limit=100`);
    refs.thread.innerHTML = "";
    if (!data.turns || !data.turns.length) {
      refs.thread.innerHTML = `<div class="thread-empty" id="threadEmpty"><div class="thread-empty-icon">&#128172;</div><div class="thread-empty-title">Empty conversation</div><div class="thread-empty-sub">Send a message to start the conversation.</div></div>`;
      return;
    }
    data.turns.forEach((turn) => {
      if (turn.role === "user") {
        appendUserMsg(turn.content);
      } else if (isSourcesTurn(turn)) {
        setEmptyState(false);
        appendSourcesTurn(refs.thread, turn.sources ?? []);
      } else {
        setEmptyState(false);
        const group = document.createElement("div");
        group.className = "msg-group";
        const ts = fmtTime(turn.timestamp_ms ?? Date.now());
        const sources = turn.sources ?? [];
        if (sources.length) cacheDocsFromSources(sources);
        const citationsHtml = sources.length ? `<div class="citations">${buildCitationsHtml(sources.map(sourceRefToChunkResult))}</div>` : "";
        group.innerHTML = `
                    <div class="msg-row assistant">
                      <div class="avatar ai-av">AI</div>
                      <div class="bubble-wrap">
                        <div class="bubble">${parseMarkdown(turn.content)}</div>
                        ${citationsHtml}
                        <div class="msg-actions">
                          <button class="msg-action-btn">&#128203; Copy</button>
                        </div>
                        <div class="msg-meta">${ts}</div>
                      </div>
                    </div>`;
        const copyBtn = group.querySelector(".msg-action-btn");
        if (copyBtn) {
          const bubbleEl = group.querySelector(".bubble");
          copyBtn.addEventListener("click", () => copyMsg(copyBtn, bubbleEl.innerText));
        }
        if (sources.length) wireCitationActions(group);
        refs.thread.appendChild(group);
      }
    });
    const activeConv = document.querySelector(`.conv-item[data-conv-id="${convId}"]`);
    if (activeConv) {
      byId("convTitle").textContent = activeConv.title ?? activeConv.textContent?.trim() ?? "Conversation";
    }
    setTimeout(() => {
      refs.thread.scrollTop = refs.thread.scrollHeight;
    }, 50);
  } catch (err) {
    appendErrorMsg("Failed to load conversation history: " + String(err));
  }
}
function createNewConversation() {
  setActiveConversation(null);
  refs.thread.innerHTML = `<div class="thread-empty" id="threadEmpty"><div class="thread-empty-icon">&#128172;</div><div class="thread-empty-title">New conversation</div><div class="thread-empty-sub">Send a message to get started.</div></div>`;
  byId("convTitle").textContent = "New conversation";
  byId("convList").querySelectorAll(".conv-item").forEach((el) => {
    el.classList.remove("active");
  });
  const input = document.getElementById("msgInput");
  if (input) input.focus();
}
async function deleteConversation(id) {
  try {
    await api("DELETE", `/console/conversations/${id}`);
    if (state.activeConversationId === id) {
      setActiveConversation(null);
      refs.thread.innerHTML = `<div class="thread-empty" id="threadEmpty"><div class="thread-empty-icon">&#9670;</div><div class="thread-empty-title">RagWeave</div><div class="thread-empty-sub">Ask anything \u2014 I'll search your knowledge base and generate a response with sources.</div></div>`;
      byId("convTitle").textContent = "RagWeave";
    }
    await loadConversations();
    showToast("Conversation deleted");
  } catch {
    showToast("Failed to delete conversation");
  }
}
function updateConvTitle() {
  if (!state.activeConversationId) return;
  const item = byId("convList").querySelector(
    `.conv-item[data-conv-id="${state.activeConversationId}"]`
  );
  if (item) {
    byId("convTitle").textContent = item.textContent?.trim().replace(/^●/, "").trim() ?? "Conversation";
  }
}
function hideConvCtxMenu() {
  const menu = byId("convCtxMenu");
  menu.classList.remove("open");
  menu.setAttribute("aria-hidden", "true");
  state.convMenuTargetId = null;
}
function showConvCtxMenu(x, y, id, title) {
  const menu = byId("convCtxMenu");
  state.convMenuTargetId = id;
  state.convMenuTargetTitle = title;
  menu.style.left = "-9999px";
  menu.style.top = "-9999px";
  menu.classList.add("open");
  menu.setAttribute("aria-hidden", "false");
  const rect = menu.getBoundingClientRect();
  const maxX = window.innerWidth - rect.width - 8;
  const maxY = window.innerHeight - rect.height - 8;
  menu.style.left = `${Math.max(8, Math.min(x, maxX))}px`;
  menu.style.top = `${Math.max(8, Math.min(y, maxY))}px`;
}
function openRenameModal(id, currentTitle) {
  state.renameTargetId = id;
  const overlay = byId("renameModal");
  const input = byId("renameInput");
  input.value = currentTitle || "";
  overlay.classList.add("open");
  overlay.setAttribute("aria-hidden", "false");
  setTimeout(() => {
    input.focus();
    input.select();
  }, 40);
}
function closeRenameModal() {
  const overlay = byId("renameModal");
  overlay.classList.remove("open");
  overlay.setAttribute("aria-hidden", "true");
  state.renameTargetId = null;
}
async function submitRename() {
  const id = state.renameTargetId;
  if (!id) return;
  const input = byId("renameInput");
  const trimmed = input.value.trim();
  if (!trimmed) {
    input.focus();
    return;
  }
  closeRenameModal();
  try {
    await api("PATCH", `/console/conversations/${encodeURIComponent(id)}`, { title: trimmed });
    if (state.activeConversationId === id) {
      byId("convTitle").textContent = trimmed;
    }
    await loadConversations();
    showToast("Conversation renamed");
  } catch (err) {
    showToast("Failed to rename: " + String(err));
  }
}
function initConversations() {
  byId("convCtxMenu").querySelectorAll("[data-action]").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const id = state.convMenuTargetId;
      const title = state.convMenuTargetTitle;
      const action = btn.dataset.action;
      hideConvCtxMenu();
      if (!id) return;
      if (action === "rename") openRenameModal(id, title);
      else if (action === "delete") void deleteConversation(id);
    });
  });
  document.addEventListener("mousedown", (e) => {
    const menu = byId("convCtxMenu");
    if (!menu.classList.contains("open")) return;
    if (e.target instanceof Node && menu.contains(e.target)) return;
    hideConvCtxMenu();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      hideConvCtxMenu();
      if (byId("renameModal").classList.contains("open")) closeRenameModal();
    }
  });
  window.addEventListener("resize", hideConvCtxMenu);
  window.addEventListener("scroll", hideConvCtxMenu, true);
  byId("renameCancel").addEventListener("click", closeRenameModal);
  byId("renameSave").addEventListener("click", () => void submitRename());
  byId("renameInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      void submitRename();
    } else if (e.key === "Escape") {
      e.preventDefault();
      closeRenameModal();
    }
  });
  byId("renameModal").addEventListener("click", (e) => {
    if (e.target === e.currentTarget) closeRenameModal();
  });
  byId("newChatBtn").addEventListener("click", createNewConversation);
  document.getElementById("convSearch")?.addEventListener("input", (e) => {
    const q = e.target.value.toLowerCase();
    byId("convList").querySelectorAll(".conv-item-wrap").forEach((wrap) => {
      const text = wrap.querySelector(".conv-item")?.textContent?.toLowerCase() || "";
      wrap.style.display = text.includes(q) ? "" : "none";
    });
  });
}

// src/feedback.ts
var pending = null;
var modalInited = false;
function captureTranscript() {
  const turns = [];
  const rows = refs.thread.querySelectorAll(".msg-row");
  rows.forEach((row) => {
    const role = row.classList.contains("user") ? "user" : "assistant";
    const bubble = row.querySelector(".bubble");
    const text = (bubble?.innerText ?? "").trim();
    if (text) turns.push({ role, text });
  });
  return turns;
}
function openModal(rating) {
  initModal();
  const overlay = byId("feedbackModal");
  const title = byId("feedbackModalTitle");
  const input = byId("feedbackInput");
  title.textContent = rating === "up" ? "Give positive feedback" : "Give negative feedback";
  input.value = "";
  overlay.classList.add("open");
  overlay.setAttribute("aria-hidden", "false");
  setTimeout(() => input.focus(), 50);
}
function closeModal() {
  const overlay = byId("feedbackModal");
  overlay.classList.remove("open");
  overlay.setAttribute("aria-hidden", "true");
  pending = null;
}
async function submitFeedback() {
  if (!pending) return closeModal();
  const { rating, ctx, upBtn, downBtn, submitted } = pending;
  const comment = byId("feedbackInput").value.trim();
  const winner = rating === "up" ? upBtn : downBtn;
  const loser = rating === "up" ? downBtn : upBtn;
  closeModal();
  submitted.value = rating;
  winner.classList.add("active");
  loser.disabled = true;
  try {
    await api("POST", "/console/feedback", {
      rating,
      conversation_id: ctx.conversationId,
      query: ctx.query,
      answer: ctx.answer(),
      comment: comment || null,
      transcript: captureTranscript()
    });
    showToast(rating === "up" ? "Thanks for the feedback!" : "Got it, we'll improve.");
  } catch (err) {
    submitted.value = null;
    winner.classList.remove("active");
    loser.disabled = false;
    showToast("Feedback failed: " + String(err));
  }
}
function initModal() {
  if (modalInited) return;
  modalInited = true;
  byId("feedbackCancel").addEventListener("click", closeModal);
  byId("feedbackSubmit").addEventListener("click", () => void submitFeedback());
  byId("feedbackModal").addEventListener("click", (e) => {
    if (e.target.id === "feedbackModal") closeModal();
  });
  document.addEventListener("keydown", (e) => {
    const overlay = byId("feedbackModal");
    if (!overlay.classList.contains("open")) return;
    if (e.key === "Escape") {
      e.preventDefault();
      closeModal();
    } else if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      void submitFeedback();
    }
  });
}
function attachFeedback(upBtn, downBtn, ctx) {
  const submitted = { value: null };
  const onClick = (rating) => () => {
    if (submitted.value) return;
    pending = { rating, ctx, upBtn, downBtn, submitted };
    openModal(rating);
  };
  upBtn.addEventListener("click", onClick("up"));
  downBtn.addEventListener("click", onClick("down"));
}

// src/streaming.ts
function buildQueryBody(queryText) {
  const s = getSettings();
  const body = {
    query: queryText,
    search_limit: parseInt(String(s.searchLimit ?? "10"), 10),
    rerank_top_k: parseInt(String(s.rerankTopK ?? "5"), 10),
    memory_enabled: s.memory_enabled !== false,
    conversation_id: state.activeConversationId ?? void 0
  };
  if (getChatMode() === "sources") {
    body.mode = "retrieval";
    body.retrieval_sub_mode = getRetrievalSubMode();
    const topK = getSourcesTopK();
    body.rerank_top_k = topK;
    body.search_limit = Math.max(parseInt(String(s.searchLimit ?? "10"), 10), topK * 2);
  }
  return body;
}
function chunkToSourceRef(c) {
  const m = c.metadata || {};
  return {
    source: String(m.source ?? ""),
    source_uri: String(m.source_uri ?? ""),
    source_key: String(m.source_key ?? ""),
    document_id: String(m.document_id ?? ""),
    section: String(m.section ?? m.heading ?? ""),
    score: c.score,
    text: c.text,
    original_char_start: typeof m.original_char_start === "number" ? m.original_char_start : void 0,
    original_char_end: typeof m.original_char_end === "number" ? m.original_char_end : void 0
  };
}
async function sourcesOnlyQuery(queryText) {
  appendUserMsg(queryText);
  try {
    const data = await api("POST", "/console/query", buildQueryBody(queryText));
    const cid = String(data.conversation_id ?? "").trim();
    if (cid) setActiveConversation(cid);
    const sources = (data.results ?? []).map(chunkToSourceRef);
    appendSourcesTurn(refs.thread, sources);
    if (data.relevant_doc_ids || data.ignored_doc_ids) {
      applyDocState(data.relevant_doc_ids ?? [], data.ignored_doc_ids ?? []);
    }
    if (data.token_budget) {
      updateContextIndicator(data.token_budget);
    }
    scrollToBottom();
    await loadConversations();
    updateConvTitle();
  } catch (err) {
    appendErrorMsg("Sources query failed: " + String(err));
  }
}
async function streamQuery(queryText) {
  if (state.isStreaming) {
    state.streamAbortCtrl?.abort();
  }
  state.isStreaming = true;
  state.streamAbortCtrl = new AbortController();
  state.pendingQueryText = queryText;
  state.pendingUserGroup = appendUserMsg(queryText);
  const pending2 = appendPendingAssistant();
  const { bubbleEl, typingEl, citationsEl, actionsEl, metaEl } = pending2;
  state.pendingAssistantGroup = pending2.group;
  attachFeedback(pending2.fbUpBtn, pending2.fbDownBtn, {
    conversationId: state.activeConversationId,
    query: queryText,
    answer: () => bubbleEl.innerText
  });
  setSendButtonStop(true);
  const url = apiBase() + "/query/stream";
  let response;
  try {
    response = await fetch(url, {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify(buildQueryBody(queryText)),
      signal: state.streamAbortCtrl.signal
    });
  } catch (err) {
    typingEl.remove();
    bubbleEl.innerHTML = "&#9888; Network error: " + escHtml(String(err));
    bubbleEl.classList.add("error-bubble");
    bubbleEl.style.display = "block";
    state.isStreaming = false;
    state.pendingUserGroup = null;
    state.pendingAssistantGroup = null;
    state.pendingQueryText = "";
    setSendButtonStop(false);
    return;
  }
  if (!response.ok || !response.body) {
    typingEl.remove();
    let detail = "";
    try {
      const body = await response.text();
      const parsed = JSON.parse(body);
      const d = parsed.detail;
      if (Array.isArray(d)) {
        detail = d.map(
          (e) => `${(e.loc ?? []).slice(1).join(".")}: ${e.msg ?? ""}`
        ).join("; ");
      } else if (typeof d === "string") {
        detail = d;
      }
    } catch {
    }
    bubbleEl.innerHTML = `&#9888; Stream error (HTTP ${response.status})` + (detail ? ` \u2014 ${escHtml(detail)}` : "");
    bubbleEl.classList.add("error-bubble");
    bubbleEl.style.display = "block";
    state.isStreaming = false;
    state.pendingUserGroup = null;
    state.pendingAssistantGroup = null;
    state.pendingQueryText = "";
    setSendButtonStop(false);
    return;
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let answer = "";
  let started = false;
  let errorShown = false;
  let pendingClarification = "";
  let firstTokenAt = 0;
  let lastTokenAt = 0;
  let tokenEventCount = 0;
  let lastBudget = null;
  clearLastTurnStats();
  let renderRaf = 0;
  const flushRender = () => {
    renderRaf = 0;
    bubbleEl.innerHTML = parseMarkdown(answer);
  };
  const scheduleRender = () => {
    if (renderRaf) return;
    renderRaf = requestAnimationFrame(flushRender);
  };
  const cancelRender = () => {
    if (renderRaf) {
      cancelAnimationFrame(renderRaf);
      renderRaf = 0;
    }
  };
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split("\n\n");
      buffer = events.pop() || "";
      for (const evt of events) {
        const lines = evt.split("\n");
        const evtType = (lines.find((l) => l.startsWith("event: ")) ?? "").slice(7);
        const dataRaw = (lines.find((l) => l.startsWith("data: ")) ?? "data: {}").slice(6);
        let data;
        try {
          data = JSON.parse(dataRaw);
        } catch {
          data = {};
        }
        if (evtType === "token") {
          if (!started) {
            typingEl.style.display = "none";
            bubbleEl.style.display = "block";
            bubbleEl.classList.add("streaming");
            started = true;
            firstTokenAt = performance.now();
          }
          lastTokenAt = performance.now();
          tokenEventCount++;
          answer += data.token || "";
          scheduleRender();
          scrollToBottom();
        } else if (evtType === "retrieval") {
          const cid = String(data.conversation_id ?? "").trim();
          if (cid) setActiveConversation(cid);
          const clar = String(data.clarification_message ?? "").trim();
          if (clar) pendingClarification = clar;
          if (data.token_budget) {
            lastBudget = data.token_budget;
            updateContextIndicator(lastBudget);
          }
          const results = data.results ?? [];
          if (results.length) {
            const sourceRefs = results.map(chunkToSourceRef);
            cacheDocsFromSources(sourceRefs);
          }
          const showCitations = byId("citationsToggle").checked;
          if (showCitations && results.length) {
            citationsEl.innerHTML = buildCitationsHtml(results);
            wireCitationActions(citationsEl);
          }
          if (data.relevant_doc_ids || data.ignored_doc_ids) {
            applyDocState(
              data.relevant_doc_ids ?? [],
              data.ignored_doc_ids ?? []
            );
          }
        } else if (evtType === "error") {
          errorShown = true;
          cancelRender();
          bubbleEl.classList.remove("streaming");
          typingEl.style.display = "none";
          bubbleEl.innerHTML = "&#9888; " + escHtml(String(data.message ?? "Unknown error"));
          bubbleEl.classList.add("error-bubble");
          bubbleEl.style.display = "block";
          scrollToBottom();
        } else if (evtType === "done") {
          const cid = String(data.conversation_id ?? "").trim();
          if (cid) setActiveConversation(cid);
          if (data.token_budget) lastBudget = data.token_budget;
          const completionTokens = Number(lastBudget?.actual_completion_tokens) || tokenEventCount;
          const promptTokens = Number(lastBudget?.actual_prompt_tokens) || Number(lastBudget?.input_tokens) || 0;
          const elapsedMs = lastTokenAt > firstTokenAt ? lastTokenAt - firstTokenAt : 0;
          const tokensPerSecond = elapsedMs > 0 && completionTokens > 0 ? completionTokens / (elapsedMs / 1e3) : 0;
          updateContextIndicator(lastBudget, {
            promptTokens,
            completionTokens,
            tokensPerSecond,
            costUsd: Number(lastBudget?.cost_usd ?? 0)
          });
          cancelRender();
          bubbleEl.classList.remove("streaming");
          typingEl.style.display = "none";
          if (!errorShown) {
            if (!started) {
              const msg = pendingClarification || "I couldn't find relevant information for that query. Could you rephrase your question or provide more details?";
              bubbleEl.innerHTML = parseMarkdown(msg);
              bubbleEl.style.display = "block";
            } else {
              bubbleEl.innerHTML = parseMarkdown(answer);
              bubbleEl.style.display = "block";
            }
          }
          const showCitations = byId("citationsToggle").checked;
          if (showCitations && citationsEl.innerHTML) {
            revealCitations(citationsEl);
          }
          actionsEl.style.display = "flex";
          metaEl.textContent = fmtTime(Date.now());
          metaEl.style.display = "block";
          scrollToBottom();
          await loadConversations();
          updateConvTitle();
        }
      }
    }
  } catch (err) {
    if (err.name !== "AbortError") {
      appendErrorMsg("Stream interrupted: " + String(err));
    }
  } finally {
    cancelRender();
    bubbleEl.classList.remove("streaming");
  }
  state.isStreaming = false;
  state.pendingUserGroup = null;
  state.pendingAssistantGroup = null;
  state.pendingQueryText = "";
  setSendButtonStop(false);
}
function setSendButtonStop(isStop) {
  const btn = byId("sendBtn");
  if (isStop) {
    btn.classList.add("stop-mode");
    btn.title = "Stop generation (cancel)";
    btn.innerHTML = "&#9632;";
  } else {
    btn.classList.remove("stop-mode");
    btn.title = "Send";
    btn.innerHTML = "&#9650;";
  }
}
function cancelStream() {
  if (!state.isStreaming) return;
  state.streamAbortCtrl?.abort();
  state.pendingUserGroup?.remove();
  state.pendingAssistantGroup?.remove();
  state.pendingUserGroup = null;
  state.pendingAssistantGroup = null;
  if (state.pendingQueryText && !refs.ta.value.trim()) {
    refs.ta.value = state.pendingQueryText;
    refs.ta.style.height = "auto";
    refs.ta.style.height = Math.min(refs.ta.scrollHeight, 220) + "px";
    refs.ta.focus();
  }
  state.pendingQueryText = "";
  state.isStreaming = false;
  setSendButtonStop(false);
}
async function nonStreamQuery(queryText) {
  appendUserMsg(queryText);
  const handles = appendPendingAssistant();
  const { bubbleEl, typingEl, citationsEl, actionsEl, metaEl } = handles;
  try {
    const data = await api("POST", "/console/query", buildQueryBody(queryText));
    const cid = String(data.conversation_id ?? "").trim();
    if (cid) setActiveConversation(cid);
    attachFeedback(handles.fbUpBtn, handles.fbDownBtn, {
      conversationId: state.activeConversationId,
      query: queryText,
      answer: () => bubbleEl.innerText
    });
    typingEl.style.display = "none";
    const answer = data.generated_answer ?? data.clarification_message ?? "No response.";
    bubbleEl.innerHTML = parseMarkdown(answer);
    bubbleEl.style.display = "block";
    const tb = data.token_budget;
    if (tb) {
      updateContextIndicator(tb, {
        promptTokens: Number(tb.actual_prompt_tokens) || Number(tb.input_tokens) || 0,
        completionTokens: Number(tb.actual_completion_tokens) || 0,
        tokensPerSecond: 0,
        costUsd: Number(tb.cost_usd) || 0
      });
    }
    const showCitations = byId("citationsToggle").checked;
    const results = data.results ?? [];
    if (results.length) {
      cacheDocsFromSources(results.map(chunkToSourceRef));
    }
    if (showCitations && results.length) {
      citationsEl.innerHTML = buildCitationsHtml(results);
      wireCitationActions(citationsEl);
      revealCitations(citationsEl);
    }
    actionsEl.style.display = "flex";
    metaEl.textContent = fmtTime(Date.now());
    metaEl.style.display = "block";
    scrollToBottom();
    await loadConversations();
    updateConvTitle();
  } catch (err) {
    typingEl.style.display = "none";
    bubbleEl.innerHTML = "&#9888; " + escHtml(String(err));
    bubbleEl.classList.add("error-bubble");
    bubbleEl.style.display = "block";
  }
}
async function sendQuery(text) {
  if (getChatMode() === "sources") {
    await sourcesOnlyQuery(text);
    return;
  }
  const s = getSettings();
  const useStreaming = s.streaming !== false;
  if (useStreaming) await streamQuery(text);
  else await nonStreamQuery(text);
}

// src/slash.ts
function renderSlashDropdown(cmds) {
  const container = byId("slashItems");
  container.innerHTML = cmds.map(
    (c) => `<div class="slash-item" data-cmd="/${escHtml(c.name)}"><span class="slash-cmd">/${escHtml(c.name)}</span><span class="slash-desc">${escHtml(c.description)}</span></div>`
  ).join("");
  state.allSlashItems = Array.from(container.querySelectorAll(".slash-item"));
  state.allSlashItems.forEach(
    (item) => item.addEventListener("click", () => executeCmd(item.dataset.cmd || ""))
  );
}
function renderCmdPicker(cmds) {
  const container = byId("cmdPickerBody");
  const grouped = {};
  cmds.forEach((c) => {
    const cat = c.category || "General";
    if (!grouped[cat]) grouped[cat] = [];
    grouped[cat].push(c);
  });
  let html = "";
  for (const [cat, items] of Object.entries(grouped)) {
    html += `<div class="cmd-group-label">${escHtml(cat)}</div>`;
    items.forEach((c) => {
      html += `<div class="cmd-picker-item" data-cmd="/${escHtml(c.name)}"><span class="cmd-picker-icon">&#47;</span><span class="cmd-picker-name">/${escHtml(c.name)}</span><span class="cmd-picker-desc">${escHtml(c.description)}</span></div>`;
    });
  }
  container.innerHTML = html;
  state.allPickerItems = Array.from(container.querySelectorAll(".cmd-picker-item"));
  state.allPickerItems.forEach(
    (item) => item.addEventListener("click", () => executePicker(item))
  );
}
async function loadCommands() {
  try {
    const data = await api("GET", "/console/commands?mode=query");
    state.dynamicCmds = data.commands || [];
    renderSlashDropdown(state.dynamicCmds);
    renderCmdPicker(state.dynamicCmds);
  } catch {
  }
}
function closeDropdown() {
  refs.dropdown.classList.remove("open");
}
function setSelected(i) {
  const vis = state.allSlashItems.filter((x) => x.style.display !== "none");
  if (!vis.length) return;
  state.slashSelIdx = (i + vis.length) % vis.length;
  vis.forEach((el, j) => el.classList.toggle("selected", j === state.slashSelIdx));
}
function executeCmd(cmd) {
  refs.ta.value = cmd + " ";
  refs.ta.focus();
  refs.ta.style.height = "auto";
  closeDropdown();
}
function handleSlashInput() {
  const val = refs.ta.value;
  if (!val.startsWith("/")) {
    closeDropdown();
    return;
  }
  const q = val.slice(1).toLowerCase();
  let vis = 0;
  state.allSlashItems.forEach((item) => {
    const cmdAttr = (item.dataset.cmd || "").toLowerCase();
    const descEl = item.querySelector(".slash-desc");
    const descText = descEl ? descEl.textContent?.toLowerCase() || "" : "";
    const match = cmdAttr.includes(q) || descText.includes(q);
    item.style.display = match ? "" : "none";
    if (match) vis++;
  });
  if (!vis) {
    closeDropdown();
    return;
  }
  refs.dropdown.classList.add("open");
  setSelected(0);
}
function setPickerSelected(idx) {
  if (!state.allPickerItems.length) return;
  state.pickerIdx = (idx + state.allPickerItems.length) % state.allPickerItems.length;
  state.allPickerItems.forEach((el, i) => el.classList.toggle("selected", i === state.pickerIdx));
  state.allPickerItems[state.pickerIdx]?.scrollIntoView({ block: "nearest" });
}
function executePicker(item) {
  const cmd = item.dataset.cmd || "";
  closeCmdPicker();
  refs.ta.value = cmd + " ";
  refs.ta.focus();
  refs.ta.style.height = "auto";
  closeDropdown();
}
function closeCmdPicker() {
  refs.cmdPicker.classList.remove("open");
  refs.cmdBtn.classList.remove("active");
}
async function submitSlashCommand(text) {
  const trimmed = text.trim();
  const spaceIdx = trimmed.indexOf(" ");
  const commandName = (spaceIdx === -1 ? trimmed.slice(1) : trimmed.slice(1, spaceIdx)).toLowerCase();
  const arg = spaceIdx === -1 ? "" : trimmed.slice(spaceIdx + 1).trim();
  appendUserMsg(trimmed);
  try {
    const result = await api("POST", "/console/command", {
      mode: "query",
      command: commandName,
      arg: arg || void 0,
      state: { conversation_id: state.activeConversationId ?? void 0 }
    });
    const action = String(result.action ?? "noop");
    if (action === "run_stream_query") {
      await streamQuery(arg || commandName);
    } else if (action === "run_non_stream_query") {
      await nonStreamQuery(arg || commandName);
    } else if (action === "new_conversation") {
      const conv = result.data?.conversation;
      if (conv?.conversation_id) setActiveConversation(conv.conversation_id);
      else createNewConversation();
      await loadConversations();
    } else if (action === "switch_conversation") {
      const cid = String(result.data?.conversation_id ?? arg).trim();
      if (cid) await selectConversation(cid);
    } else if (action === "list_conversations") {
      await loadConversations();
      appendSystemMsg("Conversation list refreshed.");
    } else if (action === "show_history") {
      await loadConversationHistory();
    } else if (action === "compact_conversation") {
      const summary = String(result.data?.summary ?? "").trim();
      appendSystemMsg(summary ? `Compacted. Summary:

${summary}` : "Conversation compacted.");
      await loadConversations();
    } else if (action === "delete_conversation") {
      const cid = String(result.data?.conversation_id ?? "").trim();
      if (cid) await deleteConversation(cid);
    } else if (action === "clear_view") {
      refs.thread.innerHTML = "";
      setEmptyState(true);
    } else if (action === "refresh_health") {
      const h = result.data?.health;
      appendSystemMsg("**Health**\n\n" + (h ? formatApiPayload(h) : "_unavailable_"));
    } else if (action === "render_help") {
      const cmds = result.data?.commands;
      if (cmds?.length) {
        const lines = cmds.map((c) => `- **/${c.name}** \u2014 ${c.description}`).join("\n");
        appendSystemMsg("**Available commands**\n\n" + lines);
      }
    } else if (result.message) {
      appendSystemMsg(String(result.message));
    } else if (result.data && Object.keys(result.data).length) {
      appendSystemMsg(formatApiPayload(result.data));
    } else {
      appendSystemMsg("Command executed.");
    }
  } catch (err) {
    appendErrorMsg("Command failed: " + String(err));
  }
}

// src/modelBadge.ts
var PROVIDER_LABELS = {
  ollama: "Ollama",
  openai: "OpenAI",
  anthropic: "Anthropic",
  openrouter: "OpenRouter",
  azure: "Azure",
  bedrock: "Bedrock",
  gemini: "Gemini",
  groq: "Groq",
  mistral: "Mistral"
};
function formatLabel(info) {
  if (!info.generation_enabled) return "Generation off";
  const provider = info.provider ? PROVIDER_LABELS[info.provider.toLowerCase()] ?? info.provider : "";
  const model = info.display || info.model || "unknown";
  return provider ? `${provider} \xB7 ${model}` : model;
}
async function loadModelInfo() {
  const badge = document.getElementById("modelBadge");
  const text = document.getElementById("modelBadgeText");
  if (!badge || !text) return;
  try {
    const info = await api("GET", "/console/model-info");
    text.textContent = formatLabel(info);
    badge.classList.toggle("disabled", !info.generation_enabled);
    badge.title = info.generation_enabled ? `${info.provider || "model"}: ${info.display}` : "Backend is in retrieval-only mode (RAG_GENERATION_ENABLED=false)";
  } catch {
    text.textContent = "Model unknown";
    badge.classList.add("disabled");
  }
}

// src/attachments.ts
function renderChips() {
  const container = byId("attachChips");
  container.innerHTML = "";
  state.attachments.forEach((a) => {
    const chip = document.createElement("div");
    chip.className = "attach-chip";
    chip.innerHTML = `<span class="attach-chip-icon">${a.icon}</span><span class="attach-chip-label">${escHtml(a.label)}</span><button class="attach-chip-remove" title="Remove">&#215;</button>`;
    chip.querySelector(".attach-chip-remove")?.addEventListener(
      "click",
      () => removeChip(a.id)
    );
    container.appendChild(chip);
  });
}
function addChip(icon, label, id) {
  if (state.attachments.find((a) => a.id === id)) return;
  state.attachments.push({ id, icon, label });
  renderChips();
}
function removeChip(id) {
  state.attachments = state.attachments.filter((a) => a.id !== id);
  renderChips();
}
function closeAllPopovers() {
  refs.attachPopover.classList.remove("open");
  refs.webInputPanel.classList.remove("open");
  refs.kbPanel.classList.remove("open");
  refs.attachBtn.classList.remove("active");
}
function toggleAttachPopover() {
  const isOpen = refs.attachPopover.classList.contains("open");
  closeAllPopovers();
  closeCmdPickerExternal();
  if (!isOpen) {
    refs.attachPopover.classList.add("open");
    refs.attachBtn.classList.add("active");
  }
}
var closeCmdPickerExternal = () => {
};
function setCloseCmdPicker(fn) {
  closeCmdPickerExternal = fn;
}
function openWebInput() {
  closeAllPopovers();
  refs.webInputPanel.classList.add("open");
  setTimeout(() => document.getElementById("webUrlInput")?.focus(), 50);
}
function openKBSelect() {
  closeAllPopovers();
  refs.kbPanel.classList.add("open");
}
function triggerFileUpload() {
  closeAllPopovers();
  byId("fileInput").click();
}
function handleFileSelect(input) {
  if (!input.files) return;
  Array.from(input.files).forEach((file) => addChip("&#128196;", file.name, "file:" + file.name));
  input.value = "";
  showToast("File added to context");
}
function attachWebUrl() {
  const input = byId("webUrlInput");
  const url = input.value.trim();
  if (!url) return;
  try {
    new URL(url);
  } catch {
    showToast("Invalid URL");
    return;
  }
  addChip("&#127760;", new URL(url).hostname.replace("www.", ""), "web:" + url);
  input.value = "";
  refs.webInputPanel.classList.remove("open");
  showToast("Web page added to context");
}
function filterKB(q) {
  document.querySelectorAll(".kb-item").forEach((el) => {
    const name = el.querySelector(".kb-item-name")?.textContent?.toLowerCase() || "";
    el.style.display = name.includes(q.toLowerCase()) ? "" : "none";
  });
}
function attachKBDocs() {
  document.querySelectorAll("#kbList input[type=checkbox]:checked").forEach((cb) => {
    addChip("&#128218;", cb.value, "kb:" + cb.value);
    cb.checked = false;
  });
  refs.kbPanel.classList.remove("open");
  showToast("Documents added to context");
}
function initAttachments() {
  refs.attachBtn.addEventListener("click", toggleAttachPopover);
  document.getElementById("attachOptFile")?.addEventListener("click", triggerFileUpload);
  document.getElementById("attachOptWeb")?.addEventListener("click", openWebInput);
  document.getElementById("attachOptKB")?.addEventListener("click", openKBSelect);
  document.getElementById("fileInput")?.addEventListener("change", (e) => {
    handleFileSelect(e.target);
  });
  document.getElementById("webUrlInput")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") attachWebUrl();
    if (e.key === "Escape") refs.webInputPanel.classList.remove("open");
  });
  document.getElementById("webAddBtn")?.addEventListener("click", attachWebUrl);
  document.getElementById("kbSearch")?.addEventListener(
    "input",
    (e) => filterKB(e.target.value)
  );
  document.getElementById("kbAddBtn")?.addEventListener("click", attachKBDocs);
  document.getElementById("kbPanelClose")?.addEventListener("click", closeAllPopovers);
  const win = window;
  win["removeChip"] = removeChip;
  win["openWebInput"] = openWebInput;
  win["openKBSelect"] = openKBSelect;
  win["triggerFileUpload"] = triggerFileUpload;
  win["filterKB"] = filterKB;
  win["attachKBDocs"] = attachKBDocs;
  win["attachWebUrl"] = attachWebUrl;
  win["handleFileSelect"] = handleFileSelect;
}

// src/input.ts
function toggleCmdPicker() {
  const isOpen = refs.cmdPicker.classList.contains("open");
  closeAllPopovers();
  if (isOpen) {
    closeCmdPicker();
  } else {
    refs.cmdPicker.classList.add("open");
    refs.cmdBtn.classList.add("active");
    setPickerSelected(0);
  }
}
function triggerSend() {
  if (state.isStreaming) {
    cancelStream();
    return;
  }
  const text = refs.ta.value.trim();
  if (!text) return;
  closeDropdown();
  refs.ta.value = "";
  refs.ta.style.height = "auto";
  if (text.startsWith("/")) {
    void submitSlashCommand(text);
  } else {
    void sendQuery(text);
  }
}
function initInput() {
  setCloseCmdPicker(closeCmdPicker);
  refs.ta.addEventListener("input", () => {
    refs.ta.style.height = "auto";
    refs.ta.style.height = Math.min(refs.ta.scrollHeight, 220) + "px";
    handleSlashInput();
  });
  let lastEscAt = 0;
  refs.ta.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      const now = Date.now();
      if (refs.dropdown.classList.contains("open")) {
        closeDropdown();
        lastEscAt = 0;
        return;
      }
      if (now - lastEscAt < 400 && refs.ta.value) {
        refs.ta.value = "";
        refs.ta.style.height = "auto";
        lastEscAt = 0;
      } else {
        lastEscAt = now;
      }
      return;
    }
    if (!refs.dropdown.classList.contains("open")) {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        triggerSend();
      }
      return;
    }
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setSelected(state.slashSelIdx + 1);
    }
    if (e.key === "ArrowUp") {
      e.preventDefault();
      setSelected(state.slashSelIdx - 1);
    }
    if (e.key === "Tab" || e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      const vis = state.allSlashItems.filter((x) => x.style.display !== "none");
      if (vis[state.slashSelIdx]) executeCmd(vis[state.slashSelIdx].dataset.cmd || "");
    }
  });
  byId("sendBtn").addEventListener("click", triggerSend);
  refs.cmdBtn.addEventListener("click", toggleCmdPicker);
  document.getElementById("cmdPickerClose")?.addEventListener("click", closeCmdPicker);
  document.addEventListener("keydown", (e) => {
    if (!refs.cmdPicker.classList.contains("open")) return;
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setPickerSelected(state.pickerIdx + 1);
    }
    if (e.key === "ArrowUp") {
      e.preventDefault();
      setPickerSelected(state.pickerIdx - 1);
    }
    if (e.key === "Enter") {
      e.preventDefault();
      executePicker(state.allPickerItems[state.pickerIdx]);
    }
    if (e.key === "Escape") closeCmdPicker();
  });
  document.addEventListener("click", (e) => {
    const target = e.target;
    if (!target.closest(".input-bar")) {
      closeAllPopovers();
      closeCmdPicker();
      closeDropdown();
      document.getElementById("ctxChip")?.classList.remove("tooltip-open");
    }
  });
}

// src/format.ts
function fmtSize(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}

// src/ingest-stream.ts
var _activeStreams = /* @__PURE__ */ new Map();
function attachStream(jobId) {
  if (_activeStreams.has(jobId)) return;
  const url = apiBase() + `/api/v1/ingest/jobs/${jobId}/stream`;
  const es = new EventSource(url);
  _activeStreams.set(jobId, es);
  const finish = () => {
    es.close();
    _activeStreams.delete(jobId);
    refreshJob(jobId);
  };
  const onEvt = (e) => {
    try {
      const data = JSON.parse(e.data);
      const buf = jobLog.get(jobId) ?? [];
      buf.push(data.message);
      jobLog.set(jobId, buf);
      const meta = jobMeta.get(jobId);
      if (meta) {
        meta.status = data.status;
        if (data.kind === "done" && data.detail?.stored_chunks) {
          meta.stored_chunks = Number(data.detail.stored_chunks);
          meta.finished_at = Date.now() / 1e3;
        }
        if (data.kind === "error") {
          meta.error = data.message;
          meta.finished_at = Date.now() / 1e3;
        }
        renderJob(meta);
      }
      if (["done", "error"].includes(data.kind)) finish();
    } catch {
    }
  };
  es.addEventListener("stage", onEvt);
  es.addEventListener("done", onEvt);
  es.addEventListener("error", (e) => {
    const ev = e;
    if (ev.data) onEvt(ev);
    else finish();
  });
}
async function refreshJob(jobId) {
  try {
    const res = await fetch(apiBase() + `/api/v1/ingest/jobs/${jobId}`, { headers: authHeaders() });
    if (!res.ok) return;
    const job = await res.json();
    jobMeta.set(job.job_id, job);
    renderJob(job);
  } catch {
  }
}
async function refreshJobsList() {
  try {
    const res = await fetch(apiBase() + "/api/v1/ingest/jobs", { headers: authHeaders() });
    if (!res.ok) return;
    const data = await res.json();
    const list = byId("ingestJobsList");
    if (data.jobs.length === 0) {
      list.innerHTML = '<div class="ingest-jobs-empty">No jobs yet.</div>';
      return;
    }
    list.innerHTML = "";
    for (const j of data.jobs) {
      jobMeta.set(j.job_id, j);
      renderJob(j);
      if (["pending", "running"].includes(j.status)) attachStream(j.job_id);
    }
  } catch {
  }
}

// src/ingest-jobs.ts
var jobLog = /* @__PURE__ */ new Map();
var jobMeta = /* @__PURE__ */ new Map();
function escapeHtml(s) {
  return s.replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]
  );
}
function jobCardHtml(job) {
  const log = (jobLog.get(job.job_id) ?? []).slice(-30).join("\n") || "Queued\u2026";
  const isTerminal = ["done", "failed", "cancelled"].includes(job.status);
  const meta = [fmtSize(job.size_bytes)];
  if (job.stored_chunks) meta.push(`${job.stored_chunks} chunks`);
  if (job.started_at && job.finished_at) {
    meta.push(`${(job.finished_at - job.started_at).toFixed(1)}s`);
  } else if (job.started_at && !isTerminal) {
    meta.push(`running ${(Date.now() / 1e3 - job.started_at).toFixed(0)}s`);
  }
  const progressClass = job.error ? "ingest-job-progress ingest-job-error" : "ingest-job-progress";
  const progressText = job.error ? job.error : log;
  const cancelBtn = !isTerminal ? `<button class="ingest-job-action" data-cancel="${job.job_id}">Cancel</button>` : "";
  return `
        <div class="ingest-job ${job.status}" data-job="${job.job_id}">
            <div class="ingest-job-head">
                <span class="ingest-job-name">${escapeHtml(job.filename)}</span>
                <span class="ingest-job-status ${job.status}">${job.status}</span>
            </div>
            <div class="ingest-job-meta">${meta.join(" \xB7 ")}</div>
            <div class="${progressClass}">${escapeHtml(progressText)}</div>
            <div class="ingest-job-actions">${cancelBtn}</div>
        </div>`;
}
function renderJob(job) {
  jobMeta.set(job.job_id, job);
  const list = byId("ingestJobsList");
  const empty = list.querySelector(".ingest-jobs-empty");
  if (empty) empty.remove();
  let card = list.querySelector(`[data-job="${job.job_id}"]`);
  if (!card) {
    const wrap = document.createElement("div");
    wrap.innerHTML = jobCardHtml(job);
    const el = wrap.firstElementChild;
    list.prepend(el);
    card = el;
  } else {
    card.outerHTML = jobCardHtml(job);
  }
  list.querySelectorAll("[data-cancel]").forEach((btn) => {
    const id = btn.dataset.cancel;
    btn.onclick = () => cancelJob(id);
  });
}
async function cancelJob(jobId) {
  const meta = jobMeta.get(jobId);
  if (meta) {
    renderJob({ ...meta, status: "cancelling" });
  }
  showToast("Cancelling job\u2026");
  try {
    const res = await fetch(apiBase() + `/api/v1/ingest/jobs/${jobId}/cancel`, {
      method: "POST",
      headers: authHeaders()
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    await refreshJob(jobId);
    showToast("Cancel sent");
  } catch (err) {
    showToast(`Cancel failed: ${String(err)}`);
    if (meta) renderJob(meta);
  }
}

// src/ingest-modes.ts
var _selected = [];
function renderSelected() {
  const list = byId("ingestSelectedList");
  const wrap = byId("ingestSelected");
  const submit = byId("ingestSubmitBtn");
  if (_selected.length === 0) {
    wrap.style.display = "none";
    submit.disabled = true;
    return;
  }
  wrap.style.display = "block";
  submit.disabled = false;
  byId("ingestSelectedCount").textContent = `${_selected.length} file${_selected.length > 1 ? "s" : ""} selected`;
  list.innerHTML = _selected.map(
    (f, i) => `<div class="ingest-selected-item">
                    <span class="name">${escapeHtml(f.name)}</span>
                    <span class="size">${fmtSize(f.size)}</span>
                    <button class="remove" data-idx="${i}" title="Remove">&times;</button>
                </div>`
  ).join("");
  list.querySelectorAll(".remove").forEach((btn) => {
    btn.addEventListener("click", () => {
      const idx = Number(btn.dataset.idx);
      _selected.splice(idx, 1);
      renderSelected();
    });
  });
}
function addFiles(files) {
  for (const f of Array.from(files)) {
    if (f.size === 0) continue;
    if (_selected.some((s) => s.name === f.name && s.size === f.size)) continue;
    _selected.push(f);
  }
  renderSelected();
}
function clearSelected() {
  _selected.length = 0;
  renderSelected();
}
async function uploadOne(file) {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("options", "{}");
  const headers = authHeaders();
  delete headers["Content-Type"];
  try {
    const res = await fetch(apiBase() + "/api/v1/ingest/upload", {
      method: "POST",
      headers,
      body: fd
    });
    if (!res.ok) {
      const txt = await res.text();
      showToast(`Upload failed (${res.status}): ${txt.slice(0, 120)}`);
      return null;
    }
    return await res.json();
  } catch (err) {
    showToast(`Network error: ${String(err)}`);
    return null;
  }
}
async function startFileIngestion() {
  if (_selected.length === 0) return;
  const submit = byId("ingestSubmitBtn");
  const statusEl = byId("ingestSubmitStatus");
  submit.disabled = true;
  statusEl.textContent = `Uploading 0/${_selected.length}\u2026`;
  const jobs = [];
  let i = 0;
  for (const file of _selected) {
    statusEl.textContent = `Uploading ${++i}/${_selected.length}: ${file.name}`;
    const job = await uploadOne(file);
    if (job) {
      jobs.push(job);
      jobMeta.set(job.job_id, job);
      renderJob(job);
      attachStream(job.job_id);
    }
  }
  _selected.length = 0;
  renderSelected();
  statusEl.textContent = jobs.length > 0 ? `Started ${jobs.length} job${jobs.length > 1 ? "s" : ""}` : "";
  submit.disabled = _selected.length === 0;
  setTimeout(() => {
    statusEl.textContent = "";
  }, 4e3);
}
async function startUrlIngestion() {
  const input = byId("ingestUrlInput");
  const btn = byId("ingestUrlBtn");
  const statusEl = byId("ingestUrlStatus");
  const url = input.value.trim();
  if (!url) return;
  btn.disabled = true;
  statusEl.textContent = "Fetching\u2026";
  try {
    const res = await fetch(apiBase() + "/api/v1/ingest/url", {
      method: "POST",
      headers: { ...authHeaders(), "Content-Type": "application/json" },
      body: JSON.stringify({ url })
    });
    const data = await res.json();
    if (!res.ok) {
      statusEl.textContent = `Error: ${data.detail ?? res.status}`;
      return;
    }
    const job = data;
    jobMeta.set(job.job_id, job);
    renderJob(job);
    attachStream(job.job_id);
    input.value = "";
    statusEl.textContent = `Job started: ${job.filename}`;
    setTimeout(() => {
      statusEl.textContent = "";
    }, 4e3);
  } catch (err) {
    statusEl.textContent = `Network error: ${String(err)}`;
  } finally {
    btn.disabled = input.value.trim().length === 0;
  }
}
async function checkDirectory() {
  const input = byId("ingestDirInput");
  const result = byId("ingestDirResult");
  const ingestBtn = byId("ingestDirBtn");
  const path = input.value.trim();
  if (!path) return;
  result.style.display = "none";
  result.className = "ingest-dir-result";
  try {
    const res = await fetch(apiBase() + "/api/v1/ingest/check-path", {
      method: "POST",
      headers: { ...authHeaders(), "Content-Type": "application/json" },
      body: JSON.stringify({ path })
    });
    const data = await res.json();
    result.style.display = "block";
    if (!data.reachable) {
      result.classList.add("bad");
      result.innerHTML = `<strong>Unreachable.</strong> ${escapeHtml(data.reason ?? "unknown")}<br>The ingestion host could not access <code>${escapeHtml(path)}</code>.`;
      ingestBtn.disabled = true;
      return;
    }
    result.classList.add("ok");
    if (data.is_file) {
      result.innerHTML = `<strong>Reachable file.</strong> <code>${escapeHtml(data.path)}</code> \xB7 ${fmtSize(data.size_bytes ?? 0)}<br>Tip: use the Files tab for single-file uploads.`;
      ingestBtn.disabled = true;
      return;
    }
    const files = data.files ?? [];
    const truncNote = data.truncated ? ` (showing first ${files.length})` : "";
    const fileList = files.length > 0 ? `<div class="ingest-dir-files">${files.map(escapeHtml).join("<br>")}</div>` : "";
    result.innerHTML = `<strong>Reachable directory.</strong> <code>${escapeHtml(data.path)}</code><br>${data.file_count ?? 0} supported file${(data.file_count ?? 0) === 1 ? "" : "s"} found${truncNote}.` + fileList;
    ingestBtn.disabled = (data.file_count ?? 0) === 0;
  } catch (err) {
    result.style.display = "block";
    result.classList.add("bad");
    result.textContent = `Network error: ${String(err)}`;
  }
}
async function startDirectoryIngestion() {
  const input = byId("ingestDirInput");
  const btn = byId("ingestDirBtn");
  const statusEl = byId("ingestDirStatus");
  const path = input.value.trim();
  if (!path) return;
  btn.disabled = true;
  statusEl.textContent = "Submitting\u2026";
  try {
    const res = await fetch(apiBase() + "/api/v1/ingest/directory", {
      method: "POST",
      headers: { ...authHeaders(), "Content-Type": "application/json" },
      body: JSON.stringify({ path })
    });
    const data = await res.json();
    if (!res.ok) {
      statusEl.textContent = `Error: ${data.detail ?? res.status}`;
      return;
    }
    const jobs = data.jobs ?? [];
    for (const job of jobs) {
      jobMeta.set(job.job_id, job);
      renderJob(job);
      attachStream(job.job_id);
    }
    statusEl.textContent = `Submitted ${jobs.length} job${jobs.length === 1 ? "" : "s"}`;
    setTimeout(() => {
      statusEl.textContent = "";
    }, 4e3);
  } catch (err) {
    statusEl.textContent = `Network error: ${String(err)}`;
  } finally {
    btn.disabled = false;
  }
}

// src/ingest.ts
function switchView(view) {
  document.querySelectorAll(".view-tab").forEach((b) => {
    b.classList.toggle("active", b.dataset.view === view);
  });
  document.querySelectorAll(".view-pane").forEach((p) => {
    p.classList.toggle("active", p.id === `view-${view}`);
  });
  if (view === "ingest") void refreshJobsList();
}
function switchMode(mode) {
  document.querySelectorAll(".ingest-mode-tab").forEach((b) => {
    b.classList.toggle("active", b.dataset.mode === mode);
  });
  document.querySelectorAll(".ingest-mode-pane").forEach((p) => {
    p.classList.toggle("active", p.dataset.modePane === mode);
  });
}
function initIngestView() {
  document.querySelectorAll(".view-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      const view = btn.dataset.view;
      if (view) switchView(view);
    });
  });
  document.querySelectorAll(".ingest-mode-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      const mode = btn.dataset.mode;
      if (mode) switchMode(mode);
    });
  });
  const dz = byId("ingestDropzone");
  const fi = byId("ingestFileInput");
  dz.addEventListener("click", () => fi.click());
  fi.addEventListener("change", () => {
    if (fi.files) addFiles(fi.files);
    fi.value = "";
  });
  dz.addEventListener("dragover", (e) => {
    e.preventDefault();
    dz.classList.add("drag");
  });
  dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    dz.classList.remove("drag");
    const dt = e.dataTransfer;
    if (dt?.files) addFiles(dt.files);
  });
  byId("ingestClearBtn").addEventListener("click", () => clearSelected());
  byId("ingestSubmitBtn").addEventListener("click", () => {
    void startFileIngestion();
  });
  const urlInput = byId("ingestUrlInput");
  const urlBtn = byId("ingestUrlBtn");
  urlInput.addEventListener("input", () => {
    urlBtn.disabled = urlInput.value.trim().length === 0;
  });
  urlInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !urlBtn.disabled) void startUrlIngestion();
  });
  urlBtn.addEventListener("click", () => {
    void startUrlIngestion();
  });
  const dirInput = byId("ingestDirInput");
  const dirCheck = byId("ingestDirCheckBtn");
  const dirBtn = byId("ingestDirBtn");
  dirInput.addEventListener("input", () => {
    dirBtn.disabled = true;
  });
  dirInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") void checkDirectory();
  });
  dirCheck.addEventListener("click", () => {
    void checkDirectory();
  });
  dirBtn.addEventListener("click", () => {
    void startDirectoryIngestion();
  });
}

// src/user-console.ts
document.addEventListener("DOMContentLoaded", () => {
  populateRefs();
  initToast();
  initCitations();
  initContextIndicator();
  initScrollFab();
  initSidebar();
  initSettings();
  initConversations();
  initAttachments();
  initInput();
  initIngestView();
  initChatMode();
  loadSettings();
  void loadModelInfo();
  void Promise.all([loadCommands(), loadConversations()]).then(() => {
    if (state.activeConversationId) {
      void loadConversationHistory(state.activeConversationId);
    }
  });
  setTimeout(() => {
    refs.thread.scrollTop = refs.thread.scrollHeight;
  }, 100);
});
//# sourceMappingURL=user-console.js.map

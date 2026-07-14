// Local legal chat UI — talks only to the in-process fine-tuned model.
// Every message is persisted server-side (SQLite); past chats reload from there.

const $ = (id) => document.getElementById(id);

const els = {
  list: $("conversationList"),
  messages: $("messageList"),
  form: $("chatForm"),
  input: $("chatInput"),
  send: $("sendButton"),
  newChat: $("newChatButton"),
  juris: $("jurisdictionSelect"),
  onlineMode: $("onlineModeSelect"),
  statusPill: $("statusPill"),
  statusText: $("statusText"),
  modelBadge: $("modelBadge"),
  attach: $("attachButton"),
  fileInput: $("fileInput"),
  docsBar: $("selectedDocsBar"),
  appShell: $("appShell"),
  openSidebar: $("openSidebar"),
  closeSidebar: $("closeSidebar"),
  modeButtons: [...document.querySelectorAll(".mode-pill[data-mode]")],
  newChatLabel: $("newChatLabel"),
  composerFoot: $("composerFoot"),
};

// ---------- sidebar collapse (restored) ----------
function setSidebar(collapsed) {
  els.appShell.classList.toggle("sidebar-collapsed", collapsed);
  if (els.openSidebar) els.openSidebar.hidden = !collapsed;
  try { localStorage.setItem("sidebarCollapsed", collapsed ? "1" : ""); } catch (_) {}
}
if (els.closeSidebar) els.closeSidebar.addEventListener("click", () => setSidebar(true));
if (els.openSidebar) els.openSidebar.addEventListener("click", () => setSidebar(false));
try { if (localStorage.getItem("sidebarCollapsed") === "1") setSidebar(true); } catch (_) {}

const state = {
  conversations: [],
  currentId: null,
  mode: "memory",
  ready: false,
  streaming: false,
};

try {
  const preferredMode = localStorage.getItem("preferredChatMode");
  if (preferredMode === "memory" || preferredMode === "private") state.mode = preferredMode;
} catch (_) {}

const modeInfo = {
  memory: {
    title: "Memory chat",
    hero: "Uses your saved Memory chats for continuity. Questions, answers, uploads, and corrections can join your improvement and training records.",
    foot: "Memory mode: other Memory chats can use this history, and it can join your improvement/training records.",
  },
  private: {
    title: "Private chat",
    hero: "Isolated to this conversation. It is never used across chats or added to improvement/training records, and you can permanently delete it.",
    foot: "Private mode: isolated from cross-chat memory and training records. Use the trash button to permanently delete it.",
  },
};

function updateModeUI() {
  const info = modeInfo[state.mode];
  for (const button of els.modeButtons) {
    const active = button.dataset.mode === state.mode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", active ? "true" : "false");
  }
  if (els.newChatLabel) els.newChatLabel.textContent = `New ${state.mode} chat`;
  if (els.composerFoot) {
    const svg = els.composerFoot.querySelector("svg")?.outerHTML || "";
    els.composerFoot.innerHTML = `${svg}<span>${info.foot}</span>`;
    els.composerFoot.classList.toggle("private", state.mode === "private");
  }
  els.input.placeholder = state.mode === "private"
    ? "Ask privately — this chat stays isolated..."
    : "Ask any legal question...";
}

function chooseMode(mode, resetChat = true) {
  if (!(mode in modeInfo)) return;
  const changed = state.mode !== mode;
  state.mode = mode;
  try { localStorage.setItem("preferredChatMode", mode); } catch (_) {}
  updateModeUI();
  if (resetChat && (changed || state.currentId)) newChat();
  else if (!state.currentId) showHero();
}

// ---------- tiny API layer ----------
async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status}`);
  return r.json();
}
async function postJSON(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  return r.json();
}

// ---------- markdown-lite ----------
function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function renderMarkdown(text) {
  let t = escapeHtml(text);
  t = t.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  t = t.replace(/\*([^*\n]+?)\*/g, "<em>$1</em>");
  t = t.replace(/`([^`]+?)`/g, "<code>$1</code>");
  t = t.replace(/\[([^\]]+?)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  const lines = t.split("\n");
  const out = [];
  let para = [];
  let list = null; // {type, items}
  const flushPara = () => { if (para.length) { out.push(`<p>${para.join("<br>")}</p>`); para = []; } };
  const flushList = () => { if (list) { out.push(`<${list.type}>${list.items.map((i) => `<li>${i}</li>`).join("")}</${list.type}>`); list = null; } };
  for (const raw of lines) {
    const line = raw.replace(/\s+$/, "");
    const ol = line.match(/^\s*\d+\.\s+(.*)$/);
    const ul = line.match(/^\s*[-*]\s+(.*)$/);
    const heading = line.match(/^\s*(#{1,4})\s+(.+)$/);
    if (heading) { flushList(); flushPara(); const level = Math.min(4, heading[1].length + 1); out.push(`<h${level}>${heading[2]}</h${level}>`); }
    else if (/^\s*---+\s*$/.test(line)) { flushList(); flushPara(); out.push("<hr>"); }
    else if (ol) { flushPara(); if (!list || list.type !== "ol") { flushList(); list = { type: "ol", items: [] }; } list.items.push(ol[1]); }
    else if (ul) { flushPara(); if (!list || list.type !== "ul") { flushList(); list = { type: "ul", items: [] }; } list.items.push(ul[1]); }
    else if (line.trim() === "") { flushList(); flushPara(); }
    else { flushList(); para.push(line); }
  }
  flushList(); flushPara();
  return out.join("") || "<p></p>";
}

function timeLabel(iso) {
  try { return new Date(iso || Date.now()).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }); }
  catch { return ""; }
}

// ---------- rendering ----------
function heroHtml() {
  const cards = [
    ["pi-doctrine", "Explain a doctrine"],
    ["pi-problem", "Answer a problem question"],
    ["pi-sqe", "SQE-style answer"],
    ["pi-essay", "Draft a critical essay"],
  ];
  const info = modeInfo[state.mode];
  return `<div class="hero">
    <div class="hero-mode ${state.mode}">${state.mode === "memory" ? "Shared memory on" : "Private & isolated"}</div>
    <h1>${info.title}</h1>
    <p><strong>Latest V11 specialist legal training is active.</strong> ${info.hero}</p>
    <div class="prompt-grid">
      ${cards.map(([c, label]) => `<button class="prompt-card" data-starter="${label}"><span class="prompt-ico ${c}"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 3h12a1 1 0 0 1 1 1v17l-7-4-7 4V4a1 1 0 0 1 1-1z"/></svg></span>${label}</button>`).join("")}
    </div>
  </div>`;
}

function showHero() {
  els.messages.innerHTML = heroHtml();
  els.messages.querySelectorAll(".prompt-card").forEach((b) =>
    b.addEventListener("click", () => { els.input.value = b.dataset.starter + ": "; els.input.focus(); autoGrow(); syncSend(); })
  );
}

function userBubble(text) {
  const wrap = document.createElement("div");
  wrap.className = "message user";
  wrap.innerHTML = `<div><div class="bubble-user">${renderMarkdown(text)}</div></div>`;
  return wrap;
}
function assistantBubble(allowFeedback = state.mode === "memory") {
  const wrap = document.createElement("div");
  wrap.className = "message assistant";
  const feedback = allowFeedback ? `<button class="ghost-icon fb-btn" type="button" title="Correct / comment (saved for training)"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"/></svg></button>` : "";
  wrap.innerHTML = `<div class="avatar">AI</div><div class="assistant-body"><div class="bubble-ai"></div><div class="msg-actions">${feedback}<span class="msg-time"></span></div></div>`;
  return wrap;
}

function attachFeedback(node, question, answer) {
  const btn = node.querySelector(".fb-btn");
  const body = node.querySelector(".assistant-body");
  if (!btn || !body) return;
  btn.addEventListener("click", () => {
    if (body.querySelector(".fb-box")) { body.querySelector(".fb-box").remove(); return; }
    const box = document.createElement("div");
    box.className = "fb-box";
    box.style.cssText = "margin-top:8px;display:flex;flex-direction:column;gap:6px";
    box.innerHTML = `<textarea class="fb-text" rows="2" placeholder="What should be corrected or improved? (saved to your improvement records)" style="width:100%;border-radius:10px;padding:8px;background:var(--surface-input);border:1px solid var(--line);color:inherit;font:inherit"></textarea><div><button type="button" class="primary-button fb-save">Save correction</button></div>`;
    body.appendChild(box);
    const ta = box.querySelector(".fb-text"); ta.focus();
    box.querySelector(".fb-save").addEventListener("click", async () => {
      const feedback = ta.value.trim();
      if (!feedback) return;
      try {
        const res = await postJSON("/api/feedback", { conversation_id: state.currentId, question, answer, feedback });
        box.innerHTML = res.ok ? `<span class="message-status">✓ Saved for training review</span>` : `<span class="message-status">Save failed</span>`;
      } catch { box.innerHTML = `<span class="message-status">Save failed</span>`; }
    });
  });
}
function scrollDown() { els.messages.scrollTop = els.messages.scrollHeight; }

function renderSourceChips(node, sources) {
  if (!sources || !sources.length) return;
  const body = node.querySelector(".assistant-body");
  if (!body || body.querySelector(".source-chips")) return;
  // de-dup by name, cap to keep it tidy
  const seen = new Set();
  const uniq = sources.filter((s) => { const k = (s.name || "") + (s.url || ""); if (seen.has(k)) return false; seen.add(k); return true; }).slice(0, 8);
  const wrap = document.createElement("div");
  wrap.className = "source-chips";
  const label = { upload: "your upload", indexed: "indexed", guidance: "writing guidance", online: "official online" };
  for (const s of uniq) {
    const name = escapeHtml((s.name || "source").replace(/\.pdf$/i, ""));
    const tag = label[s.kind] || s.kind || "source";
    if (s.url) {
      const a = document.createElement("a");
      a.className = "source-chip"; a.href = s.url; a.target = "_blank"; a.rel = "noopener";
      a.innerHTML = `<span class="sc-ico"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 4h10l6 6v10H4z"/><path d="M14 4v6h6"/></svg></span>${name} · ${tag}<span class="sc-ext"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M7 17L17 7M9 7h8v8"/></svg></span>`;
      wrap.appendChild(a);
    } else {
      const span = document.createElement("span");
      span.className = "source-chip";
      span.innerHTML = `<span class="sc-ico"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 4h10l6 6v10H4z"/><path d="M14 4v6h6"/></svg></span>${name} · ${tag}`;
      wrap.appendChild(span);
    }
  }
  body.appendChild(wrap);
  scrollDown();
}

function renderMessages(msgs) {
  els.messages.innerHTML = "";
  if (!msgs.length) { showHero(); return; }
  let lastQuestion = "";
  for (const m of msgs) {
    if (m.role === "user") { lastQuestion = m.content; els.messages.appendChild(userBubble(m.content)); }
    else {
      const node = assistantBubble(state.mode === "memory");
      node.querySelector(".bubble-ai").innerHTML = renderMarkdown(m.content);
      node.querySelector(".msg-time").textContent = timeLabel(m.created_at);
      els.messages.appendChild(node);
      attachFeedback(node, lastQuestion, m.content);
    }
  }
  scrollDown();
}

function renderConversationList() {
  els.list.innerHTML = "";
  for (const c of state.conversations) {
    const row = document.createElement("div");
    row.className = "conversation-row";
    const b = document.createElement("button");
    b.className = "conversation-item" + (c.id === state.currentId ? " active" : "");
    b.dataset.id = c.id;
    const mode = c.mode === "private" ? "private" : "memory";
    b.innerHTML = `<span class="conversation-dot"></span><div class="conversation-main"><span class="conversation-title">${escapeHtml(c.title || "New chat")}</span><span class="conversation-meta"><span class="conversation-mode ${mode}">${mode}</span>${timeLabel(c.updated_at)}</span></div>`;
    b.addEventListener("click", () => openConversation(c.id));
    row.appendChild(b);
    if (mode === "private") {
      const del = document.createElement("button");
      del.className = "conversation-delete";
      del.type = "button";
      del.title = "Permanently delete private chat";
      del.setAttribute("aria-label", `Permanently delete ${c.title || "private chat"}`);
      del.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6M10 10v6M14 10v6"/></svg>`;
      del.addEventListener("click", () => deletePrivateConversation(c));
      row.appendChild(del);
    }
    els.list.appendChild(row);
  }
}

async function deletePrivateConversation(conversation) {
  if (!window.confirm(`Permanently delete “${conversation.title || "Private chat"}”?\n\nMessages and private uploads will be removed and cannot be recovered.`)) return;
  const response = await fetch(`/api/conversations/${conversation.id}`, { method: "DELETE" });
  if (!response.ok) return;
  if (state.currentId === conversation.id) {
    state.currentId = null;
    clearDocsBar();
    showHero();
  }
  await refreshConversations();
}

// ---------- data flow ----------
async function refreshConversations() {
  const { conversations } = await getJSON("/api/conversations");
  state.conversations = conversations || [];
  renderConversationList();
}

async function openConversation(id) {
  const conversation = state.conversations.find((c) => c.id === id);
  if (conversation) chooseMode(conversation.mode === "private" ? "private" : "memory", false);
  state.currentId = id;
  clearDocsBar();
  renderConversationList();
  const { messages } = await getJSON(`/api/conversations/${id}`);
  renderMessages(messages || []);
}

function clearDocsBar() {
  if (els.docsBar) { els.docsBar.hidden = true; els.docsBar.innerHTML = ""; }
}

function newChat() {
  state.currentId = null;
  clearDocsBar();
  renderConversationList();
  showHero();
  els.input.focus();
}

async function sendMessage(text) {
  if (state.streaming || !state.ready) return;
  // Create a conversation on first message so the sidebar has no empty stubs.
  if (!state.currentId) {
    const conv = await postJSON("/api/conversations", { jurisdiction: els.juris.value, mode: state.mode });
    state.currentId = conv.id;
  }
  // first message of a fresh chat? clear hero
  if (els.messages.querySelector(".hero")) els.messages.innerHTML = "";

  els.messages.appendChild(userBubble(text));
  const node = assistantBubble(state.mode === "memory");
  els.messages.appendChild(node);
  const body = node.querySelector(".bubble-ai");
  body.innerHTML = `<span class="message-status">Thinking…</span>`;
  scrollDown();

  state.streaming = true;
  syncSend();
  let acc = "";
  let terminalError = "";
  try {
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ conversation_id: state.currentId, message: text, jurisdiction: els.juris.value, online_mode: els.onlineMode ? els.onlineMode.value : "always" }),
    });
    if (!resp.ok) {
      let detail = `Request failed (${resp.status})`;
      try { detail = (await resp.json()).error || detail; } catch (_) {}
      throw new Error(detail);
    }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split("\n\n");
      buffer = events.pop();
      for (const ev of events) {
        const line = ev.split("\n").find((l) => l.startsWith("data: "));
        if (!line) continue;
        const data = JSON.parse(line.slice(6));
        if (data.status && !acc) { body.innerHTML = `<span class="message-status">${escapeHtml(data.status)}</span>`; scrollDown(); }
        else if (data.replace !== undefined) { acc = String(data.replace || ""); body.innerHTML = renderMarkdown(acc); scrollDown(); }
        else if (data.delta) { acc += data.delta; body.innerHTML = renderMarkdown(acc); scrollDown(); }
        else if (data.sources) { renderSourceChips(node, data.sources); }
        else if (data.error) {
          terminalError = String(data.error);
          acc = "";
          body.innerHTML = `<p style="color:var(--danger,#f87171)">${escapeHtml(terminalError)}</p>`;
        }
      }
    }
  } catch (err) {
    const message = err && err.message ? err.message : String(err);
    terminalError = message;
    acc = "";
    body.innerHTML = `<p style="color:var(--danger,#f87171)">${escapeHtml(message)}</p>`;
  } finally {
    if (!terminalError && !acc.trim() && state.currentId) {
      // A very large atomic SSE event can be lost if the connection closes at
      // exactly the wrong moment. The server saves only complete supervised
      // answers, so recover that durable copy instead of leaving "Thinking…"
      // or showing an abandoned fragment.
      try {
        const saved = await getJSON(`/api/conversations/${state.currentId}`);
        const messages = saved.messages || [];
        const latest = messages.length ? messages[messages.length - 1] : null;
        if (latest && latest.role === "assistant" && latest.content) {
          acc = latest.content;
          body.innerHTML = renderMarkdown(acc);
        } else {
          terminalError = "No complete answer was saved. Please retry.";
          body.innerHTML = `<p style="color:var(--danger,#f87171)">${escapeHtml(terminalError)}</p>`;
        }
      } catch (_) {
        terminalError = "The completed answer could not be recovered. Please reopen this chat or retry.";
        body.innerHTML = `<p style="color:var(--danger,#f87171)">${escapeHtml(terminalError)}</p>`;
      }
    }
    if (!terminalError && acc.trim()) body.innerHTML = renderMarkdown(acc);
    node.querySelector(".msg-time").textContent = timeLabel();
    if (acc.trim()) attachFeedback(node, text, acc);
    state.streaming = false;
    syncSend();
    refreshConversations();
  }
}

// ---------- composer ----------
function autoGrow() {
  els.input.style.height = "auto";
  els.input.style.height = Math.min(els.input.scrollHeight, 200) + "px";
}
function syncSend() {
  els.send.disabled = state.streaming || !state.ready || !els.input.value.trim();
}

els.form.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = els.input.value.trim();
  if (!text) return;
  els.input.value = "";
  autoGrow();
  syncSend();
  sendMessage(text);
});
els.input.addEventListener("input", () => { autoGrow(); syncSend(); });
els.input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); els.form.requestSubmit(); }
});
els.newChat.addEventListener("click", newChat);
for (const button of els.modeButtons) {
  button.addEventListener("click", () => chooseMode(button.dataset.mode));
}

// ---------- uploads (saved into today's improvement-record folder) ----------
function readAsBase64(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result).split(",")[1] || "");
    r.onerror = reject;
    r.readAsDataURL(file);
  });
}
function showUploadChip(name, readable, duplicate) {
  els.docsBar.hidden = false;
  const chip = document.createElement("span");
  chip.className = "source-chip";
  const tag = duplicate ? "already added (duplicate)"
    : readable === false ? "saved (no text read)" : "ready — model can read it";
  chip.innerHTML = `<span class="sc-ico"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 4h10l6 6v10H4z"/><path d="M14 4v6h6"/></svg></span>${escapeHtml(name)} · ${tag}`;
  els.docsBar.appendChild(chip);
}
els.attach.addEventListener("click", () => els.fileInput.click());
els.fileInput.addEventListener("change", async () => {
  // Ensure a conversation exists so uploads attach to it (and the model can read them).
  if (!state.currentId) {
    const conv = await postJSON("/api/conversations", { jurisdiction: els.juris.value, mode: state.mode });
    state.currentId = conv.id;
  }
  for (const file of els.fileInput.files) {
    try {
      const content_b64 = await readAsBase64(file);
      const res = await postJSON("/api/upload", {
        filename: file.name, content_b64, conversation_id: state.currentId,
      });
      if (res.ok) showUploadChip(res.saved || file.name, res.readable, res.duplicate);
    } catch (err) { console.error("upload failed", err); }
  }
  els.fileInput.value = "";
});

document.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "n") { e.preventDefault(); newChat(); }
});

// ---------- health / boot ----------
async function pollHealth() {
  try {
    const h = await getJSON("/api/health");
    if (h.ready) {
      state.ready = true;
      els.statusPill.classList.add("active");
      els.statusText.textContent = "Local model ready";
      if (h.adapter) els.modelBadge.textContent = String(h.adapter).includes("v7_marked_gold") ? "v7 gold trained" : "fine-tuned";
      syncSend();
      return;
    }
    if (h.error) {
      els.statusText.textContent = "Model error";
      els.statusPill.title = h.error;
      return;
    }
  } catch { /* server warming up */ }
  setTimeout(pollHealth, 1500);
}

(async function boot() {
  updateModeUI();
  showHero();
  await refreshConversations();
  pollHealth();
})();

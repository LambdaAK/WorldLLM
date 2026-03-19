// Client-side conversation state
let messages = [];
let isStreaming = false;
let currentChatId = null;

const STORAGE_KEY = "tinygpt-chats";
const MAX_CHATS = 50;

// DOM refs
const messagesEl = document.getElementById("messages");
const emptyState = document.getElementById("emptyState");
const userInput = document.getElementById("userInput");
const sendBtn = document.getElementById("sendBtn");
const newChatBtn = document.getElementById("newChatBtn");
const settingsToggle = document.getElementById("settingsToggle");
const settingsBody = document.getElementById("settingsBody");
const chevron = document.getElementById("chevron");
const tempSlider = document.getElementById("temperature");
const topkSlider = document.getElementById("topk");
const tempVal = document.getElementById("tempVal");
const topkVal = document.getElementById("topkVal");
const modelInfo = document.getElementById("modelInfo");
const chatHistoryEl = document.getElementById("chatHistory");

// --- localStorage helpers ---
function loadChats() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveChats(chats) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(chats.slice(-MAX_CHATS)));
  } catch (_) {}
}

function generateId() {
  return "chat_" + Date.now() + "_" + Math.random().toString(36).slice(2, 9);
}

function getChatTitle(msgs) {
  const first = msgs.find((m) => m.role === "user");
  if (!first) return "New chat";
  const text = first.content.trim();
  return text.length > 36 ? text.slice(0, 36) + "…" : text;
}

function persistCurrentChat() {
  if (messages.length === 0) return;
  const chats = loadChats();
  const title = getChatTitle(messages);
  const payload = { id: currentChatId || generateId(), title, messages: [...messages], createdAt: Date.now() };

  if (currentChatId) {
    const idx = chats.findIndex((c) => c.id === currentChatId);
    if (idx >= 0) {
      chats[idx] = payload;
    } else {
      chats.push(payload);
    }
  } else {
    currentChatId = payload.id;
    chats.push(payload);
  }
  saveChats(chats);
  renderChatHistory();
}

function renderChatHistory() {
  if (!chatHistoryEl) return;
  const chats = loadChats();
  chatHistoryEl.innerHTML = chats
    .slice()
    .reverse()
    .map(
      (c) =>
        `<div class="chat-history-item ${c.id === currentChatId ? "active" : ""}" data-id="${c.id}">
          <button type="button" class="chat-history-title">${escapeHtml(c.title)}</button>
          <button type="button" class="chat-history-delete" title="Delete chat" aria-label="Delete chat">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 6L6 18M6 6l12 12"/></svg>
          </button>
        </div>`
    )
    .join("");

  chatHistoryEl.querySelectorAll(".chat-history-item").forEach((item) => {
    const id = item.dataset.id;
    item.querySelector(".chat-history-title").addEventListener("click", () => loadChat(id));
    item.querySelector(".chat-history-delete").addEventListener("click", (e) => {
      e.stopPropagation();
      deleteChat(id);
    });
  });
}

function deleteChat(id) {
  if (isStreaming) return;
  const chats = loadChats().filter((c) => c.id !== id);
  saveChats(chats);

  if (currentChatId === id) {
    currentChatId = null;
    messages = [];
    messagesEl.innerHTML = "";
    messagesEl.appendChild(emptyState);
    emptyState.style.display = "";
  }

  renderChatHistory();
}

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

function loadChat(id) {
  if (isStreaming) return;
  const chats = loadChats();
  const chat = chats.find((c) => c.id === id);
  if (!chat) return;

  currentChatId = chat.id;
  messages = [...chat.messages];

  messagesEl.innerHTML = "";
  emptyState.style.display = "none";

  messages.forEach((m) => {
    if (m.role === "user") {
      appendUserBubble(m.content, false);
    } else {
      appendAssistantBubble(m.content, false);
    }
  });

  scrollToBottom();
  renderChatHistory();
}

// --- Message UI ---
function appendUserBubble(text, animate = true) {
  const row = document.createElement("div");
  row.className = "message-row user";
  const bubble = document.createElement("div");
  bubble.className = "user-bubble";
  bubble.textContent = text;
  row.appendChild(bubble);
  messagesEl.appendChild(row);
  if (animate) scrollToBottom();
}

function appendAssistantBubble(text, withActions = true) {
  const row = document.createElement("div");
  row.className = "message-row assistant";
  const wrap = document.createElement("div");
  wrap.className = "assistant-message-wrap";
  const content = document.createElement("div");
  content.className = "assistant-content";
  content.textContent = text;
  wrap.appendChild(content);

  if (withActions) {
    const actions = document.createElement("div");
    actions.className = "message-actions";
    actions.innerHTML = `
      <button type="button" class="msg-action-btn copy-btn" title="Copy">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
        Copy
      </button>
      <button type="button" class="msg-action-btn regenerate-btn" title="Regenerate">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/></svg>
        Regenerate
      </button>
    `;
    actions.querySelector(".copy-btn").addEventListener("click", (e) => copyToClipboard(text, e.currentTarget));
    actions.querySelector(".regenerate-btn").addEventListener("click", () => regenerateFrom(row));
    wrap.appendChild(actions);
  }

  row.appendChild(wrap);
  messagesEl.appendChild(row);
  scrollToBottom();
}

function addMessageActions(row, text, regenerateOnly = false) {
  const wrap = row.querySelector(".assistant-message-wrap");
  const actions = document.createElement("div");
  actions.className = "message-actions";
  actions.innerHTML = regenerateOnly
    ? `
      <button type="button" class="msg-action-btn regenerate-btn" title="Regenerate">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/></svg>
        Regenerate
      </button>
    `
    : `
      <button type="button" class="msg-action-btn copy-btn" title="Copy">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
        Copy
      </button>
      <button type="button" class="msg-action-btn regenerate-btn" title="Regenerate">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/></svg>
        Regenerate
      </button>
    `;
  const copyBtn = actions.querySelector(".copy-btn");
  if (copyBtn) copyBtn.addEventListener("click", (e) => copyToClipboard(text, e.currentTarget));
  actions.querySelector(".regenerate-btn").addEventListener("click", () => regenerateFrom(row));
  wrap.appendChild(actions);
}

function copyToClipboard(text, btn) {
  navigator.clipboard.writeText(text).then(() => {
    if (btn) {
      const orig = btn.innerHTML;
      btn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 6L9 17l-5-5"/></svg> Copied!';
      setTimeout(() => { btn.innerHTML = orig; }, 1500);
    }
  });
}

function regenerateFrom(assistantRow) {
  if (isStreaming) return;
  const rows = [...messagesEl.querySelectorAll(".message-row")];
  const idx = rows.indexOf(assistantRow);
  if (idx < 0) return;

  const userRow = rows[idx - 1];
  if (!userRow || !userRow.classList.contains("user")) return;

  const userText = userRow.querySelector(".user-bubble")?.textContent;
  if (!userText) return;

  assistantRow.remove();
  userRow.remove();
  messages.pop();
  messages.pop();

  messages.push({ role: "user", content: userText });
  appendUserBubble(userText);
  streamResponse();
}

function appendAssistantRow() {
  const row = document.createElement("div");
  row.className = "message-row assistant";
  const wrap = document.createElement("div");
  wrap.className = "assistant-message-wrap";
  const content = document.createElement("div");
  content.className = "assistant-content";

  const loading = document.createElement("div");
  loading.className = "loading-dots";
  loading.innerHTML = "<span></span><span></span><span></span>";
  content.appendChild(loading);

  wrap.appendChild(content);
  row.appendChild(wrap);
  messagesEl.appendChild(row);
  scrollToBottom();
  return { row, content, loading };
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

// --- Send / Stream ---
function sendMessage() {
  const text = userInput.value.trim();
  if (!text || isStreaming) return;

  emptyState.style.display = "none";
  messages.push({ role: "user", content: text });
  appendUserBubble(text);

  userInput.value = "";
  userInput.style.height = "auto";
  sendBtn.disabled = true;

  streamResponse();
}

function sendExample(text) {
  userInput.value = text;
  userInput.style.height = "auto";
  sendBtn.disabled = false;
  sendMessage();
}

async function streamResponse() {
  isStreaming = true;
  const { row, content, loading } = appendAssistantRow();
  let responseText = "";
  let isError = false;

  try {
    const resp = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages,
        temperature: parseFloat(tempSlider.value),
        top_k: parseInt(topkSlider.value),
      }),
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      loading.remove();
      content.textContent = "[Error: " + (err.error || resp.statusText) + "]";
      content.style.color = "#e55";
      addMessageActions(row, content.textContent, true);
      isError = true;
      return;
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buf += decoder.decode(value, { stream: true });
      const parts = buf.split("\n\n");
      buf = parts.pop();

      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith("data: ")) continue;
        const token = line.slice(6);
        if (token === "[DONE]") break;

        responseText += token;
        loading.remove();
        content.querySelectorAll(".cursor").forEach((c) => c.remove());
        content.textContent = responseText;
        const cursor = document.createElement("span");
        cursor.className = "cursor";
        content.appendChild(cursor);
        scrollToBottom();
      }
    }
  } catch (err) {
    loading.remove();
    content.textContent = "[Connection error]";
    content.style.color = "#e55";
    addMessageActions(row, content.textContent, true);
    isError = true;
    return;
  } finally {
    if (isError) {
      isStreaming = false;
      sendBtn.disabled = !userInput.value.trim();
      return;
    }
    loading.remove();
    const finalText = responseText || "...";
    if (!responseText) {
      content.textContent = "...";
      content.style.color = "var(--text-muted)";
    } else {
      content.textContent = finalText;
      content.querySelectorAll(".cursor").forEach((c) => c.remove());
    }

    messages.push({ role: "assistant", content: finalText });
    addMessageActions(row, finalText, false);
    persistCurrentChat();
    isStreaming = false;
    sendBtn.disabled = !userInput.value.trim();
  }
}

// --- Init ---
function init() {
  if (!messagesEl || !userInput || !sendBtn) return;
  renderChatHistory();

  fetch("/info")
  .then((r) => r.json())
  .then((data) => {
    if (data.checkpoint && modelInfo) {
      modelInfo.innerHTML = `
        <span class="model-name">${data.checkpoint}</span>
        <span style="color:var(--text-dim);font-size:11px;display:block">
          epoch ${data.epoch} · loss ${data.val_loss} · ${(data.parameters / 1000).toFixed(0)}K params
        </span>`;
    }
  })
  .catch(() => {});

  settingsToggle?.addEventListener("click", () => {
  const open = settingsBody.classList.toggle("open");
  chevron.classList.toggle("open", open);
});

  tempSlider?.addEventListener("input", () => { tempVal.textContent = tempSlider.value; });
  topkSlider?.addEventListener("input", () => { topkVal.textContent = topkSlider.value; });

  newChatBtn?.addEventListener("click", () => {
  if (isStreaming) return;
  persistCurrentChat();
  currentChatId = null;
  messages = [];
  messagesEl.innerHTML = "";
  messagesEl.appendChild(emptyState);
  emptyState.style.display = "";
  renderChatHistory();
});

  userInput?.addEventListener("input", () => {
  userInput.style.height = "auto";
  userInput.style.height = Math.min(userInput.scrollHeight, 200) + "px";
  sendBtn.disabled = !userInput.value.trim() || isStreaming;
});

  userInput?.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    if (!sendBtn.disabled) sendMessage();
  }
});

  sendBtn?.addEventListener("click", sendMessage);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => { try { init(); } catch (e) { console.error("TinyGPT init error:", e); } });
} else {
  try { init(); } catch (e) { console.error("TinyGPT init error:", e); }
}

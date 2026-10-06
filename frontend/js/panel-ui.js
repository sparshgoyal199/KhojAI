export function initSideTabs() {
  const tabs = document.querySelectorAll(".side-tab");
  const panels = document.querySelectorAll(".side-content");

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const target = tab.dataset.tab;
      tabs.forEach((item) => item.classList.toggle("active", item === tab));
      panels.forEach((panel) => {
        panel.classList.toggle("active", panel.dataset.panel === target);
      });
    });
  });
}

export function scrollToBottom(container) {
  container.scrollTop = container.scrollHeight;
}

export function clearChatEmpty(container) {
  const empty = container.querySelector(".chat-empty");
  if (empty) empty.remove();
}

export function renderChatEmpty(container, suggestions = []) {
  container.innerHTML = `
    <div class="chat-empty">
      <div class="chat-empty-icon">✦</div>
      <h4>Ask anything about this PDF</h4>
      <p>KhojAI reads the document and answers with grounded citations.</p>
      ${suggestions.length ? `<div class="chat-suggestions">${suggestions.map((text) => `<button type="button" class="chat-suggestion" data-text="${escapeAttr(text)}">${escapeHtml(text)}</button>`).join("")}</div>` : ""}
    </div>
  `;
  container.querySelectorAll(".chat-suggestion").forEach((button) => {
    button.addEventListener("click", () => {
      const input = document.getElementById("chat-input");
      input.value = button.dataset.text;
      input.focus();
      autoResizeTextarea(input);
    });
  });
}

export function appendChatMessage(container, { role, author, content, streaming = false }) {
  clearChatEmpty(container);

  const row = document.createElement("div");
  row.className = `chat-row ${role === "user" ? "is-user" : "is-ai"}`;

  const avatar = document.createElement("div");
  avatar.className = "chat-avatar";
  avatar.textContent = role === "user" ? author.slice(0, 1).toUpperCase() : "K";

  const bubble = document.createElement("div");
  bubble.className = "chat-bubble";

  const meta = document.createElement("div");
  meta.className = "chat-meta";
  meta.textContent = author;

  const body = document.createElement("div");
  body.className = "chat-text";
  if (streaming && !content) {
    body.classList.add("is-streaming");
    body.innerHTML = `<span class="typing-dots"><span></span><span></span><span></span></span>`;
  } else {
    body.textContent = content;
  }

  bubble.append(meta, body);
  row.append(role === "user" ? bubble : avatar, role === "user" ? avatar : bubble);
  container.appendChild(row);
  scrollToBottom(container);
  return body;
}

export function updateChatStreamText(node, text) {
  node.classList.add("is-streaming");
  node.classList.remove("is-empty");
  node.textContent = text;
  const container = node.closest(".chat-messages");
  if (container) scrollToBottom(container);
}

export function finishChatStream(node, text) {
  node.classList.remove("is-streaming");
  node.textContent = text;
  const container = node.closest(".chat-messages");
  if (container) scrollToBottom(container);
}

export function appendComment(container, author, content) {
  const empty = container.querySelector(".comments-empty");
  if (empty) empty.remove();

  const item = document.createElement("article");
  item.className = "comment-card";
  item.innerHTML = `
    <div class="comment-head">
      <span class="comment-avatar">${escapeHtml(author.slice(0, 1).toUpperCase())}</span>
      <strong>${escapeHtml(author)}</strong>
    </div>
    <p>${escapeHtml(content)}</p>
  `;
  container.appendChild(item);
  scrollToBottom(container);
}

export function renderCommentsEmpty(container) {
  container.innerHTML = `<p class="comments-empty">No comments yet. Share a note with collaborators.</p>`;
}

export function setupChatComposer({ onSend }) {
  const input = document.getElementById("chat-input");
  const button = document.getElementById("chat-btn");

  autoResizeTextarea(input);

  input.addEventListener("input", () => autoResizeTextarea(input));

  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      button.click();
    }
  });

  button.addEventListener("click", async () => {
    const text = input.value.trim();
    if (!text) return;
    input.value = "";
    autoResizeTextarea(input);
    button.disabled = true;
    try {
      await onSend(text);
    } finally {
      button.disabled = false;
      input.focus();
    }
  });
}

function autoResizeTextarea(textarea) {
  textarea.style.height = "auto";
  textarea.style.height = `${Math.min(textarea.scrollHeight, 140)}px`;
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function escapeAttr(value) {
  return escapeHtml(value).replaceAll("'", "&#39;");
}

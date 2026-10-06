import { api, streamChat } from "./api.js";
import {
  initSideTabs,
  appendChatMessage,
  appendComment,
  renderChatEmpty,
  renderCommentsEmpty,
  setupChatComposer,
  updateChatStreamText,
  finishChatStream,
} from "./panel-ui.js";

const shareToken = decodeURIComponent(window.location.pathname.split("/").pop());
const commentsEl = document.getElementById("comments");
const chatEl = document.getElementById("chat");
const modal = document.getElementById("name-modal");
const storageKey = `khojai-guest-${shareToken}`;

const CHAT_SUGGESTIONS = [
  "What is this document about?",
  "Summarize the key findings",
  "What are the main sections?",
];

initSideTabs();

function getGuestName() {
  return sessionStorage.getItem(storageKey);
}

async function loadPdf() {
  const pdf = await api.getSharedPdf(shareToken);
  document.getElementById("filename").textContent = pdf.filename;
  document.getElementById("summary").textContent = pdf.summary || "No summary stored for this file.";
  document.title = `${pdf.filename} — Shared`;
  if (pdf.file_url) document.getElementById("pdf-frame").src = pdf.file_url;
}

async function loadComments() {
  commentsEl.innerHTML = "";
  const comments = await api.listGuestComments(shareToken);
  if (!comments.length) {
    renderCommentsEmpty(commentsEl);
    return;
  }
  for (const comment of comments) {
    appendComment(commentsEl, comment.guest_name || "Owner", comment.content);
  }
}

async function loadChat() {
  chatEl.innerHTML = "";
  const history = await api.guestChatHistory(shareToken, getGuestName());
  if (!history.messages.length) {
    renderChatEmpty(chatEl, CHAT_SUGGESTIONS);
    return;
  }
  for (const message of history.messages) {
    const mine = message.role === "user";
    appendChatMessage(chatEl, {
      role: mine ? "user" : "assistant",
      author: mine ? getGuestName() : "KhojAI",
      content: message.content,
    });
  }
}

document.getElementById("guest-save").addEventListener("click", async () => {
  const name = document.getElementById("guest-name").value.trim();
  const errorEl = document.getElementById("guest-error");
  errorEl.textContent = "";
  if (!name) {
    errorEl.textContent = "Name is required";
    return;
  }
  sessionStorage.setItem(storageKey, name);
  modal.classList.add("hidden");
  try {
    await Promise.all([loadPdf(), loadComments(), loadChat()]);
  } catch (err) {
    document.getElementById("summary").textContent = err.message;
  }
});

document.getElementById("comment-btn").addEventListener("click", async () => {
  const content = document.getElementById("comment-input").value.trim();
  if (!content) return;
  await api.addGuestComment(shareToken, content, getGuestName());
  document.getElementById("comment-input").value = "";
  await loadComments();
});

setupChatComposer({
  onSend: async (query) => {
    appendChatMessage(chatEl, { role: "user", author: getGuestName(), content: query });
    const answerNode = appendChatMessage(chatEl, {
      role: "assistant",
      author: "KhojAI",
      content: "",
      streaming: true,
    });
    try {
      const full = await streamChat({
        path: `/shared/${shareToken}/query`,
        json: { query, guest_name: getGuestName() },
        auth: false,
        onDelta: (text) => updateChatStreamText(answerNode, text),
      });
      finishChatStream(answerNode, full || "No answer returned.");
    } catch (err) {
      finishChatStream(answerNode, err.message);
    }
  },
});

if (getGuestName()) {
  modal.classList.add("hidden");
  Promise.all([loadPdf(), loadComments(), loadChat()]).catch((err) => {
    document.getElementById("summary").textContent = err.message;
  });
}

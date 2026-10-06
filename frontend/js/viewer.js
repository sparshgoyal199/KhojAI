import { api, requireAuth, streamChat } from "./api.js";
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

if (!requireAuth()) throw new Error("redirect");

const pdfId = decodeURIComponent(window.location.pathname.split("/").pop());
const commentsEl = document.getElementById("comments");
const chatEl = document.getElementById("chat");
const toast = document.getElementById("toast");

const CHAT_SUGGESTIONS = [
  "What is this document about?",
  "Summarize the key findings",
  "What are the main sections?",
];

initSideTabs();

function showToast(message) {
  toast.textContent = message;
  toast.classList.remove("hidden");
  setTimeout(() => toast.classList.add("hidden"), 2200);
}

async function loadPdf() {
  const pdf = await api.getPdf(pdfId);
  document.getElementById("filename").textContent = pdf.filename;
  document.getElementById("summary").textContent = pdf.summary || "No summary stored for this file.";
  document.title = `${pdf.filename} — KhojAI`;
  if (pdf.file_url) document.getElementById("pdf-frame").src = pdf.file_url;
}

async function loadComments() {
  commentsEl.innerHTML = "";
  const comments = await api.listComments(pdfId);
  if (!comments.length) {
    renderCommentsEmpty(commentsEl);
    return;
  }
  for (const comment of comments) {
    appendComment(commentsEl, comment.guest_name || "You", comment.content);
  }
}

async function loadChat() {
  chatEl.innerHTML = "";
  const history = await api.chatHistory(pdfId);
  if (!history.messages.length) {
    renderChatEmpty(chatEl, CHAT_SUGGESTIONS);
    return;
  }
  for (const message of history.messages) {
    appendChatMessage(chatEl, {
      role: message.role,
      author: message.role === "user" ? "You" : "KhojAI",
      content: message.content,
    });
  }
}

document.getElementById("comment-btn").addEventListener("click", async () => {
  const content = document.getElementById("comment-input").value.trim();
  if (!content) return;
  await api.addComment(pdfId, content);
  document.getElementById("comment-input").value = "";
  await loadComments();
});

setupChatComposer({
  onSend: async (query) => {
    appendChatMessage(chatEl, { role: "user", author: "You", content: query });
    const answerNode = appendChatMessage(chatEl, {
      role: "assistant",
      author: "KhojAI",
      content: "",
      streaming: true,
    });
    try {
      const full = await streamChat({
        path: `/pdfs/${pdfId}/query`,
        json: { query, guest_name: null },
        onDelta: (text) => updateChatStreamText(answerNode, text),
      });
      finishChatStream(answerNode, full || "No answer returned.");
    } catch (err) {
      finishChatStream(answerNode, err.message);
    }
  },
});

document.getElementById("share-btn").addEventListener("click", async () => {
  const shared = await api.sharePdf(pdfId);
  await navigator.clipboard.writeText(shared.share_url);
  showToast("Share link copied");
});

Promise.all([loadPdf(), loadComments(), loadChat()]).catch((err) => {
  document.getElementById("summary").textContent = err.message;
});

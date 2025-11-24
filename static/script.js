// render_app/static/script.js
const chatWindow = document.getElementById("chat-window");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");

function escapeHtml(unsafe) {
  return unsafe
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function appendMessage(content, classes="message user-message") {
  const div = document.createElement("div");
  div.className = classes;
  div.innerHTML = `<div class="message-text">${content}</div>`;
  chatWindow.appendChild(div);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

async function sendMessage(e){
  if (e) e.preventDefault();
  const text = chatInput.value.trim();
  if (!text) return false;
  appendMessage(escapeHtml(text), "message user-message");
  chatInput.value = "";
  appendMessage('<span class="message-text">Thinking…</span>', "message bot-message thinking");
  try {
    const form = new URLSearchParams();
    form.append("msg", text);
    const res = await fetch("/get", { method: "POST", body: form });
    const reply = await res.text();
    const thinking = document.querySelector(".message.bot-message.thinking");
    if (thinking) thinking.remove();
    appendMessage(escapeHtml(reply), "message bot-message");
  } catch (err) {
    const thinking = document.querySelector(".message.bot-message.thinking");
    if (thinking) thinking.remove();
    appendMessage("Error: unable to get response.", "message bot-message");
    console.error(err);
  }
  return false;
}

// Submit with Enter (no Shift)
chatInput.addEventListener("keydown", function(e){
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    chatForm.requestSubmit();
  }
});

// Form submit
chatForm.addEventListener("submit", sendMessage);

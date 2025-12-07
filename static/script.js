// static/script.js
// small helpers used by templates

// escape HTML
function escapeHtml(text) {
  if (!text) return "";
  return text.replace(/[&<>'"]/g, function (c) {
    return {'&':'&amp;','<':'&lt;','>':'&gt;','\'':'&#39;','"':'&quot;'}[c];
  });
}

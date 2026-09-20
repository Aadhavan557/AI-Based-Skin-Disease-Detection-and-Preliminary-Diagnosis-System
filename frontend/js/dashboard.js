// ─────────────────────────────────────────────────────────
//  DermaAI — Dashboard Page Logic
// ─────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", async () => {
  if (!Auth.requireAuth()) return;

  const user = Auth.getUser();
  if (user) {
    document.querySelectorAll(".user-display-name").forEach(el => el.textContent = user.username);
    document.querySelectorAll(".user-avatar-initial").forEach(el => el.textContent = user.username.charAt(0).toUpperCase());
    document.querySelectorAll(".user-email").forEach(el => el.textContent = user.email || "");
  }

  // Load history stats
  const history = await API.getHistory();
  const totalEl     = document.getElementById("stat-total");
  const recentEl    = document.getElementById("stat-recent");
  const avgConfEl   = document.getElementById("stat-avg-conf");
  const reportsEl   = document.getElementById("stat-reports");

  if (totalEl) totalEl.textContent = history.length;
  if (recentEl && history.length > 0) {
    recentEl.textContent = history[0].predicted_class || "–";
  }
  if (avgConfEl && history.length > 0) {
    const avg = history.reduce((s, p) => s + (p.confidence || 0), 0) / history.length;
    avgConfEl.textContent = (avg * 100).toFixed(1) + "%";
  }
  if (reportsEl) reportsEl.textContent = history.length;

  // Recent analyses table
  const recentTable = document.getElementById("recent-analyses-body");
  if (recentTable) {
    if (history.length === 0) {
      recentTable.innerHTML = `
        <tr>
          <td colspan="5" style="text-align:center; padding:40px; color:var(--text-muted)">
            <div style="font-size:2.5rem; margin-bottom:12px">🔬</div>
            <div style="font-weight:600; margin-bottom:8px">No analyses yet</div>
            <div>Start by uploading your first skin image</div>
          </td>
        </tr>`;
    } else {
      recentTable.innerHTML = history.slice(0, 5).map(pred => `
        <tr>
          <td>${formatDateShort(pred.created_at)}</td>
          <td>
            ${pred.image_path
              ? `<img class="history-thumb" src="${CONFIG.API_BASE_URL}/${pred.image_path}" onerror="this.parentElement.innerHTML='<div class=history-thumb-placeholder>🖼️</div>'" />`
              : `<div class="history-thumb-placeholder">🖼️</div>`}
          </td>
          <td><strong>${pred.predicted_class || "–"}</strong></td>
          <td>
            <span class="badge ${pred.confidence >= .8 ? "badge-success" : pred.confidence >= .6 ? "badge-warning" : "badge-danger"}">
              ${pred.confidence ? (pred.confidence * 100).toFixed(1) + "%" : "–"}
            </span>
          </td>
          <td>
            <button class="btn btn-sm btn-outline" onclick="viewResult('${pred.id || ""}')">View →</button>
          </td>
        </tr>`).join("");
    }
  }
});

function viewResult(id) {
  const history = JSON.parse(sessionStorage.getItem("dermaai_history") || "[]");
  const pred = history.find(p => p.id === id);
  if (pred) {
    sessionStorage.setItem(CONFIG.RESULT_KEY, JSON.stringify(pred));
    window.location.href = "result.html";
  }
}

// Sidebar toggle for mobile
function toggleSidebar() {
  document.getElementById("sidebar").classList.toggle("open");
}
function closeSidebar() {
  document.getElementById("sidebar").classList.remove("open");
}

// ─────────────────────────────────────────────────────────
//  DermaAI — History Page Logic
// ─────────────────────────────────────────────────────────

let allHistory = [];
let filteredHistory = [];

document.addEventListener("DOMContentLoaded", async () => {
  if (!Auth.requireAuth()) return;
  const user = Auth.getUser();
  if (user) {
    document.querySelectorAll(".user-display-name").forEach(el => el.textContent = user.username);
    document.querySelectorAll(".user-avatar-initial").forEach(el => el.textContent = user.username.charAt(0).toUpperCase());
  }

  await loadHistory();

  // Search
  const searchInput = document.getElementById("history-search");
  if (searchInput) {
    searchInput.addEventListener("input", debounce(() => {
      applyFilters();
    }, 300));
  }

  // Filter
  const filterSelect = document.getElementById("history-filter");
  if (filterSelect) filterSelect.addEventListener("change", applyFilters);

  // Sort
  const sortSelect = document.getElementById("history-sort");
  if (sortSelect) sortSelect.addEventListener("change", applyFilters);
});

async function loadHistory() {
  allHistory = await API.getHistory();
  filteredHistory = [...allHistory];
  renderHistory(filteredHistory);
}

function applyFilters() {
  const query  = (document.getElementById("history-search")?.value || "").toLowerCase();
  const disease = document.getElementById("history-filter")?.value || "";
  const sort    = document.getElementById("history-sort")?.value || "newest";

  filteredHistory = allHistory.filter(pred => {
    const matchQuery   = !query   || (pred.predicted_class || "").toLowerCase().includes(query);
    const matchDisease = !disease || pred.predicted_class === disease;
    return matchQuery && matchDisease;
  });

  if (sort === "newest") filteredHistory.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
  if (sort === "oldest") filteredHistory.sort((a, b) => new Date(a.created_at) - new Date(b.created_at));
  if (sort === "confidence-high") filteredHistory.sort((a, b) => (b.confidence || 0) - (a.confidence || 0));
  if (sort === "confidence-low")  filteredHistory.sort((a, b) => (a.confidence || 0) - (b.confidence || 0));

  renderHistory(filteredHistory);
}

function renderHistory(data) {
  const tbody = document.getElementById("history-body");
  const emptyState = document.getElementById("empty-state");
  const countEl    = document.getElementById("result-count");

  if (countEl) countEl.textContent = data.length;

  if (!data.length) {
    tbody.innerHTML = "";
    if (emptyState) emptyState.style.display = "block";
    return;
  }
  if (emptyState) emptyState.style.display = "none";

  tbody.innerHTML = data.map((pred, i) => {
    const pct = pred.confidence ? (pred.confidence * 100).toFixed(1) : "–";
    const badgeClass = pred.confidence >= .8 ? "badge-success" : pred.confidence >= .6 ? "badge-warning" : "badge-danger";
    return `
      <tr>
        <td style="white-space:nowrap">${formatDateShort(pred.created_at)}</td>
        <td>
          ${pred.image_path
            ? `<img class="history-thumb" src="${CONFIG.API_BASE_URL}/${pred.image_path}" alt="skin image" onerror="this.parentElement.innerHTML='<div class=history-thumb-placeholder>🖼️</div>'" />`
            : `<div class="history-thumb-placeholder">🖼️</div>`}
        </td>
        <td><strong>${pred.predicted_class || "–"}</strong></td>
        <td><span class="badge ${badgeClass}">${pct}%</span></td>
        <td>
          <div style="display:flex; gap:8px">
            <button class="btn btn-sm btn-outline" onclick="viewPrediction(${i})">View</button>
            ${pred.id ? `<button class="btn btn-sm btn-ghost" onclick="downloadReport('${pred.id}')">📄</button>` : ""}
          </div>
        </td>
      </tr>`;
  }).join("");
}

function viewPrediction(index) {
  const pred = filteredHistory[index];
  if (pred) {
    sessionStorage.setItem(CONFIG.RESULT_KEY, JSON.stringify(pred));
    window.location.href = "result.html";
  }
}

async function downloadReport(predId) {
  showToast("Generating PDF report…", "info");
  try {
    const res = await API.get(`${CONFIG.ENDPOINTS.REPORT}/${predId}`);
    if (res.ok) {
      const blob = await res.blob();
      const url  = URL.createObjectURL(blob);
      const a    = document.createElement("a");
      a.href = url; a.download = `DermaAI_Report_${predId}.pdf`;
      a.click(); URL.revokeObjectURL(url);
      showToast("Report downloaded!", "success");
    } else {
      showToast("Could not generate report.", "error");
    }
  } catch {
    showToast("Network error.", "error");
  }
}

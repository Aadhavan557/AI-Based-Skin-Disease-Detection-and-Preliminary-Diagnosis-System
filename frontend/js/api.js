// ─────────────────────────────────────────────────────────
//  DermaAI — Centralized API Utilities
// ─────────────────────────────────────────────────────────

const API = {
  // Authenticated GET
  async get(endpoint) {
    const token = Auth.getToken();
    const res = await fetch(apiUrl(endpoint), {
      method: "GET",
      headers: { Authorization: `Bearer ${token}` },
    });
    if (res.status === 401) { Auth.logout(); throw new Error("Unauthorized"); }
    return res;
  },

  // Authenticated POST JSON
  async post(endpoint, body) {
    const token = Auth.getToken();
    const res = await fetch(apiUrl(endpoint), {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify(body),
    });
    if (res.status === 401) { Auth.logout(); throw new Error("Unauthorized"); }
    return res;
  },

  // Authenticated POST FormData (for file upload)
  async postForm(endpoint, formData) {
    const token = Auth.getToken();
    const res = await fetch(apiUrl(endpoint), {
      method: "POST",
      headers: { Authorization: `Bearer ${token}` },
      body: formData,
    });
    if (res.status === 401) { Auth.logout(); throw new Error("Unauthorized"); }
    return res;
  },

  // Fetch all predictions (history)
  async getHistory() {
    // Backend doesn't have a dedicated /history endpoint yet;
    // we read from sessionStorage predictions array if available,
    // otherwise return an empty array.
    const cached = sessionStorage.getItem("dermaai_history");
    return cached ? JSON.parse(cached) : [];
  },

  // Save a prediction result to local history cache
  saveToHistory(prediction) {
    const history = JSON.parse(sessionStorage.getItem("dermaai_history") || "[]");
    history.unshift(prediction);
    sessionStorage.setItem("dermaai_history", JSON.stringify(history));
  }
};

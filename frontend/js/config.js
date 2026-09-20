// ─────────────────────────────────────────────────────────
//  DermaAI — API Configuration
// ─────────────────────────────────────────────────────────

const CONFIG = {
  API_BASE_URL: "http://127.0.0.1:8000",

  ENDPOINTS: {
    REGISTER:  "/api/auth/register",
    LOGIN:     "/api/auth/login",
    ME:        "/api/user/me",
    PREDICT:   "/api/prediction/upload",
    REPORT:    "/api/report",   // + /{prediction_id}
  },

  // Token storage key
  TOKEN_KEY:   "dermaai_token",
  USER_KEY:    "dermaai_user",
  RESULT_KEY:  "dermaai_result",
};

// Helper: Full URL
function apiUrl(endpoint) {
  return CONFIG.API_BASE_URL + endpoint;
}

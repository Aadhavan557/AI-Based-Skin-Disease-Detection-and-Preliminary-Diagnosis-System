// ─────────────────────────────────────────────────────────
//  DermaAI — Authentication Module
// ─────────────────────────────────────────────────────────

// ── Token Management ──────────────────────────────────────
const Auth = {
  getToken() {
    return localStorage.getItem(CONFIG.TOKEN_KEY);
  },
  setToken(token) {
    localStorage.setItem(CONFIG.TOKEN_KEY, token);
  },
  getUser() {
    const u = localStorage.getItem(CONFIG.USER_KEY);
    return u ? JSON.parse(u) : null;
  },
  setUser(user) {
    localStorage.setItem(CONFIG.USER_KEY, JSON.stringify(user));
  },
  isLoggedIn() {
    return !!this.getToken();
  },
  logout() {
    localStorage.removeItem(CONFIG.TOKEN_KEY);
    localStorage.removeItem(CONFIG.USER_KEY);
    window.location.href = "login.html";
  },
  // Redirect to login if not authenticated
  requireAuth() {
    if (!this.isLoggedIn()) {
      window.location.href = "login.html";
      return false;
    }
    return true;
  },
  // Redirect to dashboard if already authenticated
  redirectIfLoggedIn() {
    if (this.isLoggedIn()) {
      window.location.href = "dashboard.html";
    }
  }
};

// ── Register ──────────────────────────────────────────────
async function handleRegister(e) {
  e.preventDefault();
  clearAllErrors();

  const username = document.getElementById("username").value.trim();
  const email    = document.getElementById("email").value.trim();
  const password = document.getElementById("password").value;
  const confirm  = document.getElementById("confirm-password").value;

  let valid = true;
  if (!username || username.length < 3) {
    setFieldError("username", "Username must be at least 3 characters.");
    valid = false;
  }
  if (!email || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
    setFieldError("email", "Please enter a valid email address.");
    valid = false;
  }
  if (!password || password.length < 6) {
    setFieldError("password", "Password must be at least 6 characters.");
    valid = false;
  }
  if (password !== confirm) {
    setFieldError("confirm-password", "Passwords do not match.");
    valid = false;
  }
  if (!valid) return;

  const btn = document.getElementById("register-btn");
  setButtonLoading(btn, true, "Creating account...");

  try {
    const res = await fetch(apiUrl(CONFIG.ENDPOINTS.REGISTER), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, email, password }),
    });
    const data = await res.json();

    if (res.ok) {
      showToast("Account created! Redirecting to login…", "success");
      setTimeout(() => window.location.href = "login.html", 1800);
    } else {
      const msg = data.detail || "Registration failed.";
      if (msg.toLowerCase().includes("username")) setFieldError("username", msg);
      else if (msg.toLowerCase().includes("email")) setFieldError("email", msg);
      else showToast(msg, "error");
    }
  } catch (err) {
    showToast("Network error. Please check your connection.", "error");
  } finally {
    setButtonLoading(btn, false);
  }
}

// ── Login ─────────────────────────────────────────────────
async function handleLogin(e) {
  e.preventDefault();
  clearAllErrors();

  const username = document.getElementById("username").value.trim();
  const password = document.getElementById("password").value;

  if (!username) { setFieldError("username", "Username is required."); return; }
  if (!password) { setFieldError("password", "Password is required."); return; }

  const btn = document.getElementById("login-btn");
  setButtonLoading(btn, true, "Signing in...");

  try {
    // Login uses OAuth2 form-data format
    const formData = new URLSearchParams();
    formData.append("username", username);
    formData.append("password", password);

    const res = await fetch(apiUrl(CONFIG.ENDPOINTS.LOGIN), {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: formData,
    });
    const data = await res.json();

    if (res.ok && data.access_token) {
      Auth.setToken(data.access_token);

      // Fetch user profile
      const meRes = await fetch(apiUrl(CONFIG.ENDPOINTS.ME), {
        headers: { Authorization: `Bearer ${data.access_token}` }
      });
      if (meRes.ok) {
        const user = await meRes.json();
        Auth.setUser(user);
      }
      showToast("Welcome back! Redirecting…", "success");
      setTimeout(() => window.location.href = "dashboard.html", 1200);
    } else {
      const msg = data.detail || "Invalid credentials.";
      showToast(msg, "error");
      setFieldError("password", "Incorrect username or password.");
    }
  } catch (err) {
    showToast("Network error. Please check your connection.", "error");
  } finally {
    setButtonLoading(btn, false);
  }
}

// ── Update nav user name (if user-name element exists) ────
function updateNavUser() {
  const user = Auth.getUser();
  const el = document.getElementById("nav-username");
  if (el && user) el.textContent = user.username;
  const avatarEl = document.getElementById("nav-avatar");
  if (avatarEl && user) avatarEl.textContent = user.username.charAt(0).toUpperCase();
}

document.addEventListener("DOMContentLoaded", () => {
  updateNavUser();
  const regForm  = document.getElementById("register-form");
  const loginForm = document.getElementById("login-form");
  if (regForm)  regForm.addEventListener("submit", handleRegister);
  if (loginForm) loginForm.addEventListener("submit", handleLogin);
});

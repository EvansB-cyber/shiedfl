// ============================================================
// ShieldFL — Profile & Settings
// Vanilla JS. Persists to localStorage under 'shieldfl_settings'.
// Replace the persistLocal()/loadLocal() calls with real API
// calls (fetch to your FastAPI backend) when ready.
// ============================================================

const STORAGE_KEY = "shieldfl_settings";

const DEFAULT_STATE = {
  fullName: "",
  email: "",
  role: "admin",
  org: "mtn",
  twoFAEnabled: false,
  notifications: {
    email: true,
    sms: false,
    inApp: true,
    riskThreshold: 70,
    escrowFreq: "instant",
    flComplete: true,
    flFailed: true
  },
  security: {
    ipAllowlist: [],
    apiKeys: []
  },
  system: {
    sensitivity: 3,
    dataRetention: "90"
  },
  regional: {
    currency: "GHS",
    timezone: "Africa/Accra",
    prefixMtn: true,
    prefixTelecel: true,
    prefixAirtelTigo: true
  }
};

// Mock data — replace with real backend responses
const MOCK_SESSIONS = [
  { id: 1, device: "Chrome on Windows", location: "Accra, GH", lastActive: "Active now", current: true },
  { id: 2, device: "Safari on iPhone", location: "Accra, GH", lastActive: "2 hours ago", current: false }
];

const MOCK_AUDIT_LOG = [
  { action: "Released transaction", details: "TXN-88213 approved after review", timestamp: "2026-08-20 14:12" },
  { action: "Updated risk threshold", details: "Changed from 60 to 70", timestamp: "2026-08-19 09:03" },
  { action: "Held transaction", details: "TXN-88190 flagged, sent to escrow", timestamp: "2026-08-18 22:47" }
];

let state = loadLocal();
let sessions = [...MOCK_SESSIONS];
let auditLog = [...MOCK_AUDIT_LOG];

// ------------------------------------------------------------
// PERSISTENCE
// ------------------------------------------------------------
function loadLocal() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return structuredClone(DEFAULT_STATE);
    return { ...structuredClone(DEFAULT_STATE), ...JSON.parse(raw) };
  } catch (e) {
    console.error("Failed to load settings:", e);
    return structuredClone(DEFAULT_STATE);
  }
}

function persistLocal() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    showSavedIndicator();
  } catch (e) {
    console.error("Failed to save settings:", e);
  }
}

function showSavedIndicator() {
  const el = document.getElementById("ps-save-indicator");
  el.classList.remove("hidden");
  el.style.opacity = "1";
  setTimeout(() => {
    el.style.opacity = "0";
    setTimeout(() => el.classList.add("hidden"), 300);
  }, 1500);
}

// ------------------------------------------------------------
// NAV SWITCHING
// ------------------------------------------------------------
function initNav() {
  const links = document.querySelectorAll(".ps-nav-link");
  const sections = document.querySelectorAll(".ps-section");

  links.forEach(link => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      const target = link.dataset.target;

      links.forEach(l => l.classList.remove("active"));
      link.classList.add("active");

      sections.forEach(s => {
        s.classList.toggle("hidden", s.id !== target);
      });
    });
  });
}

// ------------------------------------------------------------
// ACCOUNT & IDENTITY
// ------------------------------------------------------------
function initAccountSection() {
  const fullNameInput = document.getElementById("ps-full-name");
  const emailInput = document.getElementById("ps-email");
  const roleSelect = document.getElementById("ps-role");
  const orgSelect = document.getElementById("ps-org");

  fullNameInput.value = state.fullName;
  emailInput.value = state.email;
  roleSelect.value = state.role;
  orgSelect.value = state.org;

  fullNameInput.addEventListener("input", () => {
    state.fullName = fullNameInput.value;
    updateAvatarInitials();
  });
  emailInput.addEventListener("input", () => { state.email = emailInput.value; });
  roleSelect.addEventListener("change", () => { state.role = roleSelect.value; });
  orgSelect.addEventListener("change", () => { state.org = orgSelect.value; });

  updateAvatarInitials();

  // Avatar upload (preview only — wire to real upload endpoint later)
  const avatarBtn = document.getElementById("ps-avatar-upload-btn");
  const avatarInput = document.getElementById("ps-avatar-input");
  avatarBtn.addEventListener("click", () => avatarInput.click());
  avatarInput.addEventListener("change", () => {
    const file = avatarInput.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      const preview = document.getElementById("ps-avatar-preview");
      preview.innerHTML = `<img src="${e.target.result}" alt="Avatar">`;
    };
    reader.readAsDataURL(file);
  });

  // Password change
  document.getElementById("ps-change-pass-btn").addEventListener("click", () => {
    const current = document.getElementById("ps-current-pass").value;
    const next = document.getElementById("ps-new-pass").value;
    const confirm = document.getElementById("ps-confirm-pass").value;
    const errorEl = document.getElementById("ps-pass-error");

    if (!current || !next) {
      errorEl.textContent = "Please fill in all password fields.";
      errorEl.classList.remove("hidden");
      return;
    }
    if (next !== confirm) {
      errorEl.textContent = "Passwords do not match.";
      errorEl.classList.remove("hidden");
      return;
    }
    if (next.length < 8) {
      errorEl.textContent = "New password must be at least 8 characters.";
      errorEl.classList.remove("hidden");
      return;
    }

    errorEl.classList.add("hidden");
    // TODO: call backend endpoint to actually change the password
    alert("Password updated. (Wire this to your auth endpoint.)");
    document.getElementById("ps-current-pass").value = "";
    document.getElementById("ps-new-pass").value = "";
    document.getElementById("ps-confirm-pass").value = "";
  });

  // 2FA toggle
  const twoFAToggle = document.getElementById("ps-2fa-toggle");
  const twoFAPanel = document.getElementById("ps-2fa-panel");
  twoFAToggle.checked = state.twoFAEnabled;
  twoFAPanel.classList.toggle("hidden", !state.twoFAEnabled);

  twoFAToggle.addEventListener("change", () => {
    twoFAPanel.classList.toggle("hidden", !twoFAToggle.checked);
    if (!twoFAToggle.checked) {
      state.twoFAEnabled = false;
    }
  });

  document.getElementById("ps-2fa-confirm-btn").addEventListener("click", () => {
    const code = document.getElementById("ps-2fa-code").value;
    if (code.length !== 6) {
      alert("Enter the 6-digit code from your authenticator app.");
      return;
    }
    // TODO: verify code against backend
    state.twoFAEnabled = true;
    alert("2FA enabled successfully.");
  });

  renderSessions();
}

function updateAvatarInitials() {
  const preview = document.getElementById("ps-avatar-preview");
  if (preview.querySelector("img")) return; // don't override uploaded photo
  const name = state.fullName.trim();
  const initials = name
    ? name.split(" ").map(p => p[0]).join("").slice(0, 2).toUpperCase()
    : "AR";
  preview.textContent = initials;
}

function renderSessions() {
  const container = document.getElementById("ps-sessions-list");
  container.innerHTML = "";
  sessions.forEach(s => {
    const item = document.createElement("div");
    item.className = "ps-list-item";
    item.innerHTML = `
      <div>
        <div>${s.device}</div>
        <div class="ps-list-item-meta">${s.location} · ${s.lastActive}</div>
      </div>
      ${s.current
        ? `<span class="ps-badge current">This device</span>`
        : `<button class="btn btn-danger" data-session-id="${s.id}">Revoke</button>`
      }
    `;
    container.appendChild(item);
  });

  container.querySelectorAll("button[data-session-id]").forEach(btn => {
    btn.addEventListener("click", () => {
      const id = Number(btn.dataset.sessionId);
      sessions = sessions.filter(s => s.id !== id);
      // TODO: call backend to actually revoke the session/token
      renderSessions();
    });
  });
}

// ------------------------------------------------------------
// NOTIFICATIONS
// ------------------------------------------------------------
function initNotificationsSection() {
  const emailToggle = document.getElementById("ps-notif-email");
  const smsToggle = document.getElementById("ps-notif-sms");
  const appToggle = document.getElementById("ps-notif-app");
  const riskSlider = document.getElementById("ps-risk-threshold");
  const riskValue = document.getElementById("ps-risk-threshold-value");
  const escrowFreq = document.getElementById("ps-escrow-notif-freq");
  const flCompleteToggle = document.getElementById("ps-notif-fl-complete");
  const flFailedToggle = document.getElementById("ps-notif-fl-failed");

  emailToggle.checked = state.notifications.email;
  smsToggle.checked = state.notifications.sms;
  appToggle.checked = state.notifications.inApp;
  riskSlider.value = state.notifications.riskThreshold;
  riskValue.textContent = state.notifications.riskThreshold;
  escrowFreq.value = state.notifications.escrowFreq;
  flCompleteToggle.checked = state.notifications.flComplete;
  flFailedToggle.checked = state.notifications.flFailed;

  emailToggle.addEventListener("change", () => { state.notifications.email = emailToggle.checked; });
  smsToggle.addEventListener("change", () => { state.notifications.sms = smsToggle.checked; });
  appToggle.addEventListener("change", () => { state.notifications.inApp = appToggle.checked; });
  riskSlider.addEventListener("input", () => {
    state.notifications.riskThreshold = Number(riskSlider.value);
    riskValue.textContent = riskSlider.value;
  });
  escrowFreq.addEventListener("change", () => { state.notifications.escrowFreq = escrowFreq.value; });
  flCompleteToggle.addEventListener("change", () => { state.notifications.flComplete = flCompleteToggle.checked; });
  flFailedToggle.addEventListener("change", () => { state.notifications.flFailed = flFailedToggle.checked; });
}

// ------------------------------------------------------------
// SECURITY & ACCESS
// ------------------------------------------------------------
function initSecuritySection() {
  renderApiKeys();
  renderIpList();

  document.getElementById("ps-new-api-key-btn").addEventListener("click", () => {
    const key = {
      id: Date.now(),
      label: `Key ${state.security.apiKeys.length + 1}`,
      value: generateFakeApiKey(),
      created: new Date().toISOString().slice(0, 10)
    };
    state.security.apiKeys.push(key);
    renderApiKeys();
    // TODO: request real key from backend instead of generating client-side
  });

  document.getElementById("ps-add-ip-btn").addEventListener("click", () => {
    const input = document.getElementById("ps-ip-input");
    const ip = input.value.trim();
    if (!ip) return;
    if (!isValidIp(ip)) {
      alert("Enter a valid IPv4 address.");
      return;
    }
    state.security.ipAllowlist.push(ip);
    input.value = "";
    renderIpList();
  });

  renderAuditLog();
}

function generateFakeApiKey() {
  return "sfl_" + Array.from({ length: 24 }, () =>
    "abcdefghijklmnopqrstuvwxyz0123456789"[Math.floor(Math.random() * 36)]
  ).join("");
}

function isValidIp(ip) {
  return /^(\d{1,3}\.){3}\d{1,3}$/.test(ip) && ip.split(".").every(o => Number(o) <= 255);
}

function renderApiKeys() {
  const container = document.getElementById("ps-api-keys-list");
  container.innerHTML = "";
  if (state.security.apiKeys.length === 0) {
    container.innerHTML = `<p class="ps-hint">No API keys yet.</p>`;
    return;
  }
  state.security.apiKeys.forEach(key => {
    const item = document.createElement("div");
    item.className = "ps-list-item";
    item.innerHTML = `
      <div>
        <div>${key.label} — <code>${key.value.slice(0, 10)}••••••••</code></div>
        <div class="ps-list-item-meta">Created ${key.created}</div>
      </div>
      <button class="btn btn-danger" data-key-id="${key.id}">Revoke</button>
    `;
    container.appendChild(item);
  });

  container.querySelectorAll("button[data-key-id]").forEach(btn => {
    btn.addEventListener("click", () => {
      const id = Number(btn.dataset.keyId);
      state.security.apiKeys = state.security.apiKeys.filter(k => k.id !== id);
      renderApiKeys();
    });
  });
}

function renderIpList() {
  const container = document.getElementById("ps-ip-list");
  container.innerHTML = "";
  state.security.ipAllowlist.forEach(ip => {
    const chip = document.createElement("div");
    chip.className = "ps-chip";
    chip.innerHTML = `<span>${ip}</span><button data-ip="${ip}">×</button>`;
    container.appendChild(chip);
  });

  container.querySelectorAll("button[data-ip]").forEach(btn => {
    btn.addEventListener("click", () => {
      state.security.ipAllowlist = state.security.ipAllowlist.filter(ip => ip !== btn.dataset.ip);
      renderIpList();
    });
  });
}

function renderAuditLog() {
  const tbody = document.getElementById("ps-audit-log-body");
  tbody.innerHTML = "";
  auditLog.forEach(entry => {
    const row = document.createElement("tr");
    row.innerHTML = `<td>${entry.action}</td><td>${entry.details}</td><td>${entry.timestamp}</td>`;
    tbody.appendChild(row);
  });
}

// ------------------------------------------------------------
// SYSTEM & MODEL
// ------------------------------------------------------------
const SENSITIVITY_LABELS = { 1: "Very Lenient", 2: "Lenient", 3: "Balanced", 4: "Strict", 5: "Very Strict" };

function initSystemSection() {
  const sensSlider = document.getElementById("ps-sensitivity");
  const sensValue = document.getElementById("ps-sensitivity-value");
  const retentionSelect = document.getElementById("ps-data-retention");

  sensSlider.value = state.system.sensitivity;
  sensValue.textContent = SENSITIVITY_LABELS[state.system.sensitivity];
  retentionSelect.value = state.system.dataRetention;

  sensSlider.addEventListener("input", () => {
    state.system.sensitivity = Number(sensSlider.value);
    sensValue.textContent = SENSITIVITY_LABELS[sensSlider.value];
  });
  retentionSelect.addEventListener("change", () => { state.system.dataRetention = retentionSelect.value; });

  // Aggregation strategy is read-only, fetched from backend in a real deployment.
  // TODO: replace with a fetch() to your FL config endpoint.
  document.getElementById("ps-aggregation-strategy").textContent = "Trimmed Mean";
}

// ------------------------------------------------------------
// REGIONAL & CURRENCY
// ------------------------------------------------------------
function initRegionalSection() {
  const currency = document.getElementById("ps-currency");
  const timezone = document.getElementById("ps-timezone");
  const mtn = document.getElementById("ps-prefix-mtn");
  const telecel = document.getElementById("ps-prefix-telecel");
  const airtelTigo = document.getElementById("ps-prefix-airteltigo");

  currency.value = state.regional.currency;
  timezone.value = state.regional.timezone;
  mtn.checked = state.regional.prefixMtn;
  telecel.checked = state.regional.prefixTelecel;
  airtelTigo.checked = state.regional.prefixAirtelTigo;

  currency.addEventListener("change", () => { state.regional.currency = currency.value; });
  timezone.addEventListener("change", () => { state.regional.timezone = timezone.value; });
  mtn.addEventListener("change", () => { state.regional.prefixMtn = mtn.checked; });
  telecel.addEventListener("change", () => { state.regional.prefixTelecel = telecel.checked; });
  airtelTigo.addEventListener("change", () => { state.regional.prefixAirtelTigo = airtelTigo.checked; });
}

// ------------------------------------------------------------
// SAVE ALL
// ------------------------------------------------------------
function initSaveButton() {
  document.getElementById("ps-save-all").addEventListener("click", () => {
    persistLocal();
    // TODO: also POST `state` to your backend, e.g.:
    // fetch('/api/settings', { method: 'PUT', headers: {'Content-Type':'application/json'}, body: JSON.stringify(state) })
  });
}

// ------------------------------------------------------------
// INIT
// ------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  initNav();
  initAccountSection();
  initNotificationsSection();
  initSecuritySection();
  initSystemSection();
  initRegionalSection();
  initSaveButton();
});

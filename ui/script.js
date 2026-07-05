const API_BASE = "/api";
let accuracyChart = null;
let lastAnnouncedHour = null;
let hourlyTimeAnnouncementEnabled = false;

// JWT Authentication Logic
let authToken = localStorage.getItem("auth_token") || null;

// Override fetch to inject Auth token
const originalFetch = window.fetch;
window.fetch = async function() {
    let [resource, config] = arguments;
    if (typeof resource === 'string' && resource.startsWith(API_BASE) && !resource.endsWith("/login")) {
        if (!config) {
            config = {};
        }
        if (!config.headers) {
            config.headers = {};
        }
        if (authToken) {
            config.headers['Authorization'] = `Bearer ${authToken}`;
        }
    }
    const response = await originalFetch(resource, config);
    if (response.status === 401) {
        // Unauthorized, show login
        document.getElementById("login-modal").style.display = "flex";
        authToken = null;
        localStorage.removeItem("auth_token");
    }
    return response;
};

async function handleLogin(event) {
    event.preventDefault();
    const username = document.getElementById("login-username").value;
    const password = document.getElementById("login-password").value;
    
    try {
        // Use originalFetch to avoid the interceptor
        const response = await originalFetch(`${API_BASE}/login`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ username, password })
        });
        
        if (response.ok) {
            const data = await response.json();
            authToken = data.access_token;
            localStorage.setItem("auth_token", authToken);
            document.getElementById("login-modal").style.display = "none";
            document.getElementById("login-error").style.display = "none";
            showToast("Successfully authenticated", "success");
            
            // Reload dashboard data
            loadTransfersLog();
            loadEscrowQueue();
            loadFLMetrics();
            loadDPConfig();
            loadSimulationStatus();
        } else {
            document.getElementById("login-error").style.display = "block";
        }
    } catch (err) {
        console.error("Login failed:", err);
        document.getElementById("login-error").style.display = "block";
        document.getElementById("login-error").innerText = "Connection error";
    }
}

// Initial Setup when page loads
document.addEventListener("DOMContentLoaded", () => {
    updateLiveClock();
    setInterval(updateLiveClock, 1000);

    // Enable audio after first user interaction, to comply with browser autoplay rules
    const enableAudio = () => {
        hourlyTimeAnnouncementEnabled = true;
        document.removeEventListener("click", enableAudio);
        document.removeEventListener("keydown", enableAudio);
    };
    document.addEventListener("click", enableAudio);
    document.addEventListener("keydown", enableAudio);

    // Check auth, if authenticated load data
    if (!authToken) {
        document.getElementById("login-modal").style.display = "flex";
    } else {
        document.getElementById("login-modal").style.display = "none";
        // 1. Fetch initial status and populate UI
        loadTransfersLog();
        loadEscrowQueue();
        loadFLMetrics();
        loadDPConfig();
        loadFLConfig();
        loadSimulationStatus();
        loadThreshold();
    }
    
    // Set theme from localstorage or default to dark
    const savedTheme = localStorage.getItem("theme") || "dark";
    document.documentElement.setAttribute("data-theme", savedTheme);
    updateThemeIcon(savedTheme);
});

// Tab Navigation Switching
function switchTab(tabId) {
    // Hide all tabs
    document.querySelectorAll(".tab-content").forEach(el => el.classList.remove("active"));
    document.querySelectorAll(".nav-item").forEach(el => el.classList.remove("active"));
    
    // Show selected tab
    document.getElementById(tabId).classList.add("active");
    
    // Highlight nav item (handling custom links correctly)
    const activeBtn = Array.from(document.querySelectorAll(".nav-menu button"))
        .find(btn => btn.getAttribute("onclick").includes(tabId));
    if (activeBtn) activeBtn.classList.add("active");
    
    // Update Header titles dynamically
    const headerTitle = document.getElementById("page-title");
    const headerSub = document.getElementById("page-subtitle");
    
    if (tabId === "transfer-tab") {
        headerTitle.innerText = "Transfer Portal";
        headerSub.innerText = "Simulate edge-level transactions with real-time neural network evaluation";
    } else if (tabId === "escrow-tab") {
        headerTitle.innerText = "Escrow Center";
        headerSub.innerText = "Provider-level escrow interception queue management";
        loadEscrowQueue(); // Refresh when opening
    } else if (tabId === "federated-tab") {
        headerTitle.innerText = "Federated Learning Controller";
        headerSub.innerText = "Run secure, non-IID collaborative training across decentralized nodes";
        loadFLMetrics(); // Refresh when opening
    } else if (tabId === "settings-tab") {
        headerTitle.innerText = "Profile & Settings";
        headerSub.innerText = "Manage your authentication credentials";
        loadCurrentUsername(); // Load username when opening settings
    }
}

// Fetch and Render Transfers History Log
async function loadTransfersLog() {
    try {
        const response = await fetch(`${API_BASE}/transfers`);
        const transfers = await response.json();
        
        const tbody = document.getElementById("transfers-tbody");
        if (transfers.length === 0) {
            tbody.innerHTML = `<tr><td colspan="8" class="text-center text-muted">No simulated transfers yet.</td></tr>`;
            return;
        }
        
        tbody.innerHTML = transfers.map(tx => {
            let statusClass = "status-approved";
            let statusText = tx.status;
            if (tx.status === "HELD_IN_ESCROW") {
                statusClass = "status-escrow";
                statusText = "Held in Escrow";
            } else if (tx.status === "RELEASED_FROM_ESCROW") {
                statusClass = "status-released";
                statusText = "Released";
            } else if (tx.status === "BLOCKED") {
                statusClass = "status-blocked";
                statusText = "Blocked";
            }
            
            return `
                <tr>
                    <td><code>${tx.transfer_id}</code></td>
                    <td>${tx.timestamp}</td>
                    <td><strong>${tx.sender_id}</strong></td>
                    <td><code>${tx.receiver_phone}</code></td>
                    <td>GH₵${tx.amount}</td>
                    <td>${(tx.risk_report.sms_risk_score * 100).toFixed(1)}%</td>
                    <td><strong>${(tx.risk_report.total_risk_score * 100).toFixed(1)}%</strong></td>
                    <td><span class="badge-status ${statusClass}">${statusText}</span></td>
                </tr>
            `;
        }).join("");
    } catch (err) {
        console.error("Error loading transfers history:", err);
    }
}

// Export Audit Report
async function exportReport() {
    try {
        const response = await fetch(`${API_BASE}/reports/export`);
        if (!response.ok) {
            throw new Error("Failed to export report");
        }
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "transfers_audit_report.csv";
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    } catch (err) {
        showToast("Error exporting report", "error");
        console.error(err);
    }
}

// Fetch and Render Escrow Queue
async function loadEscrowQueue() {
    try {
        const response = await fetch(`${API_BASE}/escrow`);
        const escrowItems = await response.json();
        
        // Update badge counts in sidebar
        const badge = document.getElementById("escrow-badge");
        if (escrowItems.length > 0) {
            badge.innerText = escrowItems.length;
            badge.style.display = "inline-block";
        } else {
            badge.style.display = "none";
        }
        
        const tbody = document.getElementById("escrow-tbody");
        if (escrowItems.length === 0) {
            tbody.innerHTML = `<tr><td colspan="8" class="text-center text-muted">No transactions currently held in escrow.</td></tr>`;
            return;
        }
        
        tbody.innerHTML = escrowItems.map(item => `
            <tr>
                <td><code>${item.transfer_id}</code></td>
                <td><strong>${item.decision_by}</strong></td>
                <td><strong>${item.sender_id}</strong></td>
                <td><code>${item.receiver_phone}</code></td>
                <td>GH₵${item.amount}</td>
                <td><small>${item.reason}</small></td>
                <td><strong class="text-amber">${(item.risk_report.total_risk_score * 100).toFixed(1)}%</strong></td>
                <td>
                    <div style="display: flex; gap: 8px;">
                        <button class="btn btn-small btn-release" onclick="resolveEscrow('${item.transfer_id}', 'RELEASE')">
                            <i class="fa-solid fa-check"></i> Release
                        </button>
                        <button class="btn btn-small btn-block" onclick="resolveEscrow('${item.transfer_id}', 'BLOCK')">
                            <i class="fa-solid fa-ban"></i> Block
                        </button>
                    </div>
                </td>
            </tr>
        `).join("");
    } catch (err) {
        console.error("Error loading escrow queue:", err);
    }
}

// Handle Escrow Release or Block Action
async function resolveEscrow(transferId, action) {
    try {
        const response = await fetch(`${API_BASE}/escrow/${transferId}/action`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action })
        });
        
        if (response.ok) {
            showToast(`Transaction ${transferId} has been successfully ${action === "RELEASE" ? "released" : "blocked"}!`, "success");
            loadEscrowQueue();
            loadTransfersLog();
        } else {
            showToast("Failed to resolve escrow.", "error");
        }
    } catch (err) {
        showToast("Error communicating with backend API.", "error");
        console.error(err);
    }
}

// Form Transfer Submission
async function handleTransfer(event) {
    event.preventDefault();
    
    const sender_id = document.getElementById("sender-id").value;
    const receiver_phone = document.getElementById("receiver-phone").value.trim();
    const amount = parseFloat(document.getElementById("amount").value);
    const message = document.getElementById("message-text").value.trim();
    
    if (!sender_id) {
        showToast("Please select a sender device first.", "warning");
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/transfer`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ sender_id, receiver_phone, amount, message })
        });
        
        if (!response.ok) {
            const err = await response.json();
            showToast(err.detail || "Error initiating transfer", "error");
            return;
        }
        
        const tx = await response.json();
        
        // Show result display & Hide placeholder
        document.getElementById("result-placeholder").style.display = "none";
        const resultDisplay = document.getElementById("result-display");
        resultDisplay.style.display = "block";
        
        // Set verdict banner classes based on outcome
        const banner = document.getElementById("verdict-banner");
        const title = document.getElementById("verdict-title");
        const desc = document.getElementById("verdict-desc");
        const icon = document.getElementById("verdict-icon");
        
        banner.className = "verdict-banner"; // reset
        if (tx.status === "APPROVED") {
            banner.classList.add("approved");
            title.innerText = "AUTO-APPROVED";
            desc.innerText = "Transaction cleared automatically.";
            icon.innerHTML = `<i class="fa-solid fa-circle-check"></i>`;
            showToast("Transfer approved!", "success");
        } else if (tx.status === "HELD_IN_ESCROW") {
            banner.classList.add("escrow");
            title.innerText = "HELD IN ESCROW";
            desc.innerText = "Held for inspection at Provider node.";
            icon.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i>`;
            showToast("High risk detected! Transfer held in escrow.", "warning");
        }
        
        // Populate scores
        const risk = tx.risk_report;
        document.getElementById("risk-total").innerText = risk.total_risk_score.toFixed(2);
        document.getElementById("progress-total").style.width = `${risk.total_risk_score * 100}%`;
        
        document.getElementById("risk-sms").innerText = risk.sms_risk_score.toFixed(2);
        document.getElementById("progress-sms").style.width = `${risk.sms_risk_score * 100}%`;
        
        document.getElementById("risk-contact").innerText = risk.contact_risk_score.toFixed(2);
        document.getElementById("progress-contact").style.width = `${risk.contact_risk_score * 100}%`;
        
        document.getElementById("risk-amount").innerText = risk.amount_risk_score.toFixed(2);
        document.getElementById("progress-amount").style.width = `${risk.amount_risk_score * 100}%`;
        
        document.getElementById("decision-text").innerText = tx.reason;
        
        // Reload logs & queue
        loadTransfersLog();
        loadEscrowQueue();
        
    } catch (err) {
        showToast("Error processing request.", "error");
        console.error(err);
    }
}

// Triggered when selecting sender (can be used to autofill receiver if desired)
function onSenderChange() {
    const sender = document.getElementById("sender-id").value;
    // Visually highlight corresponding client dot in Topography (FL Tab)
    document.querySelectorAll(".client-dot").forEach(el => el.classList.remove("active"));
    const dot = Array.from(document.querySelectorAll(".client-dot")).find(el => el.title === sender);
    if (dot) {
        dot.style.borderColor = "var(--primary-color)";
        dot.style.boxShadow = "0 0 10px var(--primary-color)";
    }
}

// Fetch and Plot FL Metrics & Accuracy Line Graph
async function loadFLMetrics() {
    try {
        const response = await fetch(`${API_BASE}/federated/metrics`);
        const metrics = await response.json();
        
        if (metrics.length === 0) return;
        
        const latest = metrics[metrics.length - 1];
        document.getElementById("completed-rounds").innerText = latest.round;
        document.getElementById("sms-accuracy").innerText = `${(latest.sms_accuracy * 100).toFixed(2)}%`;
        document.getElementById("call-accuracy").innerText = `${(latest.call_accuracy * 100).toFixed(2)}%`;
        if (latest.holdout_sms_accuracy !== undefined) {
            document.getElementById("holdout-sms-accuracy").innerText = `${(latest.holdout_sms_accuracy * 100).toFixed(2)}%`;
        }
        if (latest.holdout_size !== undefined) {
            document.getElementById("holdout-size").innerText = latest.holdout_size;
        }
        
        // Update Topography tree node metrics or labels if required
        renderChart(metrics);
    } catch (err) {
        console.error("Error loading FL metrics:", err);
    }
}

// Trigger Federated Learning Round manually
async function runFLRound() {
    const btn = document.querySelector(".fl-actions button");
    btn.disabled = true;
    btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Aggregating Client Parameters...`;
    
    // Add pulsing animations to nodes during training
    document.querySelectorAll(".tree-node, .client-dot").forEach(node => {
        node.style.animation = "pulse 1.2s infinite";
    });
    
    try {
        const response = await fetch(`${API_BASE}/federated/round`, { method: "POST" });
        if (response.ok) {
            const metrics = await response.json();
            showToast(`Round ${metrics.round} complete! Holdout SMS: ${((metrics.holdout_sms_accuracy || metrics.sms_accuracy) * 100).toFixed(1)}%`, "success");
            
            // Reload logs and statistics
            await loadFLMetrics();
            await loadTransfersLog();
        } else {
            showToast("FL round aggregation failed.", "error");
        }
    } catch (err) {
        showToast("Error connecting to FL coordinator.", "error");
        console.error(err);
    } finally {
        btn.disabled = false;
        btn.innerHTML = `<i class="fa-solid fa-play"></i> Trigger FL Training Round`;
        // Clear animations
        document.querySelectorAll(".tree-node, .client-dot").forEach(node => {
            node.style.animation = "";
            node.style.boxShadow = "";
            node.style.borderColor = "";
        });
    }
}

// Render Line Chart using Chart.js
function renderChart(metrics) {
    const ctx = document.getElementById("accuracyChart").getContext("2d");
    
    const rounds = metrics.map(m => `Round ${m.round}`);
    const smsAcc = metrics.map(m => m.sms_accuracy * 100);
    const callAcc = metrics.map(m => m.call_accuracy * 100);
    
    if (accuracyChart) {
        accuracyChart.destroy();
    }
    
    const isDark = document.documentElement.getAttribute("data-theme") === "dark";
    const gridColor = isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.06)";
    const textColor = isDark ? "#a0aec0" : "#4b5563";
    
    accuracyChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: rounds,
            datasets: [
                {
                    label: 'SMS Phishing CNN Accuracy',
                    data: smsAcc,
                    borderColor: '#6366f1',
                    backgroundColor: 'rgba(99, 102, 241, 0.15)',
                    borderWidth: 3,
                    tension: 0.3,
                    fill: true
                },
                {
                    label: 'Call Detection MLP Accuracy',
                    data: callAcc,
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.15)',
                    borderWidth: 3,
                    tension: 0.3,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: {
                        color: textColor,
                        font: { family: 'Outfit', size: 12 }
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: gridColor },
                    ticks: { color: textColor, font: { family: 'Outfit' } }
                },
                y: {
                    grid: { color: gridColor },
                    ticks: { color: textColor, font: { family: 'Outfit' } },
                    min: 0,
                    max: 100
                }
            }
        }
    });
}

// Light / Dark Theme Toggler
function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute("data-theme");
    const nextTheme = currentTheme === "dark" ? "light" : "dark";
    
    document.documentElement.setAttribute("data-theme", nextTheme);
    localStorage.setItem("theme", nextTheme);
    updateThemeIcon(nextTheme);
    
    // Rerender chart to apply theme-appropriate gridlines
    if (accuracyChart) {
        loadFLMetrics();
    }
}

function updateThemeIcon(theme) {
    const icon = document.getElementById("theme-icon");
    if (theme === "light") {
        icon.className = "fa-solid fa-sun";
    } else {
        icon.className = "fa-solid fa-moon";
    }
}

function updateLiveClock() {
    const clock = document.getElementById("live-clock");
    if (!clock) return;

    const now = new Date();
    clock.textContent = now.toLocaleString("en-GH", {
        weekday: "short",
        day: "2-digit",
        month: "short",
        year: "numeric",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        hour12: true
    });

    announceHourlyTime(now);
}

function announceHourlyTime(now) {
    if (!hourlyTimeAnnouncementEnabled) return;

    const currentHour = now.getHours();
    const currentMinute = now.getMinutes();
    const currentSecond = now.getSeconds();

    if (currentMinute !== 0 || currentSecond !== 0) return;
    if (lastAnnouncedHour === currentHour) return;

    const hourText = now.toLocaleString("en-GH", { hour: "numeric", hour12: true });
    const announcement = `It is now ${hourText}`;

    const speech = new SpeechSynthesisUtterance(announcement);
    speech.lang = "en-US";
    speech.rate = 1;
    speech.pitch = 1;
    window.speechSynthesis.speak(speech);

    lastAnnouncedHour = currentHour;
}

// Toast Alert System
function showToast(message, type = "info") {
    const container = document.getElementById("toast-container");
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    
    let iconClass = "fa-circle-info";
    if (type === "success") iconClass = "fa-circle-check";
    if (type === "warning") iconClass = "fa-triangle-exclamation";
    if (type === "error") iconClass = "fa-ban";
    
    toast.innerHTML = `
        <i class="fa-solid ${iconClass}"></i>
        <span>${message}</span>
    `;
    
    container.appendChild(toast);
    
    // Automatically remove after 3.5 seconds
    setTimeout(() => {
        toast.style.animation = "slideOutRight 0.3s forwards";
        toast.addEventListener("animationend", () => {
            toast.remove();
        });
    }, 3500);
}

// ==========================================
// ADVANCED FEATURES INTEGRATIONS (DP & SIM)
// ==========================================
let simPollingInterval = null;

// Fetch DP configuration settings
async function loadDPConfig() {
    try {
        const response = await fetch(`${API_BASE}/federated/config`);
        const config = await response.json();
        
        const toggle = document.getElementById("dp-toggle");
        const noiseSlider = document.getElementById("dp-noise");
        const noiseVal = document.getElementById("dp-noise-val");
        const noiseGroup = document.getElementById("dp-noise-group");
        
        toggle.checked = config.dp_enabled;
        noiseSlider.value = config.dp_noise;
        noiseVal.innerText = config.dp_noise.toFixed(2);
        
        if (config.dp_enabled) {
            noiseGroup.style.display = "block";
        } else {
            noiseGroup.style.display = "none";
        }
        
        updateEpsilonDisplay(config.dp_enabled, config.dp_noise);
    } catch (err) {
        console.error("Error loading DP config:", err);
    }
}

// Send updated DP config to backend
async function updateDPConfig() {
    const enabled = document.getElementById("dp-toggle").checked;
    const noise = parseFloat(document.getElementById("dp-noise").value);
    const noiseGroup = document.getElementById("dp-noise-group");
    
    if (enabled) {
        noiseGroup.style.display = "block";
    } else {
        noiseGroup.style.display = "none";
    }
    
    updateEpsilonDisplay(enabled, noise);
    await updateFLConfig(enabled, noise);
}

async function loadFLConfig() {
    try {
        const response = await fetch(`${API_BASE}/federated/config`);
        const config = await response.json();
        const algoSelect = document.getElementById("fl-algorithm");
        const secureToggle = document.getElementById("secure-agg-toggle");
        if (algoSelect) algoSelect.value = config.fl_algorithm || "fedprox";
        if (secureToggle) secureToggle.checked = config.secure_aggregation !== false;
    } catch (err) {
        console.error("Error loading FL config:", err);
    }
}

async function updateFLConfig(dpEnabledOverride, dpNoiseOverride) {
    const dpEnabled = dpEnabledOverride !== undefined ? dpEnabledOverride : document.getElementById("dp-toggle").checked;
    const dpNoise = dpNoiseOverride !== undefined ? dpNoiseOverride : parseFloat(document.getElementById("dp-noise").value);
    const flAlgorithm = document.getElementById("fl-algorithm")?.value || "fedprox";
    const secureAgg = document.getElementById("secure-agg-toggle")?.checked ?? true;

    try {
        const response = await fetch(`${API_BASE}/federated/config`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                fl_algorithm: flAlgorithm,
                fedprox_mu: 0.01,
                secure_aggregation: secureAgg,
                dp_enabled: dpEnabled,
                dp_noise: dpNoise,
                dp_clip: 1.0
            })
        });
        
        if (response.ok) {
            showToast(`FL config updated (${flAlgorithm}, secure agg: ${secureAgg})`, "info");
        }
    } catch (err) {
        showToast("Error updating FL config on backend.", "error");
        console.error(err);
    }
}

// Triggered on slider change in index.html
function onDPNoiseSliderInput() {
    const noiseVal = parseFloat(document.getElementById("dp-noise").value);
    document.getElementById("dp-noise-val").innerText = noiseVal.toFixed(2);
    updateEpsilonDisplay(true, noiseVal);
}

// Helper to estimate Local Differential Privacy (LDP) Epsilon budget
function updateEpsilonDisplay(enabled, noiseVal) {
    const budgetVal = document.getElementById("dp-budget-val");
    if (!enabled) {
        budgetVal.className = "text-muted";
        budgetVal.innerHTML = `&epsilon; = &infin; (No Privacy)`;
    } else {
        // Simple LDP estimation: epsilon = clip_norm / noise
        // For clip_norm = 1.0, epsilon = 1.0 / noise_multiplier
        const epsilon = (1.0 / noiseVal).toFixed(2);
        budgetVal.className = "text-purple";
        budgetVal.innerHTML = `&epsilon; = ${epsilon} (Local DP)`;
    }
}

// Fetch simulation traffic status on start
async function loadSimulationStatus() {
    try {
        const response = await fetch(`${API_BASE}/simulation/status`);
        const status = await response.json();
        
        const toggle = document.getElementById("traffic-toggle");
        const statusLabel = document.getElementById("traffic-status-label");
        
        toggle.checked = status.running;
        if (status.running) {
            statusLabel.innerText = "Active";
            statusLabel.style.color = "var(--color-emerald)";
            startSimPolling();
        } else {
            statusLabel.innerText = "Disabled";
            statusLabel.style.color = "var(--text-secondary)";
            stopSimPolling();
        }
    } catch (err) {
        console.error("Error loading simulation status:", err);
    }
}

// Toggle Simulation Traffic ON/OFF
async function toggleTrafficSimulation() {
    const toggle = document.getElementById("traffic-toggle");
    const statusLabel = document.getElementById("traffic-status-label");
    const action = toggle.checked ? "start" : "stop";
    
    try {
        const response = await fetch(`${API_BASE}/simulation/${action}`, { method: "POST" });
        const res = await response.json();
        
        if (response.ok) {
            if (toggle.checked) {
                statusLabel.innerText = "Active";
                statusLabel.style.color = "var(--color-emerald)";
                showToast("Background traffic simulation started!", "success");
                startSimPolling();
            } else {
                statusLabel.innerText = "Disabled";
                statusLabel.style.color = "var(--text-secondary)";
                showToast("Background traffic simulation stopped.", "info");
                stopSimPolling();
            }
        }
    } catch (err) {
        showToast("Error updating traffic simulation.", "error");
        console.error(err);
    }
}

function startSimPolling() {
    if (simPollingInterval) clearInterval(simPollingInterval);
    // Poll the transfers list and escrow queue every 3.5 seconds to auto-refresh UI
    simPollingInterval = setInterval(() => {
        loadTransfersLog();
        loadEscrowQueue();
    }, 3500);
}

function stopSimPolling() {
    if (simPollingInterval) {
        clearInterval(simPollingInterval);
        simPollingInterval = null;
    }
}

// ==========================================
// USER AUTHENTICATION & SETTINGS
// ==========================================
function logout() {
    authToken = null;
    localStorage.removeItem("auth_token");
    document.getElementById("login-modal").style.display = "flex";
    showToast("Logged out successfully.", "info");
}

async function loadCurrentUsername() {
    try {
        const response = await fetch(`${API_BASE}/users/me`);
        if (response.ok) {
            const data = await response.json();
            document.getElementById("current-username").value = data.username;
        }
    } catch (err) {
        console.error("Error loading current username:", err);
    }
}

function openUpdateCredentialsModal(event) {
    event.preventDefault();
    document.getElementById("update-credentials-modal").style.display = "flex";
    document.getElementById("update-credentials-form").reset();
    document.getElementById("update-credentials-error").style.display = "none";
}

function closeUpdateCredentialsModal() {
    document.getElementById("update-credentials-modal").style.display = "none";
    document.getElementById("update-credentials-form").reset();
    document.getElementById("update-credentials-error").style.display = "none";
}

async function handleUpdateCredentials(event) {
    event.preventDefault();
    
    const currentPassword = document.getElementById("update-current-password").value;
    const newUsername = document.getElementById("update-new-username").value;
    const newPassword = document.getElementById("update-new-password").value;
    
    if (!currentPassword) {
        document.getElementById("update-credentials-error").innerText = "Current password is required";
        document.getElementById("update-credentials-error").style.display = "block";
        return;
    }
    
    if (!newUsername && !newPassword) {
        document.getElementById("update-credentials-error").innerText = "Please enter at least a new username or password";
        document.getElementById("update-credentials-error").style.display = "block";
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/users/me`, {
            method: "PUT",
            headers: { 
                "Content-Type": "application/json",
                "Authorization": `Bearer ${authToken}`
            },
            body: JSON.stringify({
                current_password: currentPassword,
                new_username: newUsername || null,
                new_password: newPassword || null
            })
        });
        
        if (response.ok) {
            const data = await response.json();
            // Update token with new credentials
            authToken = data.access_token;
            localStorage.setItem("auth_token", authToken);
            
            showToast("Credentials updated successfully!", "success");
            closeUpdateCredentialsModal();
            
            // Reload current username
            loadCurrentUsername();
            
            // Clear login fields so if they return later, they'll need to re-authenticate
            document.getElementById("login-username").value = "";
            document.getElementById("login-password").value = "";
        } else {
            const err = await response.json();
            document.getElementById("update-credentials-error").innerText = err.detail || "Failed to update credentials";
            document.getElementById("update-credentials-error").style.display = "block";
        }
    } catch (err) {
        console.error("Error updating credentials:", err);
        document.getElementById("update-credentials-error").innerText = "Connection error";
        document.getElementById("update-credentials-error").style.display = "block";
    }
}

// ── Hamburger Sidebar ─────────────────────────────────────────
(function () {
  const sidebar = document.getElementById("sidebar");
  const hamburger = document.getElementById("hamburgerBtn");
  if (!sidebar || !hamburger) return;

  function openSidebar() {
    sidebar.classList.add("sidebar-open");
    hamburger.classList.add("hidden");
  }

  function closeSidebar() {
    sidebar.classList.remove("sidebar-open");
    hamburger.classList.remove("hidden");
  }

  // Hover over hamburger → open
  hamburger.addEventListener("mouseenter", openSidebar);

  // Cursor leaves sidebar → close
  sidebar.addEventListener("mouseleave", function (e) {
    if (!hamburger.contains(e.relatedTarget)) {
      closeSidebar();
    }
  });

  // Cursor leaves hamburger without entering sidebar → close
  hamburger.addEventListener("mouseleave", function (e) {
    if (!sidebar.contains(e.relatedTarget)) {
      closeSidebar();
    }
  });
})();
